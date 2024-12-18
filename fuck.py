# %%
import math
from typing import List, Optional, Tuple, Union

import datasets
import torch
import numpy as np
import torch.nn as nn
from datasets import load_dataset
# import logging
import copy

from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaConfig,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    HfArgumentParser
)

from modeling_qwen2 import (
    Qwen2RMSNorm, 
    Qwen2RotaryEmbedding, 
    Qwen2MLP, 
    Qwen2Attention, 
    Qwen2FlashAttention2, 
    Qwen2SdpaAttention, 
    Qwen2DecoderLayer, 
    Qwen2PreTrainedModel, 
    Qwen2Model, 
    Qwen2ForCausalLM,
    logger
)

from configuration_qwen2 import Qwen2Config

# Configure logger
# logger.basicConfig(level=logger.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# %%
# from utils import are_tokenizers_same
# are_tokenizers_same(
#     paths = [
#         "/workspace/models/Arcee-VyLinh/",
#         "/workspace/models/Qwen2.5-Coder-3B/"
#     ]
# )

# %%
def load_layer(path, layer_idx=33):
	state_dict = {}
	shard_paths = [f for f in os.listdir(path) if f.endswith('.safetensors')]
	for shard_path in sorted(shard_paths, key=lambda x: int(x.split('-')[1])):
		apath = os.path.join(path, shard_path)
		with safe_open(apath, framework="pt", device="cpu") as f:
			for key in f.keys():
				if f"layers.{str(layer_idx)}." in key:
					state_dict[key] = f.get_tensor(key)
	return state_dict

def strip_prefix(state_dict, prefix="model.layers."):
    """Strips 'model.layers.*.' prefix from 'input_layernorm.weight' keys."""
    return {
      k.replace(f"{prefix}{k.split('.')[2]}.", "") if k.startswith(prefix)
      else k: v for k, v in state_dict.items()
    }

# %%
def lerp(
    t: float, v0: Union[np.ndarray, torch.Tensor], v1: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    return (1 - t) * v0 + t * v1

def weighted_sum(
    factors: List[float], 
    tensors: Union[List[np.ndarray], List[torch.Tensor]]
) -> Union[np.ndarray, torch.Tensor]:
    result = 0.0
    # for factor, tensor in zip(factors, tensors):
    #     result += factor * tensor
    return sum([tensor * factor for tensor, factor in zip(tensors, factors)])

def merge_modules(modules, factors):
    """
    This is only applicable for cases where a static set of scalars
    playing as merging factor for every submodules of the passed module.
    Not recommend for fine-grained usecases.
    """
    module_out = copy.deepcopy(modules[0])
    out_dict = module_out.state_dict()
    
    tensor_dicts_list = [m.state_dict() for m in modules]
    tensor_names = [key for key in tensor_dicts_list[0].keys()]
    
    for tensor_name in tensor_names:
        tensors_list = [tensor_dicts_list[i][tensor_name]
                       for i in range(len(modules))]
        tensor_computed = (
            weighted_sum(
                factors=factors,
                tensors=tensors_list
            )
            .to(tensors_list[0].dtype)
            .to(tensors_list[0].device)
        )
        out_dict[tensor_name] = tensor_computed
    module_out.load_state_dict(out_dict)
    return module_out

def merge_linears(modules, weight_factors, bias_factors):
    param_names = sorted([name for name, _ in modules[0].named_parameters()])
    for module in modules:
        other_param_names = sorted([name for name, _ in module.named_parameters()])
        assert param_names == other_param_names, "Mismatch tensor names."
        
    module_out = copy.deepcopy(modules[0])
    out_dict = module_out.state_dict()
    
    tensor_dicts_list = [m.state_dict() for m in modules]
    tensor_names = [key for key in tensor_dicts_list[0].keys()]
    
    for tensor_name in tensor_names:
        tensors_list = [tensor_dicts_list[i][tensor_name]
                       for i in range(len(modules))]
        if "weight" in tensor_name:
            factors = weight_factors
        elif "bias" in tensor_name:
            factors = bias_factors
        else:
            raise ValueError("Hey this tensor is neither weight or bias.")
            
        tensor_computed = (
            weighted_sum(
                factors=factors,
                tensors=tensors_list
            )
            .to(tensors_list[0].dtype)
            .to(tensors_list[0].device)
        )
        out_dict[tensor_name] = tensor_computed
    module_out.load_state_dict(out_dict)
    return module_out

# %%
def place_masks(target_module, ref_modules, factors):
    """
    Recursively replaces normal components with masked components.
    
    Args:
      module: The module in which to replace layers.
    """
    assert len(ref_modules) == len(factors)
    num_components = len(ref_modules)
    for name, target_child in target_module.named_children():
        ref_children = [getattr(module, name) for module in ref_modules]
        modes = ["scalar" for _ in ref_children]
        if isinstance(target_child, nn.Linear):
            new_module = LinearsWithMasks(
                linears=ref_children,
                weight_modes=["scalar"] * num_components,
                weight_values=factors,
                bias_modes=["scalar"] * num_components,
                bias_values=factors,
            )
            setattr(target_module, name, new_module)
        elif isinstance(target_child, nn.Embedding):
            setattr(target_module, name, EmbeddingsWithMasks(
                ref_children, modes, factors
            ))
        elif type(target_child).__name__ == Qwen2RMSNorm.__name__:
            # print("Hehe placing masks to a cutie RMSNorm")
            setattr(target_module, name, RMSNormsWithMasks(
                ref_children, modes, factors
            ))
        else:
            place_masks(target_child, ref_children, factors)

# %%
from transformers import GenerationConfig, TextStreamer
def generate(prompt, model, tokenizer, max_new_tokens=1024):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    model.eval()
    with torch.no_grad():
        generation_config = GenerationConfig(
            repetition_penalty=1.13,
            max_new_tokens=max_new_tokens,
            temperature=0.4,
            top_p=0.95,
            # top_k=20,
            # bos_token_id=tokenizer.bos_token_id,
            # eos_token_id=tokenizer.eos_token_id,
            # eos_token_id=0, # for open-end generation.
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
        )
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        generated = model.generate(
            inputs=input_ids,
            generation_config=generation_config,
            streamer=streamer,
        )
    gen_tokens = generated["sequences"].cpu()[:, len(input_ids[0]):]
    output = tokenizer.batch_decode(gen_tokens)[0]
    output = output.split(tokenizer.eos_token)[0]
    return output.strip()

def get_logits(text, model, tokenizer):
    input_ids = tokenizer(text, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        logits = model(**input_ids).logits
    return logits

def get_hidden_states(text, model, tokenizer):
    input_ids = tokenizer(text, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        outputs = model(**input_ids, output_hidden_states=True)
    return outputs

# %%
class MaskConfig(PretrainedConfig):
    def __init__(
        self,
        mode: str = None,
        value: Union[float, torch.Tensor] = None,
        size: torch.Size = None,
        **kwargs,
    ):
        self.mode = mode
        self.value = value
        self.size = size
        super().__init__(**kwargs)

class Mask(nn.Module):
    def __init__(
        self, 
        mask_config: MaskConfig
    ):
        super().__init__()
        """
        now only support mode == scalar
        """
        self.mode = mask_config.mode
        if mask_config.mode == "scalar":
            value = mask_config.value if mask_config.value is not None else 1
            self.weight = nn.Parameter(torch.tensor(value))
        else:
            raise ValueError(f"Unsupported mask mode: {mask_config.mode}")
            
        self.size = mask_config.size ## Full size of the mask after broadcast.
        if self.size is not None:
            try:
                self.weight * torch.rand(self.size)
            except RuntimeError:
                print("mask initialized with an incompatible shape.")

    def forward(self, x):
        """
        Be really careful here (though I do not think it matters that much),
        When testing, it's important that the masking operation is implemented
        with `x = self.weight * x` instead of `x = x * self.weight`.

        Neither of those two implementation is superior, however I need to be
        consistent when doing testing because the phenonmenon above could lead
        to some number imprecision, which may fail `torch.testing.assert_close`
        """
        if self.size is None:
            return self.weight * x
        else:
            if self.size != x.shape:
                print("The shape of input does not match that of the mask.")
            return self.weight * x

# %%
class LinearWithMask(nn.Module):
    def __init__(self, linear, weight_mask_config: MaskConfig, bias_mask_config: MaskConfig = None):
        super().__init__()
        self.linear = linear
        self.weight_mask_config = weight_mask_config
        self.bias_mask_config = bias_mask_config

        if linear.weight.shape != weight_mask_config.size:
            print("Weight mask shape is not compatible with linear, reinitializing...")
            self.weight_mask_config.size = linear.weight.shape
        self.weight_mask = Mask(self.weight_mask_config)

        if linear.bias is not None and bias_mask_config is not None:
            if linear.bias.shape != bias_mask_config.size:
                print("Bias mask shape is not compatible with linear, reinitializing...")
                self.bias_mask_config.size = linear.bias.shape
            self.bias_mask = Mask(self.bias_mask_config)
        else:
            self.bias_mask = None

    def forward(self, x):
        masked_weight = self.weight_mask(self.linear.weight)
        if self.linear.bias is not None and self.bias_mask is not None:
            masked_bias = self.bias_mask(self.linear.bias)
        else:
            masked_bias = self.linear.bias
        return nn.functional.linear(x, masked_weight, masked_bias)

class LinearsWithMasks(nn.Module):
    def __init__(
        self,
        linears: List[nn.Module],
        weight_modes: List[str] = ["scalar"],
        weight_values: List[float] = None,
        bias_modes: List[str] = ["scalar"],
        bias_values: List[float] = None,
    ):
        super().__init__()
        
        if not all(isinstance(linear, nn.Linear) for linear in linears):
            raise ValueError("All elements in 'linears' must be instances of nn.Linear.")

        weight_sizes = [linear.weight.shape for linear in linears]
        bias_sizes = [linear.bias.shape if linear.bias is not None else None for linear in linears]
        
        if weight_values is None or len(weight_values) != len(linears):
            raise ValueError(f"weight_values for masks: {weight_values} do not match with linear layers: {linears}")
        if bias_values is None:
            bias_values = [None] * len(linears)
        if len(bias_values) != len(linears):
            raise ValueError(f"bias_values for masks: {bias_values} do not match with linear layers: {linears}")

        weight_mask_configs = [
            MaskConfig(mode, value, size)
            for mode, value, size in zip(weight_modes, weight_values, weight_sizes)
        ]
        bias_mask_configs = [
            MaskConfig(mode, value, size) if size is not None else None
            for mode, value, size in zip(bias_modes, bias_values, bias_sizes)
        ]

        self.masked_linears = nn.ModuleList(
            [LinearWithMask(linear, weight_mask_config, bias_mask_config)
             for linear, weight_mask_config, bias_mask_config 
             in zip(linears, weight_mask_configs, bias_mask_configs)]
        )

    def forward(self, x):
        weights = [linear.weight_mask(linear.linear.weight) 
                   for linear in self.masked_linears]
        # merged_weight = torch.sum(torch.stack(weights), dim=0)
        merged_weight = sum(weights)

        biases = [
            linear.bias_mask(linear.linear.bias) 
            if linear.linear.bias is not None and linear.bias_mask is not None 
            else linear.linear.bias for linear in self.masked_linears
        ]
        biases = [
            b if b is not None
            else torch.zeros_like(weights[0][:, 0])
            for b in biases
        ]
        # merged_bias = torch.sum(torch.stack(biases))
        merged_bias = sum(biases)
        if all(b is None for b in biases):
            merged_bias = None

        return nn.functional.linear(x, merged_weight, merged_bias)

# %%
class RMSNormWithMask(nn.Module):
    def __init__(self, rms_norm: Qwen2RMSNorm, mask_config: MaskConfig):
        super().__init__()
        self.rms_norm = rms_norm
        self.mask_config = mask_config
        if rms_norm.weight.shape != mask_config.size:
            print("Mask shape is not compatible with RMSNorm, reinitializing...")
        self.mask_config.size = rms_norm.weight.shape
        self.mask = Mask(self.mask_config)

    def forward(self, hidden_states):
        masked_weight = self.mask(self.rms_norm.weight)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.rms_norm.variance_epsilon)
        return masked_weight * hidden_states.to(input_dtype)

class RMSNormsWithMasks(nn.Module):
    def __init__(
        self,
        rms_norms: List[Qwen2RMSNorm],
        modes: List[str] = ["scalar"],
        values: List[float] = None
    ):
        super().__init__()
        sizes = [rms_norm.weight.shape for rms_norm in rms_norms]
        if values is None or len(values) != len(rms_norms):
            raise ValueError(f"values for masks: {values} do not match with RMSNorm layers: {rms_norms}")

        mask_configs = [
            MaskConfig(mode, value, size)
            for mode, value, size in zip(modes, values, sizes)
        ]
        self.masked_rms_norms = nn.ModuleList(
            [RMSNormWithMask(rms_norm, mask_config)
             for rms_norm, mask_config in zip(rms_norms, mask_configs)]
        )

    def forward(self, hidden_states):
        weights = [rms.mask(rms.rms_norm.weight) for rms in self.masked_rms_norms]
        merged_weight = sum(weights)
        variance_epsilon = self.masked_rms_norms[0].rms_norm.variance_epsilon
        for rms in self.masked_rms_norms:
            assert variance_epsilon == rms.rms_norm.variance_epsilon, "Variance epsilon among models must be consistent"
        
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        return merged_weight * hidden_states.to(input_dtype)

# %%
class EmbeddingWithMask(nn.Module):
    def __init__(self, embedding: nn.Embedding, mask_config: MaskConfig):
        super().__init__()
        self.embedding = embedding
        self.mask_config = mask_config
        if embedding.weight.shape != mask_config.size:
            print("Mask shape is not compatible with Embedding, reinitializing...")
        self.mask_config.size = embedding.weight.shape
        self.mask = Mask(self.mask_config)

    def forward(self, input_ids):
        masked_weight = self.mask(self.embedding.weight)
        return nn.functional.embedding(
            input_ids,
            masked_weight,
            padding_idx=self.embedding.padding_idx,
            max_norm=self.embedding.max_norm,
            norm_type=self.embedding.norm_type,
            scale_grad_by_freq=self.embedding.scale_grad_by_freq,
            sparse=self.embedding.sparse,
        )

class EmbeddingsWithMasks(nn.Module):
    def __init__(
        self,
        embeddings: List[nn.Embedding],
        modes: List[str] = ["scalar"],
        values: List[float] = None
    ):
        super().__init__()
        sizes = [embedding.weight.shape for embedding in embeddings]
        if values is None or len(values) != len(embeddings):
            raise ValueError(f"values for masks: {values} do not match with Embedding layers: {embeddings}")

        mask_configs = [
            MaskConfig(mode, value, size)
            for mode, value, size in zip(modes, values, sizes)
        ]
        self.masked_embeddings = nn.ModuleList(
            [EmbeddingWithMask(embedding, mask_config)
             for embedding, mask_config in zip(embeddings, mask_configs)]
        )

    def forward(self, input_ids):
        weights = [emb.mask(emb.embedding.weight) for emb in self.masked_embeddings]
        merged_weight = sum(weights)
        an_embedding = self.masked_embeddings[0].embedding
        for other in self.masked_embeddings:
            other_embedding = other.embedding
            assert an_embedding.padding_idx == other_embedding.padding_idx
            assert an_embedding.max_norm == other_embedding.max_norm
            assert an_embedding.norm_type == other_embedding.norm_type
            assert an_embedding.scale_grad_by_freq == other_embedding.scale_grad_by_freq
            assert an_embedding.sparse == other_embedding.sparse
            
        return nn.functional.embedding(
            input_ids,
            merged_weight,
            padding_idx=an_embedding.padding_idx,
            max_norm=an_embedding.max_norm,
            norm_type=an_embedding.norm_type,
            scale_grad_by_freq=an_embedding.scale_grad_by_freq,
            sparse=an_embedding.sparse,
        )

# %%
class MergerConfig(PretrainedConfig):
    def __init__(
        self,
        model_paths: List[str] = None,
        **kwargs,
    ):
        self.model_paths = model_paths
        super().__init__(**kwargs)

class Merger(PreTrainedModel):
    def __init__(self, merge_config):
        super().__init__(merge_config)
        """
        Need to check whether models are mergeable (having some sort of the same config)
        """
        self.merge_config = merge_config
        self.num_models = len(merge_config.model_paths)
        self.configs = [
            # AutoConfig.from_pretrained(path) 
            Qwen2Config.from_pretrained(path)
            for path in merge_config.model_paths
        ]
        # self.merger = Qwen2ForCausalLM(self.config)
        self.models = nn.ModuleList([
            Qwen2ForCausalLM.from_pretrained(
            # AutoModelForCausalLM.from_pretrained(
                merge_config.model_paths[i], 
                config=self.configs[i],
                torch_dtype=torch.bfloat16
            ) 
            for i in range(self.num_models)
        ])
        # self.__post_init__(merge_config)
        
    def __post_init__(self, merge_config, factors):
        # dummy_config = copy.deepcopy(self.configs[0])
        # dummy_config.update({"hidden_size": 1, "intermediate_size": 1})
        # self.merger = AutoModelForCausalLM.from_config(dummy_config)
        self.merger = copy.deepcopy(self.models[0])
        place_masks(self.merger, self.models, factors=factors)
        
    def forward(self, tensor, labels=None):
        pass

# %%
merge_config = MergerConfig(
    model_paths = [
        "/workspace/models/Arcee-VyLinh/",
        "/workspace/models/Qwen2.5-Coder-3B/"
    ]
)
merge_config

# %%
merger = Merger(merge_config)

# %%
merger = merger.to("cuda:0")

# %%
merger.__post_init__(merge_config, factors=[0.0, 1.0])

# %%
tokenizer = AutoTokenizer.from_pretrained(merge_config.model_paths[0])

# %%
system = "You are a helpful assistant."
prompt = "Continue this text: A dog is a cat"
messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
text

# %%
# answer = generate(text, merger.merger, tokenizer, max_new_tokens=100)

# %%
# answer = generate(text, merger.models[1], tokenizer, max_new_tokens=100)

# %%
# embeddings_with_masks = merger.merger.model.embed_tokens
# embedding1 = merger.models[1].model.embed_tokens

# %%
# torch.manual_seed(42)
# num_embeddings = embedding1.weight.data.shape[0]
# device = merger.device
# input_ids = torch.randint(0, num_embeddings, (2, 5)).to(device=device)  # Example input_ids
# input_ids

# %%
# o_merged = embeddings_with_masks(input_ids)
# o1 = embedding1(input_ids)
# torch.allclose(o_merged, o1, rtol=1e-10, atol=1e-10)

# %%
# logits_merged = get_logits(text, merger.merger, tokenizer)
# logits1 = get_logits(text, merger.models[1], tokenizer)

# %%
# logits_merged, logits1

# %%
outputs_merged = get_hidden_states(text, merger.merger, tokenizer)
outputs1 = get_hidden_states(text, merger.models[1], tokenizer)

# %%

# input_ids = tokenizer(text, return_tensors="pt").to(merger.device)
# merger.merger.model(**input_ids)

# %%
# num_decoders = len(merger.models[0].model.layers)
# num_decoders

# %%
# for i in range(num_decoders):
#     a = outputs_merged.hidden_states[i]
#     b = outputs1.hidden_states[i]
#     torch.testing.assert_close(a, b, rtol=0, atol=0)
#     logger.info(f"Hidden states at layer {i} of two models are identical.")

# %%
# merger.merger.forward?