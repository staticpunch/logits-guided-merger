import math
from typing import List, Optional, Tuple, Union

import datasets
import torch
import numpy as np
import torch.nn as nn
from datasets import load_dataset
import logging
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
    Qwen2ForCausalLM
)

from configuration_qwen2 import Qwen2Config
from utils import are_tokenizers_same

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
are_tokenizers_same(
    paths = [
        "/workspace/models/Arcee-VyLinh/",
        "/workspace/models/Qwen2.5-Coder-3B/"
    ]
)

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
            self.weight = nn.Parameter(torch.tensor(value)) # Corrected typo here
        else:
            raise ValueError(f"Unsupported mask mode: {mask_config.mode}")
            
        self.size = mask_config.size ## Full size of the mask after broadcast.
        try:
            self.weight * torch.rand(self.size)
        except RuntimeError:
            print("mask initialized with an incompatible shape.")

    def forward(self, x):
        x = self.weight * x
        return x

class LinearWithMask(nn.Module):
    def __init__(self, linear, mask_config: MaskConfig):
        super().__init__()
        self.linear = linear
        self.mask_config = mask_config
        if linear.weight.shape != mask_config.size:
            print("Mask shape is not imcompatible with linear, reinitializing...")
        self.mask_config.size = linear.weight.shape
        self.mask = Mask(self.mask_config)
        
    def forward(self, x):
        masked_linear = self.mask(self.linear.weight)
        return nn.functional.linear(x, masked_linear, self.linear.bias)

class LinearsWithMasks(nn.Module):
    def __init__(
        self, 
        linears: List[nn.Module], 
        modes: List[str] = ["scalar"], 
        values: List[float] = None
    ):
        super().__init__()
        sizes = [linear.weight.shape for linear in linears]
        if values is None or len(values) != len(linears):
            raise ValueError(f"values for masks: {values} do not match with linear layers: {linears}")
            
        mask_configs = [
            MaskConfig(mode, value, size) 
            for mode, value, size in zip(modes, values, sizes)
        ]
        self.masked_linears = nn.ModuleList(
            [LinearWithMask(linear, mask_config) 
             for linear, mask_config in zip(linears, mask_configs)]
        )
        
    def forward(self, x):
        output = 0.0
        for masked_linear in self.masked_linears:
            output += masked_linear(x)
        return output

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
        output = 0.0
        for masked_rms_norm in self.masked_rms_norms:
            output += masked_rms_norm(hidden_states)
        return output

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
        output = 0.0
        for masked_embedding in self.masked_embeddings:
            output += masked_embedding(input_ids)
        return output

def place_masks(target_module, ref_modules):
    """
    Recursively replaces normal components with masked components.
    
    Args:
      module: The module in which to replace layers.
    """
    for name, target_child in target_module.named_children():
        ref_children = [getattr(module, name) for module in ref_modules]
        modes = ["scalar" for _ in ref_children]
        values = [0.0 for _ in ref_children]
        values[0] = 1.0
        if isinstance(target_child, nn.Linear):
            setattr(target_module, name, LinearsWithMasks(
                ref_children, modes, values
            ))
        elif isinstance(target_child, nn.Embedding):
            setattr(target_module, name, EmbeddingsWithMasks(
                ref_children, modes, values
            ))
        elif type(target_child).__name__ == Qwen2RMSNorm.__name__:
            setattr(target_module, name, RMSNormsWithMasks(
                ref_children, modes, values
            ))
        else:
            place_masks(target_child, ref_children)

class MergerConfig(PretrainedConfig):
    def __init__(
        self,
        model_paths: List[str] = None,
        **kwargs,
    ):
        self.model_paths = model_paths
        super().__init__(**kwargs)

class DecoderMerger(PreTrainedModel):
    def __init__(self, merge_config):
        super().__init__(merge_config)
        """
        Need to check whether models are mergeable (having some sort of the same config)
        """
        self.merge_config = merge_config
        self.configs = [Qwen2Config.from_pretrained(path) 
                        for path in merge_config.model_paths]
        
        # self.merger = Qwen2ForCausalLM(self.config)
        self.decoders = nn.ModuleList(
            Qwen2DecoderLayer(config, layer_idx=1) for config in self.configs
        )
        for i in range(len(self.decoders)):
            path = merge_config.model_paths[i]
            state_dict = load_layer(path, layer_idx=1)
            state_dict = strip_prefix(state_dict)
            self.decoders[i].load_state_dict(
                state_dict=state_dict
            )
        self.__post_init__(merge_config)
        
    def __post_init__(self, merge_config):
        self.merger = copy.deepcopy(self.decoders[0])
        place_masks(self.merger, self.decoders)
        
    def forward(self, tensor, labels=None):
        pass
        
class Merger(PreTrainedModel):
    def __init__(self, merge_config):
        super().__init__(merge_config)
        """
        Need to check whether models are mergeable (having some sort of the same config)
        """
        self.merge_config = merge_config
        self.num_models = len(merge_config.model_paths)
        self.configs = [
            AutoConfig.from_pretrained(path) 
            for path in merge_config.model_paths
        ]
        # self.merger = Qwen2ForCausalLM(self.config)
        self.models = nn.ModuleList([
            AutoModelForCausalLM.from_pretrained(
                merge_config.model_paths[i], 
                config=self.configs[i],
                torch_dtype=torch.bfloat16
            ) 
            for i in range(self.num_models)
        ])
        self.__post_init__(merge_config)
        
    def __post_init__(self, merge_config):
        # dummy_config = copy.deepcopy(self.configs[0])
        # dummy_config.update({"hidden_size": 1, "intermediate_size": 1})
        # self.merger = AutoModelForCausalLM.from_config(dummy_config)
        self.merger = copy.deepcopy(self.models[0])
        place_masks(self.merger, self.models)
        
    def forward(self, tensor, labels=None):
        pass
        