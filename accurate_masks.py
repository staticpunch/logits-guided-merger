import math
from typing import List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import datasets
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import logging
import copy
import gc
import os

from datasets import load_dataset
from accelerate import dispatch_model, infer_auto_device_map
from tqdm import tqdm

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
    HfArgumentParser,
    default_data_collator,
    is_torch_xla_available,
    set_seed,
)

from transformers.modeling_utils import (
    is_fsdp_enabled, 
    is_deepspeed_zero3_enabled
)

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast
)

from utils import (
    generate, 
    get_hidden_states, 
    get_logits,
    free_memory
)

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def free_memory():
    if not torch.cuda.is_available():
        logger.info("CUDA is not available. No GPU memory to free.")
        return
        
    initial_memory = torch.cuda.memory_allocated()
    logger.info(f"Initial GPU memory allocated: {initial_memory / 1024**3:.2f} GB")
    gc.collect()
    torch.cuda.empty_cache()

    final_memory = torch.cuda.memory_allocated()
    logger.info(f"Final GPU memory allocated: {final_memory / 1024**3:.2f} GB")

    freed_memory = initial_memory - final_memory
    logger.info(f"Freed GPU memory: {freed_memory / 1024**3:.2f} GB")


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
    def __init__(self, mask_config: MaskConfig):
        super().__init__()
        self.config = mask_config
        self.size = mask_config.size
        assert self.size is not None, "Mask size must be specified."

        value = mask_config.value
        if value is not None:
            logger.warning("Highly reccommend initializing mask value using dedicated setup functions.")
            if not isinstance(value, torch.Tensor): 
                try: value = torch.tensor(value)
                except: raise ValueError(
                    f"Unable to convert {value} to torch.Tensor required for initializing a mask's weight."
                )

        ## TODO: might refactor later: modify _get_ones() to handle scalar mode.
        if mask_config.mode == "scalar":
            self.weight = nn.Parameter(value if value is not None else torch.tensor(1.0))
            
        elif mask_config.mode in ("vector_input", "vector_output"):
            ones = self._get_ones(mask_config.mode)
            self.weight = nn.Parameter(value if value is not None else ones)
        else:
            raise ValueError(f"Unsupported mask mode: {mask_config.mode}")

        self._check_shape_compatibility()

    def _get_ones(self, mode: str) -> torch.Tensor:
        """Generates a tensor of ones based on mode and size."""
        dim = 0 if mode == "vector_output" else -1
        features = self.size[dim]
        if len(self.size) == 2 and mode == "vector_output":
            return torch.ones(features, 1)
        else:
            return torch.ones(features)
          

    def _check_shape_compatibility(self):
        """Raises ValueError if the mask shape is incompatible with its size."""
        try:
            in_test = torch.rand(self.size)
            out_test = self.weight * in_test
            assert out_test.shape == in_test.shape, (
                "After applying mask, the shape of input weight does not stay the same."
            )
        except RuntimeError:
            raise ValueError("Mask initialized with an incompatible shape.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.size != x.shape:
            logger.warning("Warning: Input shape does not match mask shape.")
        return x * self.weight

    def extra_repr(self):
        return f"mask_mode={self.config.mode}"

class ModuleWithMask(nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super(ModuleWithMask, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

class ModulesWithMasks(nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super(ModulesWithMasks, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass
        
    @abstractmethod
    def get_raw_masks(self):
        pass
        
    @abstractmethod
    def get_constrained_masks(self):
        pass

def silu_01(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x) - F.silu(x - 1.0)

def relu_01(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x) - F.relu(x - 1.0)

def normalize(v, dim, eps: float = 1e-8):
    norm_v = torch.linalg.norm(v, dim=dim)
    norm_v[norm_v < eps] = 1.0
    v = v / norm_v
    return v

class Constrainer(nn.Module):

    def __init__(self, component_weights, constrain_mode, mask_mode):
        super().__init__()
        self.statistics = None
        self.constrain_mode = constrain_mode
        self.mask_mode = mask_mode
        if self.constrain_mode not in ("identity", "01", "-11", "spherical", "sum1"):
            raise ValueError(f"Does not support {self.constrain_mode} constraint yet!")
            
        if (self.constrain_mode == "spherical" and all([w is not None for w in component_weights])):
            self._get_spherical_stats(component_weights)

    def _get_spherical_stats(
        self, 
        component_weights: List[torch.Tensor], 
        DOT_THRESHOLD: float = 0.99995
    ):
        with torch.no_grad():
            if any([w is None for w in component_weights]):
                raise ValueError("Spherical constraint (SLERP) does not support None weights.")
            if len(component_weights) != 2: 
                raise ValueError(
                    "Spherical constraint (SLERP) only supports 2 component weights, " +
                    f"{len(component_weights)} components found."
                )

            
            dim = 0 if self.mask_mode in ("vector_input") else None
            v0 = normalize(component_weights[0], dim=dim)
            v1 = normalize(component_weights[1], dim=dim)
            self.dots = torch.sum(v0 * v1, dim=dim, keepdim=False) ## (out_features, in_features) -> (in_features, )
            self.theta_0s = torch.arccos(self.dots)
            self.sin_theta_0s = torch.sin(self.theta_0s)
        
    def _constrain_identity(self, mask_weights: List[torch.Tensor]):
        return mask_weights

    def _constrain_sumone(self, mask_weights: List[torch.Tensor]):
        W = [w / sum(mask_weights) for w in mask_weights]
        return W

    def _constrain_0_1(self, mask_weights: List[torch.Tensor]):
        W = [relu_01(w) for w in mask_weights]
        return W

    def _constrain_neg1_1(self, mask_weights: List[torch.Tensor]):
        return mask_weights

    def _constrain_spherical(self, mask_weights: List[torch.Tensor], DOT_THRESHOLD: float = 0.9995):
        assert len(mask_weights) == 2, (
            "Spherical constraint (SLERP) only supports 2 mask weights"
        )
        
        # Transform raw masks to factor t's.
        ## QUESTION MASK: Does this modify mask_weights in-place? I only 
        ## update them via backprop.
        # W = [torch.exp(w) for w in mask_weights]
        # T = W[0] / sum(W)

        ## ignore mask_weights[1]
        T = silu_01(mask_weights[0])

        # Angle at timestep t's
        theta_ts = self.theta_0s * T
        sin_theta_ts = torch.sin(theta_ts)

        # Finish calculating slerp factors
        S0 = torch.sin(self.theta_0s - theta_ts) / self.sin_theta_0s
        S1 = sin_theta_ts / self.sin_theta_0s

        # Avoid NaN
        nan_indices = self.dots.float() > DOT_THRESHOLD
        S0[nan_indices] = 1 - T[nan_indices]
        S1[nan_indices] = T[nan_indices]
        
        return [S0, S1]
        
    def forward(self, mask_weights: List[torch.Tensor]):
        if any([w is None for w in mask_weights]):
            return mask_weights
            
        if self.constrain_mode == "identity":
            return self._constrain_identity(mask_weights)
        elif self.constrain_mode == "01":
            return self._constrain_0_1(mask_weights)
        elif self.constrain_mode == "-11":
            return self._constrain_neg1_1(mask_weights)
        elif self.constrain_mode == "sum1":
            return self._constrain_sumone(mask_weights)
        elif self.constrain_mode == "spherical":
            return self._constrain_spherical(mask_weights)
        else:
            raise ValueError(f"Does not support {self.constrain_mode} constraint yet!")

    def extra_repr(self):
        return f"constrain_mode={self.constrain_mode}"
            
class LinearsWithMasks(ModulesWithMasks):
    def __init__(
        self,
        linears: List[nn.Linear],
        weight_modes: List[str] = None,
        weight_values: List[float] = None,
        bias_modes: List[str] = None,
        bias_values: List[float] = None,
        constrain_mode: str = "identity",
    ):
        super().__init__()

        if not all(isinstance(linear, nn.Linear) for linear in linears):
            raise ValueError("All elements in 'linears' must be instances of nn.Linear.")

        if weight_values is None or len(weight_values) != len(linears):
            raise ValueError(
                f"Weight values for masks: {weight_values} do not match with linear layers: {linears}"
            )
        if bias_values is None:
            bias_values = [None] * len(linears)
        if len(bias_values) != len(linears):
            raise ValueError(
                f"Bias values for masks: {bias_values} do not match with linear layers: {linears}"
            )

        self.linears = nn.ModuleList(linears)
        self.constrain_mode = constrain_mode

        self.weight_masks = nn.ModuleList([
            Mask(MaskConfig(mode, value, linear.weight.shape))
            for mode, value, linear in zip(weight_modes, weight_values, linears)
        ])
        
        self.weight_masks_constrainer = Constrainer(
            component_weights=[x.weight for x in self.linears], 
            constrain_mode=constrain_mode, mask_mode=weight_modes[0]
        )

        self.bias_masks = nn.ModuleList([
            Mask(MaskConfig(mode, value, linear.bias.shape)) if linear.bias is not None else None
            for mode, value, linear in zip(bias_modes, bias_values, linears)
        ])
        
        self.bias_masks_constrainer = Constrainer(
            component_weights=[x.bias if x.bias is not None else None for x in self.linears],
            constrain_mode=constrain_mode, mask_mode=bias_modes[0]
        )

    def forward(self, x):
        constrained_weight_masks = self.weight_masks_constrainer([m.weight for m in self.weight_masks])
        masked_weights = [
            w_mask * linear.weight for w_mask, linear in zip(constrained_weight_masks, self.linears)
        ]
        merged_weight = sum(masked_weights)

        constrained_bias_masks = self.bias_masks_constrainer(
            [m.weight if m is not None else None for m in self.bias_masks]
        )
        masked_biases = [
            b_mask * linear.bias if linear.bias is not None and b_mask is not None else linear.bias
            for b_mask, linear in zip(constrained_bias_masks, self.linears)
        ]

        merged_bias = (
            sum(b if b is not None else torch.zeros_like(merged_weight[:, 0]) for b in masked_biases)
            if not all(b is None for b in masked_biases) else None
        )

        return nn.functional.linear(x, merged_weight, merged_bias)

    def get_raw_masks(self):
        with torch.no_grad():
            return {
                "weight_masks": [m.weight for m in self.weight_masks],
                "bias_masks": [m.weight if m is not None else None for m in self.bias_masks],
            }

    def get_constrained_masks(self):
        with torch.no_grad():
            constrained_weight_masks = self.weight_masks_constrainer(
                [m.weight for m in self.weight_masks]
            )
            constrained_bias_masks = self.bias_masks_constrainer(
                [m.weight if m is not None else None for m in self.bias_masks]
            )
            return {
                "weight_masks": constrained_weight_masks,
                "bias_masks": constrained_bias_masks,
            }

class RMSNormsWithMasks(ModulesWithMasks):
    def __init__(
        self,
        rms_norms: List[nn.Module],
        modes: List[str] = None,
        values: List[float] = None,
        constrain_mode: str = "identity",
    ):
        super().__init__()
        sizes = [norm.weight.shape for norm in rms_norms]
        if any([mode != "scalar" for mode in modes]):
            logger.warning_once(
                f"Though you want to make a masks of modes {modes} " + \
                "for RMSNorms' weights, by default a mask only accepts a scalar mask. " + \
                "Converting modes to `scalar`."
            )
            modes = ["scalar"] * len(modes)
            
        if values is None or len(values) != len(rms_norms):
            raise ValueError(f"values for masks: {values} do not match with RMSNorm layers: {rms_norms}")

        self.rms_norms = nn.ModuleList(rms_norms)
        self.masks = nn.ModuleList([
            Mask(MaskConfig(mode, value, norm.weight.shape))
            for mode, value, norm in zip(modes, values, rms_norms)
        ])
        self.masks_constrainer = Constrainer(
            component_weights=[norm.weight for norm in self.rms_norms], 
            constrain_mode=constrain_mode, mask_mode=modes[0]
        )

    def forward(self, hidden_states):
        constrained_masks = self.masks_constrainer([m.weight for m in self.masks])
        masked_weights = [mask * norm.weight for mask, norm in zip(constrained_masks, self.rms_norms)]
        merged_weight = sum(masked_weights)
        variance_epsilon = self.rms_norms[0].variance_epsilon
        for norm in self.rms_norms:
            assert variance_epsilon == norm.variance_epsilon, ("Variance epsilon among models must be consistent")
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        return merged_weight * hidden_states.to(input_dtype)

    def get_raw_masks(self):
        with torch.no_grad():
            return {"masks": [m.weight for m in self.masks]}

    def get_constrained_masks(self):
        with torch.no_grad():
            constrained_masks = self.masks_constrainer([m.weight for m in self.masks])
            return {"masks": constrained_masks}

class EmbeddingsWithMasks(ModulesWithMasks):
    def __init__(
        self,
        embeddings: List[nn.Embedding],
        modes: List[str] = None,
        values: List[float] = None,
        constrain_mode: str = "identity"
    ):
        super().__init__()
        if values is None or len(values) != len(embeddings):
            raise ValueError(f"values for masks: {values} do not match with Embedding layers: {embeddings}")

        self.embeddings = nn.ModuleList(embeddings)
        sizes = [emb.weight.shape for emb in embeddings]
        self.masks = nn.ModuleList([
            Mask(MaskConfig(mode, value, size))
            for mode, value, size in zip(modes, values, sizes)
        ])
        self.masks_constrainer = Constrainer(
            component_weights=[emb.weight for emb in self.embeddings], 
            constrain_mode=constrain_mode, mask_mode=modes[0]
        )

    def forward(self, input_ids):
        constrained_masks = self.masks_constrainer([m.weight for m in self.masks])
        masked_weights = [mask * emb.weight for mask, emb in zip(constrained_masks, self.embeddings)]
        merged_weight = sum(masked_weights)
        
        an_embedding = self.embeddings[0]
        for other_embedding in self.embeddings:
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
        
    def get_raw_masks(self):
        with torch.no_grad():
            return {"masks": [m.weight for m in self.masks]}

    def get_constrained_masks(self):
        with torch.no_grad():
            constrained_masks = self.masks_constrainer([m.weight for m in self.masks])
            return {"masks": constrained_masks}

"""
BEGIN - INIT STRATEGIES
"""

def odd_one_out(masked_module: nn.Module, selected_idx: int):  
    assert selected_idx is not None and isinstance(selected_idx, int), (
        "Must provide valid model index. Check whether passed index is `int`"
    )
    masks_modules = []
    for name, child in masked_module.named_children():
        if not isinstance(child, nn.ModuleList): continue
        assert selected_idx < len(child), (
            f"There are only {len(child)} component models, passed model index is {selected_idx}"
        )
        ## exclude sub_module that is None, aka bias_masks.
        if all(isinstance(sub_module, Mask) for sub_module in child):
            masks_modules.append(child)
        
    for masks in masks_modules:
        for i, mask in enumerate(masks):
            value = 1.0 if i == selected_idx else 0.0
            with torch.no_grad():
                mask.weight.data.fill_(value)

def random_init(masked_module: nn.Module):
    masks_modules = []
    for name, child in masked_module.named_children():
        if not isinstance(child, nn.ModuleList): continue
        ## exclude sub_module that is None, aka bias_masks.
        if all(isinstance(sub_module, Mask) for sub_module in child):
            masks_modules.append(child)
        
    for masks in masks_modules:
        for i, mask in enumerate(masks):
            with torch.no_grad():
                random_value = torch.rand_like(mask.weight.data)
                mask.weight.data = random_value

def uniform_init(masked_module: nn.Module, factors: List[float]):  
    masks_modules = []
    for name, child in masked_module.named_children():
        if not isinstance(child, nn.ModuleList): continue
        assert len(factors) == len(child), (
            f"There are {len(child)} component models, but your passed factors have {len(factors)} values."
        )
        ## exclude sub_module that is None, aka bias_masks.
        if all(isinstance(sub_module, Mask) for sub_module in child):
            masks_modules.append(child)

    for masks in masks_modules:
        for factor, mask in zip(factors, masks):
            with torch.no_grad():
                mask.weight.data.fill_(factor)
"""
END - INIT STRATEGIES
"""

def find_masked_modules(module):
    masked_module_names = []
    for parent_name, parent_module in module.named_modules():
        for name, child in parent_module.named_children():
            full_child_name = f"{parent_name}.{name}" if parent_name else name
            if ("WithMasks" in type(child).__name__):
                masked_module_names.append(full_child_name)

    return masked_module_names

def get_init_method(strategy):
    MAP = {
        "random": random_init,
        "odd_one_out": odd_one_out,
        "uniform": uniform_init
    }
    selected_init_method = MAP[strategy]
    
    return selected_init_method
    
def set_masks(root_module, strategy="random", **kwargs):

    init_method = get_init_method(strategy)
    masked_module_names = find_masked_modules(root_module)
    
    for module_name in tqdm(masked_module_names, desc="Setting up masks"):
        module_names = module_name.split(".")
        target_module = root_module
        for m_name in module_names:
            target_module = getattr(target_module, m_name)

        init_method(target_module, **kwargs)

class MergerConfig(PretrainedConfig):
    def __init__(
        self,
        model_paths: List[str] = None,
        mode: str = None,
        constrain_mode: str = None,
        **kwargs,
    ):
        self.model_paths = model_paths
        self.mode = mode
        self.constrain_mode = constrain_mode
        super().__init__(**kwargs)

class NewMerger(PreTrainedModel):
    def __init__(self, config: MergerConfig):
        super().__init__(config)
        self.merge_config = config
        self.num_models = len(config.model_paths)

        # Initialize configs but don't load models yet
        self.configs = [
            AutoConfig.from_pretrained(path) 
            for path in config.model_paths
        ]
        
        # Initialize empty ModuleList for models - will be populated in from_pretrained
        self.models = nn.ModuleList()
        
        # Create merger with same architecture as first model but empty weights
        logger.info("Creating merger with dummy weights ...")
        self.merger = AutoModelForCausalLM.from_pretrained(
            config.model_paths[0],  # Use first model path
            config=self.configs[0]
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else True

        # Prepare inputs for forward pass
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
            "cache_position": cache_position,
            **kwargs,
        }
            
        # During training, we want both merger and component outputs
        merger_outputs = self.merger(**inputs)
        components_outputs = [model(**inputs) for model in self.models]
        
        return {
            "merger_outputs": merger_outputs,
            "components_outputs": components_outputs
        }

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        state_dict: Optional[dict] = None,
        **kwargs,
    ):
        """
        Every save calls during training point back to this function.
        Trainer._save_checkpoint() -> Trainer.save_model() -> Trainer._save()
        -> Merger.save_pretrained()
        Basically I only have to customize .save_pretrained()
        """
        if state_dict is None:
            state_dict = self.state_dict()
            
        # Filter for only trainable parameters (masks)
        trainable_state = {
            k: v for k, v in state_dict.items() 
            if any(trainable_key in k for trainable_key in [
                'weight_masks', 'bias_masks', 'masks'
            ])
        }
        super().save_pretrained(
            save_directory=save_directory,
            state_dict=trainable_state,
            **kwargs
        )

        # Rename the files after saving
        safe_file = os.path.join(save_directory, "model.safetensors")
        pytorch_file = os.path.join(save_directory, "pytorch_model.bin")
        
        if os.path.exists(safe_file):
            os.rename(safe_file, os.path.join(save_directory, "masks.safetensors"))
        elif os.path.exists(pytorch_file):
            os.rename(pytorch_file, os.path.join(save_directory, "masks.bin"))
        

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        *model_args,
        **kwargs,
    ):
        device_map = kwargs.pop('device_map', None)  # Remove device_map from kwargs
        
        # Initialize model instance
        model = cls(config, *model_args)
        
        # Load component models without device_map
        for i in range(model.num_models):
            loaded_model = AutoModelForCausalLM.from_pretrained(
                config.model_paths[i],
                config=model.configs[i],
                **kwargs
            )
            model.models.append(loaded_model)
            
            # Freeze component model parameters
            for param in loaded_model.parameters():
                param.requires_grad = False
        
        # Initialize masks before device mapping
        init_masks(model.merger, model.models, config)
        
        # If loading from a checkpoint, load saved trainable parameters
        if pretrained_model_name_or_path is not None:
            state_dict = self.load_masks_state_dict(pretrained_model_name_or_path)
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )
        # Apply device mapping after masks are initialized
        if device_map is None:
            return model
            
        if not is_fsdp_enabled() and not is_deepspeed_zero3_enabled():
            return model
            
        if device_map == "auto":
            # Reserve 2GB buffer per GPU
            max_memory = {
                i: f"{int(torch.cuda.get_device_properties(i).total_memory / 1024**3 - 2)}GiB" 
                for i in range(torch.cuda.device_count())
            }
            
            # Compute optimal device map
            device_map = infer_auto_device_map(
                model, 
                max_memory=max_memory,
                no_split_module_classes=[
                    "LinearsWithMasks", 
                    "EmbeddingsWithMasks", 
                    "RMSNormsWithMasks"
                ]
            )
            
        return dispatch_model(model, device_map=device_map)

    def load_masks_state_dict(
        self, 
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]]
    ):
        # Try loading from safetensors first
        trainable_path = os.path.join(pretrained_model_name_or_path, "masks.safetensors")
        if not os.path.exists(trainable_path):
            trainable_path = os.path.join(pretrained_model_name_or_path, "masks.bin")
        if os.path.exists(trainable_path):
            if trainable_path.endswith('.safetensors'):
                from safetensors.torch import load_file as safe_load_file
                state_dict = safe_load_file(trainable_path)
            else:
                state_dict = torch.load(trainable_path, map_location="cpu")

            return state_dict
        else:
            raise ValueError(f"`{trainable_path}` does not exist.")

    
    def load_masks(
        self, 
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]]
    ):  
        # Load only the trainable parameters
        state_dict = self.load_masks_state_dict(pretrained_model_name_or_path)
        missing_keys, unexpected_keys = self.merger.load_state_dict(
            state_dict, strict=False
        )

def find_modules_to_add_masks(target_module):
    module_names_to_replace = []
    for parent_name, parent_module in target_module.named_modules():
        for name, child in parent_module.named_children():
            full_child_name = f"{parent_name}.{name}" if parent_name else name
            if (isinstance(child, (nn.Linear, nn.Embedding)) 
                or "RMSNorm" in type(child).__name__):
                module_names_to_replace.append(full_child_name)

    return module_names_to_replace

def init_masks(target_module: nn.Module, ref_modules: nn.Module, merge_config: MergerConfig):
    """
    Replaces eligible submodules in target_module with masked versions, 
    using corresponding modules from ref_modules as a reference for weights.

    Args:
        target_module: The module in which to replace submodules.
        ref_modules: A list of modules to use as a reference for weights.
        strategy: The initialization strategy for factors ("naive" or others to be implemented).
    """
    mode = merge_config.mode
    constrain_mode = merge_config.constrain_mode
    module_names_to_replace = find_modules_to_add_masks(target_module)
    
    for module_name in tqdm(module_names_to_replace, desc="Initializing masks"):
        module_names = module_name.split(".")
        target_child = target_module
        ref_children = ref_modules

        for m_name in module_names:
            target_child = getattr(target_child, m_name)
            ref_children = [getattr(ref_module, m_name) for ref_module in ref_children]

        num_components = len(ref_modules)
        modes = [mode for _ in ref_children]
        factors = [None for _ in ref_children]

        if isinstance(target_child, nn.Linear):
            new_module = LinearsWithMasks(
                linears=ref_children,
                weight_modes=modes,
                weight_values=factors,
                bias_modes=modes,
                bias_values=factors,
                constrain_mode=constrain_mode
            )

        elif isinstance(target_child, nn.Embedding):
            new_module = EmbeddingsWithMasks(ref_children, modes, factors, constrain_mode)
        elif "RMSNorm" in type(target_child).__name__:
            new_module = RMSNormsWithMasks(ref_children, modes, factors, constrain_mode)

        # Move new module's mask parameters to correct dtype
        target_dtype = ref_children[0].weight.dtype
        for param in new_module.parameters():
            if param.requires_grad:  # Only convert mask parameters, not the frozen model weights
                param.data = param.data.to(dtype=target_dtype)

        # Replace the original module with the new masked module
        parent_module = target_module
        for m_name in module_names[:-1]:
            parent_module = getattr(parent_module, m_name)
        setattr(parent_module, module_names[-1], new_module)

if __name__ == "__main__":
    merge_config = MergerConfig(
        model_paths = [
            "/workspace/dont15/models/llama32_smol_rewrite_50k/",
            "/workspace/dont15/models/llama32_smol_summarize_50k/",
            # "/workspace/HUB_LLM/Llama-3.2-3B-Instruct",
        ],
        mode = "vector_input",
        # mode = "scalar",
        constrain_mode = "01"
    )
    tokenizer = AutoTokenizer.from_pretrained(merge_config.model_paths[0])
    system = "You are a helpful assistant."
    prompt = "How to attack a person with an egg. Talk like a crazy person."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    merger = Merger(merge_config)
    merger.__post_init__()
    merger = merger.to(device="cuda:1", dtype=torch.bfloat16)
    free_memory()

    ## QUICK TEST
    for selected_idx in [0, 1]:
        print(f"***** odd one out: {selected_idx}")
        # set_masks(merger.merger, strategy="odd_one_out", selected_idx=selected_idx)
        set_masks(merger.merger, strategy="uniform", factors=[0.3, 0.7])
        answer_merged = generate(text, merger.merger, tokenizer, max_new_tokens=32)
        print(f"----- {answer_merged}")
        answer_ref = generate(text, merger.models[selected_idx], tokenizer, max_new_tokens=32)
        print(f"----- {answer_ref}")
        # assert answer_merged == answer_ref
    
        logits_merged = get_logits(text, merger.merger, tokenizer)
        logits_ref = get_logits(text, merger.models[selected_idx], tokenizer)
        # assert torch.allclose(logits_merged, logits_ref, atol=0, rtol=0)

    