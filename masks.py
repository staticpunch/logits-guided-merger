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
        if self.constrain_mode not in ("identity", "01", "-11", "spherical", "sum1", "cosine"):
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
        
    def _constrain_cosine(self, mask_weights: List[torch.Tensor]):
        W = [1.0 / 2 - torch.cos(torch.pi * w) / 2 for w in mask_weights]
        return W

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
        if self.constrain_mode == "cosine":
            return self._constrain_cosine(mask_weights)
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