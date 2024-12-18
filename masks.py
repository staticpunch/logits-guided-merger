import math
from typing import List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import datasets
import torch
import numpy as np
import torch.nn as nn
import logging
import copy
import gc

from datasets import load_dataset
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
    HfArgumentParser
)

from configuration_qwen2 import Qwen2Config

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
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
        if mask_config.mode == "scalar":
            self.weight = nn.Parameter(torch.tensor(value if value is not None else 1.0))
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

class LinearWithMask(ModuleWithMask):
    def __init__(
        self, 
        linear: nn.Linear, 
        weight_mask_config: MaskConfig, 
        bias_mask_config: MaskConfig = None
    ):
        super().__init__()
        self.linear = linear
        self.weight_mask_config = weight_mask_config
        self.bias_mask_config = bias_mask_config

        if linear.weight.shape != weight_mask_config.size:
            logger.warning(
                "Weight mask shape is not compatible with linear, reinitializing..."
            )
            self.weight_mask_config.size = linear.weight.shape
        self.weight_mask = Mask(self.weight_mask_config)
        
        ## make sure things on the same page.
        self.weight_mask.to(
            device=self.linear.weight.device,
            dtype=self.linear.weight.dtype
        )

        if linear.bias is not None and bias_mask_config is not None:
            if linear.bias.shape != bias_mask_config.size:
                logger.warning(
                    "Bias mask shape is not compatible with linear, reinitializing..."
                )
                self.bias_mask_config.size = linear.bias.shape
            self.bias_mask = Mask(self.bias_mask_config)
            
            ## make sure things on the same page.
            self.bias_mask.to(
                device=self.linear.bias.device,
                dtype=self.linear.bias.dtype
            )
        else:
            self.bias_mask = None

    def forward(self, x):
        masked_weight = self.weight_mask(self.linear.weight)
        if self.linear.bias is not None and self.bias_mask is not None:
            masked_bias = self.bias_mask(self.linear.bias)
        else:
            masked_bias = self.linear.bias
        return nn.functional.linear(x, masked_weight, masked_bias)

class LinearsWithMasks(ModulesWithMasks):
    def __init__(
        self,
        linears: List[nn.Linear],
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

        self.masked_linears = nn.ModuleList([
            LinearWithMask(linear, weight_mask_config, bias_mask_config)
            for linear, weight_mask_config, bias_mask_config 
            in zip(linears, weight_mask_configs, bias_mask_configs)
        ])

        
    def forward(self, x):   
        weights = [linear.weight_mask(linear.linear.weight) 
                   for linear in self.masked_linears]
        merged_weight = sum(weights)

        biases = [
            linear.bias_mask(linear.linear.bias) 
            if linear.linear.bias is not None and linear.bias_mask is not None 
            else linear.linear.bias for linear in self.masked_linears
        ]
        
        if all(b is None for b in biases):
            merged_bias = None
        else:
            biases = [
                b if b is not None
                else torch.zeros_like(weights[0][:, 0])
                for b in biases
            ]
            merged_bias = sum(biases)

        return nn.functional.linear(x, merged_weight, merged_bias)

class RMSNormWithMask(ModuleWithMask):
    def __init__(self, rms_norm: nn.Module, mask_config: MaskConfig):
        super().__init__()
        assert "RMSNorm" in type(rms_norm).__name__
        self.rms_norm = rms_norm
        self.mask_config = mask_config
        if mask_config.mode != "scalar":
            logger.warning_once(
                f"Though you want to make a mask of mode {mask_config.mode}" + \
                "for a RMSNorm's weights, by default it only accepts a scalar mask."
            )
            self.mask_config.mode = "scalar"
        if mask_config.size != rms_norm.weight.shape:
            logger.warning_once("Mask shape is not compatible with RMSNorm, reinitializing...")
            self.mask_config.size = rms_norm.weight.shape
            
        self.mask = Mask(self.mask_config)
        
        ## make sure things on the same page.
        self.mask.to(
            device=self.rms_norm.weight.device,
            dtype=self.rms_norm.weight.dtype
        )

    def forward(self, hidden_states):
        masked_weight = self.mask(self.rms_norm.weight)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.rms_norm.variance_epsilon)
        return masked_weight * hidden_states.to(input_dtype)

class RMSNormsWithMasks(ModulesWithMasks):
    def __init__(
        self,
        rms_norms: List[nn.Module],
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
            assert variance_epsilon == rms.rms_norm.variance_epsilon, (
                "Variance epsilon among models must be consistent"
            )
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        return merged_weight * hidden_states.to(input_dtype)

class EmbeddingWithMask(ModuleWithMask):
    def __init__(self, embedding: nn.Embedding, mask_config: MaskConfig):
        super().__init__()
        self.embedding = embedding
        self.mask_config = mask_config
        if embedding.weight.shape != mask_config.size:
            logger.warning_once("Mask shape is not compatible with Embedding, reinitializing...")
            self.mask_config.size = embedding.weight.shape
            
        self.mask = Mask(self.mask_config)
        
        ## make sure things on the same page.
        self.mask.to(
            device=self.embedding.weight.device,
            dtype=self.embedding.weight.dtype
        )

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

class EmbeddingsWithMasks(ModulesWithMasks):
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

def find_modules_to_add_masks(target_module):
    module_names_to_replace = []
    for parent_name, parent_module in target_module.named_modules():
        for name, child in parent_module.named_children():
            full_child_name = f"{parent_name}.{name}" if parent_name else name
            if (isinstance(child, (nn.Linear, nn.Embedding)) 
                or "RMSNorm" in type(child).__name__):
                module_names_to_replace.append(full_child_name)

    return module_names_to_replace

def init_masks(target_module, ref_modules, mode="vector_input"):
    """
    Replaces eligible submodules in target_module with masked versions, 
    using corresponding modules from ref_modules as a reference for weights.

    Args:
        target_module: The module in which to replace submodules.
        ref_modules: A list of modules to use as a reference for weights.
        strategy: The initialization strategy for factors ("naive" or others to be implemented).
    """
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
            )

        elif isinstance(target_child, nn.Embedding):
            new_module = EmbeddingsWithMasks(ref_children, modes, factors)
        elif "RMSNorm" in type(target_child).__name__:
            new_module = RMSNormsWithMasks(ref_children, modes, factors)

        # Replace the original module with the new masked module
        parent_module = target_module
        for m_name in module_names[:-1]:
            parent_module = getattr(parent_module, m_name)
        setattr(parent_module, module_names[-1], new_module)

"""
BEGIN - Mask Initialization Strategies
"""
def random_init(module_name, masked_module, **kwargs):
    """
    Despite randomizing factors of modules, I will constrain
    sum of them to be 1.0.
    """
    module_list = masked_module.children().__next__()
    weight_masks = []
    bias_masks = []

    ## RANDOMIZING MASKS.
    for i, component in enumerate(module_list):
        assert "WithMask" in type(component).__name__, (
            f"{type(component).__name__} module does not have masks."
        )
        with torch.no_grad():
            if type(component).__name__ == LinearWithMask.__name__:
                child_names = [name for name, _ in component.named_children()]
                random_values = torch.rand_like(component.weight_mask.weight.data)
                weight_masks.append(random_values)
                
                if "bias_mask" in child_names:
                    random_values = torch.rand_like(component.bias_mask.weight.data)
                    bias_masks.append(random_values)
                    
            elif type(component).__name__ in (
                RMSNormWithMask.__name__, EmbeddingWithMask.__name__
            ):
                random_values = torch.rand_like(component.mask.weight.data)
                weight_masks.append(random_values)
            else:
                raise ValueError(f"{type(component).__name__} module does not have masks.")

    ## NORMALIZING MASKS AND ASSIGNING THEM
    weight_masks = [x / sum(weight_masks) for x in weight_masks]
    bias_masks = [x / sum(bias_masks) for x in bias_masks]
    
    for i, component in enumerate(module_list):
        with torch.no_grad():
            if type(component).__name__ == LinearWithMask.__name__:
                child_names = [name for name, _ in component.named_children()]
                component.weight_mask.weight.data = weight_masks[i]
                if "bias_mask" in child_names:
                    component.bias_mask.weight.data = bias_masks[i]
                    
            elif type(component).__name__ in (
                RMSNormWithMask.__name__, EmbeddingWithMask.__name__
            ):
                component.mask.weight.data = weight_masks[i]

def odd_one_out(module_name, masked_module, **kwargs):
    assert "selected_idx" in kwargs
    selected_idx = kwargs["selected_idx"]
    module_list = masked_module.children().__next__()
    
    assert selected_idx is not None, "Must provide index."
    assert isinstance(selected_idx, int), "Index must be int."
    assert selected_idx < len(module_list), "Out of index."

    for i, component in enumerate(module_list):
        assert "WithMask" in type(component).__name__, (
            f"{type(component).__name__} module does not have masks."
        )
        value = 1.0 if i == selected_idx else 0.0
        with torch.no_grad():
            if type(component).__name__ == LinearWithMask.__name__:
                child_names = [name for name, _ in component.named_children()]
                component.weight_mask.weight.data.fill_(value)
                if "bias_mask" in child_names:
                    component.bias_mask.weight.data.fill_(value)
            elif type(component).__name__ in (
                RMSNormWithMask.__name__, EmbeddingWithMask.__name__
            ):
                component.mask.weight.data.fill_(value)
            else:
                raise ValueError(f"{type(component).__name__} module does not have masks.")

def individual_uniform(module_name, masked_module, **kwargs):
    assert "individual_factors" in kwargs
    individual_factors = kwargs["individual_factors"]
    
    module_list = masked_module.children().__next__()
    
    assert individual_factors is not None, "Must provide index."
    assert len(individual_factors) == len(module_list), "Incorrect number of factors."

    for i, component in enumerate(module_list):
        assert "WithMask" in type(component).__name__, (
            f"{type(component).__name__} module does not have masks."
        )
        value = individual_factors[i]
        with torch.no_grad():
            if type(component).__name__ == LinearWithMask.__name__:
                child_names = [name for name, _ in component.named_children()]
                component.weight_mask.weight.data.fill_(value)
                if "bias_mask" in child_names:
                    component.bias_mask.weight.data.fill_(value)
            elif type(component).__name__ in (
                RMSNormWithMask.__name__, EmbeddingWithMask.__name__
            ):
                component.mask.weight.data.fill_(value)
            else:
                raise ValueError(f"{type(component).__name__} module does not have masks.")

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
        "slerp": slerp_init,
        "odd_one_out": odd_one_out,
        "individual_uniform": individual_uniform
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
            
        init_method(module_name, target_module, **kwargs)

"""
END - Mask Initialization Strategies
"""

class MergerConfig(PretrainedConfig):
    def __init__(
        self,
        model_paths: List[str] = None,
        mode: str = None,
        **kwargs,
    ):
        self.model_paths = model_paths
        self.mode = mode
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
            AutoConfig.from_pretrained(path) 
            for path in merge_config.model_paths
        ]
        self.models = nn.ModuleList([
            AutoModelForCausalLM.from_pretrained(
                merge_config.model_paths[i], 
                config=self.configs[i],
                torch_dtype=torch.bfloat16
            ) 
            for i in range(self.num_models)
        ])
        # self.__post_init__(merge_config)
        
    def __post_init__(self, merge_config):
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False
                
        self.merger = copy.deepcopy(self.models[0])
        init_masks(self.merger, self.models, mode=self.merge_config.mode)
        free_memory()
        
    def forward(self, tensor, labels=None):
        pass