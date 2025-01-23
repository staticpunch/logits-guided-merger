from typing import List, Mapping
from abc import ABC

import torch
import torch.nn as nn
import logging
import re
from tqdm import tqdm

from masks import (
    Mask,
    LinearsWithMasks,
    RMSNormsWithMasks,
    EmbeddingsWithMasks
)
from slerp import slerp, compute_t, blend
from logging_config import configure_logging

# Configure logger
from logging_config import configure_logging
configure_logging()
logger = logging.getLogger("merger")

class MaskInitializer:
    def __init__(self):
        self.init_map = {
            "random": self._random_init,
            "odd_one_out": self._odd_one_out,
            "uniform": self._uniform_init,
            "spherical": self._spherical_init,
            "blend": self._blend_init
        }

    def _find_masked_modules(self, module):
        """Find all modules that contain masks in the model hierarchy."""
        masked_module_names = []
        for parent_name, parent_module in module.named_modules():
            for name, child in parent_module.named_children():
                full_child_name = f"{parent_name}.{name}" if parent_name else name
                if "WithMasks" in type(child).__name__:
                    masked_module_names.append(full_child_name)
        return masked_module_names

    def _get_target_module(self, root_module, module_name):
        """Navigate module hierarchy to find target module."""
        target_module = root_module
        for m_name in module_name.split("."):
            target_module = getattr(target_module, m_name)
        return target_module

    def _init_module(self, root_module, strategy, **kwargs):
        """Initialize masks for all masked modules using specified strategy."""
        init_method = self.init_map[strategy]
        masked_module_names = self._find_masked_modules(root_module)
        
        for module_name in tqdm(masked_module_names, desc="Setting up masks"):
            target_module = self._get_target_module(root_module, module_name)
            
            if strategy in ("spherical", "blend"):
                kwargs["module_name"] = module_name
                
            init_method(target_module, **kwargs)

    def initialize(self, merger, mask_init):
        """Initialize masks based on configuration."""
        strategy = mask_init["strategy"]
        
        if strategy == "uniform":
            if not mask_init.get("factors"):
                raise ValueError("Factors must be provided for uniform strategy")
            
            logger.info(
                f"Applying uniform masks with factors = {mask_init['factors']}."
            )
            self._init_module(
                merger.merger, 
                strategy="uniform", 
                factors=mask_init["factors"]
            )
            
        elif strategy == "random":
            logger.info("Applying random masks.")
            self._init_module(merger.merger, strategy="random")
            
        elif strategy == "odd_one_out":
            logger.info(
                f"Applying odd_one_out masks with at index {mask_init['selected_idx']}."
            )
            self._init_module(
                merger.merger, 
                strategy="odd_one_out",
                selected_idx=mask_init["selected_idx"]
            )
            
        elif strategy == "spherical":
            logger.info("Applying spherical masks.")
            parameters = mask_init["parameters"]
            num_layers = len(merger.merger.model.layers)
            self._init_module(
                merger.merger,
                strategy="spherical",
                parameters=parameters,
                num_layers=num_layers
            )
            
        elif strategy == "blend":
            logger.info("Applying blended masks.")
            parameters = mask_init["parameters"]
            num_layers = len(merger.merger.model.layers)
            self._init_module(
                merger.merger,
                strategy="blend",
                parameters=parameters,
                num_layers=num_layers
            )
            
        else:
            raise ValueError(f"Unknown mask initialization strategy: {strategy}.")

    def _odd_one_out(self, masked_module: nn.Module, selected_idx: int, **kwargs):
        """Initialize masks using odd-one-out strategy."""
        assert selected_idx is not None and isinstance(selected_idx, int), (
            "Must provide valid model index. Check whether passed index is `int`"
        )
        masks_modules = []
        for name, child in masked_module.named_children():
            if not isinstance(child, nn.ModuleList):
                continue
            assert selected_idx < len(child), (
                f"There are only {len(child)} component models, "
                f"passed model index is {selected_idx}"
            )
            if all(isinstance(sub_module, Mask) for sub_module in child):
                masks_modules.append(child)
            
        for masks in masks_modules:
            for i, mask in enumerate(masks):
                value = 1.0 if i == selected_idx else 0.0
                with torch.no_grad():
                    mask.weight.data.fill_(value)

    def _random_init(self, masked_module: nn.Module, **kwargs):
        """Initialize masks with random values."""
        masks_modules = []
        for name, child in masked_module.named_children():
            if not isinstance(child, nn.ModuleList):
                continue
            if all(isinstance(sub_module, Mask) for sub_module in child):
                masks_modules.append(child)
            
        for masks in masks_modules:
            for i, mask in enumerate(masks):
                with torch.no_grad():
                    random_value = torch.rand_like(mask.weight.data)
                    mask.weight.data = random_value

    def _uniform_init(self, masked_module: nn.Module, factors: List[float], **kwargs):
        """Initialize masks with uniform values based on provided factors."""
        masks_modules = []
        for name, child in masked_module.named_children():
            if not isinstance(child, nn.ModuleList):
                continue
            assert len(factors) == len(child), (
                f"There are {len(child)} component models, "
                f"but your passed factors have {len(factors)} values."
            )
            if all(isinstance(sub_module, Mask) for sub_module in child):
                masks_modules.append(child)

        for masks in masks_modules:
            for factor, mask in zip(factors, masks):
                with torch.no_grad():
                    mask.weight.data.fill_(factor)

    def _spherical_init(
        self,
        masked_module: nn.Module,
        module_name: str,
        parameters: Mapping = None,
        num_layers: int = None,
        **kwargs
    ):
        """Initialize masks using spherical interpolation."""
        logger.info_once("You are playing with `SLURRRRRP`")
        n_children = len(next(
            child for child in masked_module.children()
            if isinstance(child, nn.ModuleList)
        ))
        assert n_children == 2, (
            f"SLERP initialization only supports 2 component modules. "
            f"Found {n_children}."
        )
        
        # t = compute_t(module_name, parameters, num_layers)
        find_layer = re.search(r"layers\.([^\.]*)\.", module_name)
        layer_idx = int(find_layer.group(1)) if find_layer else None
        t = blend(module_name, parameters, layer_idx, num_layers)
        
        if isinstance(masked_module, LinearsWithMasks):
            weight_masks = masked_module.weight_masks
            bias_masks = masked_module.bias_masks
            v0, v1 = (x.weight.data for x in masked_module.linears)
            s0, s1 = slerp(t, v0, v1)
            self._assign_two_masks(weight_masks, s0, s1)
            
            if all(isinstance(mask, Mask) for mask in bias_masks):
                v0, v1 = (x.bias.data for x in masked_module.linears)
                s0, s1 = slerp(t, v0, v1)
                self._assign_two_masks(bias_masks, s0, s1)
            
        elif isinstance(masked_module, EmbeddingsWithMasks):
            masks = masked_module.masks
            v0, v1 = (x.weight.data for x in masked_module.embeddings)
            s0, s1 = slerp(t, v0, v1)
            self._assign_two_masks(masks, s0, s1)
            
        elif isinstance(masked_module, RMSNormsWithMasks):
            masks = masked_module.masks
            v0, v1 = (x.weight.data for x in masked_module.rms_norms)
            s0, s1 = slerp(t, v0, v1)
            self._assign_two_masks(masks, s0, s1)
    
        else:
            raise ValueError(
                f"Does not support class {type(masked_module).__name__} yet."
            )

    def _blend_init(
        self,
        masked_module: nn.Module,
        module_name: str,
        parameters: Mapping = None,
        num_layers: int = None,
        **kwargs
    ):
        """Initialize masks using spherical interpolation."""
        
        find_layer = re.search(r"layers\.([^\.]*)\.", module_name)
        layer_idx = int(find_layer.group(1)) if find_layer else None
        t = blend(module_name, parameters, layer_idx, num_layers)
        
        n_children = len(next(
            child for child in masked_module.children()
            if isinstance(child, nn.ModuleList)
        ))
        assert n_children == 2, (
            f"Blending initialization only supports 2 component modules. "
            f"Found {n_children}."
        )
        masks_modules = []
        for name, child in masked_module.named_children():
            if not isinstance(child, nn.ModuleList): continue
            if all(isinstance(sub_module, Mask) for sub_module in child):
                masks_modules.append(child)
            
        for masks in masks_modules:
            self._assign_two_masks(masks, 1.0 - t, t)
            
    def _assign_two_masks(self, masks, s0, s1):
        """Assign spherical mask values to a pair of masks."""
        assert len(masks) == 2, (
            "Only supports assigning 2 mask modules. "
            f"Found {len(masks)}."
        )
        with torch.no_grad():
            masks[0].weight.data.fill_(s0)
            masks[1].weight.data.fill_(s1)
