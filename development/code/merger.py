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
import json
import copy

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
    free_memory,
    get_hf_token
)

from masks import (
    Mask, MaskConfig,
    Constrainer,
    LinearsWithMasks,
    RMSNormsWithMasks,
    EmbeddingsWithMasks
)

# Configure logger
from logging_config import configure_logging
configure_logging()
logger = logging.getLogger("merger")

HF_TOKEN = get_hf_token()
MERGER_CONFIG = "merger_config.json"
MASKS_SAFE = "masks.safetensors"
MASKS_TORCH = "masks.bin"

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
            f"There are only {len(child)} component models, "
            f"passed model index is {selected_idx}"
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
            f"There are {len(child)} component models, "
            f"but your passed factors have {len(factors)} values."
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

def set_masks(merger, mask_init):
    def _set_masks(root_module, strategy="random", **kwargs):
        init_method = get_init_method(strategy)
        masked_module_names = find_masked_modules(root_module)
        
        for module_name in tqdm(masked_module_names, desc="Setting up masks"):
            module_names = module_name.split(".")
            target_module = root_module
            for m_name in module_names:
                target_module = getattr(target_module, m_name)
    
            init_method(target_module, **kwargs)
    # Initialize masks based on config
    mask_strategy = mask_init["strategy"]
    if mask_strategy == "uniform":
        if not mask_init["factors"]:
            raise ValueError("Factors must be provided for uniform strategy")
        factors = mask_init["factors"]
        logger.info(f"Applying uniform masks with factors = {factors}.")
        _set_masks(merger.merger, strategy="uniform", factors=factors)
    elif mask_strategy == "random":
        logger.info(f"Applying random masks.")
        _set_masks(merger.merger, strategy="random")
    else:
        raise ValueError(
            f"Unknown mask initialization strategy: {mask_strategy}."
        )
    
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
        self.model_type = "merger"
        super().__init__(**kwargs)

class Merger(PreTrainedModel):
    def __init__(self, config: MergerConfig):
        super().__init__(config)
        self.merger_config = config
        self.num_models = len(config.model_paths)

        # Initialize configs but don't load models yet
        self.configs = [
            AutoConfig.from_pretrained(path, token=HF_TOKEN) 
            for path in config.model_paths
        ]
        """
        NOTE: Current implementation applies different masks to
        `embed_tokens` and `lm_head` layers. Therefore, despite
        these layers may initially be tied weights, after train-
        ing masks, the merged weights of these layers will be
        different from each other. An aesthetic pleasing solu-
        tion to this will be apply tied weight masks. However,
        for quick fix I only attempt for unsetting the untying
        weight config.
        """
        self.config = copy.deepcopy(self.configs[0])
        self.config.tie_word_embeddings = False ## Hotfix.
        
        # Initialize empty ModuleList for models - will be populated in from_pretrained
        self.models = nn.ModuleList()
        
        # Create merger with same architecture as first model but empty weights
        logger.info("Creating merger with dummy weights ...")
        self.merger = AutoModelForCausalLM.from_pretrained(
            config.model_paths[0],  # Use first model path
            config=self.configs[0],
            device_map="cpu",
            low_cpu_mem_usage=True,
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
        Every save calls during training point back to this function:
        ```
        Trainer._save_checkpoint() 
        -> Trainer.save_model() 
        -> Trainer._save()
        -> Merger.save_pretrained()
        ```
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
        save_directory = os.path.abspath(save_directory)
        self.config._name_or_path = save_directory
        super().save_pretrained(
            save_directory=save_directory,
            state_dict=trainable_state,
            **kwargs
        )

        # Rename the files after saving
        safe_file = os.path.join(save_directory, "model.safetensors")
        pytorch_file = os.path.join(save_directory, "pytorch_model.bin")
        
        if os.path.exists(safe_file):
            os.rename(safe_file, os.path.join(save_directory, MASKS_SAFE))
        elif os.path.exists(pytorch_file):
            os.rename(pytorch_file, os.path.join(save_directory, MASKS_TORCH))

        merger_config_file = os.path.join(save_directory, MERGER_CONFIG)
        # logger.info(f"Saving merger config to {merger_config_file}")
        self.merger_config.to_json_file(merger_config_file)
        config_to_copy = AutoConfig.from_pretrained(
            self.merger_config.model_paths[0]
        )
        config_to_copy._name_or_path = save_directory
        config_to_copy.save_pretrained(save_directory)
        

    @classmethod
    def from_config(cls, config, **kwargs):
        device_map = kwargs.pop('device_map', None)  # Remove device_map from kwargs
        # Initialize model instance
        model = cls(config)
        
        # Load component models without device_map
        for i in range(model.num_models):
            loaded_model = AutoModelForCausalLM.from_pretrained(
                config.model_paths[i],
                config=model.configs[i],
                token=HF_TOKEN,
                **kwargs
            )
            model.models.append(loaded_model)
            
            # Freeze component model parameters
            for param in loaded_model.parameters():
                param.requires_grad = False
        
        # Initialize masks before device mapping
        init_masks(model.merger, model.models, config)
        
        # Apply device mapping after masks are initialized
        if device_map is None:
            return model
            
        if is_fsdp_enabled() or is_deepspeed_zero3_enabled():
            logger.info(
                "Distributed training, return immediate model "
                "instead of dispatching to specified `device_map`."
            )
            return model

        if device_map == "auto":
            calculate_memory = lambda i: int(
                torch.cuda.get_device_properties(i)
                .total_memory / 1024**3 - 2
            )
            # Reserve 2GB buffer per GPU
            max_memory = {
                i: f"{calculate_memory(i)}GiB" 
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

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        **kwargs,
    ):
        config = MergerConfig.from_pretrained(
            os.path.join(pretrained_model_name_or_path, MERGER_CONFIG)
        )
        model = cls.from_config(config, **kwargs)
        assert pretrained_model_name_or_path, (
            "You must specify the path or name to your pretrained model."
        )
        state_dict = model.load_masks_state_dict(pretrained_model_name_or_path)
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict, strict=False
        )
        logger.info(f"Loaded masks from {pretrained_model_name_or_path}")
        return model
        

    def load_masks_state_dict(
        self, 
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]]
    ):
        # Try loading from safetensors first
        trainable_path = os.path.join(pretrained_model_name_or_path, MASKS_SAFE)
        if not os.path.exists(trainable_path):
            trainable_path = os.path.join(pretrained_model_name_or_path, MASKS_TORCH)
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

    def save_merged(
        self, 
        save_directory: Union[str, os.PathLike],
        state_dict: Optional[dict] = None,
        **kwargs
    ):
        """
        Compute merged weights using masks and component weights, 
        then save to directory. Removes component weights and masks 
        from the final state dict.
        """
        def compute(mask, weight):
            computed = mask * weight
            return computed.to(dtype=weight.dtype)
    
        def merge_linears(name, module):
            merged_state = {}
            keys_to_remove = set()
            for i in range(len(module.linears)):
                keys_to_remove.add(f"{name}.linears.{i}.weight")
                if module.linears[i].bias is not None:
                    keys_to_remove.add(f"{name}.linears.{i}.bias")
                keys_to_remove.add(f"{name}.weight_masks.{i}.weight")
                if module.bias_masks[i] is not None:
                    keys_to_remove.add(f"{name}.bias_masks.{i}.weight")
            
            # Get merged weights
            weight_masks = module.get_constrained_masks()["weight_masks"]
            merged_weight = sum(
                compute(mask, linear.weight)
                for mask, linear in zip(weight_masks, module.linears)
            ).cpu().detach()
            merged_state[f"{name}.weight"] = merged_weight
            
            # Get merged biases if they exist
            if hasattr(module, "bias_masks") and module.bias_masks[0] is not None:
                bias_masks = module.get_constrained_masks()["bias_masks"]
                merged_bias = sum(
                    compute(mask, linear.bias) if linear.bias is not None else 0
                    for mask, linear in zip(bias_masks, module.linears)
                ).cpu().detach()
                merged_state[f"{name}.bias"] = merged_bias
            return merged_state, keys_to_remove
    
        def merge_embeddings(name, module):
            merged_state = {}
            keys_to_remove = set()
            # Remove component embeddings and their masks
            for i in range(len(module.embeddings)):
                keys_to_remove.add(f"{name}.embeddings.{i}.weight")
                keys_to_remove.add(f"{name}.masks.{i}.weight")
            
            # Get merged weights
            masks = module.get_constrained_masks()["masks"]
            merged_weight = sum(
                compute(mask, emb.weight)
                for mask, emb in zip(masks, module.embeddings)
            ).cpu().detach()
            merged_state[f"{name}.weight"] = merged_weight
            return merged_state, keys_to_remove
    
        def merge_rmsnorms(name, module):
            merged_state = {}
            keys_to_remove = set()
            # Remove component norms and their masks
            for i in range(len(module.rms_norms)):
                keys_to_remove.add(f"{name}.rms_norms.{i}.weight")
                keys_to_remove.add(f"{name}.masks.{i}.weight")
            
            # Get merged weights
            masks = module.get_constrained_masks()["masks"]
            merged_weight = sum(
                compute(mask, norm.weight)
                for mask, norm in zip(masks, module.rms_norms)
            ).cpu().detach()
            merged_state[f"{name}.weight"] = merged_weight
            return merged_state, keys_to_remove
    
        # Initialization.
        save_directory = os.path.abspath(save_directory)
        self.config._name_or_path = save_directory
        
        merged_state = {}
        keys_to_remove = set()
        masked_modules = []
        for name, module in self.merger.named_modules():
            if any(mask_type in type(module).__name__ for mask_type in [
                "LinearsWithMasks", "EmbeddingsWithMasks", "RMSNormsWithMasks"
            ]):
                masked_modules.append((name, module))
    
        # Work, bitches. Mark component and mask keys for removal
        for name, module in tqdm(masked_modules, desc="Merging masked modules"):
            if isinstance(module, LinearsWithMasks):
                state, keys = merge_linears(name, module)
            elif isinstance(module, EmbeddingsWithMasks):
                state, keys = merge_embeddings(name, module)
            elif isinstance(module, RMSNormsWithMasks):
                state, keys = merge_rmsnorms(name, module)
            merged_state.update(state)
            keys_to_remove = keys_to_remove | keys

        # Copy over non-masked parameters
        full_state = self.merger.state_dict()
        keys_to_copy = set()
        
        for key, value in full_state.items():
            if any(remove_key in key for remove_key in keys_to_remove): continue
            if any(mask_key in key for mask_key in [
                "masks", "linears.", "embeddings.", "rms_norms."
            ]): continue
            keys_to_copy.add(key)

        if len(keys_to_copy) > 0:
            for key in tqdm(keys_to_copy, desc="Copying non-masked parameters"):
                merged_state[key] = full_state[key].to("cpu")

        ## Save the merged model.
        super().save_pretrained(
            save_directory=save_directory,
            state_dict=merged_state,
            **kwargs
        )

        ## Post process
        # Save pretrained auto assign architectures to `[Merger]`
        # However, I want it to be the architectures of component models.
        # This is critical for compability when use VLLM inference.
        architectures = [self.models[0].__class__.__name__]
        config = AutoConfig.from_pretrained(save_directory)
        config._name_or_path = save_directory
        config.architectures = architectures
        config.save_pretrained(save_directory)

def find_modules_to_add_masks(target_module):
    module_names_to_replace = []
    for parent_name, parent_module in target_module.named_modules():
        for name, child in parent_module.named_children():
            full_child_name = f"{parent_name}.{name}" if parent_name else name
            if (isinstance(child, (nn.Linear, nn.Embedding)) 
                or "RMSNorm" in type(child).__name__):
                module_names_to_replace.append(full_child_name)
    return module_names_to_replace

def init_masks(
    target_module: nn.Module, 
    ref_modules: nn.Module, 
    merger_config: MergerConfig
):
    """
    Replaces eligible submodules in target_module with masked 
    versions, using corresponding modules from ref_modules as 
    a reference for weights.

    Args:
        target_module: The module in which to replace submodules.
        ref_modules: A list of modules to use as a reference 
            for weights.
        strategy: The initialization strategy for factors 
            ("naive" or others to be implemented).
    """
    mode = merger_config.mode
    constrain_mode = merger_config.constrain_mode
    module_names_to_replace = find_modules_to_add_masks(target_module)
    
    for module_name in tqdm(
        module_names_to_replace, 
        desc="Initializing masks"
    ):
        module_names = module_name.split(".")
        target_child = target_module
        ref_children = ref_modules

        for m_name in module_names:
            target_child = getattr(target_child, m_name)
            ref_children = [
                getattr(ref_module, m_name) 
                for ref_module in ref_children
            ]

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
            new_module = EmbeddingsWithMasks(
                embeddings=ref_children, 
                modes=modes, 
                values=factors, 
                constrain_mode=constrain_mode
            )
        elif "RMSNorm" in type(target_child).__name__:
            new_module = RMSNormsWithMasks(
                rms_norms=ref_children, 
                modes=modes, 
                values=factors, 
                constrain_mode=constrain_mode
            )
        # Move new module's mask parameters to correct dtype
        target_dtype = ref_children[0].weight.dtype
        for param in new_module.parameters():
            # Only convert mask parameters, not the frozen model weights
            if param.requires_grad:
                param.data = param.data.to(dtype=target_dtype)

        # Replace the original module with the new masked module
        parent_module = target_module
        for m_name in module_names[:-1]:
            parent_module = getattr(parent_module, m_name)
        setattr(parent_module, module_names[-1], new_module)