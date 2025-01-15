"""
Model merging training implementation using PyTorch and Transformers.
Implements custom data collation and training for merged language models.
"""

from dataclasses import dataclass
from typing import (
    Any, Callable, Dict, 
    List, NewType, Optional, 
    Tuple, Union, Mapping
)
from abc import ABC, abstractmethod
from datasets import load_dataset, concatenate_datasets
from accelerate.logging import get_logger
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch
import safetensors
import math
import yaml
import logging
import copy
import gc
import os
import argparse
import sys

from tqdm import tqdm
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaConfig,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    default_data_collator,
    is_torch_xla_available,
    set_seed,
)

from transformers.utils import CONFIG_NAME
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_13

# from accurate_masks import (
# from efficient_masks import (
from merger import (
    MergerConfig,
    # Merger,
    NewMerger,
    init_masks,
    set_masks
)

from utils import (
    generate, 
    get_hidden_states, 
    get_logits,
    free_memory
)
# Configure logger
from logging_config import configure_logging
configure_logging()
logger = logging.getLogger("train")

class DataProcessor:
    """Handles data loading and preprocessing."""
    
    def __init__(self, tokenizer: PreTrainedTokenizerBase, dataset_configs: Dict[str, int], 
                 data_source_key: List[int], split: str, max_length: int):
        self.tokenizer = tokenizer
        self.dataset_configs = dataset_configs
        self.data_source_key = data_source_key
        self.split = split
        self.max_length = max_length

    def load_dataset(self):
        """Load and prepare the training dataset."""
        datasets_list = []
        
        for (data_config, num_samples), source_key in zip(
            self.dataset_configs.items(), self.data_source_key
        ):
            logger.info(f"Loading {num_samples} samples from {data_config} with source {source_key}.")
            new_dataset = load_dataset(data_config, split=self.split)
            
            # Randomly sample the specified number of examples
            if num_samples and num_samples < len(new_dataset):
                new_dataset = new_dataset.shuffle(seed=42).select(range(num_samples))
            
            new_dataset = new_dataset.add_column(
                name="data_source", 
                column=[source_key for _ in new_dataset]
            )
            datasets_list.append(new_dataset)
            
        train_dataset = concatenate_datasets(datasets_list)
        logger.info(f">>> Training on {len(train_dataset)} samples total.")
        return train_dataset.shuffle(seed=101)

    def tokenize(self, element):
        """Tokenize a single element from the dataset."""
        templated = self.tokenizer.apply_chat_template(
            element["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        return self.tokenizer(
            templated,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False
        )

@dataclass
class MergerDataCollator:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, examples):
        """
        copied from DataCollatorForLanguageModeling
        examples: List[Union[List[int], Any, Dict[str, Any]]]
        """
        # Handle dict or lists with proper padding and conversion to tensor.
        if not isinstance(examples[0], Mapping):
            raise ValueError("Data collator only processes list of dictionaries.")

        inputs_ids = []
        data_sources = []
        for i in range(len(examples)):
            _ = examples[i].pop("attention_mask")
            inputs_ids.append({"input_ids": examples[i].pop("input_ids")})
            data_sources.append(examples[i].pop("data_source"))
            
        batch = self.tokenizer.pad(
            inputs_ids, 
            # padding='max_length',  # This forces padding to max_length
            # max_length=3072,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of
        )

        labels = batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        # Handle data_source - convert to tensor
        batch["data_source"] = torch.tensor(
            [src for src in data_sources], dtype=torch.long
        )
        
        for key in examples[0]:
            if key in batch:
                raise ValueError(
                    f"`{key}` feature is collated. "
                    "Overriding it with its initial values is prohibitted."
                )
            else:
                batch[key] = [x[key] for x in examples]
        logger.info_once(f">>> Collator output keys: {batch.keys()}")
        return batch

def selective_logits_target(logits_components, data_source):
    """Select appropriate logits based on data source."""

    logits_target = torch.empty_like(logits_components[0])
    for idx, source in enumerate(data_source):
        logits_target[idx] = logits_components[source][idx]

    return logits_target

def builtin_kl_div(logits_a, logits_b, effective_idxs, temperature=1.0):
    kl_fct = nn.KLDivLoss(reduction="none")
    diff = (
        kl_fct(
            F.log_softmax(logits_b / temperature, dim=-1),
            F.softmax(logits_a / temperature, dim=-1)
        )
        * (temperature) ** 2
    )
    
    # Calculate final loss
    loss = (diff.sum(-1) * effective_idxs).sum() / effective_idxs.sum()
    return loss

class MergerTrainer(Trainer):
    """Custom trainer for merged model training."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decay = 0.1
        
    def track_masks_params(self):
        params_a, params_b = [], []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "masks.0." in name:
                    params_a.append(param.flatten())
                elif "masks.1." in name:
                    params_b.append(param.flatten())
        return {
            "params_a": torch.cat(params_a),
            "params_b": torch.cat(params_b)
        }
                
        
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        with torch.no_grad():
            all_params = self.track_masks_params()
            params_a, params_b = all_params["params_a"], all_params["params_b"]

            count_millions = (params_a.numel() + params_b.numel()) / 1_000_000
            formatted_count = f"{count_millions:.2f}M"

            logs["mean_a"] = params_a.mean().item()
            logs["mean_b"] = params_b.mean().item()
            logs["trainable_params"] = formatted_count
        
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                speed_metrics("train", start_time, num_tokens=self.state.num_input_tokens_seen)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Before computing loss
        for name, param in model.named_parameters():
            if param.requires_grad and torch.isnan(param).any():
                print(f"NaN detected in parameter {name}")
                import pdb; pdb.set_trace()
        
        labels = inputs.pop("labels")
        data_source = inputs.pop("data_source")
        effective_idxs = (labels != -100).float()
        
        outputs = model(**inputs)
        logits_merged = outputs["merger_outputs"].logits
        # logits_target = logits_merged
        logits_components = [x.logits.detach() for x in outputs["components_outputs"]]

        # Compute target logits and KL divergence
        logits_target = selective_logits_target(logits_components, data_source)
        loss = builtin_kl_div(logits_merged, logits_target, effective_idxs)

        if False:
            params_a = self.track_masks_params()["params_a"]
            params_b = self.track_masks_params()["params_b"]

            mean_a = torch.mean(params_a**2)
            mean_b = torch.mean(params_b**2)
            loss_w = self.decay * (mean_b / (mean_a + mean_b))
            
            loss = loss + loss_w
            
        # if torch.isnan(loss):
        #     import pdb; pdb.set_trace()

        return (loss, outputs) if return_outputs else loss
    
@dataclass
class TrainingConfig:
    """Configuration for training loaded from YAML."""
    # Model configuration
    model_paths: List[str]
    mode: str
    constrain_mode: str
    mask_init: dict
    
    # Dataset configuration
    dataset_configs: Dict[str, int]  # Path to dataset -> number of samples
    source_keys: List[int]
    train_split: str
    max_length: int
    
    # Training parameters
    output_dir: str
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: int
    save_steps: int
    eval_steps: int
    logging_steps: int
    logging_dir: str
    eval_strategy: str
    report_to: str
    remove_unused_columns: bool = False
    logging_first_step: bool = True
    bf16: bool = True
    gradient_checkpointing: bool = False
    validation_split: Optional[str] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            # Convert learning rate to float if it's a string
            if isinstance(config_dict['learning_rate'], str):
                config_dict['learning_rate'] = float(config_dict['learning_rate'])
        return cls(**config_dict)

def main():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='Path to YAML config file')
    config_file = parser.parse_args().config_file
    args = TrainingConfig.from_yaml(config_file)

    # Initialize configuration
    merge_config = MergerConfig(
        model_paths=args.model_paths,
        mode=args.mode,
        constrain_mode=args.constrain_mode
    )
    
    # Setup tokenizer and data processing
    tokenizer = AutoTokenizer.from_pretrained(merge_config.model_paths[0])
    tokenizer.pad_token = tokenizer.eos_token
    data_processor = DataProcessor(
        tokenizer, args.dataset_configs, 
        args.source_keys, args.train_split, args.max_length
    )
    train_dataset = data_processor.load_dataset()
    tokenized_dataset = train_dataset.map(
        data_processor.tokenize,
        remove_columns=["messages"]
    )
    
    # Initialize merger model
    merger = NewMerger.from_pretrained(
        None,
        merge_config,
        torch_dtype=torch.bfloat16,
        # device_map="auto",
        attn_implementation="flash_attention_2",
    )
    # torch distributed hack
    merger._ddp_params_and_buffers_to_ignore = [
        name for name, buffer in merger.named_buffers() 
        if buffer.dtype == torch.bool
    ]
    # set_masks(merger.merger, strategy="uniform", factors=[0.5, 0.5])
    # set_masks(merger.merger, strategy="random")

    # Initialize masks based on config
    mask_strategy = args.mask_init["strategy"]
    if mask_strategy == "uniform":
        if not args.mask_init["factors"]:
            raise ValueError("Factors must be provided for uniform strategy")
        factors = args.mask_init["factors"]
        logger.info(f"Applying uniform masks with factors = {factors}.")
        set_masks(merger.merger, strategy="uniform", factors=factors)
    elif mask_strategy == "random":
        logger.info(f"Applying random masks.")
        set_masks(merger.merger, strategy="random")
    else:
        raise ValueError(f"Unknown mask initialization strategy: {mask_strategy}.")
    
    # Setup training arguments and data collator
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        eval_strategy=args.eval_strategy if args.validation_split else "no",
        eval_steps=args.eval_steps if args.validation_split else None,
        logging_steps=args.logging_steps,
        logging_dir=args.logging_dir,
        report_to=args.report_to,  # Enable TensorBoard logging
        remove_unused_columns=args.remove_unused_columns,
        logging_first_step=args.logging_first_step,
        gradient_checkpointing=args.gradient_checkpointing,
        # bf16=args.bf16,
        # fp16=not args.bf16,
        ddp_find_unused_parameters=False
    )
    
    data_collator = MergerDataCollator(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )
    
    # Initialize and start training
    trainer = MergerTrainer(
        model=merger,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
    
    # Copy config to output directory
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.dump(args.__dict__, f)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()