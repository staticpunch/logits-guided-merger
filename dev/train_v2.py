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
import shutil

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

from merger import (
    MergerConfig,
    Merger,
)
from initializer import MaskInitializer
# Configure logger
from logging_config import configure_logging
configure_logging()
logger = logging.getLogger("train")

class DataProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, dataset_configs: Dict[str, int], 
                 data_source_key: List[int], split: str, max_length: int, 
                 train_on_inputs: bool = False):  # Add this parameter
        self.tokenizer = tokenizer
        self.dataset_configs = dataset_configs
        self.data_source_key = data_source_key
        self.split = split
        self.max_length = max_length
        self.train_on_inputs = train_on_inputs  # Store the flag

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
        logger.info(f"Training on {len(train_dataset)} samples total.")
        return train_dataset.shuffle(seed=101)

    def tokenize(self, element):
        """Tokenize a single element and mark tokens for loss computation based on train_on_inputs."""
        effective_spans = []
        current_position = 0
        
        # Track positions of assistant messages
        for message in element["messages"]:
            message_tokens = self.tokenizer.apply_chat_template(
                [message],
                tokenize=True,
                add_generation_prompt=False
            )
            
            if message["role"] == "assistant" or (self.train_on_inputs and message["role"] == "user"):
                effective_spans.append((
                    current_position,
                    current_position + len(message_tokens)
                ))
            current_position += len(message_tokens)

        # Tokenize full conversation
        tokenized = self.tokenizer(
            self.tokenizer.apply_chat_template(
                element["messages"],
                tokenize=False,
                add_generation_prompt=False
            ),
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False
        )

        # Create labels with -100 for tokens we don't want to compute loss on
        labels = [-100] * len(tokenized["input_ids"])
        for start, end in effective_spans:
            for i in range(start, min(end, len(labels))):
                labels[i] = tokenized["input_ids"][i]
                
        tokenized["labels"] = labels
        return tokenized

@dataclass 
class MergerDataCollator:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = 4
    return_tensors: str = "pt"
    
    def _format_batch_log(self, batch):
        """Format the batch sample log showing colored text chunks."""
        input_ids = batch["input_ids"][0].tolist()  
        labels = batch["labels"][0].tolist()
        
        # Build chunks of tokens with same label type (-100 or non -100)
        chunks = []
        current_chunk = {"tokens": [], "is_ignored": labels[0] == -100}
        
        for token_id, label in zip(input_ids, labels):
            is_ignored = label == -100
            # If label type changes, start new chunk
            if is_ignored != current_chunk["is_ignored"]:
                chunks.append(current_chunk)
                current_chunk = {"tokens": [], "is_ignored": is_ignored}
            current_chunk["tokens"].append(token_id)
        
        # Add final chunk
        chunks.append(current_chunk)
        
        # Format output
        log_messages = []
        log_messages.append("=== Sample text chunks ===")
        # Decode and display each chunk with appropriate color
        for i, chunk in enumerate(chunks):
            text = self.tokenizer.decode(chunk["tokens"])
            color = ("\033[90m" if (i == len(chunks) - 1) and chunk["is_ignored"] # Gray for padded.
                     else "\033[91m" if chunk["is_ignored"] # Red for ignored.
                     else "\033[92m") # Green for trained.
            log_messages.append(f"{color}{text}\033[0m")
            
        
        log_messages.append("\n========================")
        return "\n".join(log_messages)

    def __post_init__(self):
        self.first_batch = True

    def __call__(self, examples):
        """Process a batch with proper padding and selective labels for assistant tokens."""
        if not isinstance(examples[0], Mapping):
            raise ValueError("Data collator only processes list of dictionaries.")

        inputs_ids = []
        labels_list = []
        data_sources = []
        
        for i in range(len(examples)):
            _ = examples[i].pop("attention_mask")
            inputs_ids.append({"input_ids": examples[i].pop("input_ids")})
            labels_list.append({"input_ids": examples[i].pop("labels")})
            data_sources.append(examples[i].pop("data_source"))

        batch = self.tokenizer.pad(
            inputs_ids,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of
        )

        labels_batch = self.tokenizer.pad(
            labels_list,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of
        )
        
        labels = labels_batch["input_ids"]
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        batch["data_source"] = torch.tensor(
            [src for src in data_sources], dtype=torch.long
        )
        
        for key in examples[0]:
            if key in batch:
                raise ValueError(
                    f"`{key}` feature is collated. "
                    "Overriding it with its initial values is prohibited."
                )
            else:
                batch[key] = [x[key] for x in examples]

        # All logging in a single info_once call
        if self.first_batch:
            logger.info(f"Logging first batch sample:\n{self._format_batch_log(batch)}")
            self.first_batch = False
        
        return batch

def selective_logits_target(logits_components, data_source):
    """Select appropriate logits based on data source."""

    logits_target = torch.empty_like(logits_components[0])
    for idx, source in enumerate(data_source):
        logits_target[idx] = logits_components[source][idx]

    return logits_target

def compute_kl_div(logits_a, logits_b, effective_idxs, temperature=1.0):
    kl_fct = nn.KLDivLoss(reduction="none")
    diff = (
        kl_fct(
            F.log_softmax(logits_b / temperature, dim=-1),
            F.softmax(logits_a / temperature, dim=-1)
        )
        * (temperature**2)
    )
    
    # Calculate final loss
    loss = (diff.sum(-1) * effective_idxs).sum() / effective_idxs.sum()
    return loss

def compute_entropy(logits, effective_idxs, temperature=1.0):
    softmax = F.softmax(logits / temperature, dim=-1)
    log_softmax = F.log_softmax(logits / temperature, dim=-1)
    entropy = (- softmax * log_softmax) * (temperature**2)
    loss = (entropy.sum(-1) * effective_idxs).sum() / effective_idxs.sum()
    return loss

class MergerTrainer(Trainer):
    """
    Hello it's Nguyen Thanh Do.
    """
    def __init__(self, *args, loss_func_name="kl_div", mask_decay=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func_name = loss_func_name
        if self.args.should_save:
            assert self.loss_func_name in ("entropy", "kl_div"), (
                f"Loss function {self.loss_func_name} is not supported. "
                f"Please select a loss function of type `entropy` or `kl_div`."
            )
            loss_func_full_name = (
                "Minimizing Entropy loss (AdaMerging)" if loss_func_name == "entropy"
                else "Disentangled KL Divergence loss" if loss_func_name == "kl_div"
                else "Not yet supported loss"
            )
            logger.info(f"You are training masks with {loss_func_full_name}.")
            
        self.mask_decay = mask_decay
        self.track_masks_params(print_param_count=True)
        
    def track_masks_params(self, print_param_count=False):
        params_a, params_b = [], []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "masks.0." in name:
                    params_a.append(param.flatten())
                elif "masks.1." in name:
                    params_b.append(param.flatten())
                    
        statistics =  {
            "params_a": torch.cat(params_a),
            "params_b": torch.cat(params_b)
        }
        
        if print_param_count and self.args.should_save:
            count_millions = (
                statistics["params_a"].numel() 
                + statistics["params_b"].numel()
            ) / 1_000_000
            logger.info(f"Trainable parameters (masks): {count_millions:.2f}M")
            
        return statistics
            
        
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        with torch.no_grad():
            all_params = self.track_masks_params()
            params_a, params_b = all_params["params_a"], all_params["params_b"]

            logs["mean_a"] = params_a.mean().item()
            logs["mean_b"] = params_b.mean().item()
        
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                speed_metrics("train", start_time, num_tokens=self.state.num_input_tokens_seen)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
    
    def compute_kl_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        data_source = inputs.pop("data_source")
        effective_idxs = (labels != -100).float()
        
        outputs = model(**inputs)
        logits_merged = outputs["merger_outputs"].logits
        # logits_target = logits_merged
        logits_components = [x.logits.detach() for x in outputs["components_outputs"]]

        # Compute target logits and KL divergence
        logits_target = selective_logits_target(logits_components, data_source)
        loss = compute_kl_div(logits_merged, logits_target, effective_idxs)
        return loss

    def compute_entropy_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        effective_idxs = (labels != -100).float()
        
        outputs = model(**inputs)
        logits_merged = outputs["merger_outputs"].logits

        loss = compute_entropy(logits_merged, effective_idxs)
        return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        ## Debug: Before computing loss.
        # ------------------------------------------------------
        for name, param in model.named_parameters():
            if param.requires_grad and torch.isnan(param).any():
                logger.info(f"NaN detected in parameter {name}")
                import pdb; pdb.set_trace()
        # ------------------------------------------------------
        
        loss_func = (
            self.compute_kl_loss if self.loss_func_name == "kl_div"
            else self.compute_entropy_loss
        )
        loss = loss_func(model, inputs, return_outputs, num_items_in_batch)
        if self.mask_decay is not None:
            params_a = self.track_masks_params()["params_a"]
            params_b = self.track_masks_params()["params_b"]

            mean_a = torch.mean(params_a**2)
            mean_b = torch.mean(params_b**2)
            loss_w = self.mask_decay * (mean_b / (mean_a + mean_b))
            loss += loss_w
            
        ## Debug: After computing loss.
        # ------------------------------------------------------
        if torch.isnan(loss):
            logger.info(f"Loss became NaN")
            import pdb; pdb.set_trace()
        # ------------------------------------------------------

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
    eval_strategy: str
    report_to: str
    remove_unused_columns: bool = False
    logging_first_step: bool = True
    bf16: bool = True
    gradient_checkpointing: bool = False
    validation_split: Optional[str] = None
    loss_func_name: str = "kl_div"
    mask_decay: Optional[float] = None
    train_on_inputs: bool = False
    lr_scheduler: str = "constant"

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
    merger_config = MergerConfig(
        model_paths=args.model_paths,
        mode=args.mode,
        constrain_mode=args.constrain_mode
    )
    
    # Setup tokenizer and data processing
    tokenizer = AutoTokenizer.from_pretrained(merger_config.model_paths[0])
    tokenizer.pad_token = tokenizer.eos_token
    data_processor = DataProcessor(
        tokenizer, args.dataset_configs, 
        args.source_keys, args.train_split, 
        args.max_length, args.train_on_inputs
    )
    train_dataset = data_processor.load_dataset()
    tokenized_dataset = train_dataset.map(
        data_processor.tokenize,
        remove_columns=["messages"]
    )
    
    # Initialize merger model
    merger = Merger.from_config(
        merger_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    # set_masks(merger, args.mask_init)
    MaskInitializer().initialize(merger, args.mask_init)
    # import pdb; pdb.set_trace()
    
    # torch distributed hack
    merger._ddp_params_and_buffers_to_ignore = [
        name for name, buffer in merger.named_buffers() 
        if buffer.dtype == torch.bool
    ]
    
    # Setup training arguments and data collator
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type=args.lr_scheduler,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        eval_strategy=args.eval_strategy if args.validation_split else "no",
        eval_steps=args.eval_steps if args.validation_split else None,
        logging_steps=args.logging_steps,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to=args.report_to,  # Enable TensorBoard logging
        remove_unused_columns=args.remove_unused_columns,
        logging_first_step=args.logging_first_step,
        gradient_checkpointing=args.gradient_checkpointing,
        # bf16=args.bf16,
        # fp16=not args.bf16,
        ddp_find_unused_parameters=False
    )
    
    data_collator = MergerDataCollator(tokenizer)
    
    # Initialize and start training
    trainer = MergerTrainer(
        model=merger,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        ## ----- additional arguments -----
        loss_func_name=args.loss_func_name,
        mask_decay=args.mask_decay,
        ## --------------------------------
    )
    
    # Copy config to output directory
    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copy(config_file, os.path.join(args.output_dir, "train_config.yaml"))
    
    # Start training
    trainer.train()
    if trainer.args.should_save:
        ## There is a weird bug regarding saving config file.
        ## To ensure correctly saving config.json, for now
        ## you must `save_pretrained` before `save_merged`
        merger.save_pretrained(args.output_dir)
        merger.save_merged(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()