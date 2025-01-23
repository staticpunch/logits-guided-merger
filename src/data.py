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
logger = logging.getLogger(__name__)

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
            
        
        log_messages.append("==========================")
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
            pad_to_multiple_of=self.pad_to_multiple_of,
            padding_side="right",
        )

        labels_batch = self.tokenizer.pad(
            labels_list,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
            padding_side="right",
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