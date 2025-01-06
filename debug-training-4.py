# %%
"""
Model merging training implementation using PyTorch and Transformers.
Implements custom data collation and training for merged language models.
"""
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union, Mapping
from abc import ABC, abstractmethod

import datasets
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import logging
import copy
import gc

from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizerBase,
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

from accurate_masks import (
# from efficient_masks import (
    MergerConfig,
    Merger,
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
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# %%
import os
# Option 1: Set specific GPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 

# %%
class DataProcessor:
    """Handles data loading and preprocessing."""
    
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
    
    def load_dataset(self):
        """Load and prepare the training dataset."""
        summarize_train = load_dataset(
            "HuggingFaceTB/smoltalk",
            "smol-summarize",
            split="train"
        )
        summarize_train = summarize_train.add_column(
            name="data_source",
            column=[1 for _ in summarize_train]
        )
        return summarize_train.shuffle(seed=42).select(range(5000))
    
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
            max_length=2048,
            add_special_tokens=False
        )

# %%
def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded

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
            
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer, inputs_ids, return_tensors="pt", 
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

# %%
def selective_logits_target(logits_components, data_source):
    """Select appropriate logits based on data source."""
    stacked_logits = torch.stack(logits_components)
    indices = data_source.unsqueeze(-1).unsqueeze(-1)
    return stacked_logits[indices]

class MergerTrainer(Trainer):
    """Custom trainer for merged model training."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        data_source = inputs.pop("data_source")
        effective_idxs = (labels != -100).float().unsqueeze(dim=-1)
        
        outputs = model(**inputs)
        logits_merged = outputs["merger_outputs"].logits
        logits_components = [x.logits for x in outputs["components_outputs"]]

        # Compute target logits and KL divergence
        logits_target = selective_logits_target(logits_components, data_source)
        temperature = 1.0
        kl_fct = nn.KLDivLoss(reduction="none")
        diff = (
            kl_fct(
                F.log_softmax(logits_target / temperature, dim=-1),
                F.softmax(logits_merged / temperature, dim=-1)
            )
            * (temperature) ** 2
        )
        
        # Calculate final loss
        loss = (diff * effective_idxs).sum(dim=-1)
        loss = (loss / effective_idxs.sum(dim=1)).mean()

        return (loss, outputs) if return_outputs else loss

# %%
@dataclass
class Args:
    model_name: str = "..."  # You can replace this with any causal language model from HuggingFace
    dataset_name: str = "..."  # Replace with your dataset name (e.g., "your_username/your_dataset")
    train_split: str = "train"  # e.g., "train[:80%]" for an 80/20 train/validation split
    validation_split: str = None  # e.g., "train[80%:]"
    output_dir: str = "./trained_masks"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 16
    learning_rate: float = 3e-2
    num_train_epochs: int = 4
    save_steps: int = 1000
    eval_steps: int = 5000
    logging_steps: int = 10
    logging_dir: str = "./trained_masks/logs"
    evaluation_strategy: str = "steps"
    report_to: str = None
    remove_unused_columns: bool = False
    logging_first_step: bool = True

# %%
# Initialize configuration
merge_config = MergerConfig(
    model_paths=[
        "nguyenthanhdo/llama32_smol_rewrite_50k",
        "nguyenthanhdo/llama32_smol_summarize_50k",
    ],
    mode="vector_input",
    constrain_mode="01"
)

# Setup tokenizer and data processing
tokenizer = AutoTokenizer.from_pretrained(merge_config.model_paths[0])
tokenizer.pad_token = tokenizer.eos_token
data_processor = DataProcessor(tokenizer)
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
    device_map="auto",
    attn_implementation="flash_attention_2",
)
set_masks(merger.merger, strategy="uniform", factors=[0.95, 0.05])

# %%
# Monitor memory usage
initial_memory = torch.cuda.memory_allocated()
logger.info(f"Initial GPU memory allocated: {initial_memory / 1024**3:.2f} GB")

gc.collect()
torch.cuda.empty_cache()

final_memory = torch.cuda.memory_allocated()
logger.info(f"Final GPU memory allocated: {final_memory / 1024**3:.2f} GB")
logger.info(f"Freed GPU memory: {(initial_memory - final_memory) / 1024**3:.2f} GB")

# %%
# Setup training arguments and data collator
args = Args()
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    num_train_epochs=args.num_train_epochs,
    save_steps=args.save_steps,
    evaluation_strategy=args.evaluation_strategy if args.validation_split else "no",
    eval_steps=args.eval_steps if args.validation_split else None,
    logging_steps=args.logging_steps,
    logging_dir=args.logging_dir,
    report_to=args.report_to,  # Enable TensorBoard logging
    remove_unused_columns=args.remove_unused_columns,
    logging_first_step=args.logging_first_step
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

# %%
trainer.train()