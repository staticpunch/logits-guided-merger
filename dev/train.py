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
import safetensors
import numpy as np
import torch.nn as nn
import logging
import copy
import gc
import os

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
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data loading and preprocessing."""
    
    def __init__(self, tokenizer: PreTrainedTokenizerBase, dataset_configs, data_source_key, split, max_length):
        self.tokenizer = tokenizer
        self.dataset_configs = dataset_configs
        self.data_source_key = data_source_key
        self.split = split
        self.max_length = max_length

    
    def load_dataset(self):
        """Load and prepare the training dataset."""
        datasets_list = []
        for data_config, source_key in zip(self.dataset_configs, self.data_source_key):
            new_dataset = load_dataset(data_config, split=self.split)
            new_dataset = new_dataset.add_column(
                name="data_source", column=[source_key for _ in new_dataset]
            )
            datasets_list.append(new_dataset)
            
        train_dataset = datasets.concatenate_datasets([
            ds for ds in datasets_list[:] # 0 for rewrite, 1 for summarize.
        ])
        return train_dataset.shuffle(seed=42)
    
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

def selective_logits_target(logits_components, data_source):
    """Select appropriate logits based on data source."""

    logits_target = torch.empty_like(logits_components[0])
    for idx, source in enumerate(data_source):
        logits_target[idx] = logits_components[source][idx]

    return logits_target

def masked_kl_div(logits_a, logits_b, effective_idxs, temperature=1.0):
    # (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
    logits_a = logits_a.view(-1, logits_a.size(-1)) / temperature
    logits_b = logits_b.view(-1, logits_b.size(-1)) / temperature

    # (batch_size * seq_len,)
    mask = effective_idxs.view(-1)

    assert mask.size(0) == logits_a.size(0)

    log_probs_a = nn.functional.log_softmax(logits_a, dim=-1)
    log_probs_b = nn.functional.log_softmax(logits_b, dim=-1)

    # (batch_size * seq_len, vocab_size) -> (batch_size * seq_len)
    div = log_probs_a.exp() * (log_probs_a - log_probs_b)
    div = div.sum(-1)

    ## taking average on effective tokens.
    div = (div * mask).sum() / mask.sum() * (temperature ** 2)
    return div

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
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
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

        return (loss, outputs) if return_outputs else loss

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        assert model is not None, (
            "Must pass an initialized model to trainer instead of model path."
        )
        # Look for trainable parameters file
        masks_file = os.path.join(resume_from_checkpoint, "masks.safetensors")
        if not os.path.isfile(masks_file):
            masks_file = os.path.join(resume_from_checkpoint, "masks.bin")
        
        if not os.path.isfile(masks_file):
            raise ValueError(
                f"Can't find trainable parameters file in {resume_from_checkpoint}. "
                "Expected either masks.safetensors or masks.bin"
            )
    
        config_file = os.path.join(resume_from_checkpoint, CONFIG_NAME)
        if os.path.isfile(config_file):
            config = PretrainedConfig.from_json_file(config_file)
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warning(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )
    
        if os.path.isfile(masks_file):
            weights_only_kwarg = {"weights_only": True} if is_torch_greater_or_equal_than_1_13 else {}
            # If the model is on the GPU, it still works!
            # We load the model state dict on the CPU to avoid an OOM error.
            if self.args.save_safetensors and masks_file.endswith(".safetensors"):
                state_dict = safetensors.torch.load_file(masks_file, device="cpu")
            else:
                state_dict = torch.load(
                    masks_file,
                    map_location="cpu",
                    **weights_only_kwarg,
                )
    
            # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
            # which takes *args instead of **kwargs
            load_result = model.load_state_dict(state_dict, False)
            if len(load_result.missing_keys) != 0:
                logger.info(
                    "There were missing keys in the checkpoint model loaded. "
                    "However, this is an expected behavior since we are only "
                    "loading partial weights (masks)."
                )
            # release memory
            del state_dict
            gc.collect()

@dataclass
class Args:
    model_paths: List[str]
    dataset_configs: List[str]
    data_source_key: List[int]
    mode: str
    constrain_mode: str
    train_split: str
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
    remove_unused_columns: bool
    logging_first_step: bool
    bf16: bool
    gradient_checkpointing: bool
    validation_split: str = None
    max_length: int = 4096

def main():
    """Main training function."""
    parser = HfArgumentParser(Args)
    (args,) = parser.parse_args_into_dataclasses()

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
        args.data_source_key, args.train_split, args.max_length
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
    set_masks(merger.merger, strategy="uniform", factors=[0.5, 0.5])
    # set_masks(merger.merger, strategy="random")
    
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
    
    # Monitor memory usage
    initial_memory = torch.cuda.memory_allocated()
    logger.info(f"Initial GPU memory allocated: {initial_memory / 1024**3:.2f} GB")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    final_memory = torch.cuda.memory_allocated()
    logger.info(f"Final GPU memory allocated: {final_memory / 1024**3:.2f} GB")
    logger.info(f"Freed GPU memory: {(initial_memory - final_memory) / 1024**3:.2f} GB")
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()