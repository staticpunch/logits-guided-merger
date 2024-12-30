import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union, Mapping
from abc import ABC, abstractmethod
from dataclasses import dataclass

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
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer
)

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast
)

from efficient_masks import (
    MergerConfig,
    Merger,
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
        for i in range(len(examples)):
            _ = examples[i].pop("attention_mask")
            inputs_ids.append(
                {"input_ids": examples[i].pop("input_ids")}
            )
            
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer, inputs_ids, return_tensors="pt", 
            pad_to_multiple_of=self.pad_to_multiple_of
        )

        labels = batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        for key in examples[0]:
            if key in batch:
                raise ValueError(f"`{key}` feature is collated. Overriding it with its initial values is prohibitted.")
            else:
                batch[key] = [x[key] for x in examples]
        return batch

# --- Custom Trainer with Custom compute_loss ---
def get_entropy_weights(logits_a, logits_b, epsilon=1e-8):
    """
    Calculates entropy-based weights for merging two sets of logits.

    This function efficiently computes the weights for logits_a and logits_b
    based on their respective entropies. It combines the functionality of
    calculating entropy, normalizing weights, and handling potential
    division-by-zero issues.

    Args:
        logits_a: A PyTorch tensor representing the first set of logits.
        logits_b: A PyTorch tensor representing the second set of logits.
        epsilon: A small value to prevent division by zero.

    Returns:
        A tuple containing two tensors: (weight_a, weight_b), representing the
        normalized entropy-based weights for logits_a and logits_b, respectively.
    """

    # Calculate probabilities
    probs_a = F.softmax(logits_a, dim=-1)
    probs_b = F.softmax(logits_b, dim=-1)

    # Calculate entropies with epsilon for numerical stability
    entropy_a = -(probs_a * probs_a.log()).sum(dim=-1, keepdim=True)
    entropy_b = -(probs_b * probs_b.log()).sum(dim=-1, keepdim=True)

    # Calculate inverse entropies (weights)
    inv_entropy_a = 1.0 / (entropy_a + epsilon)
    inv_entropy_b = 1.0 / (entropy_b + epsilon)

    # Normalize weights
    total_inv_entropy = inv_entropy_a + inv_entropy_b
    weight_a = inv_entropy_a / (total_inv_entropy + epsilon)  
    weight_b = inv_entropy_b / (total_inv_entropy + epsilon) 

    return weight_a, weight_b
    
def compute_logits_target(logits_components):
    assert len(logits_components) == 2
    logits_a, logits_b = logits_components
    weight_a, weight_b = get_entropy_weights(logits_a, logits_b)

    logits_target = weight_a * logits_a + weight_b * logits_b

    return logits_target
    
class Mergerrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):

        ## Read inputs. Compute a simple mask `effective_idxs`
        ## to exclude non-trainable tokens (e.g., PAD tokens).
        labels = inputs.pop("labels")
        
        effective_idxs = labels.clone()
        effective_idxs[effective_idxs != -100] = 1.0
        effective_idxs[effective_idxs == -100] = 0.0

        ## Forward pass
        outputs = model(**inputs)
        logits_merged = outputs["merger_outputs"].logits
        logits_components = [x.logits for x in outputs["components_outputs"]]

        ## Compute targt logits
        logits_target = compute_logits_target(logits_components)

        ## Compute KL divergence
        temperature = 1.0
        kl_fct = nn.KLDivLoss(reduction="none")
        diff = (
            kl_fct(
                F.log_softmax(logits_target / temperature, dim=-1),
                F.softmax(logits_merged / temperature, dim=-1)
            )
            * (temperature) ** 2
        )
        
        ### Exclude non-trainable tokens from taking loss
        loss = (diff * effective_idxs).sum(dim=-1)
        loss = (loss / effective_idxs.sum(dim=-1)).mean()
        
        return (loss, outputs) if return_outputs else loss

@dataclass
class Args:
    model_name: str = "..."  # You can replace this with any causal language model from HuggingFace
    dataset_name: str = "..."  # Replace with your dataset name (e.g., "your_username/your_dataset")
    train_split: str = "train"  # e.g., "train[:80%]" for an 80/20 train/validation split
    validation_split: str = None  # e.g., "train[80%:]"
    output_dir: str = "./trained_masks"
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    logging_dir: str = "./trained_masks/logs"
    evaluation_strategy: str = "steps"
    report_to: str = "tensorboard"
    remove_unused_columns: bool = False


if __name__ == "__main__":
    
    merge_config = MergerConfig(
        model_paths = [
            "nguyenthanhdo/llama32_smol_rewrite_50k",
            "nguyenthanhdo/llama32_smol_summarize_50k",
            # "/workspace/HUB_LLM/Llama-3.2-3B-Instruct",
        ],
        mode = "vector_input",
        # mode = "scalar",
        constrain_mode = "01"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(merge_config.model_paths[0])
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = MergerDataCollator(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt"
    )
    
    args = Args()
    
    # --- Training Arguments ---
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
        remove_unused_columns=args.remove_unused_columns
    )
    
    def tokenize(element):
        templated = tokenizer.apply_chat_template(
            element["messages"], tokenize=False, add_generation_prompt=False
        )
        outputs = tokenizer(
            templated,
            truncation=True,
            max_length=4096,
            add_special_tokens=False,
            # return_tensors="pt"
        )
        return outputs
        
    train_mini = load_dataset("json", data_files=["train_mini.jsonl"], split="train")
    tokenized_mini = train_mini.map(tokenize, remove_columns=["messages"])
    merger = Merger(merge_config)
    merger.__post_init__()
    set_masks(merger.merger, strategy="uniform", factors=[0.5, 0.5])

    # --- Initialize Trainer ---
    trainer = Mergerrainer(
        model=merger,
        args=training_args,
        train_dataset=tokenized_mini,
        eval_dataset=None,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model and tokenizer saved to {args.output_dir}")