"""
Model merging training implementation using PyTorch and Transformers.
Implements custom data collation and training for merged language models.
"""
import os
import argparse
import shutil
import logging

import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    TrainingArguments,
)

from merger import MergerConfig, Merger
from initializer import MaskInitializer
from data import DataProcessor, MergerDataCollator
from trainer import MergerTrainer, TrainingConfig

# Configure logger
from logging_config import configure_logging
configure_logging()
logger = logging.getLogger("train")
        
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