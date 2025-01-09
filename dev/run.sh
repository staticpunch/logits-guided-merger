#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Define default values for arguments
MODEL_PATHS=("nguyenthanhdo/Llama-3.2-3B-Instruct" "nguyenthanhdo/llama32_3b_wizard_vi_27k_from_base")
DATASET_CONFIGS=("/workspace/logits-guided-merger/data/wizard-en-27k" "/workspace/logits-guided-merger/data/wizard-vi-27k")
DATA_SOURCE_KEY=(0 1)
MODE="vector_input"
CONSTRAIN_MODE="identity"
TRAIN_SPLIT="train"
OUTPUT_DIR="../results/wizard-run-2"
PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=32
LEARNING_RATE=5e-3
NUM_TRAIN_EPOCHS=5
SAVE_STEPS=100
EVAL_STEPS=5000
LOGGING_STEPS=10
LOGGING_DIR="${OUTPUT_DIR}/logs"
EVAL_STRATEGY="steps"
REPORT_TO="none"
REMOVE_UNUSED_COLUMNS=False
LOGGING_FIRST_STEP=True
BF16=True
GRADIENT_CHECKPOINTING=False
MAX_LENGTH=2048

# Get the name of the current script
SCRIPT_NAME=$(basename "$0")

# Construct the command to run the Python script
CMD="accelerate launch train.py \
    --model_paths ${MODEL_PATHS[*]} \
    --dataset_configs ${DATASET_CONFIGS[*]} \
    --data_source_key ${DATA_SOURCE_KEY[*]} \
    --mode $MODE \
    --constrain_mode $CONSTRAIN_MODE \
    --train_split $TRAIN_SPLIT \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS \
    --logging_dir $LOGGING_DIR \
    --eval_strategy $EVAL_STRATEGY \
    --report_to $REPORT_TO \
    --remove_unused_columns $REMOVE_UNUSED_COLUMNS \
    --logging_first_step $LOGGING_FIRST_STEP \
    --bf16 $BF16 \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --max_length $MAX_LENGTH"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Copy the script to the output directory
cp "$0" "$OUTPUT_DIR/$SCRIPT_NAME"

# Execute the command
echo "-----------------------------------------------------------------"
echo "Running command: $CMD"
echo "-----------------------------------------------------------------"
$CMD
