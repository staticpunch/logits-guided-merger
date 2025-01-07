import math
from typing import List, Optional, Tuple, Union

import datasets
import torch
import numpy as np
import torch.nn as nn
from datasets import load_dataset
import logging
import copy

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
    HfArgumentParser
)

from modeling_qwen2 import (
    Qwen2RMSNorm, 
    Qwen2RotaryEmbedding, 
    Qwen2MLP, 
    Qwen2Attention, 
    Qwen2FlashAttention2, 
    Qwen2SdpaAttention, 
    Qwen2DecoderLayer, 
    Qwen2PreTrainedModel, 
    Qwen2Model, 
    Qwen2ForCausalLM
)

from configuration_qwen2 import Qwen2Config
from utils import are_tokenizers_same
from merger import (
    MaskConfig, Mask,
    LinearWithMask, LinearsWithMasks,
    RMSNormWithMask, RMSNormsWithMasks,
    EmbeddingWithMask, EmbeddingsWithMasks,
    MergerConfig, Merger,
    place_masks
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

### TEST SUITE ###
def test_multiple_linear_components(input_size: int, output_size: int, num_components_list: List[int]):
    for num_components in num_components_list:
        linears = [nn.Linear(input_size, output_size, bias=False) for _ in range(num_components)]
        x = torch.rand(1, input_size)

        for _ in range(10):  # Reduced number of iterations for faster testing in a notebook
            values = np.random.rand(num_components).tolist() # cast to list
            weights_with_masks = LinearsWithMasks(linears=linears, modes=["scalar"] * num_components, values=values)

            individual_outputs = [linear(x) for linear in linears]
            expected_output = sum(val * out for val, out in zip(values, individual_outputs))
            actual_output = weights_with_masks(x)

            torch.testing.assert_close(actual_output, expected_output, rtol=1e-6, atol=1e-6)
        logging.info(f"Test with {num_components} Linear components passed!")

def test_multiple_rms_norm_components(hidden_size: int, num_components_list: List[int]):
    for num_components in num_components_list:
        rms_norms = [Qwen2RMSNorm(hidden_size) for _ in range(num_components)]
        hidden_states = torch.rand(2, 4, hidden_size)

        for _ in range(10):
            values = np.random.rand(num_components).tolist()
            rms_norms_with_masks = RMSNormsWithMasks(rms_norms=rms_norms, modes=["scalar"] * num_components, values=values)

            individual_outputs = [rms_norm(hidden_states) for rms_norm in rms_norms]
            expected_output = sum(val * out for val, out in zip(values, individual_outputs))
            actual_output = rms_norms_with_masks(hidden_states)
            
            torch.testing.assert_close(actual_output, expected_output, rtol=1e-6, atol=1e-6)
        logging.info(f"Test with {num_components} RMSNorm components passed!")

def test_multiple_embedding_components(num_embeddings: int, embedding_dim: int, num_components_list: List[int]):
    for num_components in num_components_list:
        embeddings = [nn.Embedding(num_embeddings, embedding_dim) for _ in range(num_components)]
        input_ids = torch.randint(0, num_embeddings, (2, 5))  # Example input_ids

        for _ in range(10):
            values = np.random.rand(num_components).tolist()
            embeddings_with_masks = EmbeddingsWithMasks(embeddings=embeddings, modes=["scalar"] * num_components, values=values)

            individual_outputs = [embedding(input_ids) for embedding in embeddings]
            expected_output = sum(val * out for val, out in zip(values, individual_outputs))
            actual_output = embeddings_with_masks(input_ids)

            torch.testing.assert_close(actual_output, expected_output, rtol=1e-6, atol=1e-6)
        logging.info(f"Test with {num_components} Embedding components passed!")

def test_logits(merger, tokenizer):
    def get_logits(text, model, tokenizer):
        input_ids = tokenizer(text, return_tensors="pt").to(model.device)
        model.eval()
        with torch.no_grad():
            logits = model(**input_ids).logits
        return logits
        
    system = "You are a helpful assistant."
    prompt = "Continue this text: A dog is a"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    logits_merger = get_logits(text, model=merger.merger, tokenizer=tokenizer)
    logits_1 = get_logits(text, model=merger.models[0], tokenizer=tokenizer)
    torch.testing.assert_close(logits_merger, logits_1, rtol=1e-6, atol=1e-6)
    logging.info("Logits test passed!")



if __name__ == "__main__":
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    num_components_list = [2, 4, 5, 8, 10, 12, 16, 20, 30]
    # Linear Test ----------------------------------------------------------------- #
    logging.info("LinearsWithMasks Tests")
    input_size = 1024
    output_size = 2048
    test_multiple_linear_components(input_size, output_size, num_components_list)
    
    # RMSNorm Test ----------------------------------------------------------------- #
    logging.info("RMSNormsWithMasks Tests")
    hidden_size = 2048
    test_multiple_rms_norm_components(hidden_size, num_components_list)

    # Embedding Test ----------------------------------------------------------------- #
    logging.info("EmbeddingsWithMasks Tests")
    num_embeddings = 2048
    embedding_dim = 2048
    test_multiple_embedding_components(num_embeddings, embedding_dim, num_components_list)

    # Final Test ----------------------------------------------------------------- #
    logging.info("Logits Test")
    merge_config = MergerConfig(
        model_paths = [
            "/workspace/models/Arcee-VyLinh/",
            "/workspace/models/Qwen2.5-Coder-3B/"
        ]
    )
    logging.info(merge_config)
    merger = Merger(merge_config)
    logging.info(merger.merger)
    merger.to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(merge_config.model_paths[0])
    test_logits(merger, tokenizer)