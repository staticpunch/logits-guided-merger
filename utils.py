from transformers import AutoTokenizer
import math
from typing import List, Optional, Tuple, Union
import torch
import logging
import numpy as np
from transformers import GenerationConfig, TextStreamer

def generate(prompt, model, tokenizer, max_new_tokens=1024):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    model.eval()
    with torch.no_grad():
        generation_config = GenerationConfig(
            repetition_penalty=1.13,
            max_new_tokens=max_new_tokens,
            temperature=0.4,
            top_p=0.95,
            # top_k=20,
            # bos_token_id=tokenizer.bos_token_id,
            # eos_token_id=tokenizer.eos_token_id,
            # eos_token_id=0, # for open-end generation.
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
        )
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        generated = model.generate(
            inputs=input_ids,
            generation_config=generation_config,
            streamer=streamer,
        )
    gen_tokens = generated["sequences"].cpu()[:, len(input_ids[0]):]
    output = tokenizer.batch_decode(gen_tokens)[0]
    output = output.split(tokenizer.eos_token)[0]
    return output.strip()

def get_logits(text, model, tokenizer):
    input_ids = tokenizer(text, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        logits = model(**input_ids).logits
    return logits

def get_hidden_states(text, model, tokenizer):
    input_ids = tokenizer(text, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        outputs = model(**input_ids, output_hidden_states=True, use_cache=False)
    return outputs

def are_tokenizers_same(paths: List[str]) -> bool:
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    """
    Tests whether some tokenizers are the same based on critical criteria.

    Args:
        paths: A list of paths to tokenizer directories or model identifiers.

    Returns:
        True if all tokenizers are the same based on the criteria, False otherwise.
    """

    if not paths:
        return True  # Empty list, consider them the same

    tokenizers = [AutoTokenizer.from_pretrained(path) for path in paths]
    first_tokenizer = tokenizers[0]

    for i, tokenizer in enumerate(tokenizers[1:]):
        logging.info(f"Comparing tokenizer at {paths[0]} with tokenizer at {paths[i+1]}")
        # 1. Check vocab size
        if first_tokenizer.vocab_size != tokenizer.vocab_size:
            logging.error(f"Vocab size mismatch: {paths[0]} has {first_tokenizer.vocab_size}, {paths[i+1]} has {tokenizer.vocab_size}")
            return False

        # 2. Check special tokens
        if sorted(first_tokenizer.all_special_tokens) != sorted(tokenizer.all_special_tokens):
            logging.error(f"Special tokens mismatch: {paths[0]} has {first_tokenizer.all_special_tokens}, {paths[i+1]} has {tokenizer.all_special_tokens}")
            return False

        # 3. Check basic tokenization on a sample input
        sample_input = "This is a sample input for testing."
        if first_tokenizer.tokenize(sample_input) != tokenizer.tokenize(
            sample_input
        ):
            logging.error(f"Tokenization mismatch for input '{sample_input}': {paths[0]} tokenizes to {first_tokenizer.tokenize(sample_input)}, {paths[i+1]} tokenizes to {tokenizer.tokenize(sample_input)}")
            return False
        
        # # 4. Check model max length
        # if first_tokenizer.model_max_length != tokenizer.model_max_length:
        #     logging.error(f"Model max length mismatch: {paths[0]} has {first_tokenizer.model_max_length}, {paths[i+1]} has {tokenizer.model_max_length}")
        #     return False
        
        logging.info(f"Tokenizer at {paths[0]} and {paths[i+1]} are the same based on the defined criteria")

    return True

def load_layer(path, layer_idx=33):
	state_dict = {}
	shard_paths = [f for f in os.listdir(path) if f.endswith('.safetensors')]
	for shard_path in sorted(shard_paths, key=lambda x: int(x.split('-')[1])):
		apath = os.path.join(path, shard_path)
		with safe_open(apath, framework="pt", device="cpu") as f:
			for key in f.keys():
				if f"layers.{str(layer_idx)}." in key:
					state_dict[key] = f.get_tensor(key)
	return state_dict
    
def lerp(
    t: float, v0: Union[np.ndarray, torch.Tensor], v1: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    return (1 - t) * v0 + t * v1

def weighted_sum(
    factors: List[float], 
    tensors: Union[List[np.ndarray], List[torch.Tensor]]
) -> Union[np.ndarray, torch.Tensor]:

    return sum([tensor * factor for tensor, factor in zip(tensors, factors)])

def merge_tensors(modules, weight_factors, bias_factors):
    param_names = sorted([name for name, _ in modules[0].named_parameters()])
    for module in modules:
        other_param_names = sorted([name for name, _ in module.named_parameters()])
        assert param_names == other_param_names, "Mismatch tensor names."
        
    module_out = copy.deepcopy(modules[0])
    out_dict = module_out.state_dict()
    
    tensor_dicts_list = [m.state_dict() for m in modules]
    tensor_names = [key for key in tensor_dicts_list[0].keys()]
    
    for tensor_name in tensor_names:
        tensors_list = [tensor_dicts_list[i][tensor_name]
                       for i in range(len(modules))]
        if "weight" in tensor_name:
            factors = weight_factors
        elif "bias" in tensor_name:
            factors = bias_factors
        else:
            raise ValueError("Hey this tensor is neither weight or bias.")
            
        tensor_computed = (
            weighted_sum(
                factors=factors,
                tensors=tensors_list
            )
            .to(tensors_list[0].dtype)
            .to(tensors_list[0].device)
        )
        out_dict[tensor_name] = tensor_computed
    module_out.load_state_dict(out_dict)
    return module_out


def find_mask_parameter_names(module, mask_param_names_list, parent_name=""):
    """
    Recursively finds full names of parameters that belong to modules of class "Mask".
    """
    for name, child in module.named_children():
        full_child_name = f"{parent_name}.{name}" if parent_name else name
        if child.__class__.__name__ == "Mask":
            for param_name, _ in child.named_parameters():
                full_param_name = f"{full_child_name}.{param_name}"
                mask_param_names_list.append(full_param_name)
        find_mask_parameter_names(child, mask_param_names_list, full_child_name)