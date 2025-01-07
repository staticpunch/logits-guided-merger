from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import os
import shutil
import json
import math
import re
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file

def load_tensors(path, signature):
    state_dict = {}
    shard_paths = [f for f in os.listdir(path) if f.endswith('.safetensors')]
    for shard_path in sorted(shard_paths, key=lambda x: int(x.split('-')[1])):
        apath = os.path.join(path, shard_path)
        with safe_open(apath, framework="pt", device="cpu") as f:
            for key in f.keys():
                if signature in key:
                    state_dict[key] = f.get_tensor(key)
    return state_dict

def get_layer_signatures(path):
    """
    Extracts and organizes layer signatures from safetensors files in the specified directory.

    Args:
        path (str): Path to the directory containing safetensors files.

    Returns:
        list: A combined list of non-layer and sorted layer signatures.
    """
    def extract_layer_signature(key):
        key = key.strip(".weight")
        match = re.search(r"layers\.[^\.]*\.", key)
        return match.group(0) if match else key

    # Collect all keys from safetensors files
    shard_paths = sorted(
        (f for f in os.listdir(path) if f.endswith('.safetensors')),
        key=lambda x: int(x.split('-')[1])
    )
    keys = []
    for shard_path in shard_paths:
        full_path = os.path.join(path, shard_path)
        with safe_open(full_path, framework="pt", device="cpu") as f:
            keys.extend(f.keys())

    # Separate and process keys
    block_signatures = sorted(
        {extract_layer_signature(key) for key in keys if "layers" in key},
        key=lambda x: int(x.split(".")[-2])
    )
    other_signatures = [extract_layer_signature(key) for 
                        key in keys if "layers" not in key]

    return other_signatures + block_signatures


def calculate_model_size(model_path):
    files = [os.path.join(model_path, filename)
            for filename in os.listdir(model_path)
            if filename.endswith(".safetensors")]
    
    total_size = sum(
        f.get_tensor(key).nbytes
        for file in files
        for f in [safe_open(file, framework="pt", device="cpu")]
        for key in f.keys()
    )
    return total_size

def make_weightmap(directory):
    files = [f for f in os.listdir(directory) 
             if f.startswith("shard_") and f.endswith(".safetensors")]
    num_shards = len(files)
    files.sort()  # Ensure proper ordering of shard files
    model_index = {
        "metadata": {
            "total_size": calculate_model_size(directory)
        },
        "weight_map": {}
    }
    for index, file in enumerate(files, start=1):
        old_path = os.path.join(directory, file)
        new_name = f"model-{index:05d}-of-{num_shards:05d}.safetensors"
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
        
        with safe_open(new_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                model_index["weight_map"][key] = new_name
                
    outfile = os.path.join(directory, "model.safetensors.index.json")
    with open(outfile, "w") as f:
        json.dump(model_index, f, indent=4)
        
    print(f"Saved model weight map to {outfile}.")
    
def shard_model(tmp_dir, output_dir, size_limit=2*1024*1024*1024):
    files = [os.path.join(tmp_dir, filename)
            for filename in sorted(os.listdir(tmp_dir))
            if filename.endswith(".safetensors")]
    
    shard, shard_size, idx = {}, 0, 1
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                tensor_size = tensor.nbytes
                if shard_size + tensor_size > size_limit:
                    output_file = os.path.join(output_dir, f"shard_{idx}.safetensors")
                    save_file(shard, output_file, metadata={"format":"pt"})
                    print(f"Saved shard {idx} with size {shard_size / 1024**3:.2f} GB")

                    # start a new shard
                    shard, shard_size = {}, 0
                    idx += 1
                    
                # add new tensor to the current shard
                shard[key] = tensor
                shard_size += tensor_size
                
    # save the last shard
    if shard:
        output_file = os.path.join(output_dir, f"shard_{idx}.safetensors")
        save_file(shard, output_file, metadata={"format":"pt"})
        print(f"Saved shard {idx} with size {shard_size / 1024**3:.2f} GB")

    # rename and add model.safetensors.index.json
    make_weightmap(output_dir)

    # remove temporary directory containing individual weights before sharding.
    shutil.rmtree(tmp_dir)

def copy_small_files(base_path, output_dir):
    print(f"Copying files to {output_dir}")
    files_to_copy = [
        "config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json"
    ]
    for filename in files_to_copy:
        src_path = os.path.join(base_path, filename)
        dst_path = os.path.join(output_dir, filename)
        try:
            shutil.copy2(src_path, dst_path)
        except FileNotFoundError:
            print(f"File {filename} not found in {base_path}. Skipping.")    
    
def save_checkpoint(config):
    base_path = config['sources'][0]['model']
    output_dir = config["output_dir"]
    tmp_dir = os.path.join(output_dir, "tmp_dir")

    ## copying files
    copy_small_files(base_path, output_dir)

    ## sharding model weights
    shard_model(tmp_dir, output_dir)