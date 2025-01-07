from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import os
import shutil
import json
import math
import yaml
import re
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file
from IO import (
    load_tensors, 
    get_layer_signatures, 
    save_checkpoint
)

def lerp(
    t: float, v0: Union[np.ndarray, torch.Tensor], v1: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    return (1 - t) * v0 + t * v1


def slerp_multiple(
    weights: List[torch.Tensor],
    eps: float = 1e-8,
):
    weights = [w.detach().cpu().float().numpy() for w in weights]
    weights_copy = [np.copy(w) for w in weights]
    weights = [normalize(w, eps) for w in weights]

    interpolated = 0
    for w in weights:
        interpolated += w
    interpolated /= len(weights)
    interpolated = normalize(interpolated, eps)
    return maybe_torch(interpolated, is_torch=True)


def maybe_torch(v: np.ndarray, is_torch: bool):
    if is_torch:
        return torch.from_numpy(v)
    return v


def normalize(v: np.ndarray, eps: float):
    norm_v = np.linalg.norm(v)
    if norm_v > eps:
        v = v / norm_v
    return v


def merge_layer(tensors, merge_config):    
    weight_names = [key for key in tensors[0].keys()]
    
    for weight_name in weight_names:
        weights = [tensor[weight_name] for tensor in tensors]
        
        # tensor_merged = tensors_merged[name]
        tensor_computed = (
            slerp_multiple(weights)
            .to(weights[0].dtype)
            .to(weights[0].device)
        )
        print("length:", np.linalg.norm(tensor_computed.float()))
        tensors[0][weight_name] = tensor_computed
    return tensors[0]

def run_merge(
    merge_config
):
    ## Read configs
    model_paths = [x['model'] for x in merge_config['sources']]
    layer_signatures = get_layer_signatures(model_paths[0])
    output_dir = merge_config["output_dir"]
    tmp_dir = os.path.join(output_dir, "tmp_dir")
        
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    ## Merge models
    for signature in (
        pbar := tqdm(
            layer_signatures,
            desc="Merging ...",
            ncols=150
        )
    ):
        pbar.set_description(f"Merging {signature}")
        models_tensors = [load_tensors(path, signature) for path in model_paths]
        merged_tensors = merge_layer(models_tensors, merge_config)
        outfile = os.path.join(tmp_dir, f"{signature.strip('.')}.safetensors")
        save_file(merged_tensors, outfile)

    ## Save models
    save_checkpoint(merge_config)

if __name__ == "__main__":
    CONFIG_FILE = "config-datht.yaml"
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        merge_config = yaml.safe_load(f)
        
    run_merge(merge_config)