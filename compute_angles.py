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

def compute_theta(
    v0: Union[np.ndarray, torch.Tensor],
    v1: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-8,
):
    """
    Spherical linear interpolation

    From: https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
    Args:
        t (float/np.ndarray): Float value between 0.0 and 1.0
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as
                               colinear. Not recommended to alter this.
    Returns:
        v2 (np.ndarray): Interpolation vector between v0 and v1
    """
    is_torch = False
    if not isinstance(v0, np.ndarray):
        is_torch = True
        v0 = v0.detach().cpu().float().numpy()
    if not isinstance(v1, np.ndarray):
        is_torch = True
        v1 = v1.detach().cpu().float().numpy()

    # Copy the vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)

    # Normalize the vectors to get the directions and angles
    v0 = normalize(v0, eps)
    v1 = normalize(v1, eps)
    # import ipdb; ipdb.set_trace()

    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = np.sum(v0 * v1)

    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    degree = np.degrees(theta_0)
    
    # return float(dot)
    # return float(theta_0)
    return float(degree)

def compute_l2(v0, v1):
    is_torch = False
    if not isinstance(v0, np.ndarray):
        is_torch = True
        v0 = v0.detach().cpu().float().numpy()
    if not isinstance(v1, np.ndarray):
        is_torch = True
        v1 = v1.detach().cpu().float().numpy()

    dist = np.linalg.norm(v0 - v1)
    return float(dist)


def maybe_torch(v: np.ndarray, is_torch: bool):
    if is_torch:
        return torch.from_numpy(v)
    return v


def normalize(v: np.ndarray, eps: float):
    return v / np.linalg.norm(v)
    # norm_v = np.linalg.norm(v)
    # if norm_v > eps:
    #     v = v / norm_v
    # return v

def compute_layers_thetas(tensors, merge_config):
    """
    Merges corresponding layers of two models using SLERP.
    
    Args:
        tensors (List[Dict[str, torch.Tensor]]): List of model weight dictionaries.
        merge_config (dict): Configuration dictionary containing merging parameters.
        
    Returns:
        Dict[str, torch.Tensor]: Merged model weights.
    """
    assert len(tensors) == 2
    conditions = merge_config["parameters"]["t"]
    conditions = {c['filter']: c['value'] for c in conditions}

    layer_begin, layer_end = merge_config["layer_range"]
    num_layers = layer_end - layer_begin
    
    weight_names = [key for key in tensors[0].keys()]
    thetas = {}
    for weight_name in weight_names:
        tensor_a = tensors[0][weight_name]
        tensor_b = tensors[1][weight_name]
        theta = compute_theta(
            v0=tensor_a,
            v1=tensor_b
        )
        thetas.update({weight_name: theta})
        # torch.testing.assert_close(tensor_merged, tensor_computed)
    return thetas

def run_compute(
    merge_config
):
    ## Read configs
    model_paths = [x['model'] for x in merge_config['sources']]
    layer_signatures = get_layer_signatures(model_paths[0])
    output_dir = merge_config["output_dir"]
    tmp_dir = os.path.join(output_dir, "tmp_dir")
    model_thetas = {}
        
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
        thetas = compute_layers_thetas(models_tensors, merge_config)
        model_thetas.update({signature: thetas})
        
    with open("thetas.json", "w") as f:
        json.dump(model_thetas, f, indent=4)

if __name__ == "__main__":
    # CONFIG_FILE = "slerp-config-custom.yaml"
    CONFIG_FILE = "config-03.yaml"
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        merge_config = yaml.safe_load(f)
        
    run_compute(merge_config)