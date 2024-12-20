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
import argparse
from datetime import datetime

def compute_theta(
    v0: Union[np.ndarray, torch.Tensor],
    v1: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-8,
    device: str = "cpu"
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
        device (str): Device to load tensors, "cpu" or "cuda".

    Returns:
        v2 (np.ndarray): Interpolation vector between v0 and v1
    """

    # Convert inputs to PyTorch tensors and move to the specified device
    if not isinstance(v0, torch.Tensor):
        v0 = torch.from_numpy(v0)
    if not isinstance(v1, torch.Tensor):
        v1 = torch.from_numpy(v1)

    v0 = v0.float().to(device)
    v1 = v1.float().to(device)

    # Copy the vectors to reuse them later   
    v0_copy = v0.clone()
    v1_copy = v1.clone()

    lengths = [
        torch.norm(v0_copy).cpu().item(),
        torch.norm(v1_copy).cpu().item()
    ]
    
    euclide = torch.norm(v0_copy - v1_copy)
    dot = torch.sum(v0_copy * v1_copy)

    # Normalize the vectors to get the directions and angles
    v0 = normalize(v0, eps)
    v1 = normalize(v1, eps)

    # Dot product with the normalized vectors (can't use np.dot in W)
    cosine = torch.sum(v0 * v1)

    # Calculate initial angle between v0 and v1
    theta_0 = torch.arccos(torch.clamp(cosine, -1.0, 1.0))
    # degree = np.degrees(theta_0)

    return {
        "lengths": lengths,
        "dot": dot.cpu().item(),
        "L2": euclide.cpu().item(),
        "cosine": cosine.cpu().item(),
        "theta": theta_0.cpu().item()
    }

def normalize(v: torch.Tensor, eps: float):
    norm_v = torch.norm(v)
    if norm_v > eps:
        v = v / norm_v
    return v

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
    
    weight_names = [key for key in tensors[0].keys()]
    thetas = {}
    for weight_name in weight_names:
        tensor_a = tensors[0][weight_name]
        tensor_b = tensors[1][weight_name]
        theta = compute_theta(
            v0=tensor_a,
            v1=tensor_b,
            device="cuda:7"
        )
        thetas.update({weight_name: theta})
    return thetas

def run_compute(
    merge_config
):
    ## Read configs
    model_paths = [x['model'] for x in merge_config['sources']]
    layer_signatures = get_layer_signatures(model_paths[0])
    output_dir = merge_config["output_dir"]
    model_thetas = {}

    ## Merge models
    for signature in (
        pbar := tqdm(
            layer_signatures,
            desc="Computing distance ...",
            ncols=150
        )
    ):
        pbar.set_description(f"Computing distance {signature}")
        models_tensors = [load_tensors(path, signature) for path in model_paths]
        thetas = compute_layers_thetas(models_tensors, merge_config)
        model_thetas.update({signature: thetas})
        
    statistics_path = os.path.join(output_dir, "statistics.json")
    with open(statistics_path, "w") as f:
        json.dump(model_thetas, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute thetas between two models.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file (YAML).",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        merge_config = yaml.safe_load(f)

    output_dir = merge_config["output_dir"]
    
    # Create a subdirectory with the current date and time
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(output_dir, current_time)
    merge_config["output_dir"] = output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    shutil.copy(args.config, output_dir)

    run_compute(merge_config)