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


def slerp(
    t: Union[float, np.ndarray],
    v0: Union[np.ndarray, torch.Tensor],
    v1: Union[np.ndarray, torch.Tensor],
    DOT_THRESHOLD: float = 0.9995,
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

    # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
    if np.abs(dot) > DOT_THRESHOLD:
        res = lerp(t, v0_copy, v1_copy)
        return maybe_torch(res, is_torch)

    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)

    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    res = s0 * v0_copy + s1 * v1_copy

    return maybe_torch(res, is_torch)


def maybe_torch(v: np.ndarray, is_torch: bool):
    if is_torch:
        return torch.from_numpy(v)
    return v


def normalize(v: np.ndarray, eps: float):
    norm_v = np.linalg.norm(v)
    if norm_v > eps:
        v = v / norm_v
    return v

def compute_t(weight_name, conditions, num_layers):
    """
    Computes the blending factor for a weight based on layer index and conditions.
    
    Args:
        weight_name (str): Name of the weight.
        conditions (dict): Mapping of conditions to blending values.
        num_layers (int): Total number of layers in the model.
        
    Returns:
        float: Computed blending value.
    """
    anchors = conditions.get("fallback_value")
    if not isinstance(anchors, list):
        anchors = [anchors]

    for filter_name in conditions.keys():
        if filter_name in weight_name:
            anchors = conditions.get(filter_name)
            break
            
    match = re.search(r"layers\.([^\.]*)\.", weight_name)
    if match:
        layer_idx = int(match.group(1))
        layer_t = layer_idx / (num_layers - 1)
        scaled = layer_t * (len(anchors) - 1)
        i0 = math.floor(scaled)
        i1 = min(len(anchors) - 1, i0 + 1)
        frac = scaled - i0
        
        blend_value = (1 - frac) * anchors[i0] + frac * anchors[i1]
    else:
        blend_value = anchors[0]
        
    return blend_value

# def compute_t(weight_name, conditions, num_layers):
#     t = np.random.rand()
#     print(t)
#     return t

def merge_layer(tensors, merge_config):
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
    
    for weight_name in weight_names:
        t = compute_t(weight_name, conditions, num_layers)
        tensor_a = tensors[0][weight_name]
        tensor_b = tensors[1][weight_name]
        
        # tensor_merged = tensors_merged[name]
        tensor_computed = (
            slerp(
                t,
                tensor_a,
                tensor_b,
            )
            .to(tensor_a.dtype)
            .to(tensor_a.device)
        )
        print(np.linalg.norm(tensor_computed.float()))
        tensors[0][weight_name] = tensor_computed
        # torch.testing.assert_close(tensor_merged, tensor_computed)
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