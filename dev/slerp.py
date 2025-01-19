from typing import (
    Any, Dict, List, 
    Optional, Union, NoneType
)

import numpy as np
import torch
import os
import shutil
import json
import math
import yaml
import re
import argparse
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file

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
        s0, s1 (float): Interpolation factors between v0 and v1
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
        s0, s1 = 1 - t, t
        return s0, s1

    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)

    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0

    return s0, s1

def maybe_torch(v: np.ndarray, is_torch: bool):
    if is_torch:
        return torch.from_numpy(v)
    return v

def normalize(v: np.ndarray, eps: float):
    norm_v = np.linalg.norm(v)
    if norm_v > eps:
        v = v / norm_v
    return v

def compute_t(weight_name, parameters, num_layers):
    """
    Computes the blending factor for a weight based on layer index and conditions.
    
    Args:
        weight_name (str): Name of the weight.
        parameters (dict): Mapping of conditions to blending values.
        num_layers (int): Total number of layers in the model.
        
    Returns:
        float: Computed blending value.
    """
    anchors = parameters.get("default")
    if not isinstance(anchors, list):
        anchors = [anchors]

    for filter_name in parameters.keys():
        if filter_name in weight_name:
            anchors = parameters.get(filter_name)
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

def blend(
    parameters: dict, 
    weight_name: str, 
    layer_idx: int, 
    num_layers: int
):
    assert isinstance(layer_idx, (int, NoneType)), (
        f"If the weight {weight_name} belongs to an i-th layer, "
        f"the argument `layer_idx` should be an integer. Otherwise "
        f"it should be a NoneType object. Found `layer_idx` = "
        f"{layer_idx}."
    )
    assert isinstance(num_layers, int), (
        f"You must specify proper argument `num_layers` "
        f"of type `int`. Found `num_layers` = {num_layers}."
    )
    assert layer_idx <= num_layers - 1, (
        f"The argument `layer_idx` must have lower value than "
        f"the argument `num_layers`. Found "
        f"`layer_idx` = {layer_idx}, `num_layers` = {num_layers}."
    )
    
    matching_filter = next(
        (f for f in parameters if f in weight_name),
        'default'
    )
    anchors = parameters[matching_filter]
    if not isinstance(anchors, list):
        anchors = [anchors]

    if layer_idx is None:
        return anchors[0]
        
    # Calculate interpolation for layer-specific weights
    layer_fraction = layer_idx / (num_layers - 1)
    anchor_position = layer_fraction * (len(anchors) - 1)
    
    lower_idx = math.floor(anchor_position)
    upper_idx = min(len(anchors) - 1, lower_idx + 1)
    fraction = anchor_position - lower_idx

    interpolated = (
        (1 - fraction) * anchors[lower_idx]
        + fraction * anchors[upper_idx]
    )
    return interpolated