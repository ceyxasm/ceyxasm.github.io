"""SIREN: Sinusoidal Representation Networks
Based on 'Implicit Neural Representations with Periodic Activation Functions'
(Sitzmann et al., 2020)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List


class Siren(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

    def extra_repr(self):
        return f'w0={self.w0}'


def siren_init(layer, w0=1.0, is_first=False):
    """Initialize weights for SIREN layers as described in the paper."""
    dim = layer.in_features
    if is_first:
        bound = 1.0 / dim
    else:
        bound = np.sqrt(6.0 / dim) / w0
    nn.init.uniform_(layer.weight, -bound, bound)
    if layer.bias is not None:
        nn.init.uniform_(layer.bias, -bound, bound)


def siren_model(dimensions: List[int], w0_initial=30.0, w0=1.0):
    """
    Create a SIREN model with the given layer dimensions.

    Args:
        dimensions: List of layer sizes, e.g. [2, 256, 128, 64, 32, 3]
        w0_initial: Frequency for the first layer (default 30.0 per paper)
        w0: Frequency for subsequent layers (default 1.0)
    """
    layers = []
    for i, (dim0, dim1) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        linear = nn.Linear(dim0, dim1)
        if i == 0:
            siren_init(linear, w0=w0_initial, is_first=True)
            layers.append(nn.Sequential(linear, Siren(w0=w0_initial)))
        else:
            siren_init(linear, w0=w0)
            layers.append(nn.Sequential(linear, Siren(w0=w0)))
    return nn.Sequential(*layers)
