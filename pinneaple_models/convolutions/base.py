from __future__ import annotations
"""Base classes for convolutional models."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class ConvOutput:
    """
    Standard output for convolutional models in Pinneaple.
    """
    y: torch.Tensor
    losses: Dict[str, torch.Tensor]
    extras: Dict[str, Any]


class ConvModelBase(nn.Module):
    """
    Base class for convolution models.

    MVP conventions:
      - Conv1D expects x: (B, C_in, L)
      - Conv2D expects x: (B, C_in, H, W)
      - Conv3D expects x: (B, C_in, D, H, W)
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean((y_hat - y) ** 2)
