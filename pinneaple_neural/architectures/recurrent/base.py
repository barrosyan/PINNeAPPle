from __future__ import annotations
"""Base classes for recurrent models."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class RNNOutput:
    """
    Standard output for recurrent models in Pinneaple.
    """
    y: torch.Tensor               # (B, H, out_dim)
    losses: Dict[str, torch.Tensor]
    extras: Dict[str, Any]


class RecurrentModelBase(nn.Module):
    """
    Base class for time-series recurrent models.

    Conventions (MVP):
      - x_past:  (B, L, in_dim)
      - y_hat:   (B, H, out_dim)
      - horizon passed at init
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean((y_hat - y) ** 2)
