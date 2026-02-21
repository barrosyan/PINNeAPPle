from __future__ import annotations
"""Base classes for classical time series models."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class ClassicalTSOutput:
    """
    Output for classical time-series / filtering models.

    y:
      - forecasts or filtered states:
        (B,T,dim) or (B,H,dim) depending on method
    extras:
      - optional covariances, gains, etc.
    """
    y: torch.Tensor
    losses: Dict[str, torch.Tensor]
    extras: Dict[str, Any]


class ClassicalTSBase(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean((y_hat - y) ** 2)
