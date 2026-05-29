from __future__ import annotations
"""Base classes for continuous-time dynamics models."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class ContOutput:
    """
    Standard output for continuous-time / probabilistic dynamics models.

    y:
      - forecasts: (B, H, out_dim) OR trajectories (B, T, out_dim)
    dist (optional):
      - if probabilistic, store params in extras["dist"]
    """
    y: torch.Tensor
    losses: Dict[str, torch.Tensor]
    extras: Dict[str, Any]


class ContinuousModelBase(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean((y_hat - y) ** 2)

    @staticmethod
    def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # diagonal Gaussian negative log-likelihood
        return 0.5 * torch.mean(logvar + (y - mu) ** 2 * torch.exp(-logvar))
