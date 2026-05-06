from __future__ import annotations
"""Base classes for time series transformer models."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class TSOutput:
    """
    Standard output for time-series transformers in Pinneaple.
    """
    y: torch.Tensor               # (B, H, out_dim) or (B, T, out_dim) depending on mode
    losses: Dict[str, torch.Tensor]
    extras: Dict[str, Any]


class TimeSeriesModelBase(nn.Module):
    """
    Base class for sequence forecasting models.

    Conventions (MVP):
      - x_past:  (B, L, in_dim)
      - x_future:(B, H, future_dim) optional known future features (calendar, forcing, etc.)
      - returns y: (B, H, out_dim)

    The models themselves are architecture-only; training module will handle splits/metrics later.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x_past: torch.Tensor,
        *,
        x_future: Optional[torch.Tensor] = None,
        y_future: Optional[torch.Tensor] = None,  # optional teacher forcing / loss computation
        return_loss: bool = False,
    ) -> TSOutput:
        raise NotImplementedError

    @staticmethod
    def mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean((y_hat - y) ** 2)
