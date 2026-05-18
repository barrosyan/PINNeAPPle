from __future__ import annotations
"""Base classes for neural operator models."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class OperatorOutput:
    """
    Standard output for neural operators.

    y:
      - grid operators: (B, ..., out_channels)
      - point operators: (B, N, out_channels)
    """
    y: torch.Tensor
    losses: Dict[str, torch.Tensor]
    extras: Dict[str, Any]


class NeuralOperatorBase(nn.Module):
    """
    Base class for Neural Operators.

    Design:
      - Input represents a function u(x)
      - Output represents a function G(u)(x)
      - Geometry / coordinates passed explicitly
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean((y_hat - y) ** 2)
