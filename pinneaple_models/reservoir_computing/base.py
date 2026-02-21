from __future__ import annotations
"""Base classes for reservoir computing models."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class RCOutput:
    """
    Standard output for reservoir computing models.

    y:
      - sequence models: (B, T, out_dim)
      - static models: (B, out_dim)
    """
    y: torch.Tensor
    losses: Dict[str, torch.Tensor]
    extras: Dict[str, Any]


class RCBase(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean((y_hat - y) ** 2)

    @staticmethod
    def ridge_solve(X: torch.Tensor, Y: torch.Tensor, l2: float = 1e-6) -> torch.Tensor:
        """
        Closed-form ridge regression: W = (X^T X + l2 I)^-1 X^T Y

        X: (N, F)
        Y: (N, O)
        returns W: (F, O)
        """
        F = X.shape[1]
        I = torch.eye(F, device=X.device, dtype=X.dtype)
        XtX = X.t() @ X
        XtY = X.t() @ Y
        W = torch.linalg.solve(XtX + l2 * I, XtY)
        return W
