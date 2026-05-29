from __future__ import annotations
"""Extreme learning machine with random hidden features and ridge output."""
from typing import Dict, Optional, Literal

import torch
import torch.nn as nn

from .base import RCBase, RCOutput


class ExtremeLearningMachine(RCBase):
    """
    Extreme Learning Machine (ELM):
      - Random hidden layer features: H = phi(X W + b)
      - Solve output weights with ridge regression (closed-form)

    Works for:
      - static regression/classification (MVP: regression)

    Input:
      x: (B, in_dim) or (N, in_dim)
    Output:
      y: (B, out_dim)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 1024,
        activation: Literal["tanh", "relu", "gelu", "silu"] = "tanh",
        l2: float = 1e-6,
        freeze_random: bool = True,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.l2 = float(l2)

        act = (activation or "tanh").lower()
        self.phi = {
            "tanh": torch.tanh,
            "relu": torch.relu,
            "gelu": torch.nn.functional.gelu,
            "silu": torch.nn.functional.silu,
        }.get(act, torch.tanh)

        W = torch.randn(self.in_dim, self.hidden_dim) * (1.0 / max(self.in_dim, 1) ** 0.5)
        b = torch.zeros(self.hidden_dim)

        self.W = nn.Parameter(W)
        self.b = nn.Parameter(b)

        # trained by closed form
        self.W_out = nn.Parameter(torch.zeros(self.hidden_dim, self.out_dim), requires_grad=False)

        if freeze_random:
            self.W.requires_grad_(False)
            self.b.requires_grad_(False)

        self._fitted = False

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.phi(x @ self.W + self.b)

    @torch.no_grad()
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "ExtremeLearningMachine":
        """
        x: (N,in_dim)
        y: (N,out_dim)
        """
        H = self._features(x)  # (N,hidden)
        W_out = self.ridge_solve(H, y, l2=self.l2)
        self.W_out.copy_(W_out)
        self._fitted = True
        return self

    def forward(self, x: torch.Tensor, *, y_true: Optional[torch.Tensor] = None, return_loss: bool = False) -> RCOutput:
        H = self._features(x)
        y = H @ self.W_out

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=x.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]

        return RCOutput(y=y, losses=losses, extras={"fitted": self._fitted})
