from __future__ import annotations
"""DeepONet operator learning with branch-trunk architecture."""
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import NeuralOperatorBase, OperatorOutput


class DeepONet(NeuralOperatorBase):
    """
    Classic DeepONet:
      G(u)(x) = sum_k B_k(u) * T_k(x)

    Branch net: encodes input function
    Trunk net: encodes coordinates
    """
    def __init__(
        self,
        branch_dim: int,
        trunk_dim: int,
        out_dim: int,
        hidden: int = 128,
        modes: int = 64,
    ):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(branch_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, modes),
        )
        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, modes),
        )
        self.out = nn.Linear(modes, out_dim)

    def forward(
        self,
        u: torch.Tensor,        # (B, branch_dim)
        coords: torch.Tensor,   # (N, trunk_dim)
        *,
        y_true: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> OperatorOutput:
        B = u.shape[0]
        N = coords.shape[0]

        b = self.branch(u)                    # (B, modes)
        t = self.trunk(coords)                # (N, modes)

        y = torch.einsum("bm,nm->bnm", b, t)
        y = self.out(y)                       # (B, N, out_dim)

        losses = {"total": torch.tensor(0.0, device=y.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]

        return OperatorOutput(y=y, losses=losses, extras={})
