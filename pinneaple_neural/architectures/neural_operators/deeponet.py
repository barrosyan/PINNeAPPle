from __future__ import annotations
"""DeepONet operator learning with branch-trunk architecture (classic form)."""
from typing import Optional

import torch
import torch.nn as nn

from .base import NeuralOperatorBase, OperatorOutput


class DeepONet(NeuralOperatorBase):
    """
    Classic DeepONet:
      For each output channel o:
        y_o(u, x) = sum_{k=1..modes} B_{o,k}(u) * T_k(x) + bias_o

    Branch net: encodes input function samples/sensors u
    Trunk net: encodes coordinates x
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
        self.out_dim = out_dim
        self.modes = modes

        # Branch outputs (out_dim * modes) so we can do a classic dot with trunk modes per output channel
        self.branch = nn.Sequential(
            nn.Linear(branch_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim * modes),
        )

        # Trunk outputs (modes)
        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, modes),
        )

        # Classic bias term per output channel
        self.bias = nn.Parameter(torch.zeros(out_dim))

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

        # (B, out_dim*modes) -> (B, out_dim, modes)
        b = self.branch(u).view(B, self.out_dim, self.modes)

        # (N, modes)
        t = self.trunk(coords)

        # Classic contraction over modes: (B, out_dim, modes) x (N, modes) -> (B, N, out_dim)
        y = torch.einsum("bom,nm->bno", b, t) + self.bias

        losses = {"total": torch.tensor(0.0, device=y.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]

        return OperatorOutput(y=y, losses=losses, extras={})