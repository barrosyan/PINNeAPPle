from __future__ import annotations
"""Multi-scale DeepONet for multi-resolution operator learning (classic contraction)."""
from typing import Sequence, Optional, Dict

import torch
import torch.nn as nn

from .base import OperatorOutput
from .deeponet import DeepONet


class MultiScaleDeepONet(DeepONet):
    """
    Multi-scale DeepONet (classic form):
      - One shared branch that outputs (out_dim * sum(scales)) coefficients
      - Multiple trunk nets, each producing modes for its scale
      - Sum contributions from each scale:
          y(u,x) = sum_s <B_s(u), T_s(x)> + bias
    """
    def __init__(
        self,
        branch_dim: int,
        trunk_dim: int,
        out_dim: int,
        *,
        hidden: int = 128,
        scales: Sequence[int] = (32, 64, 128),
    ):
        # Initialize base with a dummy modes; we'll override branch/trunk anyway.
        super().__init__(branch_dim, trunk_dim, out_dim, hidden=hidden, modes=int(scales[0]))

        self.scales = [int(s) for s in scales]
        self.total_modes = int(sum(self.scales))
        self.out_dim = int(out_dim)

        # Override branch: output all modes for all scales (out_dim * total_modes)
        self.branch = nn.Sequential(
            nn.Linear(branch_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.out_dim * self.total_modes),
        )

        # Override trunk: now we have one trunk per scale, each outputs its modes
        self.trunks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(trunk_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, s),
            )
            for s in self.scales
        ])

        # Keep classic bias per output channel (already exists in base, but ensure shape)
        self.bias = nn.Parameter(torch.zeros(self.out_dim))

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

        # (B, out_dim*total_modes) -> (B, out_dim, total_modes)
        b_all = self.branch(u).view(B, self.out_dim, self.total_modes)

        # Accumulate contributions from each scale using classic contraction over modes
        y = torch.zeros((B, N, self.out_dim), device=u.device, dtype=u.dtype)

        start = 0
        for trunk, s in zip(self.trunks, self.scales):
            end = start + s

            # (B, out_dim, s)
            b_s = b_all[:, :, start:end]

            # (N, s)
            t_s = trunk(coords)

            # (B, out_dim, s) x (N, s) -> (B, N, out_dim)
            y = y + torch.einsum("bos,ns->bno", b_s, t_s)

            start = end

        y = y + self.bias  # broadcast over (B,N,out_dim)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]

        return OperatorOutput(y=y, losses=losses, extras={})