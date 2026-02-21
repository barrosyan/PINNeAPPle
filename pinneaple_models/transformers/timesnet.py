from __future__ import annotations
"""TimesNet with multi-scale 2D-variation modeling for time series."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import TimeSeriesModelBase, TSOutput


class _InceptionBlock1D(nn.Module):
    def __init__(self, d: int, k_sizes=(3, 5, 7)):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(d, d, k, padding=k // 2) for k in k_sizes])
        self.proj = nn.Conv1d(d * len(k_sizes), d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,D,L)
        ys = [c(x) for c in self.convs]
        y = torch.cat(ys, dim=1)
        return self.proj(torch.nn.functional.gelu(y))


class TimesNet(TimeSeriesModelBase):
    """
    TimesNet (MVP approximation):
      - multi-period temporal modeling using conv/inception over the sequence.
      - we keep it simple: inception blocks + horizon head.

    Later you can add explicit period discovery / reshaping.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        horizon: int,
        d_model: int = 128,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.in_proj = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList([_InceptionBlock1D(d_model) for _ in range(int(num_blocks))])
        self.out = nn.Linear(d_model, out_dim)

    def forward(
        self,
        x_past: torch.Tensor,
        *,
        x_future: Optional[torch.Tensor] = None,
        y_future: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> TSOutput:
        B, L, _ = x_past.shape
        H = self.horizon

        h = torch.nn.functional.gelu(self.in_proj(x_past))  # (B,L,D)
        h = h.transpose(1, 2)                               # (B,D,L)
        for blk in self.blocks:
            h = h + blk(h)                                  # residual
        h = h.transpose(1, 2)                               # (B,L,D)

        last = h[:, -1:, :].repeat(1, H, 1)
        y_hat = self.out(last)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y_hat.device)}
        if return_loss and y_future is not None:
            losses["mse"] = self.mse(y_hat, y_future)
            losses["total"] = losses["mse"]
        return TSOutput(y=y_hat, losses=losses, extras={})
