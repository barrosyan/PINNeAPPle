# pinneaple_models/convolutions/conv1d.py
from __future__ import annotations
"""1D convolutional model for sequence data."""

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from .base import ConvModelBase, ConvOutput


class Conv1DModel(ConvModelBase):
    """
    Conv1D regression model.

    Input:
      x: (B, C_in, L)
    Output:
      y: (B, C_out, L) by default (same-length)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        num_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        k = int(kernel_size)
        pad = k // 2

        self.in_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList()
        self.residual = bool(residual)

        for _ in range(int(num_blocks)):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=k, padding=pad),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=k, padding=pad),
                nn.GELU(),
            ))

        self.out_proj = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        y_true: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> ConvOutput:
        h = self.in_proj(x)
        for blk in self.blocks:
            h2 = blk(h)
            h = h + h2 if self.residual else h2
        y = self.out_proj(h)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]

        return ConvOutput(y=y, losses=losses, extras={})
