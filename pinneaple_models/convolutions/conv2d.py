# pinneaple_models/convolutions/conv2d.py
from __future__ import annotations
"""2D convolutional model for spatial data."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import ConvModelBase, ConvOutput


class Conv2DModel(ConvModelBase):
    """
    Conv2D regression model.

    Input:
      x: (B, C_in, H, W)
    Output:
      y: (B, C_out, H, W) (same spatial size)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 32,
        num_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        k = int(kernel_size)
        pad = k // 2
        self.residual = bool(residual)

        self.in_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList()
        for _ in range(int(num_blocks)):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=k, padding=pad),
                nn.GELU(),
                nn.Dropout2d(float(dropout)),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=k, padding=pad),
                nn.GELU(),
            ))
        self.out_proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

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
