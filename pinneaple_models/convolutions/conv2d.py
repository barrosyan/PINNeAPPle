from __future__ import annotations
"""2D convolutional model for spatial data."""

from typing import Dict, Optional, Literal

import torch
import torch.nn as nn

from .base import ConvModelBase, ConvOutput


NormType = Literal["none", "batch", "group", "instance"]


def _make_norm(norm: NormType, channels: int, *, groups: int = 8) -> nn.Module:
    norm = str(norm).lower()
    if norm == "none":
        return nn.Identity()
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "instance":
        # InstanceNorm é ok em muitos cenários de campos; affine=True dá mais flexibilidade
        return nn.InstanceNorm2d(channels, affine=True)
    if norm == "group":
        # GroupNorm costuma funcionar melhor com batches pequenos (comum em PDE/fields)
        g = int(groups)
        g = max(1, min(g, channels))
        # garante divisibilidade
        while channels % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(num_groups=g, num_channels=channels)
    raise ValueError(f"Unknown norm='{norm}'. Use one of: none|batch|group|instance.")


class _ResBlock2D(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        padding: int,
        dropout: float,
        norm: NormType,
        norm_groups: int,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = _make_norm(norm, channels, groups=norm_groups)
        self.act1 = nn.GELU()
        self.drop = nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity()

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = _make_norm(norm, channels, groups=norm_groups)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)
        h = self.drop(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        return h


class Conv2DModel(ConvModelBase):
    """
    Conv2D regression model (ResNet-like CNN).

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
        *,
        norm: NormType = "group",
        norm_groups: int = 8,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        k = int(kernel_size)
        if k <= 0:
            raise ValueError(f"kernel_size must be > 0, got {kernel_size}")
        if k % 2 != 1:
            raise ValueError(
                f"kernel_size must be odd to preserve (H,W) with pad=k//2. Got kernel_size={kernel_size}"
            )
        pad = k // 2

        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)

        self.in_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

        self.blocks = nn.ModuleList(
            [
                _ResBlock2D(
                    hidden_channels,
                    kernel_size=k,
                    padding=pad,
                    dropout=float(dropout),
                    norm=norm,
                    norm_groups=int(norm_groups),
                )
                for _ in range(int(num_blocks))
            ]
        )

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
            if self.residual:
                # residual_scale ajuda estabilidade quando aumenta profundidade/lr
                h = h + self.residual_scale * h2
            else:
                h = h2

        y = self.out_proj(h)

        # Mantém seu contrato (sempre tem "total"), mas evita tensor CPU/dtype fixo
        losses: Dict[str, torch.Tensor] = {"total": y.new_zeros(())}
        if return_loss and (y_true is not None):
            losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]

        return ConvOutput(y=y, losses=losses, extras={})