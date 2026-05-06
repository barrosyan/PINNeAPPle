from __future__ import annotations
"""3D convolutional model for volumetric data."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ConvModelBase, ConvOutput


def _auto_groups(channels: int, max_groups: int = 8) -> int:
    """
    Choose a GroupNorm group count that divides channels.
    Prefers up to `max_groups` groups; falls back to 1.
    """
    g = min(int(max_groups), int(channels))
    while g > 1 and (channels % g) != 0:
        g -= 1
    return max(g, 1)


class ResidualBlock3D(nn.Module):
    """
    A stable 3D residual block:
      Conv -> GroupNorm -> GELU -> Dropout -> Conv -> GroupNorm -> GELU
    Residual is applied outside (so the caller can scale it).
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        k = int(kernel_size)
        pad = k // 2
        g = _auto_groups(channels, max_groups=int(norm_groups))

        self.conv1 = nn.Conv3d(channels, channels, kernel_size=k, padding=pad)
        self.gn1 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=k, padding=pad)
        self.gn2 = nn.GroupNorm(g, channels)

        self.act = nn.GELU()
        self.drop = nn.Dropout3d(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.gn1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.conv2(h)
        h = self.gn2(h)
        h = self.act(h)
        return h


class Conv3DModel(ConvModelBase):
    """
    Conv3D regression model.

    Input:
      x: (B, C_in, D, H, W)
    Output:
      y: (B, C_out, D, H, W)

    Notes (generic-for-physics defaults):
      - GroupNorm is robust for small batch sizes (common in 3D volumes).
      - Residual scaling improves stability for deeper stacks.
      - Optional global skip encourages learning residual corrections.
      - Optional grad loss promotes structural/physical smoothness without PDE-specific terms.
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
        res_scale: float = 0.1,
        norm_groups: int = 8,
        global_skip: bool = True,
        grad_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.residual = bool(residual)
        self.res_scale = float(res_scale)
        self.global_skip = bool(global_skip)
        self.grad_loss_weight = float(grad_loss_weight)

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)

        self.in_proj = nn.Conv3d(self.in_channels, self.hidden_channels, kernel_size=1)

        self.blocks = nn.ModuleList(
            [
                ResidualBlock3D(
                    channels=self.hidden_channels,
                    kernel_size=int(kernel_size),
                    dropout=float(dropout),
                    norm_groups=int(norm_groups),
                )
                for _ in range(int(num_blocks))
            ]
        )

        self.out_proj = nn.Conv3d(self.hidden_channels, self.out_channels, kernel_size=1)

        # If in/out channels match, we can optionally learn a residual correction to x.
        self._can_global_skip = (self.in_channels == self.out_channels)

    @staticmethod
    def _grad_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Generic gradient-matching loss (finite-difference).
        Helps preserve structures (edges/fronts) in physical fields.
        """
        def diffs(t: torch.Tensor):
            # forward differences (keep shapes aligned by trimming)
            dz = t[:, :, 1:, :, :] - t[:, :, :-1, :, :]
            dy = t[:, :, :, 1:, :] - t[:, :, :, :-1, :]
            dx = t[:, :, :, :, 1:] - t[:, :, :, :, :-1]
            return dz, dy, dx

        pdz, pdy, pdx = diffs(pred)
        tdz, tdy, tdx = diffs(target)

        return (
            F.mse_loss(pdz, tdz)
            + F.mse_loss(pdy, tdy)
            + F.mse_loss(pdx, tdx)
        ) / 3.0

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
                h = h + (self.res_scale * h2)
            else:
                h = h2

        y = self.out_proj(h)

        # Optional global skip: y := x + correction, when shapes match.
        if self.global_skip and self._can_global_skip:
            y = y + x

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y.device)}
        if return_loss and (y_true is not None):
            losses["mse"] = self.mse(y, y_true)
            total = losses["mse"]

            if self.grad_loss_weight > 0.0:
                losses["grad"] = self._grad_loss(y, y_true)
                total = total + (self.grad_loss_weight * losses["grad"])

            losses["total"] = total

        return ConvOutput(y=y, losses=losses, extras={})