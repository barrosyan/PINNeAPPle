from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AEBase


class Autoencoder2D(AEBase):
    """
    Conv2D autoencoder for images / 2D fields.

    Args:
      in_channels: channels in input (e.g. 1 for scalar field)
      latent_dim: latent vector dim
      img_size: (H,W) required to build final linear layers
      base_channels: width multiplier

    Notes:
      - Robust output sizing: decoder output is resized to (H, W) via interpolate if needed.
      - Optional output activation: set via output_activation.
      - Safer reshape instead of view.
      - Optional normalization (GroupNorm by default) between conv and activation.
    """
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        img_size: Tuple[int, int],
        base_channels: int = 32,
        norm: Optional[str] = "group",          # "group", "batch", or None
        gn_groups: int = 8,                    # used if norm == "group"
        output_activation: Optional[str] = None,  # None, "sigmoid", "tanh"
        resize_mode: str = "bilinear",         # interpolate mode used if output size mismatches
        align_corners: Optional[bool] = False, # only relevant for some interpolate modes
    ):
        super().__init__()
        H, W = img_size
        self._img_size = (H, W)
        self._resize_mode = resize_mode
        self._align_corners = align_corners

        def _norm(c: int) -> nn.Module:
            if norm is None:
                return nn.Identity()
            if norm == "batch":
                return nn.BatchNorm2d(c)
            if norm == "group":
                g = min(gn_groups, c)
                # ensure divisibility (GroupNorm requires channels % groups == 0)
                while g > 1 and (c % g != 0):
                    g -= 1
                return nn.GroupNorm(num_groups=g, num_channels=c)
            raise ValueError(f"Unknown norm={norm!r}. Use 'group', 'batch', or None.")

        def enc_block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=2, padding=1),
                _norm(cout),
                nn.GELU(),
            )

        self.enc = nn.Sequential(
            enc_block(in_channels, base_channels),
            enc_block(base_channels, base_channels * 2),
            enc_block(base_channels * 2, base_channels * 4),
        )

        # infer feature shape/dim for linear bottleneck
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, H, W)
            feat = self.enc(dummy)
            self._feat_shape = feat.shape[1:]  # (C,h,w)
            feat_dim = int(feat.numel())

        self.to_latent = nn.Linear(feat_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, feat_dim)

        C, _, _ = self._feat_shape

        def dec_block(cin: int, cout: int, act: bool = True) -> nn.Sequential:
            layers = [
                nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1),
                _norm(cout),
            ]
            if act:
                layers.append(nn.GELU())
            return nn.Sequential(*layers)

        self.dec = nn.Sequential(
            dec_block(C, base_channels * 2, act=True),
            dec_block(base_channels * 2, base_channels, act=True),
            nn.ConvTranspose2d(base_channels, in_channels, 4, stride=2, padding=1),
        )

        if output_activation is None:
            self.out_act = nn.Identity()
        elif output_activation == "sigmoid":
            self.out_act = nn.Sigmoid()
        elif output_activation == "tanh":
            self.out_act = nn.Tanh()
        else:
            raise ValueError(
                f"Unknown output_activation={output_activation!r}. "
                "Use None, 'sigmoid', or 'tanh'."
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        f = self.enc(x)
        f = f.reshape(x.shape[0], -1)  # safer than view
        return self.to_latent(f)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        f = self.from_latent(z)
        f = f.reshape(z.shape[0], *self._feat_shape)  # safer than view
        y = self.dec(f)

        # Robust: ensure output spatial size matches expected (H, W)
        H, W = self._img_size
        if y.shape[-2] != H or y.shape[-1] != W:
            # interpolate handles both up/down adjustments robustly
            y = F.interpolate(
                y,
                size=(H, W),
                mode=self._resize_mode,
                align_corners=self._align_corners if self._resize_mode in ("linear", "bilinear", "bicubic", "trilinear") else None,
            )

        return self.out_act(y)