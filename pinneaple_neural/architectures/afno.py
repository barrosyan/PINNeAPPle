from __future__ import annotations
"""Adaptive Fourier Neural Operator (AFNO).

Reference: Guibas et al., ICLR 2022
  "Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers"
  https://arxiv.org/abs/2111.13587
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.fft as fft

from .base import BaseModel, ModelOutput


class AFNOLayer(nn.Module):
    """Single AFNO layer: token mixing in Fourier space + channel MLP.

    The forward pass is:

    1. Apply 2-D real FFT to the spatial axes ``(H, W)`` of the input.
    2. Retain only the ``n_modes_h × n_modes_w`` lowest-frequency components.
    3. Apply a learnable *per-mode* block-diagonal complex linear transform
       (implemented as two real weight matrices for real/imag parts).
    4. Apply inverse FFT to reconstruct the spatial field.
    5. Add a channel-mixing MLP (analogous to the FFN in a Transformer).

    Args:
        hidden_dim: Channel width ``C``.
        n_modes_h: Number of Fourier modes to retain in height dimension.
        n_modes_w: Number of Fourier modes to retain in width dimension.
        mlp_ratio: Expansion ratio for the channel MLP (hidden = hidden_dim * mlp_ratio).
        dropout: Dropout probability applied inside the channel MLP.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_modes_h: int = 12,
        n_modes_w: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_modes_h = n_modes_h
        self.n_modes_w = n_modes_w

        # Learnable complex weights for the truncated Fourier space.
        # We parameterise as two real matrices (real, imag parts) for
        # the selected modes: shape (n_modes_h, n_modes_w, C, C).
        self.w_re = nn.Parameter(
            torch.empty(n_modes_h, n_modes_w, hidden_dim, hidden_dim)
        )
        self.w_im = nn.Parameter(
            torch.empty(n_modes_h, n_modes_w, hidden_dim, hidden_dim)
        )
        nn.init.xavier_uniform_(self.w_re.view(n_modes_h * n_modes_w, hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.w_im.view(n_modes_h * n_modes_w, hidden_dim, hidden_dim))

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def _fourier_mix(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learned mixing in truncated Fourier space.

        Args:
            x: ``(B, H, W, C)`` spatial field.

        Returns:
            ``(B, H, W, C)`` after Fourier mixing.
        """
        B, H, W, C = x.shape

        # 2-D real FFT over spatial axes  ->  (B, H, W//2+1, C)  complex
        x_ft = fft.rfft2(x, dim=(1, 2), norm="ortho")  # (B, H, W//2+1, C)

        nh = min(self.n_modes_h, H)
        nw = min(self.n_modes_w, x_ft.size(2))

        # Positive + negative frequency rows (top and bottom of H axis)
        nh_pos = nh // 2 + nh % 2  # positive-freq rows
        nh_neg = nh // 2            # negative-freq rows (mirrored)

        out_ft = torch.zeros_like(x_ft)

        # Helper: apply complex weight   y = (w_re + i*w_im) @ x_complex
        def _apply_w(x_c: torch.Tensor, h_slice: slice) -> torch.Tensor:
            # x_c: (B, nh_part, nw, C)  complex
            xr = x_c.real  # (B, nh_part, nw, C)
            xi = x_c.imag
            wr = self.w_re[h_slice, :nw]  # (nh_part, nw, C, C)
            wi = self.w_im[h_slice, :nw]
            yr = torch.einsum("bhwc,hwcd->bhwd", xr, wr) - torch.einsum("bhwc,hwcd->bhwd", xi, wi)
            yi = torch.einsum("bhwc,hwcd->bhwd", xr, wi) + torch.einsum("bhwc,hwcd->bhwd", xi, wr)
            return torch.complex(yr, yi)

        # Positive-frequency rows
        s_pos = slice(0, nh_pos)
        out_ft[:, :nh_pos, :nw] = _apply_w(x_ft[:, :nh_pos, :nw], s_pos)

        # Negative-frequency rows  (stored at H-nh_neg : H in rFFT output)
        if nh_neg > 0:
            s_neg = slice(nh_pos, nh_pos + nh_neg)
            out_ft[:, H - nh_neg : H, :nw] = _apply_w(x_ft[:, H - nh_neg : H, :nw], s_neg)

        # Inverse 2-D real FFT  ->  (B, H, W, C)
        return fft.irfft2(out_ft, s=(H, W), dim=(1, 2), norm="ortho")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, H, W, C)`` spatial field.

        Returns:
            ``(B, H, W, C)`` after Fourier mixing + MLP.
        """
        # Fourier branch (residual)
        x = x + self._fourier_mix(self.norm1(x))
        # Channel MLP branch (residual)
        x = x + self.mlp(self.norm2(x))
        return x


class AFNO(BaseModel):
    """Adaptive Fourier Neural Operator for structured-grid PDEs.

    Processes 2-D spatial fields represented as ``(B, H, W, C)`` tensors.
    Suitable for weather/climate forecasting, turbulence modelling, and other
    applications where inputs live on a regular spatial grid.

    Architecture:
    1. A channel-mixing **input projection** ``in_channels -> hidden_dim``.
    2. ``n_layers`` :class:`AFNOLayer` blocks, each mixing tokens in
       truncated Fourier space and channels via a point-wise MLP.
    3. A channel-mixing **output projection** ``hidden_dim -> out_channels``.

    Args:
        in_channels: Number of input field channels ``C_in``.
        out_channels: Number of output field channels ``C_out``.
        hidden_dim: Internal channel width.
        n_layers: Number of AFNO blocks.
        n_modes_h: Fourier modes kept along the height axis.
        n_modes_w: Fourier modes kept along the width axis.
        mlp_ratio: MLP expansion ratio inside each AFNO block.
        dropout: Dropout probability.
    """

    family: str = "neural_operators"
    name: str = "afno"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int = 64,
        n_layers: int = 4,
        n_modes_h: int = 12,
        n_modes_w: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                AFNOLayer(hidden_dim, n_modes_h, n_modes_w, mlp_ratio, dropout)
                for _ in range(n_layers)
            ]
        )
        self.out_proj = nn.Linear(hidden_dim, out_channels)

    def forward(self, x: torch.Tensor) -> ModelOutput:  # type: ignore[override]
        """Forward pass.

        Args:
            x: Input spatial field, shape ``(B, H, W, C_in)``.

        Returns:
            :class:`~pinneaple_models.base.ModelOutput` with ``y`` of shape
            ``(B, H, W, C_out)``.
        """
        h = self.in_proj(x)          # (B, H, W, hidden_dim)
        for block in self.blocks:
            h = block(h)
        return ModelOutput(y=self.out_proj(h))  # (B, H, W, out_channels)

    def forward_batch(self, batch: dict) -> ModelOutput:  # type: ignore[override]
        """Dict-based interface for the Arena/Trainer.

        Expects key ``'x'`` with shape ``(B, H, W, C_in)``.
        """
        return self.forward(batch["x"])
