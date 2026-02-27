from __future__ import annotations
"""Fourier neural operator for global spectral learning (1D)."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import NeuralOperatorBase, OperatorOutput


class _SpectralConv(nn.Module):
    """
    1D Spectral Convolution:
      - FFT on last dimension
      - linear transform on low-frequency modes
      - iFFT back

    x: (B, in_c, L) -> (B, out_c, L)
    """

    def __init__(self, in_c: int, out_c: int, modes: int):
        super().__init__()
        self.in_c = int(in_c)
        self.out_c = int(out_c)
        self.modes = int(modes)

        # Complex weights for low-frequency modes: (in_c, out_c, modes)
        # Scale is a common heuristic; other init strategies are also used.
        scale = 1.0 / max(1, (in_c * out_c))
        self.weights = nn.Parameter(scale * torch.randn(in_c, out_c, self.modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_c, L)
        B, C, L = x.shape
        if C != self.in_c:
            raise ValueError(f"_SpectralConv expected in_c={self.in_c}, got C={C}")

        # rFFT: (B, in_c, Lf) where Lf = L//2 + 1
        x_ft = torch.fft.rfft(x, dim=-1)
        _, _, Lf = x_ft.shape

        # In case modes > Lf (small L), clamp
        m = min(self.modes, Lf)

        # IMPORTANT: out_ft must have out_c channels (not in_c)
        out_ft = torch.zeros(
            (B, self.out_c, Lf),
            device=x.device,
            dtype=torch.cfloat,
        )

        # Multiply low modes: (B,in_c,m) x (in_c,out_c,m) -> (B,out_c,m)
        out_ft[:, :, :m] = torch.einsum("bcm,com->bom", x_ft[:, :, :m], self.weights[:, :, :m])

        # iFFT back to (B, out_c, L)
        y = torch.fft.irfft(out_ft, n=L, dim=-1)
        return y


class FourierNeuralOperator(NeuralOperatorBase):
    """
    FNO-1D (MVP) - extendable to 2D/3D by changing spectral conv + projections.

    Core block:
      x <- GELU( SpectralConv(x) + 1x1Conv(x) )

    Optional:
      - use_grid=True: concatenate 1D coordinate grid as an extra input channel
        (common in FNO literature for better generalization / positional info).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 64,
        modes: int = 16,
        layers: int = 4,
        *,
        use_grid: bool = False,
    ):
        super().__init__()
        self.use_grid = bool(use_grid)

        in_c_eff = int(in_channels) + (1 if self.use_grid else 0)
        self.in_proj = nn.Conv1d(in_c_eff, int(width), kernel_size=1)

        self.convs = nn.ModuleList([_SpectralConv(int(width), int(width), int(modes)) for _ in range(int(layers))])
        self.ws = nn.ModuleList([nn.Conv1d(int(width), int(width), kernel_size=1) for _ in range(int(layers))])

        self.out_proj = nn.Conv1d(int(width), int(out_channels), kernel_size=1)

    @staticmethod
    def _make_grid_1d(u: torch.Tensor) -> torch.Tensor:
        """
        Create a normalized 1D grid in [0,1], shaped as (B,1,L), matching u.device/u.dtype.
        """
        B, _, L = u.shape
        grid = torch.linspace(0.0, 1.0, L, device=u.device, dtype=u.dtype).view(1, 1, L)
        return grid.repeat(B, 1, 1)

    def forward(
        self,
        u: torch.Tensor,  # (B, C, L)
        *,
        y_true: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> OperatorOutput:
        x = u
        if self.use_grid:
            grid = self._make_grid_1d(u)
            x = torch.cat([x, grid], dim=1)  # (B, C+1, L)

        x = self.in_proj(x)

        for sc, w in zip(self.convs, self.ws):
            x = F.gelu(sc(x) + w(x))

        y = self.out_proj(x)

        losses = {}
        if return_loss and y_true is not None:
            mse = self.mse(y, y_true)
            losses["mse"] = mse
            losses["total"] = mse
        elif return_loss:
            # If user asked for loss but didn't provide y_true, keep total=0 for compatibility
            losses["total"] = torch.tensor(0.0, device=y.device)

        return OperatorOutput(y=y, losses=losses, extras={"use_grid": self.use_grid})