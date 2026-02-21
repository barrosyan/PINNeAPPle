from __future__ import annotations
"""Fourier neural operator for global spectral learning."""
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import NeuralOperatorBase, OperatorOutput


class _SpectralConv(nn.Module):
    def __init__(self, in_c, out_c, modes):
        super().__init__()
        self.modes = modes
        self.scale = 1 / (in_c * out_c)
        self.weights = nn.Parameter(self.scale * torch.randn(in_c, out_c, modes, dtype=torch.cfloat))

    def forward(self, x):
        # x: (B,C,L)
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros_like(x_ft)

        out_ft[:, :, : self.modes] = torch.einsum(
            "bcm,com->bom", x_ft[:, :, : self.modes], self.weights
        )
        return torch.fft.irfft(out_ft, n=x.size(-1))


class FourierNeuralOperator(NeuralOperatorBase):
    """
    FNO-1D MVP (extendable to 2D/3D).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 64,
        modes: int = 16,
        layers: int = 4,
    ):
        super().__init__()
        self.in_proj = nn.Conv1d(in_channels, width, 1)
        self.convs = nn.ModuleList([_SpectralConv(width, width, modes) for _ in range(layers)])
        self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.out_proj = nn.Conv1d(width, out_channels, 1)

    def forward(
        self,
        u: torch.Tensor,      # (B,C,L)
        *,
        y_true: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> OperatorOutput:
        x = self.in_proj(u)
        for sc, w in zip(self.convs, self.ws):
            x = torch.nn.functional.gelu(sc(x) + w(x))
        y = self.out_proj(x)

        losses = {"total": torch.tensor(0.0, device=y.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]

        return OperatorOutput(y=y, losses=losses, extras={})
