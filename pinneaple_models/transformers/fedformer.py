from __future__ import annotations
"""FEDformer with frequency-enhanced decomposition for time series."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import TimeSeriesModelBase, TSOutput


class FEDformer(TimeSeriesModelBase):
    """
    FEDformer (MVP approximation):
      - Frequency enhanced model; MVP uses FFT-based projection + lightweight MLP head.

    Later you can replace this with spectral attention blocks.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        horizon: int,
        d_model: int = 128,
        topk_freq: int = 32,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.topk = int(topk_freq)

        self.in_proj = nn.Linear(in_dim, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model + 2 * self.topk, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
        )
        self.out = nn.Linear(d_model, out_dim)

    def forward(
        self,
        x_past: torch.Tensor,
        *,
        x_future: Optional[torch.Tensor] = None,
        y_future: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> TSOutput:
        B, L, D = x_past.shape
        H = self.horizon

        h = self.in_proj(x_past)  # (B,L,d_model)
        last = h[:, -1, :]        # (B,d_model)

        # FFT on the first channel mean (cheap proxy)
        s = x_past.mean(dim=-1)   # (B,L)
        fft = torch.fft.rfft(s, dim=1)  # (B, L//2+1)
        mag = torch.abs(fft)
        idx = torch.topk(mag, k=min(self.topk, mag.shape[1]), dim=1).indices  # (B,topk)

        # gather real/imag of topk bins
        bins = torch.gather(fft, 1, idx)
        feat_freq = torch.cat([bins.real, bins.imag], dim=1)  # (B,2*topk)

        feat = torch.cat([last, feat_freq], dim=1)
        feat = self.mlp(feat)

        dec = feat[:, None, :].repeat(1, H, 1)
        y_hat = self.out(dec)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y_hat.device)}
        if return_loss and y_future is not None:
            losses["mse"] = self.mse(y_hat, y_future)
            losses["total"] = losses["mse"]
        return TSOutput(y=y_hat, losses=losses, extras={"freq_idx": idx})
