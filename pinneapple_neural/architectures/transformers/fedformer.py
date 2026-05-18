from __future__ import annotations
"""FEDformer-inspired MVP: FFT top-k features + lightweight MLP head."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import TimeSeriesModelBase, TSOutput


class FEDformer(TimeSeriesModelBase):
    """
    Fast baseline inspired by FEDformer ideas:
      - learn a 1D projection for spectral features (cheap multivariate proxy)
      - select top-k frequency bins (batch-consistent)
      - mix last-token embedding + spectral features with a small MLP
      - horizon embedding to avoid constant repeated outputs
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

        # cheap learned 1D projection for FFT features (better than mean over channels)
        self.freq_proj = nn.Linear(in_dim, 1, bias=False)
        self.freq_norm = nn.LayerNorm(2 * self.topk)

        self.mlp = nn.Sequential(
            nn.Linear(d_model + 2 * self.topk, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
        )

        # simple horizon conditioning (still very lightweight)
        self.h_emb = nn.Embedding(self.horizon, d_model)
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

        h = self.in_proj(x_past)     # (B,L,d_model)
        last = h[:, -1, :]           # (B,d_model)

        # FFT on learned 1D projection (cheap multivariate proxy)
        s = self.freq_proj(x_past).squeeze(-1)     # (B,L)
        fft = torch.fft.rfft(s, dim=1)            # (B,F)
        mag = torch.abs(fft)                       # (B,F)

        # batch-consistent top-k bins (more stable than per-sample indices)
        mag_mean = mag.mean(dim=0)                 # (F,)
        idx0 = torch.topk(mag_mean, k=min(self.topk, mag.shape[1])).indices  # (topk,)
        idx = idx0.unsqueeze(0).expand(B, -1)      # (B,topk)

        bins = torch.gather(fft, 1, idx)           # (B,topk) complex
        feat_freq = torch.cat([bins.real, bins.imag], dim=1)  # (B,2*topk)
        feat_freq = self.freq_norm(feat_freq)

        feat = torch.cat([last, feat_freq], dim=1)
        feat = self.mlp(feat)                      # (B,d_model)

        t = torch.arange(H, device=x_past.device)
        dec = feat[:, None, :] + self.h_emb(t)[None, :, :]    # (B,H,d_model)
        y_hat = self.out(dec)                      # (B,H,out_dim)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y_hat.device)}
        if return_loss and y_future is not None:
            losses["mse"] = self.mse(y_hat, y_future)
            losses["total"] = losses["mse"]

        return TSOutput(y=y_hat, losses=losses, extras={"freq_idx": idx0})