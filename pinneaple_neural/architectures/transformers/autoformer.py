from __future__ import annotations
"""Autoformer (minimal faithful) with Auto-Correlation mechanism for time series forecasting."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TimeSeriesModelBase, TSOutput


# ============================================================
# 1) Series Decomposition (Moving Average) with correct padding
# ============================================================

class SeriesDecomp(nn.Module):
    """Moving-average decomposition: x = seasonal + trend."""
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        k = int(kernel_size)
        if k < 3 or k % 2 == 0:
            raise ValueError("kernel_size should be odd and >= 3.")
        self.k = k
        self.pad = (k - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=k, stride=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, L, D)
        returns: seasonal, trend (both (B, L, D))
        """
        # replicate padding avoids edge distortion
        xt = x.transpose(1, 2)  # (B, D, L)
        xt = F.pad(xt, (self.pad, self.pad), mode="replicate")
        trend = self.avg(xt).transpose(1, 2)  # (B, L, D)
        seasonal = x - trend
        return seasonal, trend


# ============================================================
# 2) Auto-Correlation (FFT-based) + Time Delay Aggregation
# ============================================================

class AutoCorrelation(nn.Module):
    """
    Autoformer-style Auto-Correlation:
      - compute cross-correlation between Q and K via FFT
      - pick top-k delays
      - aggregate V by shifting (time-delay aggregation)

    Inputs are projected outside (like attention).
    """
    def __init__(self, topk_factor: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.topk_factor = float(topk_factor)
        self.drop = nn.Dropout(float(dropout))

    @staticmethod
    def _corr_fft(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        q, k: (B, H, L, Dh)
        return corr: (B, H, L) correlation over delays (0..L-1)
        """
        # FFT along time axis
        qf = torch.fft.rfft(q, dim=2)  # (B,H,Lf,Dh)
        kf = torch.fft.rfft(k, dim=2)
        # cross-correlation in freq: q * conj(k)
        res = qf * torch.conj(kf)
        corr = torch.fft.irfft(res, n=q.shape[2], dim=2)  # (B,H,L,Dh)
        # reduce over channel dim
        corr = corr.mean(dim=-1)  # (B,H,L)
        return corr

    @staticmethod
    def _shift_gather(v: torch.Tensor, delays: torch.Tensor) -> torch.Tensor:
        """
        v: (B, H, L, Dh)
        delays: (B, H, K) each in [0, L-1]
        returns agg: (B, H, L, Dh)
        """
        B, H, L, Dh = v.shape
        K = delays.shape[-1]

        # Build indices for gather: for each t, gather v[(t - delay) mod L]
        t = torch.arange(L, device=v.device).view(1, 1, L, 1)  # (1,1,L,1)
        d = delays.unsqueeze(2)  # (B,H,1,K)
        idx = (t - d) % L  # (B,H,L,K)

        # Expand v to gather along time dimension
        v_exp = v.unsqueeze(3).expand(B, H, L, K, Dh)  # (B,H,L,K,Dh)

        # Gather requires index shape (..., 1) for the gathered dim
        idx_exp = idx.unsqueeze(-1).expand(B, H, L, K, Dh)  # (B,H,L,K,Dh)
        gathered = torch.gather(v_exp, dim=2, index=idx_exp)  # (B,H,L,K,Dh)
        return gathered

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        q,k,v: (B, H, L, Dh)
        out:   (B, H, L, Dh)
        """
        B, H, L, Dh = q.shape

        corr = self._corr_fft(q, k)  # (B,H,L)

        # top-k delays per (B,H) from correlation over delays axis
        k_top = max(1, int(self.topk_factor * math.log(L + 1)))
        weights, delays = torch.topk(corr, k=k_top, dim=-1)  # both (B,H,K)

        # softmax over selected delays
        attn = torch.softmax(weights, dim=-1)  # (B,H,K)
        attn = self.drop(attn)

        # gather shifted values for each delay: (B,H,L,K,Dh)
        gathered = self._shift_gather(v, delays)  # (B,H,L,K,Dh)

        # weighted sum over K delays
        out = (gathered * attn.unsqueeze(2).unsqueeze(-1)).sum(dim=3)  # (B,H,L,Dh)
        return out


class AutoCorrelationLayer(nn.Module):
    """Multi-head wrapper with projections like attention."""
    def __init__(self, d_model: int, nhead: int, topk_factor: float = 1.0, dropout: float = 0.0):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.dh = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ac = AutoCorrelation(topk_factor=topk_factor, dropout=dropout)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        """
        x_q:  (B, Lq, d_model)
        x_kv: (B, Lk, d_model)
        """
        B, Lq, _ = x_q.shape
        _, Lk, _ = x_kv.shape

        q = self.q_proj(x_q).view(B, Lq, self.nhead, self.dh).transpose(1, 2)  # (B,H,Lq,Dh)
        k = self.k_proj(x_kv).view(B, Lk, self.nhead, self.dh).transpose(1, 2)  # (B,H,Lk,Dh)
        v = self.v_proj(x_kv).view(B, Lk, self.nhead, self.dh).transpose(1, 2)  # (B,H,Lk,Dh)

        # For minimal version, assume Lq == Lk inside encoder; decoder cross uses Lq!=Lk
        # Auto-corr expects same time length for FFT-based correlation.
        # Strategy: when Lq != Lk (cross), we correlate on min length and use last window of KV.
        if Lq != Lk:
            # align using last Lq positions of KV (common in forecasting)
            x_kv_aligned = x_kv[:, -Lq:, :]
            k = self.k_proj(x_kv_aligned).view(B, Lq, self.nhead, self.dh).transpose(1, 2)
            v = self.v_proj(x_kv_aligned).view(B, Lq, self.nhead, self.dh).transpose(1, 2)

        out = self.ac(q, k, v)  # (B,H,Lq,Dh)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)  # (B,Lq,d_model)
        return self.out_proj(out)


# ============================================================
# 3) Encoder/Decoder Blocks (decomp inside like Autoformer)
# ============================================================

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, kernel_size: int, dropout: float, topk_factor: float):
        super().__init__()
        self.decomp1 = SeriesDecomp(kernel_size)
        self.decomp2 = SeriesDecomp(kernel_size)

        self.ac = AutoCorrelationLayer(d_model, nhead, topk_factor=topk_factor, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L,d_model)
        # Auto-correlation "attention"
        x = x + self.drop(self.ac(x, x))
        x = self.norm1(x)

        # decomp
        seasonal, _trend = self.decomp1(x)
        x = seasonal  # keep seasonal stream

        # FFN
        x = x + self.drop(self.ff(x))
        x = self.norm2(x)

        seasonal, _trend = self.decomp2(x)
        return seasonal


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, kernel_size: int, dropout: float, topk_factor: float):
        super().__init__()
        self.decomp1 = SeriesDecomp(kernel_size)
        self.decomp2 = SeriesDecomp(kernel_size)
        self.decomp3 = SeriesDecomp(kernel_size)

        self.self_ac = AutoCorrelationLayer(d_model, nhead, topk_factor=topk_factor, dropout=dropout)
        self.cross_ac = AutoCorrelationLayer(d_model, nhead, topk_factor=topk_factor, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

        # trend projection accumulates trend residuals at each layer
        self.trend_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, trend: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B,Ld,d_model) seasonal stream
        # trend: (B,Ld,d_model) trend stream in model space

        # self auto-corr
        x = x + self.drop(self.self_ac(x, x))
        x = self.norm1(x)
        seasonal, t1 = self.decomp1(x)
        trend = trend + self.trend_proj(t1)

        # cross auto-corr (decoder attends to encoder)
        seasonal = seasonal + self.drop(self.cross_ac(seasonal, enc_out))
        seasonal = self.norm2(seasonal)
        seasonal, t2 = self.decomp2(seasonal)
        trend = trend + self.trend_proj(t2)

        # FFN
        seasonal = seasonal + self.drop(self.ff(seasonal))
        seasonal = self.norm3(seasonal)
        seasonal, t3 = self.decomp3(seasonal)
        trend = trend + self.trend_proj(t3)

        return seasonal, trend


# ============================================================
# 4) Autoformer Model
# ============================================================

class Autoformer(TimeSeriesModelBase):
    """
    Autoformer (minimal faithful):
      - Decompose input into seasonal/trend
      - Encoder: AutoCorrelation + decomposition
      - Decoder: AutoCorrelation (self + cross) + decomposition, accumulates trend
      - Output: y = seasonal_out + trend_out (projected to out_dim)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        horizon: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        kernel_size: int = 25,
        dropout: float = 0.0,
        topk_factor: float = 1.0,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        d_ff = 4 * int(d_model)

        self.decomp = SeriesDecomp(kernel_size)

        # input projections
        self.in_proj = nn.Linear(in_dim, d_model)

        # encoder/decoder stacks
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, nhead, d_ff, kernel_size, dropout, topk_factor)
            for _ in range(int(num_layers))
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, nhead, d_ff, kernel_size, dropout, topk_factor)
            for _ in range(int(num_layers))
        ])

        # output heads
        self.seasonal_out = nn.Linear(d_model, out_dim)
        self.trend_out = nn.Linear(d_model, out_dim)

        # A simple learned "future seasonal query" (instead of repeating last token)
        self.future_seasonal_query = nn.Parameter(torch.zeros(1, self.horizon, d_model))
        nn.init.normal_(self.future_seasonal_query, mean=0.0, std=0.02)

    def forward(
        self,
        x_past: torch.Tensor,
        *,
        x_future: Optional[torch.Tensor] = None,
        y_future: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> TSOutput:
        """
        x_past: (B, L, in_dim)
        returns y_hat: (B, H, out_dim)
        """
        B, L, D = x_past.shape
        H = self.horizon

        # 1) decompose in data space
        seasonal, trend = self.decomp(x_past)  # both (B,L,in_dim)

        # 2) project to model space
        enc_in = self.in_proj(seasonal)  # (B,L,d_model)

        # 3) encoder
        enc_out = enc_in
        for layer in self.encoder:
            enc_out = layer(enc_out)  # keep seasonal stream

        # 4) decoder init:
        # - seasonal stream starts from learned query (H tokens)
        # - trend stream continues from last observed trend (repeat), projected into d_model space
        dec_seasonal = self.future_seasonal_query.expand(B, H, -1)  # (B,H,d_model)

        # trend continuation in model space (simple but consistent with decoder accumulating updates)
        trend_last = trend[:, -1:, :]  # (B,1,in_dim)
        trend_init = trend_last.repeat(1, H, 1)  # (B,H,in_dim)
        dec_trend = self.in_proj(trend_init)  # (B,H,d_model)

        # 5) decoder
        for layer in self.decoder:
            dec_seasonal, dec_trend = layer(dec_seasonal, enc_out, dec_trend)

        # 6) project outputs
        y_season = self.seasonal_out(dec_seasonal)  # (B,H,out_dim)
        y_trend = self.trend_out(dec_trend)         # (B,H,out_dim)
        y_hat = y_season + y_trend

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y_hat.device)}
        if return_loss and y_future is not None:
            losses["mse"] = self.mse(y_hat, y_future)
            losses["total"] = losses["mse"]

        extras = {
            "enc_out": enc_out,
            "dec_seasonal": dec_seasonal,
            "dec_trend": dec_trend,
        }
        return TSOutput(y=y_hat, losses=losses, extras=extras)