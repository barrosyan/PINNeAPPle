from __future__ import annotations
"""TimesNet-like forecasting with multi-period 2D variation modeling."""

from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TimeSeriesModelBase, TSOutput


class _InceptionBlock2D(nn.Module):
    """Simple multi-kernel Conv2D inception block with projection."""
    def __init__(self, d: int, k_sizes: Tuple[int, ...] = (3, 5, 7), dropout: float = 0.0):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(d, d, kernel_size=(k, k), padding=(k // 2, k // 2))
            for k in k_sizes
        ])
        self.proj = nn.Conv2d(d * len(k_sizes), d, kernel_size=1)
        self.norm = nn.BatchNorm2d(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,D,H,W)
        ys = [c(x) for c in self.convs]
        y = torch.cat(ys, dim=1)
        y = F.gelu(y)
        y = self.proj(y)
        y = self.norm(y)
        return self.drop(y)


def _topk_periods_fft(x: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate Top-K periods using FFT magnitudes.

    Args:
        x: (B, L, D) input sequence (projected features).
        top_k: number of periods.

    Returns:
        periods: (B, K) integer periods (>=2 and <=L)
        weights: (B, K) nonnegative weights (softmaxed)
    """
    B, L, D = x.shape
    # Use mean over channels for period discovery (stable for multivariate)
    s = x.mean(dim=-1)  # (B, L)

    # rfft over time
    fft = torch.fft.rfft(s, dim=1)                 # (B, L//2+1)
    mag = torch.abs(fft)                           # (B, F)
    mag[:, 0] = 0.0                                # remove DC

    Fbins = mag.shape[1]
    k = min(top_k, max(1, Fbins - 1))

    # top-k frequencies (indices)
    vals, idx = torch.topk(mag, k=k, dim=1)        # (B, K)
    # convert frequency index to period: p ≈ L / f
    # clamp to [2, L]
    periods = torch.clamp((L / idx.clamp(min=1)).round().long(), min=2, max=L)

    weights = torch.softmax(vals, dim=1)           # (B, K)
    return periods, weights


def _reshape_to_2d(x: torch.Tensor, period: int) -> Tuple[torch.Tensor, int]:
    """
    Reshape (B, L, D) -> (B, D, n, period) with padding to multiple of period.

    Returns:
        x2d: (B, D, n, period)
        Lpad: padded length
    """
    B, L, D = x.shape
    if L % period != 0:
        Lpad = ((L // period) + 1) * period
        pad_len = Lpad - L
        x = F.pad(x, (0, 0, 0, pad_len))  # pad time dimension (L)
    else:
        Lpad = L

    n = Lpad // period
    x = x.transpose(1, 2)                # (B, D, Lpad)
    x2d = x.reshape(B, D, n, period)     # (B, D, n, period)
    return x2d, Lpad


def _reshape_back_1d(x2d: torch.Tensor, L: int) -> torch.Tensor:
    """
    Reshape (B, D, n, period) -> (B, L, D) and crop to original L.
    """
    B, D, n, p = x2d.shape
    x = x2d.reshape(B, D, n * p)         # (B, D, Lpad)
    x = x[:, :, :L]                      # crop
    x = x.transpose(1, 2)                # (B, L, D)
    return x


class TimesNet(TimeSeriesModelBase):
    """
    TimesNet-like forecasting:
      1) FFT period discovery (Top-K)
      2) Reshape into 2D maps per period
      3) 2D inception blocks (multi-scale)
      4) Weighted aggregation across periods
      5) Flatten head for horizon prediction

    Works for uni- or multivariate series:
      x_past: (B, L, in_dim)
      y_hat : (B, H, out_dim)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        horizon: int,
        d_model: int = 128,
        num_blocks: int = 2,
        top_k_periods: int = 3,
        head_window: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.top_k_periods = int(top_k_periods)
        self.num_blocks = int(num_blocks)
        self.head_window = int(head_window)

        self.in_proj = nn.Linear(in_dim, d_model)

        self.blocks2d = nn.ModuleList([
            _InceptionBlock2D(d_model, dropout=dropout) for _ in range(self.num_blocks)
        ])
        self.norm1d = nn.LayerNorm(d_model)

        # Flatten head: last W steps -> H*out_dim
        self.head = nn.Sequential(
            nn.Linear(d_model * self.head_window, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim * self.horizon),
        )

    def forward(
        self,
        x_past: torch.Tensor,
        *,
        x_future: Optional[torch.Tensor] = None,
        y_future: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> TSOutput:
        B, L, _ = x_past.shape
        H = self.horizon

        # Project
        x = F.gelu(self.in_proj(x_past))  # (B, L, D)

        # Discover Top-K periods & weights
        periods, weights = _topk_periods_fft(x, self.top_k_periods)  # (B,K), (B,K)

        # Multi-period 2D modeling + aggregation
        agg = torch.zeros_like(x)
        for k in range(periods.shape[1]):
            # Use a single scalar period for each sample; loop per-sample is expensive.
            # We approximate by using the batch median period for this k.
            pk = int(periods[:, k].median().item())

            x2d, _ = _reshape_to_2d(x, pk)          # (B,D,n,pk)
            y2d = x2d
            for blk in self.blocks2d:
                y2d = y2d + blk(y2d)                # residual 2D
            y1d = _reshape_back_1d(y2d, L)          # (B,L,D)

            wk = weights[:, k].view(B, 1, 1)        # (B,1,1)
            agg = agg + wk * y1d

        x = self.norm1d(agg)  # (B,L,D)

        # Head window (robust for short L)
        W = min(self.head_window, L)
        tail = x[:, -W:, :]                              # (B,W,D)
        if W < self.head_window:
            # left-pad to fixed size for head
            pad = self.head_window - W
            tail = F.pad(tail, (0, 0, pad, 0))          # pad time on the left

        flat = tail.reshape(B, self.head_window * x.shape[-1])  # (B, W*D)
        y_hat = self.head(flat).reshape(B, H, -1)               # (B,H,out_dim)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y_hat.device)}
        if return_loss and y_future is not None:
            losses["mse"] = self.mse(y_hat, y_future)
            losses["total"] = losses["mse"]

        return TSOutput(y=y_hat, losses=losses, extras={"periods": periods, "weights": weights})