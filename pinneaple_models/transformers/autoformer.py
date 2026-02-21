from __future__ import annotations
"""Autoformer with auto-correlation mechanism for time series."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import TimeSeriesModelBase, TSOutput


class _SeriesDecomp(nn.Module):
    """
    Simple moving-average decomposition (trend + seasonal).
    """
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.k = int(kernel_size)
        self.pad = (self.k - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=self.k, stride=1, padding=self.pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L,D) -> trend: moving avg
        t = self.avg(x.transpose(1, 2)).transpose(1, 2)
        return t


class Autoformer(TimeSeriesModelBase):
    """
    Autoformer (MVP approximation):
      - Series decomposition into trend/seasonal
      - Then apply attention-style encoder (standard Transformer in MVP)
      - Predict horizon by extending trend + forecasting seasonal

    This is a scaffold; later you can replace attention with AutoCorrelation.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        horizon: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        kernel_size: int = 25,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.decomp = _SeriesDecomp(kernel_size=kernel_size)

        self.in_proj = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=float(dropout), batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head_seasonal = nn.Linear(d_model, out_dim)
        self.head_trend = nn.Linear(in_dim, out_dim)

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

        trend = self.decomp(x_past)
        seasonal = x_past - trend

        h = self.enc(self.in_proj(seasonal))
        last = h[:, -1:, :].repeat(1, H, 1)
        y_season = self.head_seasonal(last)

        # extend trend with last known slope (MVP: constant continuation)
        trend_last = trend[:, -1:, :].repeat(1, H, 1)
        y_trend = self.head_trend(trend_last)

        y_hat = y_trend + y_season

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y_hat.device)}
        if return_loss and y_future is not None:
            losses["mse"] = self.mse(y_hat, y_future)
            losses["total"] = losses["mse"]
        return TSOutput(y=y_hat, losses=losses, extras={"trend": trend, "seasonal": seasonal})
