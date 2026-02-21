from __future__ import annotations
"""Temporal Fusion Transformer for interpretable time series forecasting."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import TimeSeriesModelBase, TSOutput


class _GatedResidualNetwork(nn.Module):
    def __init__(self, d: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)
        self.gate = nn.Linear(d, d)
        self.drop = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.gelu(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        g = torch.sigmoid(self.gate(x))
        y = x + g * h
        return self.norm(y)


class TemporalFusionTransformer(TimeSeriesModelBase):
    """
    TFT (MVP):
      - GRN blocks + self-attention + output head
      - Focus on practicality; missing some TFT specifics (static covariates, variable selection) in MVP
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        horizon: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.horizon = int(horizon)

        self.in_proj = nn.Linear(in_dim, d_model)
        self.grn_in = _GatedResidualNetwork(d_model, hidden=2 * d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.grn_out = _GatedResidualNetwork(d_model, hidden=2 * d_model, dropout=dropout)
        self.out = nn.Linear(d_model, out_dim)

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

        h = self.in_proj(x_past)
        h = self.grn_in(h)
        h = self.enc(h)
        h = self.grn_out(h)

        last = h[:, -1:, :]
        dec = last.repeat(1, H, 1)
        y_hat = self.out(dec)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y_hat.device)}
        if return_loss and y_future is not None:
            losses["mse"] = self.mse(y_hat, y_future)
            losses["total"] = losses["mse"]
        return TSOutput(y=y_hat, losses=losses, extras={"enc": h})
