from __future__ import annotations
"""Vanilla Transformer encoder for time series."""

import torch
import torch.nn as nn

from .base import TimeSeriesModelBase


class VanillaTransformer(TimeSeriesModelBase):
    """
    Vanilla Transformer encoder for time series (MVP).

    Input:
      x: (B, T, D)

    Output:
      - If pool="none": (B, T, out_dim)
      - If pool="mean" or "last": (B, out_dim)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pool: str = "none",  # "none" | "mean" | "last"
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.pool = str(pool).lower().strip()

        self.in_proj = nn.Linear(self.in_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.head = nn.Linear(d_model, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.encoder(h)
        y = self.head(h)

        if self.pool == "none":
            return y
        if self.pool == "mean":
            return y.mean(dim=1)
        if self.pool == "last":
            return y[:, -1, :]
        raise ValueError(f"Unknown pool='{self.pool}'. Use: none | mean | last")
