from __future__ import annotations
"""Causal Transformer encoder for autoregressive forecasting."""

import math
import torch
import torch.nn as nn

from .base import TimeSeriesModelBase


class VanillaTransformer(TimeSeriesModelBase):
    """
    Causal Transformer encoder for autoregressive forecasting.

    Input:
      x: (B, T, D)

    Output:
      - If pool="none": (B, T, out_dim)  (each t predicts y_t using x_{<=t})
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
        max_len: int = 4096,  # max T supported for learned positional encoding
        pos_encoding: str = "learned",  # "learned" | "sinusoidal"
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.pool = str(pool).lower().strip()
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.pos_encoding = str(pos_encoding).lower().strip()

        self.in_proj = nn.Linear(self.in_dim, self.d_model)
        self.in_dropout = nn.Dropout(dropout)
        self.in_norm = nn.LayerNorm(self.d_model)

        if self.pos_encoding == "learned":
            self.pos_emb = nn.Parameter(torch.zeros(1, self.max_len, self.d_model))
            nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
            self.register_buffer("_sin_pe", None, persistent=False)
        elif self.pos_encoding == "sinusoidal":
            self.pos_emb = None
            pe = self._build_sinusoidal_pe(self.max_len, self.d_model)
            self.register_buffer("_sin_pe", pe, persistent=False)
        else:
            raise ValueError("pos_encoding must be 'learned' or 'sinusoidal'")

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,  # common in modern practice; improves stability
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.head = nn.Linear(self.d_model, self.out_dim)

    @staticmethod
    def _build_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        # True where positions should be masked (i.e., future positions)
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, _ = x.shape
        if T > self.max_len:
            raise ValueError(f"T={T} > max_len={self.max_len}. Increase max_len.")

        h = self.in_proj(x)  # (B, T, d_model)

        # Add positional encoding
        if self.pos_encoding == "learned":
            h = h + self.pos_emb[:, :T, :]
        else:  # sinusoidal
            h = h + self._sin_pe[:, :T, :]

        h = self.in_norm(self.in_dropout(h))

        # Causal mask: block attention to future tokens
        mask = self._causal_mask(T, x.device)

        h = self.encoder(h, mask=mask)  # (B, T, d_model)
        y = self.head(h)                # (B, T, out_dim)

        if self.pool == "none":
            return y
        if self.pool == "mean":
            return y.mean(dim=1)
        if self.pool == "last":
            return y[:, -1, :]
        raise ValueError(f"Unknown pool='{self.pool}'. Use: none | mean | last")