from __future__ import annotations
"""Efficient encoder-only baseline (Informer-like distill) for long TS."""

from typing import Dict, Optional

import math
import torch
import torch.nn as nn

from .base import TimeSeriesModelBase, TSOutput


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        L = x.size(1)
        if L > self.pe.size(1):
            raise ValueError(f"Sequence length {L} exceeds max_len {self.pe.size(1)}")
        return x + self.pe[:, :L, :]


class Informer(TimeSeriesModelBase):
    """
    Efficient encoder-only baseline:
      - Transformer encoder (standard attention)
      - Optional distilling (Conv1d stride=2) on early layers only
      - Attention pooling to get sequence context
      - Forecast head uses context + (optional) known future covariates per horizon
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        horizon: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        distill: bool = True,
        distill_layers: int = 2,   # only first k layers downsample
        future_dim: int = 0,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.future_dim = int(future_dim)

        self.in_proj = nn.Linear(in_dim, d_model)
        self.future_proj = nn.Linear(future_dim, d_model) if future_dim > 0 else None

        self.pos = SinusoidalPositionalEncoding(d_model=d_model, max_len=10000)

        self.layers = nn.ModuleList()
        self.down = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=float(dropout),
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,  # usually helps stability
                )
            )
            use_distill = distill and (i < int(distill_layers))
            self.down.append(
                nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
                if use_distill else nn.Identity()
            )

        # cheap attention pooling: one learned query -> weights over time
        self.pool_q = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pool = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=float(dropout), batch_first=True)

        # head: combine pooled context with each horizon's future embedding (or zeros)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
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

        h = self.in_proj(x_past)
        h = self.pos(h)

        for layer, down in zip(self.layers, self.down):
            h = layer(h)
            if not isinstance(down, nn.Identity):
                # (B, L, D) -> (B, D, L) -> downsample -> (B, L', D)
                h = down(h.transpose(1, 2)).transpose(1, 2)

        # attention pooling: q attends to all tokens -> context (B, 1, D)
        q = self.pool_q.expand(B, 1, -1)
        ctx, _ = self.pool(q, h, h, need_weights=False)
        ctx = ctx.expand(B, H, -1)  # (B, H, D)

        if self.future_proj is not None and x_future is not None:
            # expect x_future: (B, H, future_dim)
            fut = self.future_proj(x_future)
        else:
            fut = torch.zeros(B, H, ctx.size(-1), device=ctx.device, dtype=ctx.dtype)

        y_hat = self.head(torch.cat([ctx, fut], dim=-1))  # (B, H, out_dim)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y_hat.device)}
        if return_loss and y_future is not None:
            losses["mse"] = self.mse(y_hat, y_future)
            losses["total"] = losses["mse"]

        return TSOutput(y=y_hat, losses=losses, extras={"enc": h, "ctx": ctx[:, :1, :]})