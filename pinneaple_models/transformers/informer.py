from __future__ import annotations
"""Informer transformer with efficient attention for long sequences."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import TimeSeriesModelBase, TSOutput


class Informer(TimeSeriesModelBase):
    """
    Informer (MVP approximation):
      - Transformer encoder with lightweight attention (we approximate with standard attention in MVP)
      - Distilling via Conv1d downsampling between encoder layers (optional)

    This is a usable scaffold; you can later replace attention with ProbSparse.
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
        future_dim: int = 0,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.future_dim = int(future_dim)

        self.in_proj = nn.Linear(in_dim, d_model)
        self.future_proj = nn.Linear(future_dim, d_model) if future_dim > 0 else None

        self.pos = nn.Parameter(torch.zeros(1, 4096, d_model))

        self.layers = nn.ModuleList()
        self.down = nn.ModuleList()

        for _ in range(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=float(dropout),
                batch_first=True,
                activation="gelu",
            )
            self.layers.append(layer)
            if distill:
                self.down.append(nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1))
            else:
                self.down.append(nn.Identity())

        self.head = nn.Linear(d_model, out_dim)

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

        h = self.in_proj(x_past) + self.pos[:, :L, :]
        for layer, down in zip(self.layers, self.down):
            h = layer(h)
            # distill: downsample sequence length
            if not isinstance(down, nn.Identity):
                h = down(h.transpose(1, 2)).transpose(1, 2)

        # simple projection to horizon: take last token and repeat (MVP)
        last = h[:, -1:, :]
        dec = last.repeat(1, H, 1)
        y_hat = self.head(dec)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y_hat.device)}
        if return_loss and y_future is not None:
            losses["mse"] = self.mse(y_hat, y_future)
            losses["total"] = losses["mse"]
        return TSOutput(y=y_hat, losses=losses, extras={"enc": h})
