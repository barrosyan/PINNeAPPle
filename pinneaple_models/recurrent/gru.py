from __future__ import annotations
"""GRU and bidirectional GRU models."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import RecurrentModelBase, RNNOutput


class GRUModel(RecurrentModelBase):
    """
    GRU forecaster.

    Inputs:
      x_past: (B, L, in_dim)
    Output:
      y_hat:  (B, H, out_dim) using the last hidden state repeated (MVP).
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        horizon: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=float(dropout) if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x_past: torch.Tensor,
        *,
        y_future: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> RNNOutput:
        B, L, _ = x_past.shape
        H = self.horizon

        _, h = self.gru(x_past)        # h: (num_layers, B, hidden_dim)
        h_last = h[-1]                 # (B, hidden_dim)

        dec = h_last[:, None, :].repeat(1, H, 1)  # (B,H,hidden_dim)
        y_hat = self.head(dec)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y_hat.device)}
        if return_loss and y_future is not None:
            losses["mse"] = self.mse(y_hat, y_future)
            losses["total"] = losses["mse"]

        return RNNOutput(y=y_hat, losses=losses, extras={"h_last": h_last})


class BiGRUModel(RecurrentModelBase):
    """
    Bidirectional GRU forecaster.

    MVP decoding:
      - concatenates last layer forward+backward hidden and repeats over horizon.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        horizon: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=float(dropout) if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Linear(2 * hidden_dim, out_dim)

    def forward(
        self,
        x_past: torch.Tensor,
        *,
        y_future: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> RNNOutput:
        B, L, _ = x_past.shape
        H = self.horizon

        _, h = self.gru(x_past)   # (num_layers*2, B, hidden_dim)
        # last layer forward/backward:
        h_f = h[-2]
        h_b = h[-1]
        h_last = torch.cat([h_f, h_b], dim=-1)  # (B,2H)

        dec = h_last[:, None, :].repeat(1, H, 1)
        y_hat = self.head(dec)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y_hat.device)}
        if return_loss and y_future is not None:
            losses["mse"] = self.mse(y_hat, y_future)
            losses["total"] = losses["mse"]

        return RNNOutput(y=y_hat, losses=losses, extras={"h_last": h_last})
