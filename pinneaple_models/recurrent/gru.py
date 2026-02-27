from __future__ import annotations
"""GRU and bidirectional GRU models (baseline+)."""

from typing import Dict, Optional, Literal

import torch
import torch.nn as nn

from .base import RecurrentModelBase, RNNOutput


PoolMode = Literal["last", "mean", "max"]


class GRUModel(RecurrentModelBase):
    """
    GRU forecaster (baseline+).

    Inputs:
      x_past: (B, L, in_dim)
    Output:
      y_hat:  (B, H, out_dim)

    Options:
      - pool: "last" uses last hidden; "mean"/"max" pool over time using GRU outputs.
      - time_embedding: if enabled, concatenates an embedding of step t=0..H-1 to allow horizon-varying outputs.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        horizon: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        *,
        pool: PoolMode = "last",
        time_embedding_dim: int = 0,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.pool: PoolMode = pool

        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=float(dropout) if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.time_embedding_dim = int(time_embedding_dim)
        if self.time_embedding_dim > 0:
            self.t_embed = nn.Embedding(self.horizon, self.time_embedding_dim)
            head_in = hidden_dim + self.time_embedding_dim
        else:
            self.t_embed = None
            head_in = hidden_dim

        self.head = nn.Linear(head_in, out_dim)

    def _pool_context(self, out: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # out: (B, L, hidden_dim), h: (num_layers, B, hidden_dim)
        if self.pool == "last":
            return h[-1]  # (B, hidden_dim)
        if self.pool == "mean":
            return out.mean(dim=1)  # (B, hidden_dim)
        if self.pool == "max":
            return out.max(dim=1).values  # (B, hidden_dim)
        raise ValueError(f"Unknown pool mode: {self.pool}")

    def forward(
        self,
        x_past: torch.Tensor,
        *,
        y_future: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> RNNOutput:
        B, L, _ = x_past.shape
        H = self.horizon

        out, h = self.gru(x_past)
        ctx = self._pool_context(out, h)  # (B, hidden_dim)

        ctx_rep = ctx[:, None, :].repeat(1, H, 1)  # (B, H, hidden_dim)

        if self.t_embed is not None:
            t = torch.arange(H, device=x_past.device)  # (H,)
            t_emb = self.t_embed(t)[None, :, :].repeat(B, 1, 1)  # (B, H, time_embedding_dim)
            dec_in = torch.cat([ctx_rep, t_emb], dim=-1)  # (B, H, hidden_dim+time_emb)
        else:
            dec_in = ctx_rep

        y_hat = self.head(dec_in)  # (B, H, out_dim)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y_hat.device)}
        if return_loss and y_future is not None:
            losses["mse"] = self.mse(y_hat, y_future)
            losses["total"] = losses["mse"]

        return RNNOutput(y=y_hat, losses=losses, extras={"ctx": ctx})


class BiGRUModel(RecurrentModelBase):
    """
    Bidirectional GRU forecaster (baseline+).

    Inputs:
      x_past: (B, L, in_dim)
    Output:
      y_hat:  (B, H, out_dim)

    Notes:
      - For pool="last": concatenates last-layer forward+backward hidden.
      - For pool="mean"/"max": pools over time from bidirectional outputs.
      - Optional time embedding for horizon-varying outputs.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        horizon: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        *,
        pool: PoolMode = "last",
        time_embedding_dim: int = 0,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.pool: PoolMode = pool

        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=float(dropout) if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.time_embedding_dim = int(time_embedding_dim)
        base_ctx_dim = 2 * hidden_dim
        if self.time_embedding_dim > 0:
            self.t_embed = nn.Embedding(self.horizon, self.time_embedding_dim)
            head_in = base_ctx_dim + self.time_embedding_dim
        else:
            self.t_embed = None
            head_in = base_ctx_dim

        self.head = nn.Linear(head_in, out_dim)

    def _pool_context(self, out: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # out: (B, L, 2*hidden_dim), h: (num_layers*2, B, hidden_dim)
        if self.pool == "last":
            # last layer forward/backward:
            h_f = h[-2]  # (B, hidden_dim)
            h_b = h[-1]  # (B, hidden_dim)
            return torch.cat([h_f, h_b], dim=-1)  # (B, 2*hidden_dim)
        if self.pool == "mean":
            return out.mean(dim=1)  # (B, 2*hidden_dim)
        if self.pool == "max":
            return out.max(dim=1).values  # (B, 2*hidden_dim)
        raise ValueError(f"Unknown pool mode: {self.pool}")

    def forward(
        self,
        x_past: torch.Tensor,
        *,
        y_future: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> RNNOutput:
        B, L, _ = x_past.shape
        H = self.horizon

        out, h = self.gru(x_past)
        ctx = self._pool_context(out, h)  # (B, 2*hidden_dim)

        ctx_rep = ctx[:, None, :].repeat(1, H, 1)  # (B, H, 2*hidden_dim)

        if self.t_embed is not None:
            t = torch.arange(H, device=x_past.device)  # (H,)
            t_emb = self.t_embed(t)[None, :, :].repeat(B, 1, 1)  # (B, H, time_embedding_dim)
            dec_in = torch.cat([ctx_rep, t_emb], dim=-1)  # (B, H, 2*hidden_dim+time_emb)
        else:
            dec_in = ctx_rep

        y_hat = self.head(dec_in)  # (B, H, out_dim)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y_hat.device)}
        if return_loss and y_future is not None:
            losses["mse"] = self.mse(y_hat, y_future)
            losses["total"] = losses["mse"]

        return RNNOutput(y=y_hat, losses=losses, extras={"ctx": ctx})