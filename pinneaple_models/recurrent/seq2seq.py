from __future__ import annotations
"""Seq2Seq encoder-decoder RNN for sequence-to-sequence tasks."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import RecurrentModelBase, RNNOutput


class Seq2SeqRNN(RecurrentModelBase):
    """
    Seq2Seq RNN forecaster (MVP).

    Encoder: GRU/LSTM
    Decoder: GRU/LSTM generating H steps autoregressively (optionally teacher forcing).

    Inputs:
      x_past:  (B, L, in_dim)
      x_future:(B, H, future_dim) optional known future features (calendar/forcing)
      y_future:(B, H, out_dim) optional teacher forcing targets

    Output:
      y_hat:   (B, H, out_dim)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        horizon: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        cell: str = "gru",     # "gru" or "lstm"
        future_dim: int = 0,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.out_dim = int(out_dim)
        self.future_dim = int(future_dim)
        self.cell = cell.lower().strip()

        if self.cell not in ("gru", "lstm"):
            raise ValueError("cell must be 'gru' or 'lstm'")

        rnn_cls = nn.GRU if self.cell == "gru" else nn.LSTM

        self.encoder = rnn_cls(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=float(dropout) if num_layers > 1 else 0.0,
        )

        # decoder input: previous y + known future features (if any)
        dec_in = out_dim + self.future_dim
        self.decoder = rnn_cls(
            input_size=dec_in,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=float(dropout) if num_layers > 1 else 0.0,
        )

        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x_past: torch.Tensor,
        *,
        x_future: Optional[torch.Tensor] = None,
        y_future: Optional[torch.Tensor] = None,
        teacher_forcing: float = 0.0,
        return_loss: bool = False,
    ) -> RNNOutput:
        B, L, _ = x_past.shape
        H = self.horizon

        if self.future_dim > 0:
            if x_future is None:
                raise ValueError("future_dim > 0 but x_future is None.")
            if x_future.shape[1] != H:
                raise ValueError(f"x_future horizon mismatch: expected H={H}, got {x_future.shape[1]}")
        else:
            x_future = None

        # encode
        enc_out = self.encoder(x_past)
        if self.cell == "gru":
            _, h = enc_out
            state = h
        else:
            _, (h, c) = enc_out
            state = (h, c)

        # decode autoregressively
        y_hat_steps = []
        y_prev = torch.zeros((B, self.out_dim), device=x_past.device, dtype=x_past.dtype)

        for t in range(H):
            if self.future_dim > 0:
                fut = x_future[:, t, :]
                dec_in = torch.cat([y_prev, fut], dim=-1)  # (B, out_dim+future_dim)
            else:
                dec_in = y_prev

            dec_in = dec_in[:, None, :]  # (B,1,dec_in)

            if self.cell == "gru":
                out, state = self.decoder(dec_in, state)
            else:
                out, state = self.decoder(dec_in, state)

            y_step = self.head(out[:, 0, :])  # (B,out_dim)
            y_hat_steps.append(y_step)

            # teacher forcing (stochastic)
            if y_future is not None and teacher_forcing > 0.0:
                use_tf = (torch.rand(B, device=x_past.device) < float(teacher_forcing)).float()[:, None]
                y_prev = use_tf * y_future[:, t, :] + (1.0 - use_tf) * y_step
            else:
                y_prev = y_step

        y_hat = torch.stack(y_hat_steps, dim=1)  # (B,H,out_dim)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y_hat.device)}
        if return_loss and y_future is not None:
            losses["mse"] = self.mse(y_hat, y_future)
            losses["total"] = losses["mse"]

        return RNNOutput(y=y_hat, losses=losses, extras={})
