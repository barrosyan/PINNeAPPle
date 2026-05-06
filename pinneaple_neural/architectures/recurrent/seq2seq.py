from __future__ import annotations
"""Seq2Seq encoder-decoder RNN for sequence-to-sequence tasks."""

from typing import Dict, Optional, Literal

import torch
import torch.nn as nn

from .base import RecurrentModelBase, RNNOutput


class Seq2SeqRNN(RecurrentModelBase):
    """
    Seq2Seq RNN forecaster.

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
        cell: Literal["gru", "lstm"] = "gru",
        future_dim: int = 0,
        # --- improvements ---
        init_y: Literal["zeros", "last_past", "learned"] = "zeros",
        last_past_y_slice: Optional[slice] = None,
        input_dropout: float = 0.0,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.out_dim = int(out_dim)
        self.future_dim = int(future_dim)
        self.cell = str(cell).lower().strip()

        if self.cell not in ("gru", "lstm"):
            raise ValueError("cell must be 'gru' or 'lstm'")

        self.init_y = str(init_y).lower().strip()
        if self.init_y not in ("zeros", "last_past", "learned"):
            raise ValueError("init_y must be one of: 'zeros', 'last_past', 'learned'")

        # If your x_past contains the target in some slice, set last_past_y_slice.
        # Default assumes the first out_dim channels correspond to y.
        self.last_past_y_slice = last_past_y_slice or slice(0, out_dim)

        rnn_cls = nn.GRU if self.cell == "gru" else nn.LSTM

        self.encoder = rnn_cls(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=float(dropout) if num_layers > 1 else 0.0,
        )

        dec_in = out_dim + self.future_dim
        self.decoder = rnn_cls(
            input_size=dec_in,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=float(dropout) if num_layers > 1 else 0.0,
        )

        self.in_drop = nn.Dropout(float(input_dropout)) if input_dropout > 0 else nn.Identity()
        self.out_drop = nn.Dropout(float(head_dropout)) if head_dropout > 0 else nn.Identity()
        self.head = nn.Linear(hidden_dim, out_dim)

        # learned start token for y_prev, if requested
        if self.init_y == "learned":
            self.y0 = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("y0", None)

    def _validate_inputs(
        self,
        x_past: torch.Tensor,
        x_future: Optional[torch.Tensor],
        y_future: Optional[torch.Tensor],
    ) -> None:
        if x_past.ndim != 3:
            raise ValueError(f"x_past must be (B,L,in_dim). Got shape={tuple(x_past.shape)}")
        B, L, _ = x_past.shape
        if B <= 0 or L <= 0:
            raise ValueError(f"x_past has invalid batch/length: B={B}, L={L}")

        H = self.horizon

        if self.future_dim > 0:
            if x_future is None:
                raise ValueError("future_dim > 0 but x_future is None.")
            if x_future.ndim != 3:
                raise ValueError(f"x_future must be (B,H,future_dim). Got shape={tuple(x_future.shape)}")
            if x_future.shape[0] != B:
                raise ValueError(f"x_future batch mismatch: {x_future.shape[0]} != {B}")
            if x_future.shape[1] != H:
                raise ValueError(f"x_future horizon mismatch: expected H={H}, got {x_future.shape[1]}")
            if x_future.shape[2] != self.future_dim:
                raise ValueError(
                    f"x_future feature mismatch: expected future_dim={self.future_dim}, got {x_future.shape[2]}"
                )
        else:
            # normalize: ignore any provided x_future
            if x_future is not None:
                # silently ignore to avoid surprising crashes in pipelines
                pass

        if y_future is not None:
            if y_future.ndim != 3:
                raise ValueError(f"y_future must be (B,H,out_dim). Got shape={tuple(y_future.shape)}")
            if y_future.shape[0] != B:
                raise ValueError(f"y_future batch mismatch: {y_future.shape[0]} != {B}")
            if y_future.shape[1] != H:
                raise ValueError(f"y_future horizon mismatch: expected H={H}, got {y_future.shape[1]}")
            if y_future.shape[2] != self.out_dim:
                raise ValueError(
                    f"y_future out_dim mismatch: expected out_dim={self.out_dim}, got {y_future.shape[2]}"
                )

    def _init_y_prev(self, x_past: torch.Tensor) -> torch.Tensor:
        B = x_past.shape[0]
        if self.init_y == "zeros":
            return torch.zeros((B, self.out_dim), device=x_past.device, dtype=x_past.dtype)
        if self.init_y == "learned":
            return self.y0[None, :].expand(B, -1).to(device=x_past.device, dtype=x_past.dtype)
        # last_past
        y_last = x_past[:, -1, self.last_past_y_slice]
        if y_last.shape[-1] != self.out_dim:
            raise ValueError(
                f"init_y='last_past' expects x_past[:, -1, last_past_y_slice] to have out_dim={self.out_dim}, "
                f"got {y_last.shape[-1]}. Adjust last_past_y_slice."
            )
        return y_last

    def forward(
        self,
        x_past: torch.Tensor,
        *,
        x_future: Optional[torch.Tensor] = None,
        y_future: Optional[torch.Tensor] = None,
        teacher_forcing: float = 0.0,
        return_loss: bool = False,
    ) -> RNNOutput:
        self._validate_inputs(x_past, x_future, y_future)

        B, _, _ = x_past.shape
        H = self.horizon

        # normalize x_future when not used
        if self.future_dim <= 0:
            x_future = None

        # encode
        enc_out = self.encoder(x_past)
        if self.cell == "gru":
            _, state = enc_out  # h: (num_layers, B, hidden)
        else:
            _, state = enc_out  # (h, c)

        # decode autoregressively
        y_hat_steps = []
        y_prev = self._init_y_prev(x_past)

        # clamp to [0,1] just in case caller passes weird values
        tf_p = float(max(0.0, min(1.0, teacher_forcing)))

        for t in range(H):
            if self.future_dim > 0:
                fut = x_future[:, t, :]  # (B,future_dim)
                dec_in = torch.cat([y_prev, fut], dim=-1)  # (B,out_dim+future_dim)
            else:
                dec_in = y_prev  # (B,out_dim)

            dec_in = self.in_drop(dec_in)[:, None, :]  # (B,1,dec_in)
            out, state = self.decoder(dec_in, state)   # out: (B,1,hidden)
            out0 = self.out_drop(out[:, 0, :])         # (B,hidden)

            y_step = self.head(out0)                   # (B,out_dim)
            y_hat_steps.append(y_step)

            # teacher forcing (stochastic scheduled sampling)
            if y_future is not None and tf_p > 0.0:
                # Bernoulli per-sample per-step
                use_tf = (torch.rand(B, device=x_past.device) < tf_p).to(dtype=x_past.dtype)[:, None]
                y_prev = use_tf * y_future[:, t, :] + (1.0 - use_tf) * y_step
            else:
                y_prev = y_step

        y_hat = torch.stack(y_hat_steps, dim=1)  # (B,H,out_dim)

        losses: Dict[str, torch.Tensor] = {}
        if return_loss:
            if y_future is None:
                # Explicitly signal that we can't compute supervised loss
                losses["total"] = torch.tensor(0.0, device=y_hat.device, dtype=y_hat.dtype)
            else:
                losses["mse"] = self.mse(y_hat, y_future)
                losses["total"] = losses["mse"]

        return RNNOutput(y=y_hat, losses=losses, extras={})