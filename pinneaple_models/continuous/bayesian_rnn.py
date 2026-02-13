from __future__ import annotations

from typing import Dict, Optional, Literal

import torch
import torch.nn as nn

from .base import ContinuousModelBase, ContOutput


class LockedDropout(nn.Module):
    """
    Locked (aka variational) dropout for sequences:
    uses the same dropout mask across the time dimension T.

    Input:  (B, T, D)
    Output: (B, T, D)
    """
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p <= 0.0:
            return x
        if x.dim() != 3:
            raise ValueError(f"LockedDropout expects (B,T,D), got shape={tuple(x.shape)}")

        B, T, D = x.shape
        # mask shared across T
        mask = x.new_empty((B, 1, D)).bernoulli_(1.0 - self.p)
        mask = mask / (1.0 - self.p)
        return x * mask


class BayesianRNN(ContinuousModelBase):
    """
    Bayesian RNN (MVP via MC Dropout / variational dropout approximation).

    Improvements vs original:
      - LockedDropout for input/head to better match "variational dropout in RNNs"
        (same mask across time).
      - MC inference kept separate and safe: forward() remains grad-friendly.
      - Loss zeros created with x.new_zeros(()) to match dtype/device.
      - Optional return_samples in forward if you want (still default off).

    Output:
      y: (B,T,out_dim)
      extras (MC):
        - "logvar": predictive log-variance (B,T,out_dim) from MC samples
        - "samples": (S,B,T,out_dim) if requested
    """
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        cell: Literal["gru", "lstm"] = "gru",
        min_logvar: float = -10.0,
        max_logvar: float = 2.0,
        *,
        # locked dropout controls (kept simple, defaults follow `dropout`)
        locked_input_dropout: Optional[float] = None,
        locked_head_dropout: Optional[float] = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

        cell = (cell or "gru").lower().strip()
        if cell not in ("gru", "lstm"):
            raise ValueError("cell must be 'gru' or 'lstm'")
        self.cell = cell

        # Locked/variational dropout (mask shared across time)
        p_in = self.dropout if locked_input_dropout is None else float(locked_input_dropout)
        p_head = self.dropout if locked_head_dropout is None else float(locked_head_dropout)
        self.locked_in = LockedDropout(p_in)
        self.locked_head = LockedDropout(p_head)

        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            # PyTorch's built-in dropout applies BETWEEN layers (not recurrent).
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )

        # Head: keep it simple and deterministic layers + locked dropout on hidden seq
        self.head_fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.head_act = nn.GELU()
        self.head_fc2 = nn.Linear(self.hidden_dim, self.out_dim)

    def _single_pass(self, x: torch.Tensor) -> torch.Tensor:
        """
        One stochastic pass (stochastic if model is in train() and dropout p>0).
        """
        if x.dim() != 3 or x.size(-1) != self.input_dim:
            raise ValueError(
                f"x must be (B,T,{self.input_dim}), got shape={tuple(x.shape)}"
            )

        # Locked dropout on inputs (same mask over T)
        x = self.locked_in(x)

        h, _ = self.rnn(x)  # (B,T,H)

        # Head
        h = self.head_fc1(h)
        h = self.head_act(h)
        h = self.locked_head(h)  # locked dropout on hidden sequence
        y = self.head_fc2(h)     # (B,T,out_dim)
        return y

    def _mc_forward(
        self,
        x: torch.Tensor,
        *,
        mc_samples: int,
        return_samples: bool,
    ) -> ContOutput:
        """
        MC forward helper (no mode restore here; caller manages train/eval state).
        """
        S = int(mc_samples)
        ys = [self._single_pass(x) for _ in range(S)]
        Y = torch.stack(ys, dim=0)  # (S,B,T,out_dim)

        mu = Y.mean(dim=0)
        var = Y.var(dim=0, unbiased=False).clamp_min(1e-12)
        logvar = torch.log(var).clamp(self.min_logvar, self.max_logvar)

        extras = {"logvar": logvar}
        if return_samples:
            extras["samples"] = Y

        # Losses are dummy here (MC is typically inference-only)
        losses: Dict[str, torch.Tensor] = {"total": x.new_zeros(())}
        return ContOutput(y=mu, losses=losses, extras=extras)

    @torch.no_grad()
    def predict_mc(
        self,
        x: torch.Tensor,
        *,
        mc_samples: int = 16,
        return_samples: bool = False,
    ) -> ContOutput:
        """
        Inference-time MC dropout:
          - forces train(True) to keep dropout ON
          - returns mean and log-variance
        """
        was_training = self.training
        self.train(True)
        out = self._mc_forward(x, mc_samples=mc_samples, return_samples=return_samples)
        self.train(was_training)
        return out

    def forward(
        self,
        x: torch.Tensor,  # (B,T,input_dim)
        *,
        y_true: Optional[torch.Tensor] = None,  # (B,T,out_dim)
        return_loss: bool = False,
        mc_samples: int = 0,         # if >0, does MC in forward (GRAD-FRIENDLY)
        return_samples: bool = False # only relevant if mc_samples > 0
    ) -> ContOutput:
        """
        Normal forward:
          - if mc_samples == 0: single pass, optionally computes MSE loss
          - if mc_samples  > 0: MC forward (keeps current mode; if you want dropout,
            call model.train(True) before calling forward)
        """
        if mc_samples and mc_samples > 0:
            # IMPORTANT: no no_grad here. This is grad-friendly.
            # If caller wants stochasticity, they should set model.train(True).
            return self._mc_forward(x, mc_samples=mc_samples, return_samples=return_samples)

        y_hat = self._single_pass(x)

        losses: Dict[str, torch.Tensor] = {"total": x.new_zeros(())}
        if return_loss and y_true is not None:
            if y_true.shape != y_hat.shape:
                raise ValueError(f"y_true shape {tuple(y_true.shape)} != y_hat {tuple(y_hat.shape)}")
            losses["mse"] = self.mse(y_hat, y_true)
            losses["total"] = losses["mse"]

        return ContOutput(y=y_hat, losses=losses, extras={})
