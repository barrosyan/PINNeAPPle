from __future__ import annotations
"""ODE-RNN hybrid for continuous-time sequence modeling."""
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import ContinuousModelBase, ContOutput
from .neural_ode import NeuralODE


class ODERNN(ContinuousModelBase):
    """
    ODE-RNN MVP:
      - Integrate hidden state with Neural ODE between observation times
      - Update with GRUCell when observation arrives
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.ode = NeuralODE(state_dim=hidden_dim, hidden=hidden_dim, num_layers=2, method="rk4")
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(
        self,
        x: torch.Tensor,     # (B,T,input_dim)
        t: torch.Tensor,     # (T,)
        *,
        y_true: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> ContOutput:
        B, T, D = x.shape
        h = torch.zeros((B, self.hidden_dim), device=x.device, dtype=x.dtype)

        hs = []
        for i in range(T):
            if i > 0:
                # integrate from t[i-1] -> t[i]
                tt = t[i-1:i+1]
                h = self.ode(h, tt).y[:, -1, :]
            h = self.gru(x[:, i, :], h)
            hs.append(h)

        h_path = torch.stack(hs, dim=1)     # (B,T,H)
        y_hat = self.head(h_path)           # (B,T,D)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=x.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y_hat, y_true)
            losses["total"] = losses["mse"]

        return ContOutput(y=y_hat, losses=losses, extras={"h": h_path})
