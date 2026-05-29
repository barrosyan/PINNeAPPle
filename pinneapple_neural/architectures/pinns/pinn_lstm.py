from __future__ import annotations
"""PINN-LSTM hybrid combining physics-informed losses with LSTM dynamics."""

from typing import Dict, Optional, List
import torch
import torch.nn as nn


class PINNLSTM(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        seq_len: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        inverse_params_names: Optional[List[str]] = None,
        initial_guesses: Optional[Dict[str, float]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.seq_len = int(seq_len)

        self.lstm = nn.LSTM(
            input_size=self.in_dim,
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        self.head = nn.Linear(int(hidden_dim), self.out_dim)

        self.inverse_params = nn.ParameterDict()
        if inverse_params_names:
            initial_guesses = initial_guesses or {}
            for name in inverse_params_names:
                init = float(initial_guesses.get(name, 0.1))
                self.inverse_params[name] = nn.Parameter(torch.tensor(init, dtype=dtype))

    @staticmethod
    def _ensure_n1(a: torch.Tensor) -> torch.Tensor:
        if a.dim() == 1:
            return a.unsqueeze(1)
        if a.dim() == 2 and a.shape[1] == 1:
            return a
        raise ValueError(f"Expected (N,) or (N,1), got {tuple(a.shape)}")

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) != self.in_dim:
            raise ValueError(f"Expected {self.in_dim} inputs, got {len(inputs)}")

        xs = [self._ensure_n1(t) for t in inputs]
        N = xs[0].shape[0]
        T = self.seq_len

        if any(t.shape[0] != N for t in xs):
            raise ValueError("All inputs must have same N")

        if N % T != 0:
            raise RuntimeError(f"N={N} must be multiple of seq_len={T}")

        B = N // T

        x_seq = torch.cat(xs, dim=1).reshape(B, T, self.in_dim)

        h, _ = self.lstm(x_seq)             # (B,T,H)
        y_seq = self.head(h)                # (B,T,out_dim)

        y = y_seq.reshape(N, self.out_dim)  # (N,out_dim)
        return y
