"""LSTM and GRU forecasters (PyTorch, GPU-ready)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..base import TSModelBase, TSOutput


@dataclass
class RecurrentConfig:
    input_len:   int   = 64
    horizon:     int   = 16
    n_features:  int   = 1
    n_targets:   int   = 1
    hidden_size: int   = 128
    num_layers:  int   = 2
    dropout:     float = 0.1
    bidirectional: bool = False


class _RecurrentCore(TSModelBase):
    """Shared LSTM/GRU forecaster backbone."""

    def __init__(self, cfg: RecurrentConfig, cell: str = "lstm"):
        super().__init__()
        self.cfg = cfg
        D = 2 if cfg.bidirectional else 1
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=cfg.n_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_size * D),
            nn.Linear(cfg.hidden_size * D, cfg.horizon * cfg.n_targets),
        )

    def forward(self, x: torch.Tensor) -> TSOutput:
        # x: (B, L, F)
        out, _ = self.rnn(x)
        last = out[:, -1, :]                          # (B, H_rnn)
        pred = self.head(last)                        # (B, horizon * n_targets)
        pred = pred.view(x.shape[0], self.cfg.horizon, self.cfg.n_targets)
        return TSOutput(y_hat=pred, extras={})


class LSTMForecaster(_RecurrentCore):
    """LSTM-based multi-step forecaster."""
    def __init__(self, cfg: Optional[RecurrentConfig] = None, **kw):
        super().__init__(cfg or RecurrentConfig(**kw), cell="lstm")


class GRUForecaster(_RecurrentCore):
    """GRU-based multi-step forecaster."""
    def __init__(self, cfg: Optional[RecurrentConfig] = None, **kw):
        super().__init__(cfg or RecurrentConfig(**kw), cell="gru")
