"""Temporal Convolutional Network (TCN) forecaster."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..base import TSModelBase, TSOutput


@dataclass
class TCNConfig:
    input_len:    int   = 64
    horizon:      int   = 16
    n_features:   int   = 1
    n_targets:    int   = 1
    n_channels:   int   = 64
    n_layers:     int   = 6
    kernel_size:  int   = 3
    dropout:      float = 0.1


class _TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(out_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) — causal: chop extra padding
        L = x.shape[-1]
        out = self.conv1(x)[..., :L]
        out = self.drop(self.relu(self.norm1(out.transpose(1, 2)).transpose(1, 2)))
        out = self.conv2(out)[..., :L]
        out = self.drop(self.relu(self.norm2(out.transpose(1, 2)).transpose(1, 2)))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNForecaster(TSModelBase):
    """
    TCN with exponentially increasing dilations (1,2,4,8,...).
    Input  : (B, L, F)
    Output : TSOutput with y_hat (B, H, n_targets)
    """

    def __init__(self, cfg: Optional[TCNConfig] = None, **kw):
        super().__init__()
        self.cfg = cfg or TCNConfig(**kw)
        c = self.cfg
        layers = []
        in_ch = c.n_features
        for i in range(c.n_layers):
            dil = 2 ** i
            layers.append(_TCNBlock(in_ch, c.n_channels, c.kernel_size, dil, c.dropout))
            in_ch = c.n_channels
        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(c.n_channels, c.horizon * c.n_targets)

    def forward(self, x: torch.Tensor) -> TSOutput:
        # x: (B, L, F) → (B, F, L) for Conv1d
        h = self.network(x.transpose(1, 2))   # (B, C, L)
        last = h[:, :, -1]                     # (B, C)
        pred = self.head(last)                 # (B, H*T)
        pred = pred.view(x.shape[0], self.cfg.horizon, self.cfg.n_targets)
        return TSOutput(y_hat=pred, extras={})
