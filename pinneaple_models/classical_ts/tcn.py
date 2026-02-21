from __future__ import annotations
"""Temporal convolutional network for time series."""
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ClassicalTSBase, ClassicalTSOutput


class _Chomp1d(nn.Module):
    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = int(chomp)

    def forward(self, x):
        if self.chomp == 0:
            return x
        return x[:, :, :-self.chomp]


class _TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel - 1) * dilation  # causal padding
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, dilation=dilation, padding=padding)
        self.chomp1 = _Chomp1d(padding)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel, dilation=dilation, padding=padding)
        self.chomp2 = _Chomp1d(padding)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.out_act = nn.GELU()

    def forward(self, x):
        y = self.drop1(self.act1(self.chomp1(self.conv1(x))))
        y = self.drop2(self.act2(self.chomp2(self.conv2(y))))
        res = x if self.down is None else self.down(x)
        return self.out_act(y + res)


class TCN(ClassicalTSBase):
    """
    TCN for sequence-to-sequence regression.

    Input:
      x: (B,T,in_dim)
    Output:
      y: (B,T,out_dim)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        channels: List[int] = (64, 64, 64),
        kernel: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        layers = []
        c_prev = self.in_dim
        for i, c in enumerate(list(channels)):
            layers.append(_TCNBlock(c_prev, int(c), kernel=int(kernel), dilation=2**i, dropout=float(dropout)))
            c_prev = int(c)
        self.net = nn.Sequential(*layers)
        self.head = nn.Conv1d(c_prev, self.out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, *, y_true: Optional[torch.Tensor] = None, return_loss: bool = False) -> ClassicalTSOutput:
        # x: (B,T,D) -> (B,D,T)
        xt = x.transpose(1, 2)
        h = self.net(xt)
        y = self.head(h).transpose(1, 2)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=x.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]

        return ClassicalTSOutput(y=y, losses=losses, extras={})
