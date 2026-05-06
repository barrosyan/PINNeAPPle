"""N-BEATS: Neural Basis Expansion Analysis for Time Series (Oreshkin et al., 2019)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import numpy as np

from ..base import TSModelBase, TSOutput


@dataclass
class NBeatsConfig:
    input_len:      int   = 64
    horizon:        int   = 16
    n_features:     int   = 1
    stack_types:    List[str] = field(default_factory=lambda: ["trend", "seasonality", "generic"])
    n_blocks:       int   = 3
    n_layers:       int   = 4
    layer_width:    int   = 256
    degree_of_polynomial: int = 3
    n_harmonics:    int   = 1


class _NBeatsBlock(nn.Module):
    def __init__(
        self,
        input_len: int,
        horizon: int,
        n_features: int,
        block_type: str,
        n_layers: int,
        layer_width: int,
        degree: int,
        n_harmonics: int,
    ):
        super().__init__()
        self.block_type = block_type
        self.input_len = input_len
        self.horizon = horizon

        in_dim = input_len * n_features
        self.fc_stack = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim if i == 0 else layer_width, layer_width), nn.ReLU())
            for i in range(n_layers)
        ])

        if block_type == "trend":
            self.theta_b_dim = degree + 1
            self.theta_f_dim = degree + 1
        elif block_type == "seasonality":
            self.theta_b_dim = 2 * n_harmonics + 1
            self.theta_f_dim = 2 * n_harmonics + 1
        else:
            self.theta_b_dim = input_len
            self.theta_f_dim = horizon

        self.theta_b = nn.Linear(layer_width, self.theta_b_dim, bias=False)
        self.theta_f = nn.Linear(layer_width, self.theta_f_dim, bias=False)

        # Basis matrices
        if block_type == "trend":
            t_b = torch.arange(input_len).float() / input_len
            t_f = torch.arange(horizon).float() / horizon
            self.register_buffer("T_b", torch.stack([t_b ** d for d in range(degree + 1)], dim=0))
            self.register_buffer("T_f", torch.stack([t_f ** d for d in range(degree + 1)], dim=0))
        elif block_type == "seasonality":
            t_b = torch.arange(input_len).float() / input_len
            t_f = torch.arange(horizon).float() / horizon
            H = n_harmonics
            basis_b = [torch.ones(input_len)]
            basis_f = [torch.ones(horizon)]
            for h in range(1, H + 1):
                basis_b += [torch.cos(2 * np.pi * h * t_b), torch.sin(2 * np.pi * h * t_b)]
                basis_f += [torch.cos(2 * np.pi * h * t_f), torch.sin(2 * np.pi * h * t_f)]
            self.register_buffer("T_b", torch.stack(basis_b, dim=0))
            self.register_buffer("T_f", torch.stack(basis_f, dim=0))
        else:
            self.register_buffer("T_b", torch.eye(input_len))
            self.register_buffer("T_f", torch.eye(horizon))

    def forward(self, x: torch.Tensor) -> tuple:
        # x: (B, L*F)
        h = x
        for layer in self.fc_stack:
            h = layer(h)
        theta_b = self.theta_b(h)   # (B, theta_b_dim)
        theta_f = self.theta_f(h)   # (B, theta_f_dim)
        backcast = theta_b @ self.T_b   # (B, L)
        forecast = theta_f @ self.T_f   # (B, H)
        return backcast, forecast


class NBeats(TSModelBase):
    """
    N-BEATS with trend + seasonality + generic stacks.
    Input  : (B, L, F)
    Output : TSOutput with y_hat (B, H, 1)
    """

    def __init__(self, cfg: Optional[NBeatsConfig] = None, **kw):
        super().__init__()
        self.cfg = cfg or NBeatsConfig(**kw)
        c = self.cfg
        self.blocks = nn.ModuleList()
        for stype in c.stack_types:
            for _ in range(c.n_blocks):
                self.blocks.append(_NBeatsBlock(
                    c.input_len, c.horizon, c.n_features,
                    stype, c.n_layers, c.layer_width,
                    c.degree_of_polynomial, c.n_harmonics,
                ))

    def forward(self, x: torch.Tensor) -> TSOutput:
        # x: (B, L, F) — use only first feature for univariate path
        B, L, F = x.shape
        residual = x[:, :, 0].reshape(B, -1)   # (B, L)
        forecast = torch.zeros(B, self.cfg.horizon, device=x.device)

        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        return TSOutput(y_hat=forecast.unsqueeze(-1), extras={"residual": residual})
