
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn

from pinneaple_models.base import BaseModel, ModelOutput


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 128, depth: int = 4, act: str = "tanh"):
        super().__init__()
        acts = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        a = acts.get(act, nn.Tanh())
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), a]
            d = width
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, num_f: int = 64, scale: float = 10.0):
        super().__init__()
        B = torch.randn((in_dim, num_f)) * float(scale)
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N,in)->(N,2*num_f)
        proj = 2.0 * torch.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class GenericMLP(BaseModel):
    def __init__(self, in_dim: int = 3, out_dim: int = 1, width: int = 128, depth: int = 4, act: str = "tanh"):
        super().__init__()
        self.net = _MLP(in_dim, out_dim, width=width, depth=depth, act=act)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        y = self.net(x)
        return ModelOutput(y=y, losses={}, extras={})


class GenericFourierMLP(BaseModel):
    def __init__(self, in_dim: int = 3, out_dim: int = 1, num_f: int = 64, scale: float = 10.0, width: int = 128, depth: int = 4):
        super().__init__()
        self.ff = FourierFeatures(in_dim, num_f=num_f, scale=scale)
        self.net = _MLP(2 * num_f, out_dim, width=width, depth=depth, act="gelu")

    def forward(self, x: torch.Tensor) -> ModelOutput:
        z = self.ff(x)
        y = self.net(z)
        return ModelOutput(y=y, losses={}, extras={})


class Sine(nn.Module):
    def __init__(self, w0: float = 30.0):
        super().__init__()
        self.w0 = float(w0)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class GenericSIREN(BaseModel):
    def __init__(self, in_dim: int = 3, out_dim: int = 1, width: int = 128, depth: int = 4, w0: float = 30.0):
        super().__init__()
        layers = []
        d = in_dim
        for i in range(depth):
            lin = nn.Linear(d, width)
            nn.init.uniform_(lin.weight, -1.0 / d, 1.0 / d)
            layers += [lin, Sine(w0)]
            d = width
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        y = self.net(x)
        return ModelOutput(y=y, losses={}, extras={})


class GenericResMLP(BaseModel):
    def __init__(self, in_dim: int = 3, out_dim: int = 1, width: int = 128, depth: int = 6):
        super().__init__()
        self.inp = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width)) for _ in range(depth)])
        self.out = nn.Linear(width, out_dim)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h = self.inp(x)
        for b in self.blocks:
            h = self.norm(h + b(h))
        y = self.out(h)
        return ModelOutput(y=y, losses={}, extras={})


class GenericLinear(BaseModel):
    def __init__(self, in_dim: int = 3, out_dim: int = 1):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        return ModelOutput(y=self.lin(x), losses={}, extras={})
