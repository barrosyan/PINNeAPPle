"""Heat equation 1D benchmark task.

PDE  : u_t = alpha * u_xx,  x in [0,1], t in [0,1]
IC   : u(x,0) = sin(pi*x)
BC   : u(0,t) = u(1,t) = 0
Exact: u(x,t) = exp(-pi^2 * alpha * t) * sin(pi*x)
alpha: 0.01 (default)
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from pinneaple_arena.benchmark import BenchmarkTaskBase


class Heat1DTask(BenchmarkTaskBase):
    task_id = "heat_1d"
    in_dim = 2
    out_dim = 1

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = float(alpha)

    # ── Exact solution ────────────────────────────────────────────────────────

    def exact(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.exp(-np.pi ** 2 * self.alpha * t) * np.sin(np.pi * x)

    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample_collocation(self, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.0, 1.0, n)
        t = rng.uniform(0.0, 1.0, n)
        return np.stack([x, t], axis=-1).astype(np.float32)

    def sample_boundary(self, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        n_half = n // 2
        t0 = rng.uniform(0.0, 1.0, n_half)
        t1 = rng.uniform(0.0, 1.0, n - n_half)
        X_left = np.stack([np.zeros(n_half), t0], axis=-1)
        X_right = np.stack([np.ones(n - n_half), t1], axis=-1)
        X_bc = np.concatenate([X_left, X_right], axis=0).astype(np.float32)
        Y_bc = np.zeros((n, 1), dtype=np.float32)
        return X_bc, Y_bc

    def sample_ic(self, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.0, 1.0, n)
        t = np.zeros(n)
        X_ic = np.stack([x, t], axis=-1).astype(np.float32)
        U_ic = np.sin(np.pi * x).reshape(-1, 1).astype(np.float32)
        return X_ic, U_ic

    # ── PDE residual ──────────────────────────────────────────────────────────

    def pde_residual(self, model: nn.Module, X: torch.Tensor) -> torch.Tensor:
        u = model(X)
        if hasattr(u, "y"):
            u = u.y

        grads = torch.autograd.grad(u.sum(), X, create_graph=True)[0]
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]
        u_xx = torch.autograd.grad(u_x.sum(), X, create_graph=True)[0][:, 0:1]

        res = u_t - self.alpha * u_xx
        return (res ** 2).mean()

    # ── Evaluation ────────────────────────────────────────────────────────────

    def eval_grid(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        side = int(n ** 0.5)
        x = np.linspace(0, 1, side)
        t = np.linspace(0, 1, side)
        xx, tt = np.meshgrid(x, t)
        X_eval = np.stack([xx.ravel(), tt.ravel()], axis=-1).astype(np.float32)
        U_exact = self.exact(X_eval[:, 0], X_eval[:, 1]).reshape(-1, 1).astype(np.float32)
        return X_eval, U_exact
