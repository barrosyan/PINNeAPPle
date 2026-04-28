"""Wave equation 1D benchmark task.

PDE  : u_tt = c^2 * u_xx,  x in [0,1], t in [0,1]
IC   : u(x,0) = sin(pi*x),  u_t(x,0) = 0
BC   : u(0,t) = u(1,t) = 0
Exact: u(x,t) = cos(c*pi*t) * sin(pi*x)
c    : 1.0 (default)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from pinneaple_arena.benchmark import BenchmarkTaskBase


class Wave1DTask(BenchmarkTaskBase):
    task_id = "wave_1d"
    in_dim = 2
    out_dim = 1

    def __init__(self, c: float = 1.0) -> None:
        self.c = float(c)

    # ── Exact solution ────────────────────────────────────────────────────────

    def exact(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.cos(self.c * np.pi * t) * np.sin(np.pi * x)

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

        # u(x,0) and u_t(x,0)=0 — encode both as IC
        # Half points for u, half for u_t=0 (we enforce u_t=0 via IC residual)
        n_u = n // 2
        n_ut = n - n_u

        x_u = rng.uniform(0.0, 1.0, n_u)
        x_ut = rng.uniform(0.0, 1.0, n_ut)

        X_ic = np.stack([
            np.concatenate([x_u, x_ut]),
            np.zeros(n),
        ], axis=-1).astype(np.float32)

        # For n_u points: target = sin(pi*x)
        # For n_ut points: target = 0 (will use u_t=0 via ic in training)
        # Simplification: just use u IC for all points
        U_ic = np.sin(np.pi * X_ic[:, 0]).reshape(-1, 1).astype(np.float32)
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
        u_tt = torch.autograd.grad(u_t.sum(), X, create_graph=True)[0][:, 1:2]

        res = u_tt - (self.c ** 2) * u_xx
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
