"""Poisson 2D benchmark task.

PDE  : -u_xx - u_yy = f(x,y),  (x,y) in [0,1]^2
f    : 2*pi^2 * sin(pi*x) * sin(pi*y)
BC   : u = 0 on all boundaries
Exact: u(x,y) = sin(pi*x) * sin(pi*y)
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from pinneaple_arena.benchmark import BenchmarkTaskBase


class Poisson2DTask(BenchmarkTaskBase):
    task_id = "poisson_2d"
    in_dim = 2
    out_dim = 1

    _PI = np.pi

    # ── Exact solution ────────────────────────────────────────────────────────

    @staticmethod
    def exact(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sin(Poisson2DTask._PI * x) * np.sin(Poisson2DTask._PI * y)

    @staticmethod
    def forcing(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 2.0 * Poisson2DTask._PI ** 2 * np.sin(Poisson2DTask._PI * x) * np.sin(Poisson2DTask._PI * y)

    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample_collocation(self, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        pts = rng.uniform(0.0, 1.0, (n, 2))
        return pts.astype(np.float32)

    def sample_boundary(self, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        n_side = n // 4
        r = n - 3 * n_side  # last side gets remainder

        sides = []
        # x=0
        y = rng.uniform(0, 1, n_side)
        sides.append(np.stack([np.zeros(n_side), y], -1))
        # x=1
        y = rng.uniform(0, 1, n_side)
        sides.append(np.stack([np.ones(n_side), y], -1))
        # y=0
        x = rng.uniform(0, 1, n_side)
        sides.append(np.stack([x, np.zeros(n_side)], -1))
        # y=1
        x = rng.uniform(0, 1, r)
        sides.append(np.stack([x, np.ones(r)], -1))

        X_bc = np.concatenate(sides, axis=0).astype(np.float32)
        Y_bc = np.zeros((X_bc.shape[0], 1), dtype=np.float32)
        return X_bc, Y_bc

    def sample_ic(self, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        return np.empty((0, self.in_dim), np.float32), np.empty((0, self.out_dim), np.float32)

    # ── PDE residual ──────────────────────────────────────────────────────────

    def pde_residual(self, model: nn.Module, X: torch.Tensor) -> torch.Tensor:
        u = model(X)
        if hasattr(u, "y"):
            u = u.y

        u_x = torch.autograd.grad(u.sum(), X, create_graph=True)[0][:, 0:1]
        u_y_all = torch.autograd.grad(u.sum(), X, create_graph=True)[0][:, 1:2]

        u_xx = torch.autograd.grad(u_x.sum(), X, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y_all.sum(), X, create_graph=True)[0][:, 1:2]

        pi = torch.tensor(float(self._PI), device=X.device)
        f = 2.0 * pi ** 2 * torch.sin(pi * X[:, 0:1]) * torch.sin(pi * X[:, 1:2])

        res = -u_xx - u_yy - f
        return (res ** 2).mean()

    # ── Evaluation ────────────────────────────────────────────────────────────

    def eval_grid(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        side = int(n ** 0.5)
        x = np.linspace(0, 1, side)
        y = np.linspace(0, 1, side)
        xx, yy = np.meshgrid(x, y)
        X_eval = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
        U_exact = self.exact(X_eval[:, 0], X_eval[:, 1]).reshape(-1, 1).astype(np.float32)
        return X_eval, U_exact
