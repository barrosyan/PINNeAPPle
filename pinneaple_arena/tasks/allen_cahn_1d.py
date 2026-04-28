"""Allen-Cahn 1D benchmark task.

PDE  : u_t = eps^2 * u_xx + u - u^3,  x in [-1,1], t in [0,1]
IC   : u(x,0) = x^2 * cos(pi*x)
BC   : u(-1,t) = u(1,t) = -1
eps  : 0.01 (standard benchmark, thin interface)

Reference solution computed with method-of-lines + RK4 on fine grid.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from pinneaple_arena.benchmark import BenchmarkTaskBase


class AllenCahn1DTask(BenchmarkTaskBase):
    task_id = "allen_cahn_1d"
    in_dim = 2
    out_dim = 1

    def __init__(self, eps: float = 0.01, build_reference: bool = True) -> None:
        self.eps = float(eps)
        self._ref_x: np.ndarray = np.array([])
        self._ref_t: np.ndarray = np.array([])
        self._ref_u: np.ndarray = np.array([])
        if build_reference:
            self._build_reference()

    # ── Reference solution ────────────────────────────────────────────────────

    def _build_reference(self, N: int = 512, nt_store: int = 101, T: float = 1.0) -> None:
        x = np.linspace(-1.0, 1.0, N)
        dx = x[1] - x[0]
        u = (x ** 2 * np.cos(np.pi * x)).copy()
        u[0] = -1.0
        u[-1] = -1.0

        # Stable dt for diffusion: dt < 0.5 * dx^2 / eps^2
        dt = min(2e-4, 0.3 * dx ** 2 / max(self.eps ** 2, 1e-10))
        nt = int(T / dt) + 1
        store_every = max(1, nt // (nt_store - 1))

        solutions = [u.copy()]

        def rhs(u_: np.ndarray) -> np.ndarray:
            du = np.zeros_like(u_)
            u_xx = np.zeros_like(u_)
            u_xx[1:-1] = (u_[2:] - 2.0 * u_[1:-1] + u_[:-2]) / dx ** 2
            # Dirichlet BCs: u[0]=u[-1]=-1, so u_xx at boundaries is 0 (no flux update needed)
            du[1:-1] = self.eps ** 2 * u_xx[1:-1] + u_[1:-1] - u_[1:-1] ** 3
            return du

        t_current = 0.0
        for step in range(nt):
            k1 = rhs(u)
            k2 = rhs(u + 0.5 * dt * k1)
            k3 = rhs(u + 0.5 * dt * k2)
            k4 = rhs(u + dt * k3)
            u = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            u[0] = -1.0
            u[-1] = -1.0
            u = np.clip(u, -2.0, 2.0)  # numerical safety
            t_current += dt
            if (step + 1) % store_every == 0 and len(solutions) < nt_store:
                solutions.append(u.copy())

        while len(solutions) < nt_store:
            solutions.append(u.copy())

        t_store = np.linspace(0, T, nt_store)
        self._ref_x = x
        self._ref_t = t_store
        self._ref_u = np.stack(solutions[:nt_store], axis=0)  # (nt_store, N)

    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample_collocation(self, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1.0, 1.0, n)
        t = rng.uniform(0.0, 1.0, n)
        return np.stack([x, t], axis=-1).astype(np.float32)

    def sample_boundary(self, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        n_half = n // 2
        t_left = rng.uniform(0.0, 1.0, n_half)
        t_right = rng.uniform(0.0, 1.0, n - n_half)
        X_left = np.stack([-np.ones(n_half), t_left], axis=-1)
        X_right = np.stack([np.ones(n - n_half), t_right], axis=-1)
        X_bc = np.concatenate([X_left, X_right], axis=0).astype(np.float32)
        Y_bc = -np.ones((n, 1), dtype=np.float32)
        return X_bc, Y_bc

    def sample_ic(self, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1.0, 1.0, n)
        t = np.zeros(n)
        X_ic = np.stack([x, t], axis=-1).astype(np.float32)
        U_ic = (x ** 2 * np.cos(np.pi * x)).reshape(-1, 1).astype(np.float32)
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

        eps2 = torch.tensor(self.eps ** 2, device=X.device)
        res = u_t - eps2 * u_xx - u + u ** 3
        return (res ** 2).mean()

    # ── Evaluation ────────────────────────────────────────────────────────────

    def eval_grid(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._ref_u.size == 0:
            self._build_reference()
        nt, nx = self._ref_u.shape
        rng = np.random.default_rng(0)
        ti = rng.integers(0, nt, n)
        xi = rng.integers(0, nx, n)
        X_eval = np.stack([self._ref_x[xi], self._ref_t[ti]], axis=-1).astype(np.float32)
        U_ref = self._ref_u[ti, xi].reshape(-1, 1).astype(np.float32)
        return X_eval, U_ref
