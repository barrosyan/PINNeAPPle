"""Burgers 1D benchmark task.

PDE  : u_t + u*u_x = nu*u_xx,  x in [-1,1], t in [0,1]
IC   : u(x,0) = -sin(pi*x)
BC   : u(-1,t) = u(1,t) = 0
nu   : 0.01/pi  (standard Raissi benchmark)

Reference solution computed with pseudo-spectral RK4.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from pinneaple_arena.benchmark import BenchmarkTaskBase


class Burgers1DTask(BenchmarkTaskBase):
    task_id = "burgers_1d"
    in_dim = 2
    out_dim = 1

    def __init__(self, nu: float = 0.01 / np.pi, build_reference: bool = True) -> None:
        self.nu = float(nu)
        self._ref_x: np.ndarray = np.array([])
        self._ref_t: np.ndarray = np.array([])
        self._ref_u: np.ndarray = np.array([])
        if build_reference:
            self._build_reference()

    # ── Reference solution ────────────────────────────────────────────────────

    def _build_reference(self, N: int = 512, nt_store: int = 101, T: float = 1.0) -> None:
        """Compute reference via method-of-lines with RK4."""
        x = np.linspace(-1.0, 1.0, N)
        dx = x[1] - x[0]
        u = -np.sin(np.pi * x).copy()
        u[0] = 0.0
        u[-1] = 0.0

        t_store = np.linspace(0, T, nt_store)
        dt = 5e-4
        nt = int(T / dt) + 1

        solutions = [u.copy()]
        store_every = max(1, nt // (nt_store - 1))
        t_current = 0.0

        def rhs(u_: np.ndarray) -> np.ndarray:
            du = np.zeros_like(u_)
            u_x = (u_[2:] - u_[:-2]) / (2.0 * dx)
            u_xx = (u_[2:] - 2.0 * u_[1:-1] + u_[:-2]) / dx ** 2
            du[1:-1] = -u_[1:-1] * u_x + self.nu * u_xx
            return du

        for step in range(nt):
            k1 = rhs(u)
            k2 = rhs(u + 0.5 * dt * k1)
            k3 = rhs(u + 0.5 * dt * k2)
            k4 = rhs(u + dt * k3)
            u = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            u[0] = 0.0
            u[-1] = 0.0
            t_current += dt
            if (step + 1) % store_every == 0 and len(solutions) < nt_store:
                solutions.append(u.copy())

        while len(solutions) < nt_store:
            solutions.append(u.copy())

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
        Y_bc = np.zeros((n, 1), dtype=np.float32)
        return X_bc, Y_bc

    def sample_ic(self, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1.0, 1.0, n)
        t = np.zeros(n)
        X_ic = np.stack([x, t], axis=-1).astype(np.float32)
        u_ic = (-np.sin(np.pi * x)).reshape(-1, 1).astype(np.float32)
        return X_ic, u_ic

    # ── PDE residual ──────────────────────────────────────────────────────────

    def pde_residual(self, model: nn.Module, X: torch.Tensor) -> torch.Tensor:
        u = model(X)
        if hasattr(u, "y"):
            u = u.y

        grads = torch.autograd.grad(u.sum(), X, create_graph=True)[0]
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]
        u_xx = torch.autograd.grad(u_x.sum(), X, create_graph=True)[0][:, 0:1]

        res = u_t + u * u_x - self.nu * u_xx
        return (res ** 2).mean()

    # ── Evaluation ────────────────────────────────────────────────────────────

    def eval_grid(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample (X, U_ref) from FDM reference grid."""
        if self._ref_u.size == 0:
            self._build_reference()
        nt, nx = self._ref_u.shape
        rng = np.random.default_rng(0)
        ti = rng.integers(0, nt, n)
        xi = rng.integers(0, nx, n)
        X_eval = np.stack([self._ref_x[xi], self._ref_t[ti]], axis=-1).astype(np.float32)
        U_ref = self._ref_u[ti, xi].reshape(-1, 1).astype(np.float32)
        return X_eval, U_ref

    def evaluate(self, model: nn.Module, device: torch.device) -> Dict[str, float]:
        metrics = super().evaluate(model, device)

        # Also compute bc_error explicitly
        model.eval()
        with torch.no_grad():
            t_test = torch.linspace(0, 1, 200, device=device).unsqueeze(-1)
            x_left = torch.full_like(t_test, -1.0)
            x_right = torch.full_like(t_test, 1.0)
            X_left = torch.cat([x_left, t_test], dim=-1)
            X_right = torch.cat([x_right, t_test], dim=-1)
            bc_err = float(
                (model(X_left).pow(2).mean() + model(X_right).pow(2).mean()).item() / 2
            )
        metrics["bc_error"] = bc_err
        return metrics
