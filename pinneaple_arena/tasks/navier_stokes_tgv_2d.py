"""Navier-Stokes 2D Taylor-Green Vortex benchmark task.

PDE  : u_t + u*u_x + v*u_y = -p_x + nu*(u_xx + u_yy)
       v_t + u*v_x + v*v_y = -p_y + nu*(v_xx + v_yy)
       u_x + v_y = 0  (continuity)

Domain: (x,y) in [0, 2*pi]^2,  t in [0, 1]
Fields: (u, v, p)  (3 outputs)

Exact Taylor-Green Vortex solution:
  u(x,y,t) = -cos(x)*sin(y)*exp(-2*nu*t)
  v(x,y,t) =  sin(x)*cos(y)*exp(-2*nu*t)
  p(x,y,t) = -1/4*(cos(2*x) + cos(2*y))*exp(-4*nu*t)

nu   : 0.01 (default)
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from pinneaple_arena.benchmark import BenchmarkTaskBase


class NavierStokesTGV2DTask(BenchmarkTaskBase):
    task_id = "ns_tgv_2d"
    in_dim = 3   # (x, y, t)
    out_dim = 3  # (u, v, p)

    def __init__(self, nu: float = 0.01) -> None:
        self.nu = float(nu)
        self._L = 2.0 * np.pi

    # ── Exact solution ────────────────────────────────────────────────────────

    def exact_uvp(
        self, x: np.ndarray, y: np.ndarray, t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        decay = np.exp(-2.0 * self.nu * t)
        decay2 = np.exp(-4.0 * self.nu * t)
        u = -np.cos(x) * np.sin(y) * decay
        v = np.sin(x) * np.cos(y) * decay
        p = -0.25 * (np.cos(2 * x) + np.cos(2 * y)) * decay2
        return u, v, p

    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample_collocation(self, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.0, self._L, n)
        y = rng.uniform(0.0, self._L, n)
        t = rng.uniform(0.0, 1.0, n)
        return np.stack([x, y, t], axis=-1).astype(np.float32)

    def sample_boundary(self, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """Periodic BC implemented via exact solution values on domain edges."""
        rng = np.random.default_rng(seed)
        n_side = n // 4
        r = n - 3 * n_side

        pts = []
        # x=0 and x=2pi (periodic, use exact values)
        for x_val, n_pts in [(0.0, n_side), (self._L, n_side)]:
            y = rng.uniform(0.0, self._L, n_pts)
            t = rng.uniform(0.0, 1.0, n_pts)
            pts.append(np.stack([np.full(n_pts, x_val), y, t], axis=-1))
        for y_val, n_pts in [(0.0, n_side), (self._L, r)]:
            x = rng.uniform(0.0, self._L, n_pts)
            t = rng.uniform(0.0, 1.0, n_pts)
            pts.append(np.stack([x, np.full(n_pts, y_val), t], axis=-1))

        X_bc = np.concatenate(pts, axis=0).astype(np.float32)
        u, v, p = self.exact_uvp(X_bc[:, 0], X_bc[:, 1], X_bc[:, 2])
        Y_bc = np.stack([u, v, p], axis=-1).astype(np.float32)
        return X_bc, Y_bc

    def sample_ic(self, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.0, self._L, n)
        y = rng.uniform(0.0, self._L, n)
        t = np.zeros(n)
        X_ic = np.stack([x, y, t], axis=-1).astype(np.float32)
        u, v, p = self.exact_uvp(x, y, t)
        Y_ic = np.stack([u, v, p], axis=-1).astype(np.float32)
        return X_ic, Y_ic

    # ── PDE residual ──────────────────────────────────────────────────────────

    def pde_residual(self, model: nn.Module, X: torch.Tensor) -> torch.Tensor:
        out = model(X)
        if hasattr(out, "y"):
            out = out.y

        u_vel = out[:, 0:1]
        v_vel = out[:, 1:2]
        p_fld = out[:, 2:3]

        # First-order derivatives
        grad_u = torch.autograd.grad(u_vel.sum(), X, create_graph=True)[0]
        u_x, u_y, u_t = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]

        grad_v = torch.autograd.grad(v_vel.sum(), X, create_graph=True)[0]
        v_x, v_y, v_t = grad_v[:, 0:1], grad_v[:, 1:2], grad_v[:, 2:3]

        grad_p = torch.autograd.grad(p_fld.sum(), X, create_graph=True)[0]
        p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]

        # Second-order derivatives (Laplacian components)
        u_xx = torch.autograd.grad(u_x.sum(), X, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y.sum(), X, create_graph=True)[0][:, 1:2]
        v_xx = torch.autograd.grad(v_x.sum(), X, create_graph=True)[0][:, 0:1]
        v_yy = torch.autograd.grad(v_y.sum(), X, create_graph=True)[0][:, 1:2]

        nu = torch.tensor(self.nu, device=X.device)

        # Momentum equations
        mom_u = u_t + u_vel * u_x + v_vel * u_y + p_x - nu * (u_xx + u_yy)
        mom_v = v_t + u_vel * v_x + v_vel * v_y + p_y - nu * (v_xx + v_yy)
        # Continuity
        cont = u_x + v_y

        return (mom_u ** 2 + mom_v ** 2 + cont ** 2).mean()

    # ── Evaluation ────────────────────────────────────────────────────────────

    def eval_grid(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        side = int((n / 5) ** 0.5)  # fewer points since 3D input, 3 outputs
        x = np.linspace(0, self._L, side)
        y = np.linspace(0, self._L, side)
        t_vals = np.linspace(0, 1, 5)
        pts = []
        for t_val in t_vals:
            xx, yy = np.meshgrid(x, y)
            tt = np.full_like(xx, t_val)
            pts.append(np.stack([xx.ravel(), yy.ravel(), tt.ravel()], axis=-1))
        X_eval = np.concatenate(pts, axis=0).astype(np.float32)
        u_ex, v_ex, p_ex = self.exact_uvp(X_eval[:, 0], X_eval[:, 1], X_eval[:, 2])
        U_exact = np.stack([u_ex, v_ex, p_ex], axis=-1).astype(np.float32)
        return X_eval, U_exact

    def evaluate(self, model: nn.Module, device: torch.device) -> Dict[str, float]:
        X_eval, U_exact = self.eval_grid(5000)
        X_t = torch.tensor(X_eval, dtype=torch.float32, device=device)
        U_ref = torch.tensor(U_exact, dtype=torch.float32, device=device)

        model.eval()
        with torch.no_grad():
            U_pred = model(X_t)
            if hasattr(U_pred, "y"):
                U_pred = U_pred.y

        diff = U_pred - U_ref
        rel_l2 = float((diff.pow(2).sum() / (U_ref.pow(2).sum() + 1e-10)).sqrt().item())
        l_inf = float(diff.abs().max().item())
        mse = float(diff.pow(2).mean().item())

        # Per-field metrics
        u_err = float((diff[:, 0].pow(2).mean()).item() ** 0.5)
        v_err = float((diff[:, 1].pow(2).mean()).item() ** 0.5)
        p_err = float((diff[:, 2].pow(2).mean()).item() ** 0.5)

        # PDE residual
        X_req = X_t[:500].detach().requires_grad_(True)
        model.train()
        try:
            pde_res = float(self.pde_residual(model, X_req).item())
        except Exception:
            pde_res = float("nan")
        model.eval()

        return {
            "rel_l2": rel_l2,
            "l_inf": l_inf,
            "mse": mse,
            "pde_residual": pde_res,
            "u_rmse": u_err,
            "v_rmse": v_err,
            "p_rmse": p_err,
        }
