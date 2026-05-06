"""DoMINO + Time-Marching demo on a 2D wave equation.

Problem
-------
1D wave equation (spatial x, temporal t)::

    u_tt - c^2 u_xx = 0,  (t, x) in [0, 1] x [0, 1]
    u(0, x) = sin(pi x)          (initial displacement)
    u_t(0, x) = 0                (zero initial velocity)
    u(t, 0) = u(t, 1) = 0       (fixed ends)

Exact solution::

    u(t, x) = sin(pi x) cos(pi c t),  with c = 1.0

We demonstrate two solvers in sequence:

1. **DoMINO** — solve the *full* space-time domain [0,1]x[0,1] with a 2x2
   domain decomposition (4 subdomains) plus interface continuity constraints.

2. **TimeMarchingTrainer** — solve the same wave equation by marching through
   10 equal time windows, using each window's trained model as the IC for the
   next.

Run::

    python examples/pinneaple_pinn/06_domino_time_marching_demo.py
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from pinneaple_pinn import DoMINO
from pinneaple_train import TimeMarchingTrainer


# ---------------------------------------------------------------------------
# Problem constants
# ---------------------------------------------------------------------------

C_WAVE = 1.0          # wave speed
PI = math.pi


# ---------------------------------------------------------------------------
# Shared network architecture
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Simple fully-connected network with Tanh activations."""

    def __init__(self, in_dim: int = 2, out_dim: int = 1,
                 width: int = 64, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Analytic solution & helper functions
# ---------------------------------------------------------------------------

def u_exact(X: torch.Tensor) -> torch.Tensor:
    """u(t, x) = sin(pi x) * cos(pi c t).  X shape [N, 2]: cols = [t, x]."""
    t, x = X[:, 0:1], X[:, 1:2]
    return torch.sin(PI * x) * torch.cos(PI * C_WAVE * t)


def u_exact_np(X: np.ndarray) -> np.ndarray:
    t, x = X[:, 0], X[:, 1]
    return (np.sin(PI * x) * np.cos(PI * C_WAVE * t))[:, None].astype(np.float32)


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def sample_collocation(n: int, seed: int = 0,
                       t_lo: float = 0.0, t_hi: float = 1.0) -> torch.Tensor:
    r = _rng(seed)
    t = r.uniform(t_lo, t_hi, (n, 1)).astype(np.float32)
    x = r.uniform(0.0, 1.0, (n, 1)).astype(np.float32)
    return torch.from_numpy(np.hstack([t, x]))


def sample_boundary(n: int, seed: int = 2,
                    t_lo: float = 0.0, t_hi: float = 1.0) -> torch.Tensor:
    r = _rng(seed)
    t = r.uniform(t_lo, t_hi, (n, 1)).astype(np.float32)
    side = r.integers(0, 2, (n, 1))
    x = np.where(side == 0, 0.0, 1.0).astype(np.float32)
    return torch.from_numpy(np.hstack([t, x]))


def sample_ic(n: int, seed: int = 1) -> torch.Tensor:
    r = _rng(seed)
    t = np.zeros((n, 1), dtype=np.float32)
    x = r.uniform(0.0, 1.0, (n, 1)).astype(np.float32)
    return torch.from_numpy(np.hstack([t, x]))


# ---------------------------------------------------------------------------
# PDE residual (wave equation)
# ---------------------------------------------------------------------------

def wave_residual(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return u_tt - c^2 u_xx at collocation points x = [t, x_sp].

    x must have requires_grad=True.
    """
    u = model(x)
    ones = torch.ones_like(u)

    # First-order derivatives
    grads1 = torch.autograd.grad(u, x, ones, create_graph=True)[0]
    u_t = grads1[:, 0:1]
    u_x = grads1[:, 1:2]

    # Second-order derivatives
    u_tt = torch.autograd.grad(u_t, x, ones, create_graph=True)[0][:, 0:1]
    u_xx = torch.autograd.grad(u_x, x, ones, create_graph=True)[0][:, 1:2]

    return u_tt - (C_WAVE ** 2) * u_xx


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def relative_l2(pred: np.ndarray, true: np.ndarray) -> float:
    return float(
        np.linalg.norm(pred.flatten() - true.flatten())
        / (np.linalg.norm(true.flatten()) + 1e-12)
    )


# ===========================================================================
# Part 1: DoMINO — domain decomposition across space × time
# ===========================================================================

def run_domino(device: torch.device, dtype: torch.dtype) -> None:
    print("=" * 60)
    print("Part 1: DoMINO — 2×2 domain decomposition of (t,x) ∈ [0,1]²")
    print("=" * 60)

    torch.manual_seed(42)

    # Domain: (t in [0,1], x in [0,1]) — note that DoMINO bounds are
    # listed as [(t_lo, t_hi), (x_lo, x_hi)].
    subdomains = DoMINO.partition(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        n_splits=(2, 2),
        overlap=0.15,
    )
    print(f"  Created {len(subdomains)} subdomains:")
    for i, sd in enumerate(subdomains):
        print(f"    [{i}] {sd}")

    domino = DoMINO(
        subdomains=subdomains,
        model_factory=lambda: MLP(2, 1, 64, 4),
        interface_weight=10.0,
    ).to(device=device, dtype=dtype)

    # Collocation & boundary data
    x_col = sample_collocation(8000, seed=0).to(device=device, dtype=dtype)
    x_bc = sample_boundary(1024, seed=2).to(device=device, dtype=dtype)
    x_ic_pts = sample_ic(1024, seed=1).to(device=device, dtype=dtype)

    # BC function: homogeneous Dirichlet at x=0 and x=1
    def bc_fn(x_b: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x_b.shape[0], 1, device=device, dtype=dtype)

    # IC target (used only for evaluation; DoMINO trains IC via PDE + BC)
    x_all_bc = torch.cat([x_bc, x_ic_pts], dim=0)

    def residual_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return wave_residual(model, x)

    # Extended BC enforcer that also covers the IC
    def extended_bc_fn(x_b: torch.Tensor) -> torch.Tensor:
        """Zero BC everywhere + sin(pi x) IC at t = 0."""
        out = torch.zeros(x_b.shape[0], 1, device=device, dtype=dtype)
        # IC: t column is 0
        ic_mask = x_b[:, 0].abs() < 1e-4
        if ic_mask.any():
            x_ic_here = x_b[ic_mask, 1:2]
            out[ic_mask] = torch.sin(PI * x_ic_here)
        return out

    history = domino.train_domino(
        residual_fn=residual_fn,
        bc_fn=extended_bc_fn,
        x_col=x_col,
        x_bc=x_all_bc,
        n_epochs=3000,
        lr=1e-3,
        print_every=500,
    )

    # Evaluate on regular grid
    ts = torch.linspace(0, 1, 64, device=device, dtype=dtype)
    xs = torch.linspace(0, 1, 64, device=device, dtype=dtype)
    TT, XX = torch.meshgrid(ts, xs, indexing="ij")
    Xg = torch.stack([TT.flatten(), XX.flatten()], dim=1)

    with torch.no_grad():
        u_pred = domino(Xg).cpu().numpy()

    u_true = u_exact_np(Xg.cpu().numpy())
    rl2 = relative_l2(u_pred, u_true)
    print(f"\n  DoMINO relative L2 error on 64×64 grid: {rl2:.3e}")
    print(f"  Final total loss: {history['total'][-1]:.4e}")
    print(f"  Final interface loss: {history['interface'][-1]:.4e}")


# ===========================================================================
# Part 2: Time-marching — sequential training through 5 time windows
# ===========================================================================

def run_time_marching(device: torch.device, dtype: torch.dtype) -> None:
    print("\n" + "=" * 60)
    print("Part 2: Time-marching — 5 windows over t ∈ [0, 1]")
    print("=" * 60)

    torch.manual_seed(0)

    # -------------------------------------------------------------------
    # PDE residual for TimeMarchingTrainer
    #   Input x has shape [N, 2]: columns = [x_spatial, t]
    #   (time is the LAST column for TimeMarchingTrainer convention)
    # -------------------------------------------------------------------

    def tm_residual(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Wave residual when input is [x_sp, t] (time last)."""
        u = model(x)
        ones = torch.ones_like(u)
        g1 = torch.autograd.grad(u, x, ones, create_graph=True)[0]
        u_x = g1[:, 0:1]
        u_t = g1[:, 1:2]
        u_xx = torch.autograd.grad(u_x, x, ones, create_graph=True)[0][:, 0:1]
        u_tt = torch.autograd.grad(u_t, x, ones, create_graph=True)[0][:, 1:2]
        return u_tt - (C_WAVE ** 2) * u_xx

    # -------------------------------------------------------------------
    # Initial condition: u(x, 0) = sin(pi x)
    # Input x: [N, 2] = [x_sp, t=0]
    # -------------------------------------------------------------------

    def tm_ic_fn(x: torch.Tensor) -> torch.Tensor:
        x_sp = x[:, 0:1]
        return torch.sin(PI * x_sp)

    # -------------------------------------------------------------------
    # Boundary conditions factory for TimeMarchingTrainer:
    #   bc_fns = [callable(t_lo, t_hi, device, dtype) -> (x_bc, u_bc)]
    # -------------------------------------------------------------------

    def make_bc_fn(n_bc: int = 256, seed: int = 5):
        def bc_fn(t_lo: float, t_hi: float,
                  dev: torch.device, dt: torch.dtype):
            r = _rng(seed)
            t = r.uniform(t_lo, t_hi, (n_bc, 1)).astype(np.float32)
            side = r.integers(0, 2, (n_bc, 1))
            x_sp = np.where(side == 0, 0.0, 1.0).astype(np.float32)
            # Input format for TimeMarchingTrainer: [x_sp, t]
            x_bc = torch.from_numpy(np.hstack([x_sp, t])).to(device=dev, dtype=dt)
            u_bc = torch.zeros(n_bc, 1, device=dev, dtype=dt)
            return x_bc, u_bc
        return bc_fn

    # -------------------------------------------------------------------
    # Spatial grid for collocation (x in [0,1], 1D)
    # -------------------------------------------------------------------
    x_domain = torch.linspace(0.0, 1.0, 500, device=device, dtype=dtype).unsqueeze(1)

    trainer = TimeMarchingTrainer(
        model_factory=lambda: MLP(2, 1, 64, 4),
        t_start=0.0,
        t_end=1.0,
        n_windows=5,
        epochs_per_window=1500,
        n_col=1500,
        ic_weight=20.0,
    )

    models = trainer.march(
        pde_residual_fn=tm_residual,
        ic_fn=tm_ic_fn,
        bc_fns=[make_bc_fn()],
        x_domain=x_domain,
        device=device,
        dtype=dtype,
    )

    # -------------------------------------------------------------------
    # Evaluate: build [x, t] grid and query trainer.evaluate
    # -------------------------------------------------------------------
    n_eval = 64
    xs_eval = torch.linspace(0.0, 1.0, n_eval, device=device, dtype=dtype)
    ts_eval = torch.linspace(0.0, 1.0, n_eval, device=device, dtype=dtype)
    XX_e, TT_e = torch.meshgrid(xs_eval, ts_eval, indexing="ij")

    x_flat = XX_e.flatten().unsqueeze(1)  # [N, 1]
    t_flat = TT_e.flatten()               # [N]

    u_pred = trainer.evaluate(x_flat, t_flat).cpu().numpy()

    # Exact solution on same grid (format: [x_sp, t] → [t, x_sp] for u_exact_np)
    xt_eval = torch.stack([t_flat, x_flat.flatten()], dim=1).cpu().numpy()
    u_true = u_exact_np(xt_eval)

    rl2 = relative_l2(u_pred, u_true)
    print(f"\n  Time-marching relative L2 error on {n_eval}×{n_eval} grid: {rl2:.3e}")

    # Print per-window info
    for i, (t_lo, t_hi) in enumerate(trainer.window_edges):
        window_mask = (ts_eval >= t_lo) & (
            ts_eval <= t_hi if i == len(trainer.window_edges) - 1
            else ts_eval < t_hi
        )
        if window_mask.any():
            idx = window_mask.nonzero().flatten()
            t_idxs = idx.tolist()
            # Gather the spatial predictions for this window
            u_w = u_pred.reshape(n_eval, n_eval)[:, t_idxs]
            u_t_w = u_true.reshape(n_eval, n_eval)[:, t_idxs]
            w_rl2 = relative_l2(u_w, u_t_w)
            print(f"    Window {i + 1}: t=[{t_lo:.2f},{t_hi:.2f}]  rel-L2={w_rl2:.3e}")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"Device: {device}")

    run_domino(device, dtype)
    run_time_marching(device, dtype)

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
