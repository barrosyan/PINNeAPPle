"""
01_pde_comparison.py — PINNeAPPle PDE Architecture Benchmark

Three canonical PDEs, three architectures — same setup, same conditions.
Automatic comparison pipeline: train → evaluate → rank → plot.

PDEs
----
  (1) Burgers 1D+t   — nonlinear advection-diffusion, shock formation
  (2) Heat 2D+t      — pure diffusion, smooth solution (analytical reference)
  (3) Kovasznay NS   — 2D steady incompressible flow (analytical reference)

Architectures
-------------
  (A) VanillaPINN  — fully connected MLP + tanh (classical baseline)
  (B) SIREN        — sinusoidal activations (Sitzmann et al. 2020)
  (C) ModifiedMLP  — random Fourier features + highway gating

Workflow
--------
  Understand → Compare → Validate → then commit to production training.

  Instead of guessing which architecture to use, run this script once:
  a summary table and comparison plots let you see what actually works
  before spending hours on full-scale training.

Usage
-----
  FAST_MODE = True   ~5-10 min, indicative results
  FAST_MODE = False  ~30-60 min, production-quality comparison
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ============================================================
# Global settings
# ============================================================

FAST_MODE = False  # True ≈ 5-10 min  |  False ≈ 30-60 min

_ARCHS = ["VanillaPINN", "SIREN", "ModifiedMLP"]

if FAST_MODE:
    _EPOCHS = {"Burgers 1D+t": 1000, "Heat 2D+t": 1000, "Kovasznay NS 2D": 800}
    _N_COL  = {"Burgers 1D+t": 1500, "Heat 2D+t": 1500, "Kovasznay NS 2D": 600}
    _HIDDEN = [32, 32, 32]
    _SIREN_DIM, _MODMLP_DIM = 64, 32
else:
    _EPOCHS = {"Burgers 1D+t": 3000, "Heat 2D+t": 2500, "Kovasznay NS 2D": 3000}
    _N_COL  = {"Burgers 1D+t": 4000, "Heat 2D+t": 3000, "Kovasznay NS 2D": 2000}
    _HIDDEN = [64, 64, 64, 64]
    _SIREN_DIM, _MODMLP_DIM = 128, 64


# ============================================================
# Utilities
# ============================================================

def _pred(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Call model and return prediction tensor (handles PINNOutput / ModelOutput)."""
    out = model(x)
    if isinstance(out, torch.Tensor):
        return out
    return out.y  # PINNOutput.y or ModelOutput.y


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y.sum(), x, create_graph=True, retain_graph=True
    )[0]


# ============================================================
# Model factory
# ============================================================

def build_model(arch: str, in_dim: int, out_dim: int) -> nn.Module:
    """Instantiate an architecture by name."""
    if arch == "VanillaPINN":
        from pinneapple_neural.architectures.pinns.vanilla import VanillaPINN
        return VanillaPINN(in_dim=in_dim, out_dim=out_dim, hidden=_HIDDEN)

    if arch == "SIREN":
        from pinneapple_neural.architectures.siren import SIREN
        return SIREN(
            in_dim=in_dim, out_dim=out_dim,
            hidden_dim=_SIREN_DIM, n_layers=4, omega_0=30.0,
        )

    if arch == "ModifiedMLP":
        from pinneapple_neural.architectures.modified_mlp import ModifiedMLP
        return ModifiedMLP(
            in_dim=in_dim, out_dim=out_dim,
            hidden_dim=_MODMLP_DIM, n_layers=4, n_fourier=16, sigma=1.0,
        )

    raise ValueError(f"Unknown architecture: {arch!r}")


# ============================================================
# Result container
# ============================================================

@dataclass
class BenchmarkResult:
    arch:    str
    problem: str
    l2_rel:  float
    time_s:  float
    losses:  List[float] = field(default_factory=list)


# ============================================================
# PDE Problem 1 — Burgers equation (1D + time)
#
#   ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
#   x ∈ [−1, 1],  t ∈ [0, 1],  ν = 0.01/π
#   IC:  u(x, 0) = −sin(πx)
#   BC:  u(±1, t) = 0
#   Reference: scipy solve_ivp method-of-lines (or decaying-sine fallback)
# ============================================================

class BurgersProblem:
    name    = "Burgers 1D+t"
    in_dim  = 2   # (x, t)
    out_dim = 1

    nu = 0.01 / math.pi

    def __init__(self) -> None:
        self._build_reference()

    def _build_reference(self) -> None:
        try:
            from scipy.integrate import solve_ivp
            Nx  = 128
            x   = np.linspace(-1.0, 1.0, Nx)
            dx  = x[1] - x[0]
            nu  = self.nu

            def rhs(t, u):
                u = u.copy()
                u[0] = u[-1] = 0.0
                # Upwind for advection (stability near shock)
                u_fwd = (np.roll(u, -1) - u) / dx
                u_bwd = (u - np.roll(u,  1)) / dx
                u_x   = np.where(u >= 0, u_bwd, u_fwd)
                u_x[0] = u_x[-1] = 0.0
                # Central for diffusion
                u_xx  = (np.roll(u, -1) - 2*u + np.roll(u, 1)) / dx**2
                u_xx[0] = u_xx[-1] = 0.0
                return -u * u_x + nu * u_xx

            u0  = -np.sin(np.pi * x)
            sol = solve_ivp(rhs, [0.0, 1.0], u0,
                            t_eval=np.linspace(0, 1, 101),
                            method="Radau", rtol=1e-5, atol=1e-7)
            # Use sol.t (actual times) — Radau always reaches t_end
            self._x  = x
            self._t  = sol.t       # actual times from solver
            self._u  = sol.y.T     # (n_t, Nx) — matches self._t
        except Exception:
            Nx, Nt = 100, 101
            x      = np.linspace(-1.0, 1.0, Nx)
            t      = np.linspace(0.0,  1.0, Nt)
            X, T   = np.meshgrid(x, t)
            self._x = x
            self._t = t
            self._u = -np.sin(np.pi * X) * np.exp(-self.nu * np.pi**2 * T)

    def sample_interior(self, n: int) -> torch.Tensor:
        x = torch.FloatTensor(n, 1).uniform_(-1.0, 1.0)
        t = torch.FloatTensor(n, 1).uniform_( 0.0, 1.0)
        return torch.cat([x, t], dim=1)

    def sample_bc(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.FloatTensor(n, 1).uniform_(0.0, 1.0)
        pts = torch.cat([
            torch.cat([torch.full((n, 1), -1.0), t], dim=1),
            torch.cat([torch.full((n, 1),  1.0), t], dim=1),
        ], dim=0)
        return pts, torch.zeros(2 * n, 1)

    def sample_ic(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x   = torch.FloatTensor(n, 1).uniform_(-1.0, 1.0)
        pts = torch.cat([x, torch.zeros(n, 1)], dim=1)
        u0  = -torch.sin(math.pi * x)
        return pts, u0

    def physics_residual(
        self, model_fn: Callable, pts: torch.Tensor
    ) -> torch.Tensor:
        u    = model_fn(pts)
        du   = _grad(u, pts)
        u_x  = du[:, 0:1]
        u_t  = du[:, 1:2]
        u_xx = _grad(u_x, pts)[:, 0:1]
        return u_t + u * u_x - self.nu * u_xx

    def evaluate(self, model: nn.Module) -> float:
        X, T = np.meshgrid(self._x, self._t)
        pts  = torch.tensor(
            np.stack([X.flatten(), T.flatten()], axis=1), dtype=torch.float32
        )
        with torch.no_grad():
            u_pred = _pred(model, pts).cpu().numpy().flatten()
        u_true = self._u.flatten()
        return float(
            np.sqrt(np.mean((u_pred - u_true)**2))
            / (np.sqrt(np.mean(u_true**2)) + 1e-12)
        )

    def ref_colormap(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(t, x, u) for pcolormesh — rows=time, cols=space."""
        return self._t, self._x, self._u


# ============================================================
# PDE Problem 2 — Heat equation (2D + time)
#
#   ∂u/∂t = α (∂²u/∂x² + ∂²u/∂y²)
#   (x,y) ∈ [0,1]²,  t ∈ [0, 0.1],  α = 0.1
#   IC:  u(x, y, 0) = sin(πx)·sin(πy)
#   BC:  u = 0 on all boundaries
#   Exact: u = sin(πx)·sin(πy)·exp(−2π²αt)
# ============================================================

class HeatProblem:
    name    = "Heat 2D+t"
    in_dim  = 3   # (x, y, t)
    out_dim = 1

    alpha = 0.1
    t_end = 0.1

    @staticmethod
    def _exact_np(x: np.ndarray, y: np.ndarray, t: float, alpha: float = 0.1) -> np.ndarray:
        return np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-2 * np.pi**2 * alpha * t)

    def sample_interior(self, n: int) -> torch.Tensor:
        x = torch.FloatTensor(n, 1).uniform_(0.0, 1.0)
        y = torch.FloatTensor(n, 1).uniform_(0.0, 1.0)
        t = torch.FloatTensor(n, 1).uniform_(0.0, self.t_end)
        return torch.cat([x, y, t], dim=1)

    def sample_bc(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        nb = n // 4
        t  = torch.FloatTensor(nb, 1).uniform_(0.0, self.t_end)
        r  = torch.FloatTensor(nb, 1).uniform_(0.0, 1.0)
        pts = torch.cat([
            torch.cat([torch.zeros(nb, 1), r, t], dim=1),   # x=0
            torch.cat([torch.ones(nb, 1),  r, t], dim=1),   # x=1
            torch.cat([r, torch.zeros(nb, 1), t], dim=1),   # y=0
            torch.cat([r, torch.ones(nb, 1),  t], dim=1),   # y=1
        ], dim=0)
        return pts, torch.zeros(4 * nb, 1)

    def sample_ic(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x   = torch.FloatTensor(n, 1).uniform_(0.0, 1.0)
        y   = torch.FloatTensor(n, 1).uniform_(0.0, 1.0)
        pts = torch.cat([x, y, torch.zeros(n, 1)], dim=1)
        u0  = torch.sin(math.pi * x) * torch.sin(math.pi * y)
        return pts, u0

    def physics_residual(
        self, model_fn: Callable, pts: torch.Tensor
    ) -> torch.Tensor:
        u    = model_fn(pts)
        d1   = _grad(u, pts)
        u_t  = d1[:, 2:3]
        u_x  = d1[:, 0:1];  u_xx = _grad(u_x, pts)[:, 0:1]
        u_y  = d1[:, 1:2];  u_yy = _grad(u_y, pts)[:, 1:2]
        return u_t - self.alpha * (u_xx + u_yy)

    def evaluate(self, model: nn.Module, t_v: float = 0.05) -> float:
        Ng   = 50
        x, y = np.linspace(0, 1, Ng), np.linspace(0, 1, Ng)
        X, Y = np.meshgrid(x, y)
        pts  = torch.tensor(
            np.stack([X.flatten(), Y.flatten(),
                      np.full(Ng * Ng, t_v)], axis=1),
            dtype=torch.float32,
        )
        with torch.no_grad():
            u_pred = _pred(model, pts).cpu().numpy().flatten()
        u_true = self._exact_np(X.flatten(), Y.flatten(), t_v)
        return float(
            np.sqrt(np.mean((u_pred - u_true)**2))
            / (np.sqrt(np.mean(u_true**2)) + 1e-12)
        )

    def ref_colormap(self, t_v: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x, y, u) 2D snapshot at t_v."""
        Ng   = 80
        x, y = np.linspace(0, 1, Ng), np.linspace(0, 1, Ng)
        X, Y = np.meshgrid(x, y)
        return x, y, self._exact_np(X, Y, t_v)


# ============================================================
# PDE Problem 3 — Kovasznay flow (2D steady incompressible NS)
#
#   ν = 1/40,  λ = Re/2 − √(Re²/4 + 4π²)
#   Exact (Kovasznay 1948):
#     u = 1 − exp(λx) cos(2πy)
#     v = λ/(2π) exp(λx) sin(2πy)
#     p = (1 − exp(2λx)) / 2
#   Domain: (x,y) ∈ [−0.5, 1.0] × [−0.5, 1.5]
#   Physics: continuity + x/y momentum
# ============================================================

class KovasznayProblem:
    name    = "Kovasznay NS 2D"
    in_dim  = 2   # (x, y)
    out_dim = 3   # (u, v, p)

    Re  = 40.0
    nu  = 1.0 / 40.0
    lam = Re / 2.0 - math.sqrt(Re**2 / 4.0 + 4.0 * math.pi**2)

    x_lo, x_hi = -0.5, 1.0
    y_lo, y_hi = -0.5, 1.5

    def _exact_np(
        self, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        lam = self.lam
        u   = 1.0 - np.exp(lam * x) * np.cos(2 * np.pi * y)
        v   = lam / (2 * np.pi) * np.exp(lam * x) * np.sin(2 * np.pi * y)
        p   = (1.0 - np.exp(2 * lam * x)) / 2.0
        return np.stack([u, v, p], axis=-1)

    def sample_interior(self, n: int) -> torch.Tensor:
        x = torch.FloatTensor(n, 1).uniform_(self.x_lo, self.x_hi)
        y = torch.FloatTensor(n, 1).uniform_(self.y_lo, self.y_hi)
        return torch.cat([x, y], dim=1)

    def sample_bc(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        nb = n // 4
        pts_list, val_list = [], []
        for side in range(4):
            if side == 0:
                xs = torch.full((nb, 1), self.x_lo)
                ys = torch.FloatTensor(nb, 1).uniform_(self.y_lo, self.y_hi)
            elif side == 1:
                xs = torch.full((nb, 1), self.x_hi)
                ys = torch.FloatTensor(nb, 1).uniform_(self.y_lo, self.y_hi)
            elif side == 2:
                xs = torch.FloatTensor(nb, 1).uniform_(self.x_lo, self.x_hi)
                ys = torch.full((nb, 1), self.y_lo)
            else:
                xs = torch.FloatTensor(nb, 1).uniform_(self.x_lo, self.x_hi)
                ys = torch.full((nb, 1), self.y_hi)
            pts = torch.cat([xs, ys], dim=1)
            uvp = self._exact_np(pts[:, 0].numpy(), pts[:, 1].numpy())
            pts_list.append(pts)
            val_list.append(torch.tensor(uvp, dtype=torch.float32))
        return torch.cat(pts_list, 0), torch.cat(val_list, 0)

    def sample_ic(self, n: int) -> None:
        return None

    def physics_residual(
        self, model_fn: Callable, pts: torch.Tensor
    ) -> torch.Tensor:
        uvp  = model_fn(pts)
        u, v, p = uvp[:, 0:1], uvp[:, 1:2], uvp[:, 2:3]
        nu   = self.nu

        du    = _grad(u, pts)
        u_x   = du[:, 0:1];  u_y = du[:, 1:2]
        u_xx  = _grad(u_x, pts)[:, 0:1]
        u_yy  = _grad(u_y, pts)[:, 1:2]

        dv    = _grad(v, pts)
        v_x   = dv[:, 0:1];  v_y = dv[:, 1:2]
        v_xx  = _grad(v_x, pts)[:, 0:1]
        v_yy  = _grad(v_y, pts)[:, 1:2]

        dp    = _grad(p, pts)
        p_x   = dp[:, 0:1];  p_y = dp[:, 1:2]

        r_div = u_x + v_y
        r_mom_x = u*u_x + v*u_y + p_x - nu*(u_xx + u_yy)
        r_mom_y = u*v_x + v*v_y + p_y - nu*(v_xx + v_yy)

        return torch.cat([r_div, r_mom_x, r_mom_y], dim=1)

    def evaluate(self, model: nn.Module) -> float:
        Ng   = 50
        x    = np.linspace(self.x_lo, self.x_hi, Ng)
        y    = np.linspace(self.y_lo, self.y_hi, Ng)
        X, Y = np.meshgrid(x, y)
        pts  = torch.tensor(
            np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32
        )
        with torch.no_grad():
            uvp_pred = _pred(model, pts).cpu().numpy()
        uvp_true = self._exact_np(X.flatten(), Y.flatten())
        return float(
            np.sqrt(np.mean((uvp_pred - uvp_true)**2))
            / (np.sqrt(np.mean(uvp_true**2)) + 1e-12)
        )

    def ref_colormap(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x, y, u_velocity) 2D map."""
        Ng   = 80
        x    = np.linspace(self.x_lo, self.x_hi, Ng)
        y    = np.linspace(self.y_lo, self.y_hi, Ng)
        X, Y = np.meshgrid(x, y)
        uvp  = self._exact_np(X.flatten(), Y.flatten())
        return x, y, uvp[:, 0].reshape(Ng, Ng)


# ============================================================
# Unified training loop
# ============================================================

def run_experiment(
    problem,
    model: nn.Module,
    *,
    epochs: int,
    n_col:  int,
    n_bc:   int = 300,
    n_ic:   int = 300,
    lr:     float = 2e-3,
    w_bc:   float = 10.0,
    w_ic:   float = 10.0,
    log_every: int = 200,
) -> BenchmarkResult:
    """Train one (problem, model) pair and return evaluation metrics."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    # Pre-sample BC / IC (static across epochs)
    pts_bc, u_bc    = problem.sample_bc(n_bc)
    ic_data         = problem.sample_ic(n_ic)

    losses: List[float] = []
    model.train()
    t0 = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Resample interior points each epoch for good coverage
        pts_col = problem.sample_interior(n_col)
        pts_col.requires_grad_(True)

        def model_fn(x: torch.Tensor) -> torch.Tensor:
            return _pred(model, x)

        # Physics loss (mean of squared residuals)
        res        = problem.physics_residual(model_fn, pts_col)
        loss_phys  = res.pow(2).mean()

        # BC loss
        u_bc_pred  = _pred(model, pts_bc)
        loss_bc    = F.mse_loss(u_bc_pred, u_bc)

        # IC loss (None for steady problems like Kovasznay)
        loss_ic    = torch.tensor(0.0)
        if ic_data is not None:
            pts_ic, u_ic = ic_data
            loss_ic = F.mse_loss(_pred(model, pts_ic), u_ic)

        loss = loss_phys + w_bc * loss_bc + w_ic * loss_ic
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))

        if (epoch + 1) % log_every == 0:
            print(f"      [{epoch+1:5d}/{epochs}]  loss={losses[-1]:.4e}")

    elapsed = time.time() - t0
    model.eval()
    l2_rel  = problem.evaluate(model)

    return BenchmarkResult(
        arch=model.__class__.__name__,
        problem=problem.name,
        l2_rel=l2_rel,
        time_s=elapsed,
        losses=losses,
    )


# ============================================================
# Benchmark runner
# ============================================================

PROBLEMS = [BurgersProblem(), HeatProblem(), KovasznayProblem()]
results:       Dict[Tuple[str, str], BenchmarkResult] = {}
trained_models: Dict[Tuple[str, str], nn.Module]      = {}

print("=" * 68)
print("  PINNeAPPle — PDE Architecture Benchmark")
print(f"  Mode: {'FAST' if FAST_MODE else 'FULL'}")
print("=" * 68)

for prob in PROBLEMS:
    epochs = _EPOCHS[prob.name]
    n_col  = _N_COL[prob.name]
    print(f"\n── {prob.name}  (epochs={epochs}, n_col={n_col}) ──")

    for arch in _ARCHS:
        print(f"  [{arch}]")
        try:
            model = build_model(arch, prob.in_dim, prob.out_dim)
            n_par = sum(p.numel() for p in model.parameters())
            print(f"    params = {n_par:,}")
        except Exception as exc:
            print(f"    SKIP — {exc}")
            continue

        res = run_experiment(
            prob, model,
            epochs=epochs,
            n_col=n_col,
            log_every=max(1, epochs // 4),
        )
        results[(prob.name, arch)]       = res
        trained_models[(prob.name, arch)] = model
        print(f"    → L2 rel = {res.l2_rel:.4f}  |  {res.time_s:.1f}s")


# ============================================================
# Summary table
# ============================================================

SEP  = "─" * 72
HDR  = f"  {'Problem':<22}{'VanillaPINN':>18}{'SIREN':>18}{'ModifiedMLP':>18}"

print(f"\n{SEP}")
print("  PINNeAPPle PDE Benchmark — Results")
print(SEP)
print(HDR)
print(SEP)

for prob in PROBLEMS:
    cells = []
    best  = float("inf")
    best_arch = ""
    for arch in _ARCHS:
        key = (prob.name, arch)
        if key in results:
            r = results[key]
            cells.append(f"L2={r.l2_rel:.4f} ({r.time_s:.0f}s)")
            if r.l2_rel < best:
                best = r.l2_rel
                best_arch = arch
        else:
            cells.append("N/A")
    row = f"  {prob.name:<22}" + "".join(f"{c:>18}" for c in cells)
    print(row)

print(SEP)
print(f"\n  Tip: lower L2 relative error is better.")
print(f"  FAST_MODE={FAST_MODE} — set False for higher-quality results.")


# ============================================================
# Plots — 3×3 grid
#   Row 0: reference solution colormaps
#   Row 1: best-model prediction  vs  reference (absolute error)
#   Row 2: training loss curves (3 models per problem)
# ============================================================

def _best_model(prob_name: str) -> Optional[nn.Module]:
    best_l2, best_m = float("inf"), None
    for arch in _ARCHS:
        key = (prob_name, arch)
        if key in results and results[key].l2_rel < best_l2:
            best_l2 = results[key].l2_rel
            best_m  = trained_models.get(key)
    return best_m


def _pred_grid(model: nn.Module, pts: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return _pred(model, pts).cpu().numpy()


try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    COLORS = {
        "VanillaPINN": "#e41a1c",
        "SIREN":       "#377eb8",
        "ModifiedMLP": "#4daf4a",
    }
    LS = {
        "VanillaPINN": "-",
        "SIREN":       "--",
        "ModifiedMLP": "-.",
    }

    fig = plt.figure(figsize=(16, 13))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.36)

    for col, prob in enumerate(PROBLEMS):

        # ── Row 0: reference solution ──────────────────────────────────
        ax_ref = fig.add_subplot(gs[0, col])

        if isinstance(prob, BurgersProblem):
            t_r, x_r, u_r = prob.ref_colormap()   # (Nt,) (Nx,) (Nt, Nx)
            im = ax_ref.pcolormesh(
                t_r, x_r, u_r.T, cmap="RdBu_r", shading="auto"
            )
            ax_ref.set_xlabel("t"); ax_ref.set_ylabel("x")
            plt.colorbar(im, ax=ax_ref, fraction=0.046, pad=0.04, label="u")

        elif isinstance(prob, HeatProblem):
            x_r, y_r, u_r = prob.ref_colormap(t_v=0.05)
            X, Y = np.meshgrid(x_r, y_r)
            im   = ax_ref.pcolormesh(X, Y, u_r, cmap="hot_r", shading="auto")
            ax_ref.set_xlabel("x"); ax_ref.set_ylabel("y")
            plt.colorbar(im, ax=ax_ref, fraction=0.046, pad=0.04, label="u")
            ax_ref.set_aspect("equal")

        elif isinstance(prob, KovasznayProblem):
            x_r, y_r, u_r = prob.ref_colormap()
            X, Y = np.meshgrid(x_r, y_r)
            im   = ax_ref.pcolormesh(X, Y, u_r, cmap="viridis", shading="auto")
            ax_ref.set_xlabel("x"); ax_ref.set_ylabel("y")
            plt.colorbar(im, ax=ax_ref, fraction=0.046, pad=0.04, label="u")

        ax_ref.set_title(f"{prob.name}\n(reference)", fontsize=9)

        # ── Row 1: best-model prediction  vs  reference (error) ────────
        ax_pred = fig.add_subplot(gs[1, col])
        best_m  = _best_model(prob.name)

        if best_m is not None:
            if isinstance(prob, BurgersProblem):
                t_r, x_r, u_r = prob.ref_colormap()
                X, T = np.meshgrid(x_r, t_r)
                pts  = torch.tensor(
                    np.stack([X.flatten(), T.flatten()], axis=1),
                    dtype=torch.float32,
                )
                u_p  = _pred_grid(best_m, pts).reshape(len(t_r), len(x_r))
                err  = np.abs(u_p - u_r)
                im   = ax_pred.pcolormesh(t_r, x_r, err.T,
                                          cmap="Oranges", shading="auto")
                ax_pred.set_xlabel("t"); ax_pred.set_ylabel("x")
                plt.colorbar(im, ax=ax_pred, fraction=0.046, pad=0.04,
                             label="|error|")

            elif isinstance(prob, HeatProblem):
                x_r, y_r, u_r = prob.ref_colormap(t_v=0.05)
                X, Y = np.meshgrid(x_r, y_r)
                Ng   = len(x_r)
                pts  = torch.tensor(
                    np.stack([X.flatten(), Y.flatten(),
                              np.full(Ng*Ng, 0.05)], axis=1),
                    dtype=torch.float32,
                )
                u_p  = _pred_grid(best_m, pts).reshape(Ng, Ng)
                err  = np.abs(u_p - u_r)
                im   = ax_pred.pcolormesh(X, Y, err, cmap="Oranges",
                                          shading="auto")
                ax_pred.set_xlabel("x"); ax_pred.set_ylabel("y")
                plt.colorbar(im, ax=ax_pred, fraction=0.046, pad=0.04,
                             label="|error|")
                ax_pred.set_aspect("equal")

            elif isinstance(prob, KovasznayProblem):
                x_r, y_r, u_ref = prob.ref_colormap()
                X, Y = np.meshgrid(x_r, y_r)
                pts  = torch.tensor(
                    np.stack([X.flatten(), Y.flatten()], axis=1),
                    dtype=torch.float32,
                )
                uvp  = _pred_grid(best_m, pts)
                u_p  = uvp[:, 0].reshape(len(y_r), len(x_r))
                err  = np.abs(u_p - u_ref)
                im   = ax_pred.pcolormesh(X, Y, err, cmap="Oranges",
                                          shading="auto")
                ax_pred.set_xlabel("x"); ax_pred.set_ylabel("y")
                plt.colorbar(im, ax=ax_pred, fraction=0.046, pad=0.04,
                             label="|error|")

            # Find which arch was best
            best_arch_name = min(
                (a for a in _ARCHS if (prob.name, a) in results),
                key=lambda a: results[(prob.name, a)].l2_rel,
                default="?",
            )
            ax_pred.set_title(
                f"{prob.name}\n|pred − ref|  ({best_arch_name})", fontsize=9
            )
        else:
            ax_pred.set_visible(False)

        # ── Row 2: training loss curves ────────────────────────────────
        ax_l = fig.add_subplot(gs[2, col])
        has_any = False

        for arch in _ARCHS:
            key  = (prob.name, arch)
            if key not in results:
                continue
            hist = results[key].losses

            win  = max(1, len(hist) // 100)
            hs   = np.convolve(hist, np.ones(win) / win, mode="valid")
            ax_l.semilogy(
                hs,
                color=COLORS[arch], ls=LS[arch], lw=1.4,
                label=f"{arch}  L2={results[key].l2_rel:.3f}",
            )
            has_any = True

        ax_l.set_xlabel("Epoch"); ax_l.set_ylabel("Loss (log)")
        ax_l.set_title(f"{prob.name}\n(loss convergence)", fontsize=9)
        if has_any:
            ax_l.legend(fontsize=7, loc="upper right")
        ax_l.grid(True, alpha=0.25)

    fig.suptitle(
        "PINNeAPPle — PDE Architecture Benchmark\n"
        "VanillaPINN  ·  SIREN  ·  ModifiedMLP",
        fontsize=12, fontweight="bold",
    )

    out_path = Path("outputs") / "02_pde_benchmark.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\n  Plot saved: {out_path.resolve()}")
    plt.close()

except ImportError:
    print("  (matplotlib not available — plots skipped)")

print("\nBenchmark complete.")
