# -*- coding: utf-8 -*-
"""SHIFT-Crash Inspired — Full-Vehicle Crashworthiness Physics AI
=================================================================

Demonstrates how PINNeAPPle can replicate the key capabilities of
physics-informed crash surrogate systems (e.g. Luminary's SHIFT-Crash):

  1. Abramowicz-Jones thin-walled tube crush model → synthetic dataset
  2. Parameter-conditioned PINN surrogate training
  3. Transfer learning: SUV Program (large dataset) → Sedan Program (small data)
  4. Ensemble uncertainty quantification
  5. Design space exploration & Bayesian-inspired optimisation

Physical Model — Abramowicz-Jones (1984)
-----------------------------------------
  Progressive fold crush of a square thin-walled tube:

    P_mean = C_AJ · σ_y · t^(5/3) / b^(1/3)          [mean crush force, N]

  where
    t   — wall thickness [mm]
    b   — cross-section side length [mm]
    σ_y — yield strength [MPa]
    C_AJ ≈ 13.06  (empirical Abramowicz-Jones constant for square tubes)

  Instantaneous force (sinusoidal fold model):
    P(x, d) = P_mean · (1 + β · sin(π · x / H_fold))

  where x ∈ [0, d_max] is crush displacement, H_fold = 2πt is half-wavelength,
  β = 0.3 is amplitude coefficient, d_max = 0.7·L (effective crush distance).

  Energy absorbed:   E_abs(d) = ∫₀ᵈ P(x) dx
  Specific energy:   SEA = E_abs / (ρ · V_tube)

  Von Mises effective stress (simplified tube model):
    σ_vm(x) = σ_y · min(1, P(x)/P_mean) · (1 + 0.1·(x/d_max))

  Effective plastic strain:
    ε_p(x) = (P(x) / (E_mod · A_cross)) + 0.002 · (x/d_max)

Pipeline Steps
--------------
  [1] Generate synthetic crash data for SUV program (5 000 samples)
  [2] Train base PINN on SUV data
  [3] Evaluate surrogate vs. analytical solution
  [4] Transfer-learn to Sedan program (300 samples)
  [5] Ensemble UQ — report 90% PI coverage and mean RMSE
  [6] Bayesian-inspired optimisation over design space
  [7] Export plots & model checkpoint

References
----------
  Abramowicz & Jones (1984) Int. J. Impact Eng. 2, 263–281
  Wierzbicki & Abramowicz (1983) J. Appl. Mech. 50, 727–734
  Luminary Cloud SHIFT-Crash (2024) — physics-AI crash surrogate
  Jones (2011) Structural Impact, Cambridge University Press
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── PINNeAPPle imports (graceful fallback) ─────────────────────────────────
try:
    from pinneapple_train import best_device, maybe_compile
except ImportError:
    def best_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def maybe_compile(m, **_):
        return m

try:
    from pinneapple_analysis.uncertainty import EnsembleUQ
    _UQ_AVAILABLE = True
except ImportError:
    _UQ_AVAILABLE = False

try:
    from pinneapple_adaptation.transfer_learning import freeze_layers, layer_lr_groups
    _ADAPT_AVAILABLE = True
except ImportError:
    _ADAPT_AVAILABLE = False

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DARK_BG = "#0d1117"
ACCENT  = "#58a6ff"
ACCENT2 = "#f78166"
ACCENT3 = "#3fb950"

# ══════════════════════════════════════════════════════════════════════════════
# 1. PHYSICAL MODEL — Abramowicz-Jones Analytical Solver
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TubeParams:
    """Design parameters for a square thin-walled crush tube."""
    t_mm:     float = 2.0      # wall thickness [mm]
    b_mm:     float = 60.0     # section side length [mm]
    sigma_y:  float = 300.0    # yield strength [MPa]
    L_mm:     float = 400.0    # tube length [mm]
    rho:      float = 7850.0   # material density [kg/m³]
    E_GPa:    float = 200.0    # Young's modulus [GPa]
    v0_kmh:   float = 56.0     # impact velocity [km/h]


C_AJ   = 13.06   # Abramowicz-Jones constant, square tube
BETA   = 0.30    # fold-wave amplitude coefficient


def abramowicz_jones_solver(
    params: TubeParams,
    n_points: int = 200,
) -> Dict[str, np.ndarray]:
    """Analytical Abramowicz-Jones crush response."""
    t   = params.t_mm * 1e-3          # m
    b   = params.b_mm * 1e-3          # m
    sig = params.sigma_y * 1e6        # Pa
    L   = params.L_mm  * 1e-3         # m
    E   = params.E_GPa * 1e9          # Pa
    rho = params.rho

    d_max = 0.70 * L                  # effective crush distance [m]
    H_fold = 2.0 * math.pi * t       # fold half-wavelength [m]
    A_cross = 4.0 * b * t            # cross-section area [m²]
    V_tube  = 4.0 * b * t * L        # tube volume [m³]

    P_mean = C_AJ * sig * t**(5.0/3.0) / b**(1.0/3.0)

    x = np.linspace(0.0, d_max, n_points)   # crush displacement [m]
    P = P_mean * (1.0 + BETA * np.sin(math.pi * x / H_fold))
    P = np.maximum(P, 0.0)

    E_abs = np.cumsum(P) * (d_max / n_points)
    SEA   = E_abs / (rho * V_tube) / 1e3    # kJ/kg

    sigma_vm = sig * np.minimum(1.0, P / P_mean) * (1.0 + 0.1 * x / d_max) / 1e6  # MPa
    eps_p    = P / (E * A_cross) + 0.002 * (x / d_max)

    return {
        "x_m":       x,
        "P_N":       P,
        "E_abs_J":   E_abs,
        "SEA_kJ_kg": SEA,
        "sigma_vm_MPa": sigma_vm,
        "eps_p":     eps_p,
        "P_mean_N":  np.full_like(x, P_mean),
        "d_max_m":   np.full_like(x, d_max),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATASET GENERATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProgramSpec:
    """Dataset specification for a vehicle program."""
    name:    str
    n_tubes: int                         # number of design samples
    t_range:     Tuple[float, float] = (1.0, 4.0)    # mm
    b_range:     Tuple[float, float] = (40.0, 100.0)  # mm
    sigma_range: Tuple[float, float] = (200.0, 500.0) # MPa
    v0_range:    Tuple[float, float] = (40.0, 80.0)   # km/h
    n_crush_pts: int = 50


SUV_PROGRAM   = ProgramSpec("SUV",   n_tubes=5000, n_crush_pts=50)
SEDAN_PROGRAM = ProgramSpec("Sedan", n_tubes=300,  n_crush_pts=50,
                             t_range=(1.2, 3.0), b_range=(40.0, 80.0))


def _lhs_sample(n: int, ranges: List[Tuple[float, float]], rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube Sampling (LHS) from unit hypercube → physical space."""
    d = len(ranges)
    pts = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        u    = (perm + rng.uniform(size=n)) / n
        lo, hi = ranges[j]
        pts[:, j] = lo + u * (hi - lo)
    return pts


def generate_dataset(prog: ProgramSpec, seed: int = 0) -> Dict[str, np.ndarray]:
    """Generate crash surrogate dataset via LHS + Abramowicz-Jones solver."""
    rng = np.random.default_rng(seed)
    ranges = [prog.t_range, prog.b_range, prog.sigma_range, prog.v0_range]
    design_pts = _lhs_sample(prog.n_tubes, ranges, rng)

    N  = prog.n_tubes
    Nx = prog.n_crush_pts
    X_all = np.zeros((N * Nx, 6), dtype=np.float32)
    Y_all = np.zeros((N * Nx, 3), dtype=np.float32)

    for i, (t, b, sy, v0) in enumerate(design_pts):
        params = TubeParams(t_mm=t, b_mm=b, sigma_y=sy, v0_kmh=v0)
        sol    = abramowicz_jones_solver(params, n_points=Nx)

        d_max = sol["d_max_m"][0]
        x_n   = sol["x_m"] / d_max         # [0, 1]
        t_n   = (t  - prog.t_range[0])  / (prog.t_range[1]  - prog.t_range[0])
        b_n   = (b  - prog.b_range[0])  / (prog.b_range[1]  - prog.b_range[0])
        sy_n  = (sy - prog.sigma_range[0]) / (prog.sigma_range[1] - prog.sigma_range[0])
        v0_n  = (v0 - prog.v0_range[0]) / (prog.v0_range[1] - prog.v0_range[0])

        idx = slice(i * Nx, (i + 1) * Nx)
        X_all[idx, 0] = x_n.astype(np.float32)
        X_all[idx, 1] = t_n
        X_all[idx, 2] = b_n
        X_all[idx, 3] = sy_n
        X_all[idx, 4] = v0_n
        X_all[idx, 5] = 0.0    # reserved: time-history index (quasi-static → 0)

        P_scale   = sol["P_mean_N"][0]
        sig_scale = sy
        eps_scale = 1.0

        Y_all[idx, 0] = (sol["P_N"]          / P_scale).astype(np.float32)
        Y_all[idx, 1] = (sol["sigma_vm_MPa"] / sig_scale).astype(np.float32)
        Y_all[idx, 2] = (sol["eps_p"]        / 1e-2).astype(np.float32)

    perm = rng.permutation(N * Nx)
    return {"X": X_all[perm], "Y": Y_all[perm]}


# ══════════════════════════════════════════════════════════════════════════════
# 3. CRASH SURROGATE PINN
# ══════════════════════════════════════════════════════════════════════════════

class CrashPINN(nn.Module):
    """
    Parameter-conditioned crash surrogate.

    Input  : (x_norm, t_norm, b_norm, σy_norm, v0_norm, τ_norm) — 6 features
    Output : (P_norm, σ_vm_norm, ε_p_norm) — 3 physical quantities

    Architecture: Fourier feature embedding + residual MLP
    """

    def __init__(
        self,
        in_dim:        int   = 6,
        out_dim:       int   = 3,
        hidden:        int   = 256,
        n_layers:      int   = 6,
        n_fourier:     int   = 32,
        fourier_scale: float = 4.0,
    ):
        super().__init__()
        torch.manual_seed(42)
        self.B = nn.Parameter(
            torch.randn(n_fourier, in_dim) * fourier_scale,
            requires_grad=False,
        )
        feat_dim = 2 * n_fourier

        layers: List[nn.Module] = [nn.Linear(feat_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        self.trunk = nn.Sequential(*layers)
        self.skip  = nn.Linear(feat_dim, hidden)
        self.head  = nn.Linear(hidden, out_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B.T * (2.0 * math.pi)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z   = self._encode(x)
        h   = self.trunk(z) + self.skip(z)
        out = self.head(h)
        return torch.softplus(out)   # enforce positivity (P, σ, ε all ≥ 0)


# ══════════════════════════════════════════════════════════════════════════════
# 4. PHYSICS RESIDUAL — quasi-static equilibrium + energy consistency
# ══════════════════════════════════════════════════════════════════════════════

def physics_residual(model: CrashPINN, x_batch: torch.Tensor) -> torch.Tensor:
    """
    Enforce quasi-static equilibrium and energy monotonicity as soft constraints.

    dP/dx ≈ P_mean·β·(π/H) cos(πx/H) — handled implicitly by data;
    we enforce:
      (a) ∂σ_vm/∂x ≥ 0  (damage accumulation)
      (b) ε_p ≥ 0        (plastic strain non-negative — already via softplus)
      (c) P ≤ (1+β)·P_mean  (force bounded by peak fold force)
    """
    x = x_batch.clone().requires_grad_(True)
    out = model(x)
    P, sigma, eps = out[:, 0], out[:, 1], out[:, 2]

    grad_sigma = torch.autograd.grad(
        sigma.sum(), x, create_graph=True
    )[0][:, 0]                         # ∂σ_vm / ∂x_norm

    r_mono  = torch.relu(-grad_sigma).pow(2).mean()          # monotonicity
    r_bound = torch.relu(P - (1.0 + BETA) * 1.05).pow(2).mean()  # force cap

    return r_mono + r_bound


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    model:      CrashPINN,
    data:       Dict[str, np.ndarray],
    device:     torch.device,
    epochs:     int   = 8_000,
    batch_size: int   = 2_048,
    lr:         float = 3e-4,
    phys_weight: float = 0.05,
    label:      str   = "base",
) -> Dict[str, List[float]]:

    model = model.to(device)
    X = torch.tensor(data["X"], dtype=torch.float32, device=device)
    Y = torch.tensor(data["Y"], dtype=torch.float32, device=device)

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    mse   = nn.MSELoss()

    history: Dict[str, List[float]] = {"total": [], "data": [], "phys": []}
    n = X.shape[0]

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        idx   = torch.randperm(n, device=device)[:batch_size]
        xb, yb = X[idx], Y[idx]

        model.train()
        pred = model(xb)
        l_data = mse(pred, yb)
        l_phys = physics_residual(model, xb)
        loss   = l_data + phys_weight * l_phys

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        history["total"].append(loss.item())
        history["data"].append(l_data.item())
        history["phys"].append(l_phys.item())

        if epoch % 1000 == 0:
            elapsed = time.time() - t0
            print(
                f"[{label}] epoch {epoch:>6}/{epochs} | "
                f"loss {loss.item():.4e} | data {l_data.item():.4e} | "
                f"phys {l_phys.item():.4e} | {elapsed:.0f}s"
            )

    return history


# ══════════════════════════════════════════════════════════════════════════════
# 6. TRANSFER LEARNING — SUV → Sedan
# ══════════════════════════════════════════════════════════════════════════════

def transfer_to_sedan(
    base_model:   CrashPINN,
    sedan_data:   Dict[str, np.ndarray],
    device:       torch.device,
    fine_tune_epochs: int = 2_000,
) -> Tuple[CrashPINN, Dict[str, List[float]]]:
    """Freeze trunk, fine-tune head on small sedan dataset."""
    import copy
    sedan_model = copy.deepcopy(base_model).to(device)

    if _ADAPT_AVAILABLE:
        freeze_layers(sedan_model, names=["trunk", "skip"])
        param_groups = layer_lr_groups(sedan_model, base_lr=1e-4, head_multiplier=5.0)
        opt = torch.optim.AdamW(param_groups, weight_decay=1e-5)
    else:
        for name, p in sedan_model.named_parameters():
            if "head" not in name:
                p.requires_grad_(False)
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, sedan_model.parameters()),
            lr=5e-4, weight_decay=1e-5,
        )

    X = torch.tensor(sedan_data["X"], dtype=torch.float32, device=device)
    Y = torch.tensor(sedan_data["Y"], dtype=torch.float32, device=device)
    mse = nn.MSELoss()

    history: Dict[str, List[float]] = {"total": [], "data": [], "phys": []}
    batch_size = min(512, X.shape[0])

    t0 = time.time()
    for epoch in range(1, fine_tune_epochs + 1):
        idx = torch.randperm(X.shape[0], device=device)[:batch_size]
        xb, yb = X[idx], Y[idx]
        sedan_model.train()
        pred   = sedan_model(xb)
        l_data = mse(pred, yb)
        l_phys = physics_residual(sedan_model, xb)
        loss   = l_data + 0.03 * l_phys
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(sedan_model.parameters(), 1.0)
        opt.step()

        history["total"].append(loss.item())
        history["data"].append(l_data.item())
        history["phys"].append(l_phys.item())

        if epoch % 500 == 0:
            print(
                f"[Sedan TL] epoch {epoch:>5}/{fine_tune_epochs} | "
                f"loss {loss.item():.4e} | {time.time()-t0:.0f}s"
            )

    return sedan_model, history


# ══════════════════════════════════════════════════════════════════════════════
# 7. ENSEMBLE UNCERTAINTY QUANTIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def ensemble_uq(
    models:  List[CrashPINN],
    X_test:  np.ndarray,
    device:  torch.device,
    alpha:   float = 0.10,
) -> Dict[str, np.ndarray]:
    """Compute ensemble mean, std and (1-α) coverage prediction interval."""
    Xt = torch.tensor(X_test, dtype=torch.float32, device=device)
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            preds.append(m(Xt).cpu().numpy())
    preds  = np.stack(preds, axis=0)          # (n_ens, N, 3)
    mean   = preds.mean(axis=0)
    std    = preds.std(axis=0)
    lo     = np.percentile(preds, 100 * alpha / 2,       axis=0)
    hi     = np.percentile(preds, 100 * (1 - alpha / 2), axis=0)
    return {"mean": mean, "std": std, "lo": lo, "hi": hi}


# ══════════════════════════════════════════════════════════════════════════════
# 8. DESIGN SPACE EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════

def design_space_sweep(
    model:     CrashPINN,
    device:    torch.device,
    n_grid:    int = 40,
    x_norm:    float = 0.5,
) -> Dict[str, np.ndarray]:
    """Evaluate surrogate over (t, b) grid at fixed x_norm, σy_norm=0.5, v0_norm=0.5."""
    t_vals  = np.linspace(0.0, 1.0, n_grid)
    b_vals  = np.linspace(0.0, 1.0, n_grid)
    T, B    = np.meshgrid(t_vals, b_vals, indexing="ij")
    flat_t  = T.ravel().astype(np.float32)
    flat_b  = B.ravel().astype(np.float32)
    N       = flat_t.shape[0]

    X_sweep = np.column_stack([
        np.full(N, x_norm, dtype=np.float32),
        flat_t, flat_b,
        np.full(N, 0.5, dtype=np.float32),
        np.full(N, 0.5, dtype=np.float32),
        np.zeros(N, dtype=np.float32),
    ])
    Xt = torch.tensor(X_sweep, device=device)
    model.eval()
    with torch.no_grad():
        out = model(Xt).cpu().numpy()

    return {
        "T_grid": T, "B_grid": B,
        "P_map":     out[:, 0].reshape(n_grid, n_grid),
        "sigma_map": out[:, 1].reshape(n_grid, n_grid),
        "eps_map":   out[:, 2].reshape(n_grid, n_grid),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 9. PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def _dark_fig(nrows=1, ncols=1, figsize=(10, 4)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=DARK_BG)
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes[None, :]
    for ax in axes.flat:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
    return fig, axes


def plot_training_history(history_suv, history_sedan, out_path):
    fig, axes = _dark_fig(1, 2, figsize=(12, 4))
    for ax, hist, label, color in zip(
        axes[0],
        [history_suv, history_sedan],
        ["SUV Base Model", "Sedan Transfer"],
        [ACCENT, ACCENT2],
    ):
        e = np.arange(1, len(hist["total"]) + 1)
        ax.semilogy(e, hist["total"], color=color,       lw=1.5, label="total")
        ax.semilogy(e, hist["data"],  color=ACCENT3,     lw=1.0, ls="--", label="data MSE")
        ax.semilogy(e, hist["phys"],  color="#d29922",   lw=1.0, ls=":",  label="physics")
        ax.set_xlabel("Epoch");  ax.set_ylabel("Loss")
        ax.set_title(label)
        ax.legend(facecolor=DARK_BG, labelcolor="white", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  saved → {out_path}")


def plot_force_displacement(
    params: TubeParams,
    model:  CrashPINN,
    device: torch.device,
    out_path: Path,
):
    sol    = abramowicz_jones_solver(params, n_points=100)
    d_max  = sol["d_max_m"][0]
    x_norm = (sol["x_m"] / d_max).astype(np.float32)

    t_n  = float((params.t_mm  - 1.0)  / (4.0 - 1.0))
    b_n  = float((params.b_mm  - 40.0) / (100.0 - 40.0))
    sy_n = float((params.sigma_y - 200.0) / (500.0 - 200.0))
    v0_n = float((params.v0_kmh - 40.0) / (80.0 - 40.0))

    X_q = np.column_stack([
        x_norm,
        np.full_like(x_norm, t_n),
        np.full_like(x_norm, b_n),
        np.full_like(x_norm, sy_n),
        np.full_like(x_norm, v0_n),
        np.zeros_like(x_norm),
    ]).astype(np.float32)

    P_mean = sol["P_mean_N"][0]
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_q, device=device)).cpu().numpy()

    fig, axes = _dark_fig(1, 3, figsize=(15, 4))
    x_mm = sol["x_m"] * 1e3

    for ax, y_ref, y_pred, scale, ylabel, col in zip(
        axes[0],
        [sol["P_N"] / P_mean, sol["sigma_vm_MPa"] / params.sigma_y, sol["eps_p"] / 1e-2],
        [pred[:, 0],          pred[:, 1],                            pred[:, 2]],
        [1.0, 1.0, 1.0],
        ["P / P_mean", "σ_vm / σ_y", "ε_p  (×10⁻²)"],
        [ACCENT, ACCENT2, ACCENT3],
    ):
        ax.plot(x_mm, y_ref,  color="white", lw=2,   label="AJ Analytical")
        ax.plot(x_mm, y_pred, color=col,     lw=1.5, ls="--", label="PINN Surrogate")
        ax.set_xlabel("Crush displacement [mm]")
        ax.set_ylabel(ylabel)
        ax.legend(facecolor=DARK_BG, labelcolor="white", fontsize=8)

    fig.suptitle(
        f"Crash Response  |  t={params.t_mm}mm  b={params.b_mm}mm  "
        f"σ_y={params.sigma_y}MPa  v₀={params.v0_kmh}km/h",
        color="white", fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  saved → {out_path}")


def plot_design_map(sweep: Dict[str, np.ndarray], out_path: Path):
    fig, axes = _dark_fig(1, 3, figsize=(15, 5))
    t_range = (1.0, 4.0);  b_range = (40.0, 100.0)

    for ax, zmap, title, cmap in zip(
        axes[0],
        [sweep["P_map"],     sweep["sigma_map"],  sweep["eps_map"]],
        ["P / P_mean",       "σ_vm / σ_y",        "ε_p  (×10⁻²)"],
        ["plasma",           "inferno",            "viridis"],
    ):
        t_phys = t_range[0] + sweep["T_grid"] * (t_range[1] - t_range[0])
        b_phys = b_range[0] + sweep["B_grid"] * (b_range[1] - b_range[0])
        cf = ax.contourf(b_phys, t_phys, zmap, levels=20, cmap=cmap)
        fig.colorbar(cf, ax=ax)
        ax.set_xlabel("b — section side [mm]")
        ax.set_ylabel("t — wall thickness [mm]")
        ax.set_title(title)

    fig.suptitle("Design Space Map  |  x_norm=0.5  σy_norm=0.5  v0_norm=0.5",
                 color="white", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  saved → {out_path}")


def plot_uq(uq: Dict[str, np.ndarray], Y_test: np.ndarray, out_path: Path):
    """Plot ensemble mean ± 90% PI for crush force prediction."""
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.tick_params(colors="white")

    n  = min(500, Y_test.shape[0])
    xs = np.arange(n)
    ax.fill_between(xs, uq["lo"][:n, 0], uq["hi"][:n, 0],
                    alpha=0.3, color=ACCENT, label="90% PI")
    ax.plot(xs, uq["mean"][:n, 0], color=ACCENT,  lw=1.5, label="Ensemble mean")
    ax.plot(xs, Y_test[:n, 0],     color="white",  lw=1.0, ls="--", label="Ground truth")
    ax.set_xlabel("Sample index"); ax.set_ylabel("P / P_mean")
    ax.set_title("Ensemble UQ — Crush Force Prediction", color="white")
    ax.legend(facecolor=DARK_BG, labelcolor="white", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    model:  CrashPINN,
    data:   Dict[str, np.ndarray],
    device: torch.device,
    label:  str = "",
) -> Dict[str, float]:
    Xt = torch.tensor(data["X"], dtype=torch.float32, device=device)
    Yt = data["Y"]
    model.eval()
    with torch.no_grad():
        pred = model(Xt).cpu().numpy()

    names = ["P_norm", "sigma_vm_norm", "eps_p_norm"]
    metrics: Dict[str, float] = {}
    for i, nm in enumerate(names):
        rmse = float(np.sqrt(np.mean((pred[:, i] - Yt[:, i]) ** 2)))
        rel  = float(rmse / (Yt[:, i].std() + 1e-8) * 100)
        metrics[f"{nm}_rmse"] = rmse
        metrics[f"{nm}_rel%"] = rel
        print(f"  [{label}] {nm:20s}  RMSE={rmse:.4f}  Rel={rel:.2f}%")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    device = best_device()
    print(f"\n{'='*70}")
    print("  SHIFT-Crash Inspired — Crashworthiness Physics AI")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")

    # ── Step 1: Generate datasets ──────────────────────────────────────────
    print("[1/7] Generating SUV dataset (5 000 samples, LHS) …")
    suv_data   = generate_dataset(SUV_PROGRAM, seed=0)
    print(f"      X: {suv_data['X'].shape}  Y: {suv_data['Y'].shape}")

    print("[1/7] Generating Sedan dataset (300 samples, LHS) …")
    sedan_data = generate_dataset(SEDAN_PROGRAM, seed=99)

    n_test = 2_000
    rng    = np.random.default_rng(7)
    test_idx = rng.choice(suv_data["X"].shape[0], n_test, replace=False)
    test_data = {"X": suv_data["X"][test_idx], "Y": suv_data["Y"][test_idx]}

    # ── Step 2: Train base model on SUV data ───────────────────────────────
    print("\n[2/7] Training base PINN on SUV program …")
    base_model = CrashPINN(in_dim=6, out_dim=3, hidden=256, n_layers=6, n_fourier=32)
    base_model = maybe_compile(base_model)
    history_suv = train_model(
        base_model, suv_data, device,
        epochs=8_000, batch_size=2_048, lr=3e-4, phys_weight=0.05, label="SUV",
    )

    # ── Step 3: Evaluate base model ────────────────────────────────────────
    print("\n[3/7] Evaluating base model on test set …")
    metrics_base = evaluate(base_model, test_data, device, label="SUV-base")

    print("\n[3/7] Plotting force-displacement curve …")
    demo_params = TubeParams(t_mm=2.5, b_mm=65.0, sigma_y=350.0, v0_kmh=56.0)
    plot_force_displacement(
        demo_params, base_model, device,
        OUT_DIR / "01_force_displacement.png",
    )

    # ── Step 4: Transfer learning → Sedan ─────────────────────────────────
    print("\n[4/7] Transfer learning SUV → Sedan (300 samples) …")
    sedan_model, history_sedan = transfer_to_sedan(
        base_model, sedan_data, device, fine_tune_epochs=2_000
    )

    print("\n[4/7] Evaluating sedan model …")
    metrics_sedan = evaluate(sedan_model, sedan_data, device, label="Sedan-TL")

    print("\n[4/7] Plotting training history …")
    plot_training_history(
        history_suv, history_sedan,
        OUT_DIR / "02_training_history.png",
    )

    # ── Step 5: Ensemble UQ ────────────────────────────────────────────────
    print("\n[5/7] Building 5-model ensemble for UQ …")
    import copy
    ensemble = [base_model]
    for seed in range(1, 5):
        m = CrashPINN(in_dim=6, out_dim=3, hidden=256, n_layers=6, n_fourier=32)
        torch.manual_seed(seed)
        for p in m.parameters():
            if p.requires_grad:
                nn.init.normal_(p, std=0.01)
        m.load_state_dict(base_model.state_dict())
        m.to(device)
        train_model(
            m, suv_data, device,
            epochs=2_000, batch_size=2_048, lr=1e-4, phys_weight=0.02,
            label=f"ens-{seed}",
        )
        ensemble.append(m)

    uq = ensemble_uq(ensemble, test_data["X"], device, alpha=0.10)
    coverage = float(np.mean(
        (test_data["Y"][:, 0] >= uq["lo"][:, 0]) &
        (test_data["Y"][:, 0] <= uq["hi"][:, 0])
    ) * 100)
    print(f"  90% PI coverage on crush force: {coverage:.1f}%  (target ≥ 90%)")
    plot_uq(uq, test_data["Y"], OUT_DIR / "03_ensemble_uq.png")

    # ── Step 6: Design space map ───────────────────────────────────────────
    print("\n[6/7] Design space exploration …")
    sweep = design_space_sweep(base_model, device, n_grid=40, x_norm=0.5)
    plot_design_map(sweep, OUT_DIR / "04_design_map.png")

    # ── Step 7: Save results ───────────────────────────────────────────────
    print("\n[7/7] Saving model checkpoint and metrics …")
    torch.save(base_model.state_dict(), OUT_DIR / "crash_surrogate_suv.pt")
    torch.save(sedan_model.state_dict(), OUT_DIR / "crash_surrogate_sedan_tl.pt")

    all_metrics = {
        "suv_base":   metrics_base,
        "sedan_tl":   metrics_sedan,
        "uq_90pi_coverage_pct": coverage,
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "="*70)
    print("  Pipeline complete.  Outputs in:", OUT_DIR)
    for k, v in all_metrics.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                print(f"    {k}/{kk}: {vv:.4f}")
        else:
            print(f"    {k}: {v:.2f}")
    print("="*70)


if __name__ == "__main__":
    main()
