"""
generate_chassis_results.py
===========================
Chassis 3D linear-elasticity PINN via PINNeAPPle.

Produces:
  examples/results_chassis/
    field_uz.png        vertical displacement (y-dir)
    field_ux.png        longitudinal displacement (x-dir)
    field_uy.png        lateral displacement (z-dir)
    field_vm.png        von Mises stress
    loss_history.png    training loss curves
    loss_data.json      loss history (for HTML chart)
    field_data.json     field arrays (for HTML canvas)
    collocation.png     collocation points distribution

Normalisation
-------------
  Length scale  L = 4.5 m   (chassis length)
  Stiffness     E = 1        (normalised)
  ν = 0.30  →  λ = 0.577, μ = 0.385
  Normalised domain: x̂∈[0,1], ŷ∈[-0.056,0.056], ẑ∈[-0.2,0.2]
  Body-force magnitude F̂ = 1  at engine-mount patch (x̂≈0.2)
  Physical scale-back: u_phys = û * (F * L / E) = û * (8 kN * 4.5 m / 210 GPa)
                               = û * 1.71×10⁻⁷ m  [μm]
"""
from __future__ import annotations

import json
import os, sys
# Ensure project root is on path when script is run directly
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from pinneaple_environment.conditions import DirichletBC
from pinneaple_environment.presets import get_preset
from pinneaple_models.siren import SIREN
from pinneaple_pinn.compiler import LossWeights, compile_problem
from pinneaple_viz import (
    plot_multi_loss,
    plot_scalar,
    use_cfd_style,
)

# ── Output directory ──────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__), "results_chassis")
os.makedirs(OUT, exist_ok=True)

use_cfd_style()

# ── Hyper-parameters ──────────────────────────────────────────────────────
DEVICE      = torch.device("cpu")
N_COL       = 600      # interior collocation points (CPU-feasible ~0.4s/ep)
N_BC        = 400      # fixed-support BC points (4 patches × 100)
EPOCHS      = 1_200
LR          = 1e-3
LOG_EVERY   = 200
SEED        = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Normalised material & geometry ───────────────────────────────────────
# Physical: E=210 GPa, ν=0.30, chassis 4.5×0.25×1.8 m, F=8 kN
E_NORM  = 1.0
NU      = 0.30
LAM     = E_NORM * NU / ((1 + NU) * (1 - 2 * NU))   # ≈ 0.577
MU      = E_NORM / (2 * (1 + NU))                    # ≈ 0.385

# Normalised domain  [0,1] × [-0.056,0.056] × [-0.20,0.20]
X_BOUNDS = (0.0,  1.0)
Y_BOUNDS = (-0.056, 0.056)
Z_BOUNDS = (-0.200, 0.200)

# Suspension mount locations (normalised)
SUPP_XZ = [
    (0.111, +0.133),   # front-left
    (0.111, -0.133),   # front-right
    (0.889, +0.133),   # rear-left
    (0.889, -0.133),   # rear-right
]
SUPP_RADIUS = 0.028    # patch radius in normalised coords

# Engine-mount body-force patch (normalised x≈0.2, Gaussian spread)
LOAD_X  = 0.200
LOAD_Z  = 0.000
LOAD_SX = 0.040        # Gaussian σ in x
LOAD_SZ = 0.090        # Gaussian σ in z
F_MAG   = 1.0          # normalised force magnitude (−y direction)


# ── Body-force function (engine mount load, −y direction) ─────────────────
def body_force_fn(X: np.ndarray, ctx: dict) -> np.ndarray:
    """Gaussian body force in −y direction at engine-mount location."""
    bx = np.exp(-((X[:, 0] - LOAD_X) ** 2) / (2 * LOAD_SX ** 2)
                -((X[:, 2] - LOAD_Z) ** 2) / (2 * LOAD_SZ ** 2))
    b = np.zeros((len(X), 3), dtype=np.float32)
    b[:, 1] = -F_MAG * bx            # −y direction
    return b


# ── Build ProblemSpec ─────────────────────────────────────────────────────
print("[1/6] Building problem specification …")
spec = get_preset("linear_elasticity_3d", E=E_NORM, nu=NU)

# Override BCs: only fixed Dirichlet at suspension mounts
def _supp_sel(X: np.ndarray, ctx: dict) -> np.ndarray:
    mask = np.zeros(len(X), dtype=bool)
    for (xs, zs) in SUPP_XZ:
        d = np.sqrt((X[:, 0] - xs) ** 2 + (X[:, 2] - zs) ** 2)
        mask |= d < SUPP_RADIUS
    return mask

bc_fixed = DirichletBC(
    "suspension_mounts",
    ("ux", "uy", "uz"),
    "callable",
    _supp_sel,
    lambda X, ctx: np.zeros((len(X), 3), dtype=np.float32),
    weight=200.0,
)

# Use a tuple so the spec is mutable (dataclass frozen=True → replace)
from dataclasses import replace
spec = replace(spec, conditions=(bc_fixed,))


# ── Collocation sampler ───────────────────────────────────────────────────
print("[2/6] Sampling collocation and BC points …")

rng = np.random.default_rng(SEED)


def lhs_sample(n: int, bounds: list[tuple[float, float]]) -> np.ndarray:
    """Simple Latin Hypercube Sampling."""
    d = len(bounds)
    pts = np.zeros((n, d), dtype=np.float32)
    for j, (lo, hi) in enumerate(bounds):
        perm = rng.permutation(n)
        pts[:, j] = lo + (perm + rng.random(n)) / n * (hi - lo)
    return pts


X_col_np = lhs_sample(N_COL, [X_BOUNDS, Y_BOUNDS, Z_BOUNDS])

# BC points: rings around each suspension mount
bc_pts = []
for (xs, zs) in SUPP_XZ:
    n_patch = max(N_BC // 4, 1)
    theta = rng.uniform(0, 2 * np.pi, n_patch)
    r     = rng.uniform(0, SUPP_RADIUS, n_patch)
    x_p   = xs + r * np.cos(theta)
    y_p   = rng.uniform(Y_BOUNDS[0], Y_BOUNDS[1], n_patch)
    z_p   = zs + r * np.sin(theta)
    bc_pts.append(np.stack([x_p, y_p, z_p], axis=1))
X_bc_np  = np.vstack(bc_pts).astype(np.float32)
Y_bc_np  = np.zeros((len(X_bc_np), 3), dtype=np.float32)

X_col_t = torch.tensor(X_col_np, device=DEVICE)
X_bc_t  = torch.tensor(X_bc_np,  device=DEVICE)
Y_bc_t  = torch.tensor(Y_bc_np,  device=DEVICE)

print(f"   Interior points : {len(X_col_np):,}")
print(f"   BC points       : {len(X_bc_np):,}")


# ── SIREN model ───────────────────────────────────────────────────────────
print("[3/6] Building SIREN model …")
model = SIREN(
    in_dim          = 3,
    out_dim         = 3,
    hidden_dim      = 32,
    n_layers        = 3,
    omega_0         = 30.0,
    outermost_linear= True,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"   Parameters : {n_params:,}")

loss_fn = compile_problem(
    spec,
    weights=LossWeights(w_pde=1.0, w_bc=1.0),   # weights already in cond.weight
)


# ── Training loop ─────────────────────────────────────────────────────────
print("[4/6] Training …")
opt = torch.optim.Adam(model.parameters(), lr=LR)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

history: list[dict] = []
t0 = time.time()

for ep in range(EPOCHS + 1):
    opt.zero_grad()

    batch = {
        "x_col": X_col_t.clone().detach().requires_grad_(True),
        "x_bc" : X_bc_t,
        "y_bc" : Y_bc_t,
        "ctx"  : {"body_force_fn": body_force_fn},
    }

    losses = loss_fn(model, None, batch)
    total  = sum(v for v in losses.values())
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sch.step()

    if ep % LOG_EVERY == 0:
        rec = {k: float(v.detach()) for k, v in losses.items()}
        rec["total"] = float(total.detach())
        rec["epoch"] = ep
        history.append(rec)
        elapsed = time.time() - t0
        print(f"   ep {ep:5d}/{EPOCHS}  total={rec['total']:.3e}  "
              f"pde={rec.get('pde', 0):.3e}  "
              f"[{elapsed:.0f}s]")

print(f"   Training done in {time.time()-t0:.1f} s")


# ── Dense grid inference ─────────────────────────────────────────────────
print("[5/6] Inference on dense grid (y=0 plane) …")
NX, NZ = 200, 80
xg = np.linspace(*X_BOUNDS, NX, dtype=np.float32)
zg = np.linspace(*Z_BOUNDS, NZ, dtype=np.float32)
XX, ZZ = np.meshgrid(xg, zg)
YY = np.zeros_like(XX)
X_grid_np = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)

model.eval()
with torch.no_grad():
    Xg_t = torch.tensor(X_grid_np, device=DEVICE)
    pred_out = model(Xg_t)
    pred_np = (pred_out.y if hasattr(pred_out, "y") else pred_out).cpu().numpy()

ux_raw = pred_np[:, 0].reshape(NZ, NX)
uy_raw = pred_np[:, 1].reshape(NZ, NX)
uz_raw = pred_np[:, 2].reshape(NZ, NX)

# Normalised units — show raw PINN output scaled ×1e3 for readability
# Physical: u_phys [mm] = u_norm × (f_body * L^3 / E) ≈ u_norm × 3.9 mm
#   (F=8 kN over patch area ≈ 0.05 m², f_body=F/V≈420 kN/m³, L=4.5 m, E=210 GPa)
DISP_SCALE = 1e3          # ×1e3 so typical values are O(1–10)
ux_um = ux_raw * DISP_SCALE
uy_um = uy_raw * DISP_SCALE
uz_um = uz_raw * DISP_SCALE


# ── Von Mises stress via autograd ────────────────────────────────────────
print("   Computing von Mises stress via autograd …")
model.train()
Xg_ad = torch.tensor(X_grid_np, device=DEVICE, requires_grad=True)
pred_ad = model(Xg_ad)
U = pred_ad.y if hasattr(pred_ad, "y") else pred_ad      # (N,3)

# Jacobian rows
JU_rows = []
for i in range(3):
    ui = U[:, i:i + 1]
    gi = torch.autograd.grad(
        ui.sum(), Xg_ad,
        create_graph=False, retain_graph=(i < 2)
    )[0]                              # (N,3)
    JU_rows.append(gi)
# JU[i,j] = ∂u_i/∂x_j
JU = torch.stack(JU_rows, dim=1)     # (N,3,3)

eps = 0.5 * (JU + JU.permute(0, 2, 1))   # strain tensor

tr = eps[:, 0, 0] + eps[:, 1, 1] + eps[:, 2, 2]   # volumetric strain

# Cauchy stress σ_ij = λ tr(ε) δ_ij + 2μ ε_ij
sigma = 2.0 * MU * eps
sigma[:, 0, 0] += LAM * tr
sigma[:, 1, 1] += LAM * tr
sigma[:, 2, 2] += LAM * tr

s11, s22, s33 = sigma[:, 0, 0], sigma[:, 1, 1], sigma[:, 2, 2]
s12, s23, s13 = sigma[:, 0, 1], sigma[:, 1, 2], sigma[:, 0, 2]
vm_norm = torch.sqrt(
    0.5 * ((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2
           + 6 * (s12**2 + s23**2 + s13**2))
).detach().cpu().numpy()

# Normalised stress (units of E_norm = 1); scale ×1e3 for display
STRESS_SCALE = 1e3
vm_plot = vm_norm.reshape(NZ, NX) * STRESS_SCALE

model.eval()

print(f"   Max |uz|  (x1e3) = {np.abs(uz_um).max():.4f}")
print(f"   Max vm    (x1e3) = {vm_plot.max():.4f}")


# ── Plots ─────────────────────────────────────────────────────────────────
print("[6/6] Saving plots …")

# Coordinates in physical units [m]
# X: normalised [0,1] → physical [0, 4.5 m]
# Z: normalised [-0.2, 0.2] → physical [-0.9, 0.9 m]  (scale = 0.9/0.2 = 4.5)
X_phys = XX * 4.5
Z_phys = ZZ * 4.5

def savefig(fig: plt.Figure, name: str) -> None:
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved {path}")


# ── uz ────────────────────────────────────────────────────────────────────
fig_uz = plot_scalar(
    X_phys.ravel(), Z_phys.ravel(), uz_um.ravel(),
    title="Vertical Displacement $u_z$ (norm. x1e3)",
    label="$u_z$ (norm.)",
    cmap="Blues",
    n_levels=128,
    show_contour_lines=True,
    n_lines=12,
    figsize=(9, 4),
    show=False,
)
fig_uz.axes[0].set_xlabel("x [m]")
fig_uz.axes[0].set_ylabel("z [m]")
# Mark suspension mounts
ax_uz = fig_uz.axes[0]
for (xs, zs) in SUPP_XZ:
    ax_uz.plot(xs * 4.5, zs * 4.5, "rv", ms=8, zorder=5, label="_BC fixed")
ax_uz.plot(LOAD_X * 4.5, LOAD_Z * 4.5, "y^", ms=9, zorder=5, label="Load")
savefig(fig_uz, "field_uz.png")

# ── ux ────────────────────────────────────────────────────────────────────
fig_ux = plot_scalar(
    X_phys.ravel(), Z_phys.ravel(), ux_um.ravel(),
    title="Longitudinal Displacement $u_x$ (norm. x1e3)",
    label="$u_x$ (norm.)",
    cmap="RdBu_r",
    n_levels=128,
    show_contour_lines=True,
    n_lines=12,
    figsize=(9, 4),
    show=False,
)
fig_ux.axes[0].set_xlabel("x [m]")
fig_ux.axes[0].set_ylabel("z [m]")
savefig(fig_ux, "field_ux.png")

# ── uy ────────────────────────────────────────────────────────────────────
fig_uy = plot_scalar(
    X_phys.ravel(), Z_phys.ravel(), uy_um.ravel(),
    title="Lateral Displacement $u_y$ (norm. x1e3)",
    label="$u_y$ (norm.)",
    cmap="PiYG",
    n_levels=128,
    show_contour_lines=True,
    n_lines=12,
    figsize=(9, 4),
    show=False,
)
fig_uy.axes[0].set_xlabel("x [m]")
fig_uy.axes[0].set_ylabel("z [m]")
savefig(fig_uy, "field_uy.png")

# ── Von Mises ─────────────────────────────────────────────────────────────
fig_vm = plot_scalar(
    X_phys.ravel(), Z_phys.ravel(), vm_plot.ravel(),
    title="Von Mises Stress $\\sigma_{VM}$ (norm. x1e3)",
    label="$\\sigma_{VM}$ (norm.)",
    cmap="plasma",
    n_levels=128,
    show_contour_lines=True,
    n_lines=14,
    figsize=(9, 4),
    show=False,
)
fig_vm.axes[0].set_xlabel("x [m]")
fig_vm.axes[0].set_ylabel("z [m]")
ax_vm = fig_vm.axes[0]
for (xs, zs) in SUPP_XZ:
    ax_vm.plot(xs * 4.5, zs * 4.5, "rv", ms=8, zorder=5)
ax_vm.plot(LOAD_X * 4.5, LOAD_Z * 4.5, "y^", ms=9, zorder=5)
savefig(fig_vm, "field_vm.png")

# ── Loss history ──────────────────────────────────────────────────────────
fig_loss = plot_multi_loss(
    history,
    groups={"Physics residual": ["pde"],
            "Boundary (Dirichlet)": [k for k in history[-1] if k.startswith("bc_")]},
    log_scale=True,
    title="PINN Training — Linear Elasticity 3D Chassis",
    show=False,
)
savefig(fig_loss, "loss_history.png")

# ── Collocation points ────────────────────────────────────────────────────
use_cfd_style()
fig_col, ax_col = plt.subplots(figsize=(9, 4))
ax_col.scatter(X_col_np[:, 0] * 4.5, X_col_np[:, 2] * 4.5,
               s=0.8, c="#58a6ff", alpha=0.4, linewidths=0, label="Interior")
ax_col.scatter(X_bc_np[:, 0] * 4.5,  X_bc_np[:, 2] * 4.5,
               s=2.5, c="#f85149", alpha=0.9, linewidths=0, label="BC (fixed)")
ax_col.set_xlabel("x [m]")
ax_col.set_ylabel("z [m]")
ax_col.set_title("Collocation Points — Chassis Domain (y=0 projection)")
ax_col.legend(fontsize=9, markerscale=4)
ax_col.set_aspect("equal")
savefig(fig_col, "collocation.png")

# ── Save JSON data for HTML ───────────────────────────────────────────────
# Loss history JSON
loss_json = [
    {"epoch": r["epoch"],
     "total": r["total"],
     "pde":   r.get("pde", 0.0),
     **{k: r[k] for k in r if k.startswith("bc_")}}
    for r in history
]
with open(os.path.join(OUT, "loss_data.json"), "w") as f:
    json.dump(loss_json, f)

# Field summary JSON (min/max for colorbar rescaling in HTML)
summary = {
    "uz_um": {"min": float(uz_um.min()), "max": float(uz_um.max()),
              "data": uz_um.ravel().tolist()},
    "ux_um": {"min": float(ux_um.min()), "max": float(ux_um.max()),
              "data": ux_um.ravel().tolist()},
    "uy_um": {"min": float(uy_um.min()), "max": float(uy_um.max()),
              "data": uy_um.ravel().tolist()},
    "vm_plot": {"min": float(vm_plot.min()), "max": float(vm_plot.max()),
               "data": vm_plot.ravel().tolist()},
    "grid": {"Nx": NX, "Nz": NZ,
             "x_phys": xg.tolist(),
             "z_phys": (zg * 4.5).tolist()},
    "supp_xz_phys": [[xs*4.5, zs*4.5] for (xs, zs) in SUPP_XZ],
    "load_xz_phys": [LOAD_X*4.5, LOAD_Z*4.5],
    "disp_scale": DISP_SCALE,
    "stress_scale": STRESS_SCALE,
    "n_params": n_params,
    "epochs_run": EPOCHS,
    "final_loss": history[-1]["total"],
}
with open(os.path.join(OUT, "field_data.json"), "w") as f:
    json.dump(summary, f, indent=2)

print()
print("=" * 55)
print("  Results saved to:", OUT)
print(f"  Final loss       : {history[-1]['total']:.3e}")
print(f"  Max |uz| (x1e3)  : {np.abs(uz_um).max():.4f}")
print(f"  Max vm   (x1e3)  : {vm_plot.max():.4f}")
print("=" * 55)
