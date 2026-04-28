"""NACA 0012 aerodynamic surrogate — Joukowski mapping + PINN.

Reference solution: Joukowski conformal mapping (exact potential flow, 2D
inviscid, incompressible). A VanillaPINN then learns to reproduce u, v, Cp.

Outputs saved to  examples/pinneaple_arena/_out/naca0012/
"""
from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPoly
from matplotlib.collections import PatchCollection
import torch
import torch.nn as nn

plt.style.use("dark_background")
CMAP_VEL  = "plasma"
CMAP_PRES = "RdBu_r"
ACCENT    = "#58a6ff"
GREEN     = "#3fb950"
ORANGE    = "#ffa657"

OUT = os.path.join(os.path.dirname(__file__), "_out", "naca0012")
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Joukowski airfoil geometry + exact potential-flow solution
# ─────────────────────────────────────────────────────────────────────────────

def joukowski_transform(zeta: np.ndarray, c: float = 1.0) -> np.ndarray:
    return zeta + c**2 / zeta

def joukowski_airfoil(alpha_deg: float = 5.0, n_pts: int = 400):
    alpha = np.deg2rad(alpha_deg)
    # Circle in ζ-plane that maps to a symmetric airfoil
    R = 1.02          # slightly > 1 so trailing-edge curvature is finite
    eps = 0.03        # vertical offset for camber / thickness
    center = complex(-eps, eps * np.sin(alpha))
    R_circ = abs(1.0 - center) + 0.01   # ensure it passes near z=1

    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    zeta_circle = center + R_circ * np.exp(1j * theta)
    z_airfoil   = joukowski_transform(zeta_circle)
    return z_airfoil, center, R_circ

def potential_flow_velocity(zeta: np.ndarray, center: complex, R: float,
                             U_inf: float = 1.0, alpha_deg: float = 5.0,
                             c_j: float = 1.0):
    """
    Velocity field in the physical (z) plane via complex potential:
      f(ζ) = U(e^{-iα}ζ + e^{iα}R²/ζ) - iΓ/(2π) ln ζ
    Kutta condition: Γ = 4π U R sin(α + β)  where β is a small correction
    """
    alpha = np.deg2rad(alpha_deg)
    Gamma = 4 * np.pi * U_inf * R * np.sin(alpha)

    # Shift ζ relative to circle centre
    zeta_s = zeta - center

    # df/dζ — derivative of complex potential w.r.t. ζ
    df_dzeta = U_inf * (np.exp(-1j * alpha) - np.exp(1j * alpha) * R**2 / zeta_s**2) \
               - 1j * Gamma / (2 * np.pi * zeta_s)

    # dz/dζ — Joukowski derivative
    dz_dzeta = 1.0 - c_j**2 / zeta_s**2

    # Conjugate velocity in z-plane
    w_z = df_dzeta / dz_dzeta  # df/dz = (df/dζ)/(dz/dζ)

    u =  np.real(w_z)
    v = -np.imag(w_z)
    return u, v

def make_flow_grid(n: int = 80, xlim=(-2.5, 2.5), ylim=(-2.0, 2.0)):
    x = np.linspace(*xlim, n)
    y = np.linspace(*ylim, n)
    XX, YY = np.meshgrid(x, y)
    return XX, YY

def is_inside_airfoil(z_grid: np.ndarray, z_foil: np.ndarray) -> np.ndarray:
    """Mask points roughly inside the airfoil contour (ray-casting)."""
    from matplotlib.path import Path
    path = Path(np.column_stack([z_foil.real, z_foil.imag]))
    pts  = np.column_stack([z_grid.real.ravel(), z_grid.imag.ravel()])
    return path.contains_points(pts).reshape(z_grid.shape)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Reference data generation
# ─────────────────────────────────────────────────────────────────────────────

ALPHA  = 5.0
U_INF  = 1.0
C_J    = 1.0
N_GRID = 80

z_foil_full, center, R_circ = joukowski_airfoil(ALPHA)
z_foil = z_foil_full   # full perimeter

XX, YY = make_flow_grid(N_GRID)
ZZ = XX + 1j * YY

# Need ζ from z: simple approximate inversion via Newton (z = ζ + 1/ζ → ζ² - zζ + 1 = 0)
def z_to_zeta(z_pts, center):
    # ζ = (z ± sqrt(z²-4))/2  — choose branch far from origin
    disc = np.sqrt(z_pts**2 - 4 * C_J**2 + 0j)
    zeta1 = (z_pts + disc) / 2
    zeta2 = (z_pts - disc) / 2
    # pick the one outside the unit circle
    return np.where(np.abs(zeta1) >= np.abs(zeta2), zeta1, zeta2)

ZETA_GRID = z_to_zeta(ZZ, center)
U_REF, V_REF = potential_flow_velocity(ZETA_GRID, center, R_circ, U_INF, ALPHA, C_J)
SPEED = np.sqrt(U_REF**2 + V_REF**2)
CP_REF = 1.0 - SPEED**2 / U_INF**2

# Mask interior
inside = is_inside_airfoil(ZZ, z_foil)
U_REF[inside] = 0.0
V_REF[inside] = 0.0
CP_REF[inside] = np.nan

# ─────────────────────────────────────────────────────────────────────────────
# 3.  FIGURE 1 — Geometry + Reference flow
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0d1117")
fig.suptitle("NACA 0012 — Joukowski Potential Flow Reference",
             color=ACCENT, fontsize=14, fontweight="bold", y=1.01)

ax0, ax1 = axes
for ax in axes:
    ax.set_facecolor("#0d1117")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.tick_params(colors="#8b949e", labelsize=8)

# — Velocity magnitude contour
speed_plot = np.where(inside, np.nan, np.sqrt(U_REF**2 + V_REF**2))
cf0 = ax0.contourf(XX, YY, speed_plot, levels=50, cmap=CMAP_VEL)
ax0.streamplot(XX, YY,
               np.where(inside, 0.0, U_REF),
               np.where(inside, 0.0, V_REF),
               color="#ffffff", linewidth=0.5, density=1.2, arrowsize=0.8,
               arrowstyle="-")
ax0.fill(z_foil.real, z_foil.imag, color="#161b22", zorder=5)
ax0.plot(z_foil.real, z_foil.imag, color=ACCENT, linewidth=1.2, zorder=6)
cb0 = fig.colorbar(cf0, ax=ax0, pad=0.02, fraction=0.046)
cb0.ax.tick_params(colors="#8b949e", labelsize=7)
cb0.set_label("|V| / U∞", color="#8b949e", fontsize=9)
ax0.set_title("Velocity magnitude + streamlines", color="#e6edf3", fontsize=10)
ax0.set_aspect("equal"); ax0.set_xlim(-2.5, 2.5); ax0.set_ylim(-2.0, 2.0)
ax0.set_xlabel("x / c", color="#8b949e", fontsize=9)
ax0.set_ylabel("y / c", color="#8b949e", fontsize=9)
ax0.axhline(0, color="#30363d", lw=0.5, zorder=0)
ax0.axvline(0, color="#30363d", lw=0.5, zorder=0)

# — Cp contour
cp_clipped = np.clip(CP_REF, -4, 1.5)
cf1 = ax1.contourf(XX, YY, cp_clipped, levels=50, cmap=CMAP_PRES,
                   norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-4, vmax=1.5))
ax1.fill(z_foil.real, z_foil.imag, color="#161b22", zorder=5)
ax1.plot(z_foil.real, z_foil.imag, color=ORANGE, linewidth=1.2, zorder=6)
cb1 = fig.colorbar(cf1, ax=ax1, pad=0.02, fraction=0.046)
cb1.ax.tick_params(colors="#8b949e", labelsize=7)
cb1.set_label("Cp", color="#8b949e", fontsize=9)
ax1.set_title("Pressure coefficient Cp", color="#e6edf3", fontsize=10)
ax1.set_aspect("equal"); ax1.set_xlim(-2.5, 2.5); ax1.set_ylim(-2.0, 2.0)
ax1.set_xlabel("x / c", color="#8b949e", fontsize=9)
ax1.set_ylabel("y / c", color="#8b949e", fontsize=9)
ax1.axhline(0, color="#30363d", lw=0.5, zorder=0)
ax1.axvline(0, color="#30363d", lw=0.5, zorder=0)

plt.tight_layout(pad=0.8)
fig.savefig(os.path.join(OUT, "naca0012_reference_flow.png"), dpi=140,
            bbox_inches="tight", facecolor="#0d1117")
plt.close(fig)
print("Saved: naca0012_reference_flow.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  PINN surrogate — learns (x,y) → (u, v, Cp)
# ─────────────────────────────────────────────────────────────────────────────

class VanillaPINN(nn.Module):
    def __init__(self, hidden: int = 128, depth: int = 5):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 3))  # u, v, Cp
        self.net = nn.Sequential(*layers)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(xy)

def build_training_data():
    mask = ~inside
    x_flat = XX[mask].ravel().astype(np.float32)
    y_flat = YY[mask].ravel().astype(np.float32)
    u_flat = U_REF[mask].ravel().astype(np.float32)
    v_flat = V_REF[mask].ravel().astype(np.float32)
    cp_flat_raw = CP_REF[mask]
    # replace NaN in cp with 0
    cp_flat = np.where(np.isfinite(cp_flat_raw), cp_flat_raw, 0.0).astype(np.float32)

    xy = torch.from_numpy(np.column_stack([x_flat, y_flat]))
    uvcp = torch.from_numpy(np.column_stack([u_flat, v_flat, cp_flat]))
    return xy, uvcp

DEVICE = "cpu"
xy_train, uvcp_train = build_training_data()
xy_train  = xy_train.to(DEVICE)
uvcp_train = uvcp_train.to(DEVICE)

# Normalize inputs / targets
xy_mean, xy_std = xy_train.mean(0), xy_train.std(0) + 1e-8
t_mean, t_std   = uvcp_train.mean(0), uvcp_train.std(0) + 1e-8

xy_n   = (xy_train  - xy_mean) / xy_std
uvcp_n = (uvcp_train - t_mean) / t_std

model = VanillaPINN(hidden=128, depth=5).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3000, eta_min=1e-4)

EPOCHS   = 3000
BATCH    = 512
hist     = []

for ep in range(1, EPOCHS + 1):
    idx = torch.randperm(xy_n.shape[0])[:BATCH]
    xb, yb = xy_n[idx], uvcp_n[idx]
    opt.zero_grad()
    pred = model(xb)
    loss = nn.functional.mse_loss(pred, yb)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    hist.append(float(loss))

print(f"Final training loss: {hist[-1]:.4e}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  FIGURE 2 — Training loss curve
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0d1117")
ax.set_facecolor("#161b22")
for sp in ax.spines.values():
    sp.set_edgecolor("#30363d")
ax.tick_params(colors="#8b949e", labelsize=9)
ax.semilogy(hist, color=ACCENT, linewidth=1.5, label="MSE (normalised targets)")
ax.set_xlabel("Epoch", color="#e6edf3", fontsize=11)
ax.set_ylabel("MSE loss", color="#e6edf3", fontsize=11)
ax.set_title("NACA 0012 PINN — Training convergence", color=ACCENT, fontsize=12, fontweight="bold")
ax.legend(fontsize=10, framealpha=0.2)
ax.grid(True, color="#30363d", linewidth=0.5, alpha=0.7)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "naca0012_training_loss.png"), dpi=140,
            bbox_inches="tight", facecolor="#0d1117")
plt.close(fig)
print("Saved: naca0012_training_loss.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  FIGURE 3 — True vs Predicted fields
# ─────────────────────────────────────────────────────────────────────────────

model.eval()
with torch.no_grad():
    xy_all = torch.from_numpy(
        np.column_stack([XX.ravel().astype(np.float32),
                         YY.ravel().astype(np.float32)])
    ).to(DEVICE)
    xy_all_n = (xy_all - xy_mean) / xy_std
    pred_n   = model(xy_all_n).cpu().numpy()

pred_phys = pred_n * t_std.numpy() + t_mean.numpy()
U_pred = pred_phys[:, 0].reshape(XX.shape)
V_pred = pred_phys[:, 1].reshape(XX.shape)
CP_pred = pred_phys[:, 2].reshape(XX.shape)

U_pred[inside] = np.nan
V_pred[inside] = np.nan
CP_pred[inside] = np.nan
U_err  = np.abs(U_pred - np.where(inside, np.nan, U_REF))
CP_err = np.abs(CP_pred - np.where(inside, np.nan, np.where(np.isfinite(CP_REF), CP_REF, 0.0)))

fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor="#0d1117")
fig.suptitle("NACA 0012 PINN — True vs Predicted Flow Fields",
             color=ACCENT, fontsize=14, fontweight="bold")

titles = ["u (true)", "u (PINN)", "|u error|",
          "Cp (true)", "Cp (PINN)", "|Cp error|"]
fields = [np.where(inside, np.nan, U_REF), U_pred, U_err,
          np.where(inside, np.nan, np.where(np.isfinite(CP_REF), CP_REF, 0.0)),
          CP_pred, CP_err]
cmaps  = [CMAP_VEL, CMAP_VEL, "hot",
          CMAP_PRES, CMAP_PRES, "hot"]

for ax, title, field, cmap in zip(axes.ravel(), titles, fields, cmaps):
    ax.set_facecolor("#0d1117")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.tick_params(colors="#8b949e", labelsize=7)
    vmin, vmax = np.nanpercentile(field, [2, 98])
    cf = ax.contourf(XX, YY, field, levels=40, cmap=cmap,
                     vmin=vmin, vmax=vmax)
    ax.fill(z_foil.real, z_foil.imag, color="#1c2330", zorder=5)
    ax.plot(z_foil.real, z_foil.imag, color=ACCENT, linewidth=0.8, zorder=6)
    cb = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.046)
    cb.ax.tick_params(colors="#8b949e", labelsize=6)
    ax.set_title(title, color="#e6edf3", fontsize=9)
    ax.set_aspect("equal"); ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.0, 2.0)

plt.tight_layout(pad=1.0)
fig.savefig(os.path.join(OUT, "naca0012_true_vs_pred.png"), dpi=140,
            bbox_inches="tight", facecolor="#0d1117")
plt.close(fig)
print("Saved: naca0012_true_vs_pred.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  FIGURE 4 — Cp distribution on airfoil surface
# ─────────────────────────────────────────────────────────────────────────────

# Sample Cp along upper/lower surface from reference
n_surf = 200
theta_surf = np.linspace(0, 2 * np.pi, n_surf)
zeta_surf  = center + R_circ * np.exp(1j * theta_surf)
z_surf     = joukowski_transform(zeta_surf, C_J)
U_surf, V_surf = potential_flow_velocity(zeta_surf, center, R_circ, U_INF, ALPHA, C_J)
SPEED_surf = np.sqrt(U_surf**2 + V_surf**2)
CP_surf_ref = 1.0 - SPEED_surf**2 / U_INF**2

# PINN Cp on surface
xy_surf_np = np.column_stack([z_surf.real.astype(np.float32),
                               z_surf.imag.astype(np.float32)])
with torch.no_grad():
    xy_surf_t = torch.from_numpy(xy_surf_np)
    xy_surf_n = (xy_surf_t - xy_mean) / xy_std
    cp_pred_surf_n = model(xy_surf_n)[:, 2].numpy()
CP_surf_pinn = cp_pred_surf_n * t_std[2].item() + t_mean[2].item()

# chordwise coordinate (x normalised 0→1)
x_norm = (z_surf.real - z_surf.real.min()) / (z_surf.real.max() - z_surf.real.min())

fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0d1117")
ax.set_facecolor("#161b22")
for sp in ax.spines.values():
    sp.set_edgecolor("#30363d")
ax.tick_params(colors="#8b949e", labelsize=9)
ax.plot(x_norm, -CP_surf_ref, color=ACCENT,   linewidth=1.8, label="Joukowski ref (−Cp)")
ax.plot(x_norm, -CP_surf_pinn, color=ORANGE, linewidth=1.4, linestyle="--", label="PINN (−Cp)")
ax.invert_yaxis()
ax.set_xlabel("x/c", color="#e6edf3", fontsize=11)
ax.set_ylabel("−Cp", color="#e6edf3", fontsize=11)
ax.set_title("NACA 0012 — Cp distribution (α = 5°)", color=ACCENT, fontsize=12, fontweight="bold")
ax.legend(fontsize=10, framealpha=0.2)
ax.grid(True, color="#30363d", linewidth=0.5, alpha=0.7)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "naca0012_cp_distribution.png"), dpi=140,
            bbox_inches="tight", facecolor="#0d1117")
plt.close(fig)
print("Saved: naca0012_cp_distribution.png")

print(f"\nAll outputs in: {OUT}")
