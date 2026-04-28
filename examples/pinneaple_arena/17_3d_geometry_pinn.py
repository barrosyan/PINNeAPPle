"""3D geometry PINN examples.

1. Laplace 3D with exact solution — cross-section visualisation
2. 3D sphere domain — radial temperature profile

Outputs saved to  examples/pinneaple_arena/_out/3d_pinn/
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import torch
import torch.nn as nn

plt.style.use("dark_background")
CMAP_MAIN = "plasma"
CMAP_ERR  = "hot"
ACCENT = "#58a6ff"
GREEN  = "#3fb950"
ORANGE = "#ffa657"
PURPLE = "#d2a8ff"

OUT = os.path.join(os.path.dirname(__file__), "_out", "3d_pinn")
os.makedirs(OUT, exist_ok=True)

DEVICE = "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

class MLP3D(nn.Module):
    def __init__(self, in_d: int = 3, hidden: int = 96, depth: int = 5):
        super().__init__()
        layers = [nn.Linear(in_d, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def dark_ax(ax):
    ax.set_facecolor("#0d1117")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.tick_params(colors="#8b949e", labelsize=8)


# ═════════════════════════════════════════════════════════════════════════════
# PROBLEM 1 — Laplace 3D
#   Δu = 0  in [0,1]³,  BCs from manufactured solution
#   u*(x,y,z) = sin(πx)·sin(πy)·sinh(√2·π·z) / sinh(√2·π)
#   Δu* = 0  exactly
# ═════════════════════════════════════════════════════════════════════════════

SQRT2 = np.sqrt(2.0)

def u_true_3d(x, y, z):
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.sinh(SQRT2 * np.pi * z) / np.sinh(SQRT2 * np.pi)

def u_true_3d_t(xyz):
    x, y, z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]
    return (torch.sin(np.pi * x) * torch.sin(np.pi * y)
            * torch.sinh(SQRT2 * np.pi * z) / np.sinh(SQRT2 * np.pi))

def laplace3d_residual(model, xyz):
    xyz = xyz.requires_grad_(True)
    u   = model(xyz)
    g   = torch.autograd.grad(u, xyz, torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(g[:, 0:1], xyz, torch.ones_like(g[:, 0:1]), create_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(g[:, 1:2], xyz, torch.ones_like(g[:, 1:2]), create_graph=True)[0][:, 1:2]
    uzz = torch.autograd.grad(g[:, 2:3], xyz, torch.ones_like(g[:, 2:3]), create_graph=True)[0][:, 2:3]
    return (uxx + uyy + uzz) ** 2

rng = np.random.default_rng(42)
xyz_col = torch.from_numpy(rng.uniform(0.0, 1.0, (3000, 3)).astype(np.float32))

# BCs: all 6 faces
nb = 50
def face(ax0, val0, ax1, ax2):
    coords = rng.uniform(0.0, 1.0, (nb, 3)).astype(np.float32)
    coords[:, ax0] = val0
    return coords

faces = np.concatenate([
    face(0, 0.0, 1, 2), face(0, 1.0, 1, 2),
    face(1, 0.0, 0, 2), face(1, 1.0, 0, 2),
    face(2, 0.0, 0, 1), face(2, 1.0, 0, 1),
], axis=0)
xyz_bc = torch.from_numpy(faces)
u_bc   = u_true_3d_t(xyz_bc).detach()

print("Training Laplace 3D...")
model_laplace = MLP3D(hidden=96, depth=5).to(DEVICE)
opt   = torch.optim.Adam(model_laplace.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3000, eta_min=1e-4)
hist_laplace = []

for ep in range(3000):
    opt.zero_grad()
    loss_pde = laplace3d_residual(model_laplace, xyz_col).mean()
    loss_bc  = nn.functional.mse_loss(model_laplace(xyz_bc), u_bc)
    loss = loss_pde + 20.0 * loss_bc
    loss.backward()
    nn.utils.clip_grad_norm_(model_laplace.parameters(), 1.0)
    opt.step()
    sched.step()
    hist_laplace.append(float(loss.detach()))

print(f"Laplace 3D — final loss: {hist_laplace[-1]:.4e}")

# Cross-section evaluation: z = 0.25, 0.50, 0.75
n_e = 80
xs = np.linspace(0, 1, n_e, dtype=np.float32)
XE, YE = np.meshgrid(xs, xs)
xy_flat = np.column_stack([XE.ravel(), YE.ravel()])

fig, axes = plt.subplots(2, 3, figsize=(14, 9), facecolor="#0d1117")
fig.suptitle("Laplace 3D PINN — Cross-sections at z = 0.25, 0.50, 0.75",
             color=ACCENT, fontsize=13, fontweight="bold")

for col_i, z_val in enumerate([0.25, 0.50, 0.75]):
    xyz_eval = np.column_stack([xy_flat, np.full(xy_flat.shape[0], z_val, dtype=np.float32)])
    xyz_t    = torch.from_numpy(xyz_eval)
    model_laplace.eval()
    with torch.no_grad():
        u_pred = model_laplace(xyz_t).numpy().reshape(n_e, n_e)
    u_ref  = u_true_3d(XE, YE, z_val)
    err    = np.abs(u_pred - u_ref)
    rel_l2 = np.linalg.norm(err) / (np.linalg.norm(u_ref) + 1e-12)

    # True
    ax = axes[0, col_i]; dark_ax(ax)
    cf = ax.contourf(XE, YE, u_ref, levels=40, cmap=CMAP_MAIN)
    fig.colorbar(cf, ax=ax, pad=0.02).ax.tick_params(colors="#8b949e", labelsize=7)
    ax.set_title(f"True u, z={z_val}", color="#e6edf3", fontsize=9)
    ax.set_xlabel("x", color="#8b949e"); ax.set_ylabel("y", color="#8b949e")
    ax.set_aspect("equal")

    # Error
    ax = axes[1, col_i]; dark_ax(ax)
    cf = ax.contourf(XE, YE, err, levels=40, cmap=CMAP_ERR)
    fig.colorbar(cf, ax=ax, pad=0.02).ax.tick_params(colors="#8b949e", labelsize=7)
    ax.set_title(f"|Error|, z={z_val}  (rel-L²={rel_l2:.3f})", color="#e6edf3", fontsize=9)
    ax.set_xlabel("x", color="#8b949e"); ax.set_ylabel("y", color="#8b949e")
    ax.set_aspect("equal")

plt.tight_layout(pad=0.8)
fig.savefig(os.path.join(OUT, "laplace_3d_cross_sections.png"), dpi=140,
            bbox_inches="tight", facecolor="#0d1117")
plt.close(fig)
print("Saved: laplace_3d_cross_sections.png")


# ═════════════════════════════════════════════════════════════════════════════
# PROBLEM 2 — Radial heat conduction in a sphere
#   d²T/dr² + (2/r)·dT/dr = 0,  T(1) = 0,  dT/dr|r=0 bounded
#   Exact: T*(r) = 1 - r  (linear)
#   PINN uses 3D Cartesian coords; exploits spherical symmetry as test
# ═════════════════════════════════════════════════════════════════════════════

def sphere_laplace_residual(model, xyz):
    xyz = xyz.requires_grad_(True)
    T   = model(xyz)
    g   = torch.autograd.grad(T, xyz, torch.ones_like(T), create_graph=True)[0]
    Txx = torch.autograd.grad(g[:, 0:1], xyz, torch.ones_like(g[:, 0:1]), create_graph=True)[0][:, 0:1]
    Tyy = torch.autograd.grad(g[:, 1:2], xyz, torch.ones_like(g[:, 1:2]), create_graph=True)[0][:, 1:2]
    Tzz = torch.autograd.grad(g[:, 2:3], xyz, torch.ones_like(g[:, 2:3]), create_graph=True)[0][:, 2:3]
    return (Txx + Tyy + Tzz) ** 2

# Collocation: random points inside unit sphere
rng2 = np.random.default_rng(99)
n_sphere = 3000
pts = rng2.standard_normal((n_sphere * 3, 3))
r_pts = np.linalg.norm(pts, axis=1, keepdims=True)
pts_unit = pts / r_pts
r_rand   = rng2.uniform(0.0, 1.0, (n_sphere * 3, 1)) ** (1/3)
xyz_sphere = (pts_unit * r_rand)[:n_sphere].astype(np.float32)
xyz_sphere_col = torch.from_numpy(xyz_sphere)

# BC: T = 0 at r = 1
n_bc_sph = 300
theta = rng2.uniform(0, np.pi, n_bc_sph)
phi   = rng2.uniform(0, 2*np.pi, n_bc_sph)
x_s = np.sin(theta) * np.cos(phi)
y_s = np.sin(theta) * np.sin(phi)
z_s = np.cos(theta)
xyz_bc_sph = torch.from_numpy(np.column_stack([x_s, y_s, z_s]).astype(np.float32))
T_bc_sph   = torch.zeros(n_bc_sph, 1)

# IC/center: T(0,0,0) ~ 1
xyz_center = torch.zeros(1, 3)
T_center   = torch.ones(1, 1)

print("Training sphere heat 3D...")
model_sphere = MLP3D(hidden=64, depth=4).to(DEVICE)
opt_s   = torch.optim.Adam(model_sphere.parameters(), lr=1e-3)
sched_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=2000, eta_min=1e-4)
hist_sphere = []

for ep in range(2000):
    opt_s.zero_grad()
    loss_pde    = sphere_laplace_residual(model_sphere, xyz_sphere_col).mean()
    loss_bc_sph = nn.functional.mse_loss(model_sphere(xyz_bc_sph), T_bc_sph)
    loss_ctr    = nn.functional.mse_loss(model_sphere(xyz_center), T_center)
    loss = loss_pde + 20.0 * loss_bc_sph + 50.0 * loss_ctr
    loss.backward()
    nn.utils.clip_grad_norm_(model_sphere.parameters(), 1.0)
    opt_s.step(); sched_s.step()
    hist_sphere.append(float(loss.detach()))

print(f"Sphere heat — final loss: {hist_sphere[-1]:.4e}")

# Evaluate along radial line (x-axis)
r_eval = np.linspace(0, 1, 200, dtype=np.float32)
xyz_line = np.column_stack([r_eval, np.zeros_like(r_eval), np.zeros_like(r_eval)])
model_sphere.eval()
with torch.no_grad():
    T_pred_line = model_sphere(torch.from_numpy(xyz_line)).numpy().ravel()
T_true_line = 1.0 - r_eval

# Mid-plane T(x, y, 0) for 2D contour
n_mp = 80
x_mp = np.linspace(-1, 1, n_mp, dtype=np.float32)
X_MP, Y_MP = np.meshgrid(x_mp, x_mp)
Z_MP = np.zeros_like(X_MP)
xyz_mp = np.column_stack([X_MP.ravel(), Y_MP.ravel(), Z_MP.ravel()])
r_mp   = np.linalg.norm(xyz_mp, axis=1)
with torch.no_grad():
    T_mp = model_sphere(torch.from_numpy(xyz_mp)).numpy().reshape(n_mp, n_mp)
T_mp_masked = np.where((X_MP**2 + Y_MP**2) <= 1.0, T_mp, np.nan)
T_true_mp   = np.where((X_MP**2 + Y_MP**2) <= 1.0, 1.0 - np.sqrt(X_MP**2 + Y_MP**2), np.nan)

fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor="#0d1117")
fig.suptitle("Sphere Heat Conduction 3D PINN — ΔT = 0",
             color=ACCENT, fontsize=13, fontweight="bold")

# Radial profile
ax = axes[0]; dark_ax(ax)
ax.plot(r_eval, T_true_line, color=ACCENT, lw=2.0, label="Exact T = 1−r")
ax.plot(r_eval, T_pred_line, color=ORANGE, lw=1.5, linestyle="--", label="PINN")
ax.set_xlabel("r", color="#e6edf3", fontsize=10)
ax.set_ylabel("T", color="#e6edf3", fontsize=10)
ax.set_title("Radial profile T(r)", color="#e6edf3", fontsize=10)
ax.legend(fontsize=9, framealpha=0.2)
ax.grid(True, color="#30363d", lw=0.4, alpha=0.6)
rel_l2_sph = np.linalg.norm(T_pred_line - T_true_line) / np.linalg.norm(T_true_line)
ax.text(0.98, 0.05, f"rel-L²={rel_l2_sph:.3f}", transform=ax.transAxes,
        ha="right", va="bottom", color=GREEN, fontsize=9)

# Mid-plane PINN
ax = axes[1]; dark_ax(ax)
cf = ax.contourf(X_MP, Y_MP, T_mp_masked, levels=40, cmap=CMAP_MAIN)
circle = plt.Circle((0, 0), 1.0, color=ACCENT, fill=False, linewidth=1.2)
ax.add_patch(circle)
fig.colorbar(cf, ax=ax, pad=0.02).ax.tick_params(colors="#8b949e", labelsize=7)
ax.set_title("PINN T(x,y,0) midplane", color="#e6edf3", fontsize=10)
ax.set_xlabel("x", color="#8b949e"); ax.set_ylabel("y", color="#8b949e")
ax.set_aspect("equal"); ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)

# Error
ax = axes[2]; dark_ax(ax)
err_mp = np.abs(T_mp_masked - T_true_mp)
cf = ax.contourf(X_MP, Y_MP, err_mp, levels=40, cmap=CMAP_ERR)
circle2 = plt.Circle((0, 0), 1.0, color=ACCENT, fill=False, linewidth=1.2)
ax.add_patch(circle2)
fig.colorbar(cf, ax=ax, pad=0.02).ax.tick_params(colors="#8b949e", labelsize=7)
ax.set_title(f"|Error| midplane (rel-L²={rel_l2_sph:.3f})", color="#e6edf3", fontsize=10)
ax.set_xlabel("x", color="#8b949e"); ax.set_ylabel("y", color="#8b949e")
ax.set_aspect("equal"); ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)

plt.tight_layout(pad=0.8)
fig.savefig(os.path.join(OUT, "sphere_heat_3d.png"), dpi=140,
            bbox_inches="tight", facecolor="#0d1117")
plt.close(fig)
print("Saved: sphere_heat_3d.png")


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY — training loss curves
# ═════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0d1117")
dark_ax(ax)
ax.semilogy(hist_laplace, color=ACCENT,  lw=1.6, label="Laplace 3D (cube)")
ax.semilogy(hist_sphere,  color=ORANGE,  lw=1.6, label="Heat 3D (sphere)")
ax.set_xlabel("Epoch", color="#e6edf3", fontsize=11)
ax.set_ylabel("Total loss", color="#e6edf3", fontsize=11)
ax.set_title("3D PINN training convergence",
             color=ACCENT, fontsize=12, fontweight="bold")
ax.legend(fontsize=10, framealpha=0.2)
ax.grid(True, color="#30363d", lw=0.5, alpha=0.6)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "3d_pinn_losses.png"), dpi=140,
            bbox_inches="tight", facecolor="#0d1117")
plt.close(fig)
print("Saved: 3d_pinn_losses.png")

print(f"\nAll outputs in: {OUT}")
