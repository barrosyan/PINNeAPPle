"""3D geometry PINNs — Laplace cube + radial sphere heat.

(1) Laplace 3D in [0,1]³ — manufactured solution sin(πx)sin(πy)sinh(√2πz)/sinh(√2π)
(2) Sphere heat — 1D radial PINN (d²T/dr² + 2/r dT/dr = 0), T(1)=0, T(0)=1
    Exact: T(r) = 1 − r. Visualised as 3D midplane contour.

Outputs saved to  examples/pinneaple_arena/_out/3d_pinn/
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

BG   = "#0d1117"
SURF = "#161b22"
BORD = "#30363d"
MUTED= "#8b949e"
TEXT = "#e6edf3"
ACCENT = "#58a6ff"
GREEN  = "#3fb950"
ORANGE = "#ffa657"
PURPLE = "#d2a8ff"

OUT = os.path.join(os.path.dirname(__file__), "_out", "3d_pinn")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cpu"


def dark_ax(ax, fc=BG):
    ax.set_facecolor(fc)
    for sp in ax.spines.values():
        sp.set_color(BORD)
    ax.tick_params(colors=MUTED, which="both")
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


class MLP(nn.Module):
    def __init__(self, in_d, h=128, d=6):
        super().__init__()
        layers = [nn.Linear(in_d, h), nn.Tanh()]
        for _ in range(d - 1):
            layers += [nn.Linear(h, h), nn.Tanh()]
        layers.append(nn.Linear(h, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


# ═════════════════════════════════════════════════════════════════════════════
# LAPLACE 3D — Δu = 0 in [0,1]³
# Manufactured:  u* = sin(πx)·sin(πy)·sinh(√2·π·z) / sinh(√2·π)
# ═════════════════════════════════════════════════════════════════════════════

SQ2 = np.sqrt(2.0)
DENOM = np.sinh(SQ2 * np.pi)

def u_exact(x, y, z):
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.sinh(SQ2 * np.pi * z) / DENOM

def u_exact_t(xyz):
    x, y, z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]
    return torch.sin(np.pi * x) * torch.sin(np.pi * y) \
           * torch.sinh(torch.tensor(SQ2 * np.pi) * z) / DENOM

def laplace_res(model, xyz):
    xyz = xyz.requires_grad_(True)
    u   = model(xyz)
    g   = torch.autograd.grad(u, xyz, torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(g[:, 0:1], xyz, torch.ones_like(g[:, 0:1]), create_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(g[:, 1:2], xyz, torch.ones_like(g[:, 1:2]), create_graph=True)[0][:, 1:2]
    uzz = torch.autograd.grad(g[:, 2:3], xyz, torch.ones_like(g[:, 2:3]), create_graph=True)[0][:, 2:3]
    return (uxx + uyy + uzz) ** 2

rng = np.random.default_rng(42)
N_COL = 5000
xyz_col = torch.from_numpy(rng.uniform(0, 1, (N_COL, 3)).astype(np.float32))

# BC: all 6 faces — values from exact solution
NF = 80
def face_pts(ax_fix, val, ax_free1, ax_free2):
    n = NF
    pts = rng.uniform(0, 1, (n, 3)).astype(np.float32)
    pts[:, ax_fix] = val
    return pts

faces_np = np.vstack([face_pts(0,0,1,2), face_pts(0,1,1,2),
                      face_pts(1,0,0,2), face_pts(1,1,0,2),
                      face_pts(2,0,0,1), face_pts(2,1,0,1)])
xyz_bc = torch.from_numpy(faces_np)
u_bc   = u_exact_t(xyz_bc).detach()

print("Training Laplace 3D…")
model_L = MLP(3, h=128, d=6)
opt_L   = torch.optim.Adam(model_L.parameters(), lr=1e-3)
sch_L   = torch.optim.lr_scheduler.CosineAnnealingLR(opt_L, 5000, eta_min=5e-5)
hist_L  = []

for ep in range(5000):
    opt_L.zero_grad()
    loss = laplace_res(model_L, xyz_col).mean() + \
           25.0 * nn.functional.mse_loss(model_L(xyz_bc), u_bc)
    loss.backward()
    nn.utils.clip_grad_norm_(model_L.parameters(), 1.0)
    opt_L.step(); sch_L.step()
    hist_L.append(float(loss.detach()))

print(f"  Laplace 3D final loss: {hist_L[-1]:.4e}")

# Evaluate at z = 0.25, 0.50, 0.75
NE = 100
xs = np.linspace(0, 1, NE, dtype=np.float32)
XE, YE = np.meshgrid(xs, xs)
flat = np.column_stack([XE.ravel(), YE.ravel()])

results_L = {}
for z_val in [0.25, 0.50, 0.75]:
    xyz_e = np.column_stack([flat, np.full(flat.shape[0], z_val, dtype=np.float32)])
    model_L.eval()
    with torch.no_grad():
        u_p = model_L(torch.from_numpy(xyz_e)).numpy().reshape(NE, NE)
    u_t = u_exact(XE, YE, z_val)
    err = np.abs(u_p - u_t)
    rel = np.linalg.norm(err) / (np.linalg.norm(u_t) + 1e-12)
    results_L[z_val] = (u_t, u_p, err, rel)

# ── Figure: 2 rows × 3 cols  (row 0: exact, row 1: PINN, with error bar)
fig, axes = plt.subplots(2, 3, figsize=(17, 10), facecolor=BG)
fig.suptitle("Laplace 3D — Δu = 0  in [0,1]³\n"
             "Exact: sin(πx)·sin(πy)·sinh(√2πz) / sinh(√2π)",
             color=ACCENT, fontsize=14, fontweight="bold", y=1.01)

for col_i, z_val in enumerate([0.25, 0.50, 0.75]):
    u_t, u_p, err, rel = results_L[z_val]
    vmin_z = min(u_t.min(), u_p.min())
    vmax_z = max(u_t.max(), u_p.max())

    # Row 0 — exact
    ax = axes[0, col_i]; dark_ax(ax, fc=SURF)
    cf = ax.contourf(XE, YE, u_t, levels=50, cmap="plasma",
                     vmin=vmin_z, vmax=vmax_z)
    ax.contour(XE, YE, u_t, levels=8, colors=[BORD], linewidths=0.7, alpha=0.7)
    cb = fig.colorbar(cf, ax=ax, pad=0.01, fraction=0.05, shrink=0.9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(f"Exact u*  (z = {z_val})", fontsize=11, pad=6)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")

    # Row 1 — PINN
    ax = axes[1, col_i]; dark_ax(ax, fc=SURF)
    cf2 = ax.contourf(XE, YE, u_p, levels=50, cmap="plasma",
                      vmin=vmin_z, vmax=vmax_z)
    ax.contour(XE, YE, u_p, levels=8, colors=[BORD], linewidths=0.7, alpha=0.7)
    cb2 = fig.colorbar(cf2, ax=ax, pad=0.01, fraction=0.05, shrink=0.9)
    cb2.ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(f"PINN prediction  (z = {z_val})", fontsize=11, pad=6)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.text(0.97, 0.04, f"rel-L² = {rel:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            color=GREEN if rel < 0.10 else ORANGE, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=BORD, alpha=0.85))

plt.tight_layout(pad=1.5)
fig.savefig(os.path.join(OUT, "laplace_3d_cross_sections.png"), dpi=160,
            bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved laplace_3d_cross_sections.png")


# ═════════════════════════════════════════════════════════════════════════════
# SPHERE HEAT — radial 1D PINN
# d²T/dr² + (2/r) dT/dr = 0,   T(0) = 1,  T(1) = 0
# Exact: T*(r) = 1 − r
# The radial PINN is 1D in r, then visualised as 3D midplane contour.
# ═════════════════════════════════════════════════════════════════════════════

class RadialMLP(nn.Module):
    """PINN input: (r,). Uses soft-tanh boundary fix: T̃ = r*(1-r)*net(r) + (1-r)
    so that T(0)=1 and T(1)=0 are exactly satisfied by construction."""
    def __init__(self, h=64, d=5):
        super().__init__()
        layers = [nn.Linear(1, h), nn.Tanh()]
        for _ in range(d - 1):
            layers += [nn.Linear(h, h), nn.Tanh()]
        layers.append(nn.Linear(h, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, r):
        # Exact BC satisfaction: T = (1-r) + r*(1-r)*net(r)
        T_hom = (1.0 - r) + r * (1.0 - r) * self.net(r)
        return T_hom

def sphere_res(model, r_col):
    r = r_col.requires_grad_(True)
    T = model(r)
    dT  = torch.autograd.grad(T, r, torch.ones_like(T), create_graph=True)[0]
    d2T = torch.autograd.grad(dT, r, torch.ones_like(dT), create_graph=True)[0]
    return (d2T + (2.0 / (r + 1e-6)) * dT) ** 2

rng4 = np.random.default_rng(99)
# Avoid r=0 singularity in 2/r term — use (0.005, 1)
r_col = torch.from_numpy(rng4.uniform(0.005, 1.0, (3000, 1)).astype(np.float32))

print("Training sphere heat (1D radial PINN)…")
model_S = RadialMLP(h=64, d=5)
opt_S   = torch.optim.Adam(model_S.parameters(), lr=1e-3)
sch_S   = torch.optim.lr_scheduler.CosineAnnealingLR(opt_S, 3000, eta_min=1e-5)
hist_S  = []

for ep in range(3000):
    opt_S.zero_grad()
    loss = sphere_res(model_S, r_col).mean()
    loss.backward()
    nn.utils.clip_grad_norm_(model_S.parameters(), 1.0)
    opt_S.step(); sch_S.step()
    hist_S.append(float(loss.detach()))

print(f"  Sphere heat final loss: {hist_S[-1]:.4e}")

# Evaluate on radial line
r_eval = torch.from_numpy(np.linspace(0, 1, 500, dtype=np.float32).reshape(-1, 1))
model_S.eval()
with torch.no_grad():
    T_pred_1d = model_S(r_eval).numpy().ravel()
T_true_1d = 1.0 - r_eval.numpy().ravel()
rel_l2_S  = np.linalg.norm(T_pred_1d - T_true_1d) / np.linalg.norm(T_true_1d)

# Build 3D midplane (z=0) contour by revolving the 1D solution: T(x,y) = T(sqrt(x²+y²))
NMP = 120
x_mp = np.linspace(-1, 1, NMP, dtype=np.float32)
X_MP, Y_MP = np.meshgrid(x_mp, x_mp)
r_mp = np.sqrt(X_MP**2 + Y_MP**2)
r_mp_t = torch.from_numpy(r_mp.ravel().reshape(-1, 1))
with torch.no_grad():
    T_mp_pred = model_S(r_mp_t).numpy().reshape(NMP, NMP)
T_mp_true = 1.0 - r_mp
T_mp_pred = np.where(r_mp <= 1.0, T_mp_pred, np.nan)
T_mp_true = np.where(r_mp <= 1.0, T_mp_true, np.nan)
err_mp    = np.abs(T_mp_pred - T_mp_true)

r_v = r_eval.numpy().ravel()

# ── Figure: 3 panels — radial profile / midplane true / midplane pred
fig, axes = plt.subplots(1, 3, figsize=(17, 6.5), facecolor=BG)
fig.suptitle("Spherical Heat Conduction — ΔT = 0  (T = 1 − r)\n"
             "1D radial PINN with exact boundary-condition enforcement",
             color=ACCENT, fontsize=14, fontweight="bold")

# Radial profile
ax = axes[0]; dark_ax(ax, fc=SURF)
ax.plot(r_v, T_true_1d, color=ACCENT, lw=2.5, label="Exact  T = 1 − r")
ax.plot(r_v, T_pred_1d, color=ORANGE, lw=1.8, ls="--", label="PINN")
ax.fill_between(r_v, T_true_1d, T_pred_1d, alpha=0.2, color=ORANGE)
ax.set_xlabel("r"); ax.set_ylabel("T(r)")
ax.set_title("Radial temperature profile", fontsize=12, pad=8)
ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.1)
ax.legend(fontsize=11, framealpha=0.3, loc="upper right")
ax.grid(True, color=BORD, lw=0.5, alpha=0.7)
ax.text(0.97, 0.55, f"rel-L² = {rel_l2_S:.5f}",
        transform=ax.transAxes, ha="right",
        color=GREEN, fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=BORD, alpha=0.85))

# True midplane
ax = axes[1]; dark_ax(ax, fc=SURF)
cf = ax.contourf(X_MP, Y_MP, T_mp_true, levels=50, cmap="plasma", vmin=0, vmax=1)
ax.contour(X_MP, Y_MP, T_mp_true, levels=10, colors=[BORD], lw=0.7, alpha=0.7)
circle = plt.Circle((0,0), 1.0, color=ACCENT, fill=False, lw=1.5, zorder=5)
ax.add_patch(circle)
cb = fig.colorbar(cf, ax=ax, pad=0.015, fraction=0.05, shrink=0.9)
cb.set_label("T", color=MUTED, fontsize=10); cb.ax.tick_params(colors=MUTED, labelsize=9)
ax.set_title("Exact T(x,y,0) — midplane", fontsize=12, pad=8)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_aspect("equal"); ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)

# PINN midplane
ax = axes[2]; dark_ax(ax, fc=SURF)
cf2 = ax.contourf(X_MP, Y_MP, T_mp_pred, levels=50, cmap="plasma", vmin=0, vmax=1)
ax.contour(X_MP, Y_MP, T_mp_pred, levels=10, colors=[BORD], lw=0.7, alpha=0.7)
circle2 = plt.Circle((0,0), 1.0, color=ACCENT, fill=False, lw=1.5, zorder=5)
ax.add_patch(circle2)
cb2 = fig.colorbar(cf2, ax=ax, pad=0.015, fraction=0.05, shrink=0.9)
cb2.set_label("T", color=MUTED, fontsize=10); cb2.ax.tick_params(colors=MUTED, labelsize=9)
ax.set_title("PINN T(x,y,0) — midplane", fontsize=12, pad=8)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_aspect("equal"); ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
ax.text(0.97, 0.04, f"rel-L² = {rel_l2_S:.5f}",
        transform=ax.transAxes, ha="right", va="bottom",
        color=GREEN, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=BORD, alpha=0.85))

plt.tight_layout(pad=1.2)
fig.savefig(os.path.join(OUT, "sphere_heat_3d.png"), dpi=160,
            bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved sphere_heat_3d.png")

print(f"\nAll outputs in: {OUT}")
