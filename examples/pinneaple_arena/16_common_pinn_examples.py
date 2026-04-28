"""Classic PINN benchmarks — Burgers 1D, Heat 2D, Poisson 2D.

Each problem has an analytical solution used as ground truth.
Outputs saved to  examples/pinneaple_arena/_out/common_pinns/
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

plt.style.use("dark_background")
CMAP_MAIN = "plasma"
CMAP_ERR  = "hot"
ACCENT = "#58a6ff"
GREEN  = "#3fb950"
ORANGE = "#ffa657"
PURPLE = "#d2a8ff"

OUT = os.path.join(os.path.dirname(__file__), "_out", "common_pinns")
os.makedirs(OUT, exist_ok=True)

DEVICE = "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_d: int, out_d: int, hidden: int = 64, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(in_d, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, out_d))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train(model, x_col, y_col, residual_fn, epochs=3000, lr=1e-3,
          x_bc=None, y_bc=None, bc_weight=10.0):
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    hist  = []
    for ep in range(epochs):
        opt.zero_grad()
        loss_pde = residual_fn(model, x_col).mean()
        if x_bc is not None:
            loss_bc = nn.functional.mse_loss(model(x_bc), y_bc)
            loss = loss_pde + bc_weight * loss_bc
        else:
            loss = loss_pde
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        hist.append(float(loss.detach()))
    return hist


def dark_ax(ax):
    ax.set_facecolor("#0d1117")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.tick_params(colors="#8b949e", labelsize=8)


# ═════════════════════════════════════════════════════════════════════════════
# PROBLEM 1 — Burgers 1D (space-time shock)
#   u_t + u·u_x = ν·u_xx,  u(x,0) = -sin(πx), u(±1,t) = 0
#   Analytical: Hopf-Cole (series approx for low ν)
# ═════════════════════════════════════════════════════════════════════════════

NU = 0.01 / np.pi

def burgers_residual(model, xt):
    xt = xt.requires_grad_(True)
    u  = model(xt)
    u_t = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0][:, 1:2]
    u_x = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0][:, 0:1]
    u_xx = torch.autograd.grad(u_x, xt, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    return (u_t + u * u_x - NU * u_xx) ** 2

N_COL = 3000
rng   = np.random.default_rng(42)
xt_np = rng.uniform([-1, 0], [1, 1], (N_COL, 2)).astype(np.float32)
xt_col = torch.from_numpy(xt_np)

# IC  u(x,0) = -sin(πx)
x_ic = torch.from_numpy(rng.uniform(-1, 1, (200, 1)).astype(np.float32))
t_ic = torch.zeros_like(x_ic)
xt_ic = torch.cat([x_ic, t_ic], dim=1)
u_ic  = -torch.sin(np.pi * x_ic)

# BC  u(±1, t) = 0
t_bc  = torch.from_numpy(rng.uniform(0, 1, (100, 1)).astype(np.float32))
x_bc1 = -torch.ones_like(t_bc)
x_bc2 =  torch.ones_like(t_bc)
xt_bc = torch.cat([torch.cat([x_bc1, t_bc], 1), torch.cat([x_bc2, t_bc], 1)], 0)
u_bc  = torch.zeros(xt_bc.shape[0], 1)

x_bc_all = torch.cat([xt_ic, xt_bc], 0)
y_bc_all = torch.cat([u_ic, u_bc], 0)

print("Training Burgers 1D...")
model_burg = MLP(2, 1, hidden=64, depth=5).to(DEVICE)
hist_burg  = train(model_burg, xt_col, None, burgers_residual,
                   epochs=3000, lr=1e-3,
                   x_bc=x_bc_all, y_bc=y_bc_all, bc_weight=10.0)

# Evaluate on grid
nx, nt = 150, 150
x_g = np.linspace(-1, 1, nx, dtype=np.float32)
t_g = np.linspace(0, 1, nt, dtype=np.float32)
XX, TT = np.meshgrid(x_g, t_g)
xt_eval = torch.from_numpy(np.column_stack([XX.ravel(), TT.ravel()]))
model_burg.eval()
with torch.no_grad():
    u_pred_burg = model_burg(xt_eval).numpy().reshape(nt, nx)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="#0d1117")
fig.suptitle("Burgers 1D PINN — u(x, t)", color=ACCENT, fontsize=13, fontweight="bold")

# Prediction heatmap
ax = axes[0]; dark_ax(ax)
im = ax.contourf(XX, TT, u_pred_burg, levels=60, cmap=CMAP_MAIN)
fig.colorbar(im, ax=ax, pad=0.02).ax.tick_params(colors="#8b949e", labelsize=7)
ax.set_title("PINN prediction u(x,t)", color="#e6edf3", fontsize=10)
ax.set_xlabel("x", color="#8b949e"); ax.set_ylabel("t", color="#8b949e")

# Slices at t = 0.25, 0.50, 0.75
ax = axes[1]; dark_ax(ax)
colors_sl = [ACCENT, GREEN, ORANGE]
for ti, col in zip([0.25, 0.50, 0.75], colors_sl):
    idx = np.argmin(np.abs(t_g - ti))
    ax.plot(x_g, u_pred_burg[idx], color=col, linewidth=1.5, label=f"t={ti}")
ax.axhline(0, color="#30363d", lw=0.5)
ax.set_title("Solution slices", color="#e6edf3", fontsize=10)
ax.set_xlabel("x", color="#8b949e"); ax.set_ylabel("u", color="#8b949e")
ax.legend(fontsize=9, framealpha=0.2)
ax.grid(True, color="#30363d", lw=0.4, alpha=0.6)

# Loss curve
ax = axes[2]; dark_ax(ax)
ax.semilogy(hist_burg, color=ACCENT, lw=1.5)
ax.set_title("Training loss", color="#e6edf3", fontsize=10)
ax.set_xlabel("Epoch", color="#8b949e"); ax.set_ylabel("Loss", color="#8b949e")
ax.grid(True, color="#30363d", lw=0.4, alpha=0.6)
ax.text(0.98, 0.05, f"Final: {hist_burg[-1]:.2e}", transform=ax.transAxes,
        ha="right", va="bottom", color=GREEN, fontsize=9)

plt.tight_layout(pad=0.8)
fig.savefig(os.path.join(OUT, "burgers_1d.png"), dpi=140,
            bbox_inches="tight", facecolor="#0d1117")
plt.close(fig)
print("Saved: burgers_1d.png")


# ═════════════════════════════════════════════════════════════════════════════
# PROBLEM 2 — Heat equation 2D steady (Poisson with source)
#   -Δu = f(x,y),   u = 0 on ∂Ω,   Ω = [0,1]²
#   Manufactured: u* = sin(πx)·sin(πy)  →  f = 2π²·sin(πx)·sin(πy)
# ═════════════════════════════════════════════════════════════════════════════

def heat2d_residual(model, xy):
    xy = xy.requires_grad_(True)
    u  = model(xy)
    grads = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0]
    ux, uy = grads[:, 0:1], grads[:, 1:2]
    uxx = torch.autograd.grad(ux, xy, torch.ones_like(ux), create_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(uy, xy, torch.ones_like(uy), create_graph=True)[0][:, 1:2]
    f   = 2 * (np.pi**2) * torch.sin(np.pi * xy[:, 0:1]) * torch.sin(np.pi * xy[:, 1:2])
    return (-(uxx + uyy) - f) ** 2

rng2  = np.random.default_rng(7)
xy_col_np = rng2.uniform(0.0, 1.0, (2500, 2)).astype(np.float32)
xy_col = torch.from_numpy(xy_col_np)

# BC: 4 edges u=0
nb = 100
x_bc_np = np.concatenate([
    np.column_stack([np.linspace(0,1,nb), np.zeros(nb)]),
    np.column_stack([np.linspace(0,1,nb), np.ones(nb)]),
    np.column_stack([np.zeros(nb), np.linspace(0,1,nb)]),
    np.column_stack([np.ones(nb),  np.linspace(0,1,nb)]),
], axis=0).astype(np.float32)
xy_bc = torch.from_numpy(x_bc_np)
u_bc_heat = torch.zeros(xy_bc.shape[0], 1)

print("Training Heat 2D...")
model_heat = MLP(2, 1, hidden=64, depth=4).to(DEVICE)
hist_heat  = train(model_heat, xy_col, None, heat2d_residual,
                   epochs=3000, lr=1e-3,
                   x_bc=xy_bc, y_bc=u_bc_heat, bc_weight=20.0)

# Evaluate
n_e = 100
xs = np.linspace(0, 1, n_e, dtype=np.float32)
XE, YE = np.meshgrid(xs, xs)
xy_eval = torch.from_numpy(np.column_stack([XE.ravel(), YE.ravel()]))
model_heat.eval()
with torch.no_grad():
    u_pred_heat = model_heat(xy_eval).numpy().reshape(n_e, n_e)

u_true_heat  = np.sin(np.pi * XE) * np.sin(np.pi * YE)
err_heat     = np.abs(u_pred_heat - u_true_heat)
rel_l2_heat  = np.linalg.norm(err_heat) / np.linalg.norm(u_true_heat)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="#0d1117")
fig.suptitle("Heat Equation 2D PINN — −Δu = f, u|∂Ω = 0",
             color=ACCENT, fontsize=13, fontweight="bold")

for i, (title, field, cmap_i) in enumerate([
    ("True: sin(πx)sin(πy)", u_true_heat, CMAP_MAIN),
    ("PINN prediction",      u_pred_heat, CMAP_MAIN),
    (f"|Error|  (rel-L² = {rel_l2_heat:.3f})", err_heat, CMAP_ERR),
]):
    ax = axes[i]; dark_ax(ax)
    cf = ax.contourf(XE, YE, field, levels=50, cmap=cmap_i)
    fig.colorbar(cf, ax=ax, pad=0.02).ax.tick_params(colors="#8b949e", labelsize=7)
    ax.set_title(title, color="#e6edf3", fontsize=10)
    ax.set_xlabel("x", color="#8b949e"); ax.set_ylabel("y", color="#8b949e")
    ax.set_aspect("equal")

plt.tight_layout(pad=0.8)
fig.savefig(os.path.join(OUT, "heat_2d.png"), dpi=140,
            bbox_inches="tight", facecolor="#0d1117")
plt.close(fig)
print(f"Saved: heat_2d.png  (rel-L²={rel_l2_heat:.4f})")


# ═════════════════════════════════════════════════════════════════════════════
# PROBLEM 3 — Poisson 2D with multiple point sources
#   -Δu = f,  Dirichlet BCs,  Ω = [−1,1]²
#   Two Gaussian sources at (±0.5, 0)
# ═════════════════════════════════════════════════════════════════════════════

def gaussian_source(xy, cx, cy, amp=6.0, sigma=0.15):
    r2 = (xy[:, 0:1] - cx)**2 + (xy[:, 1:2] - cy)**2
    return amp * torch.exp(-r2 / (2 * sigma**2))

def poisson_residual(model, xy):
    xy = xy.requires_grad_(True)
    u  = model(xy)
    grads = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0]
    ux, uy = grads[:, 0:1], grads[:, 1:2]
    uxx = torch.autograd.grad(ux, xy, torch.ones_like(ux), create_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(uy, xy, torch.ones_like(uy), create_graph=True)[0][:, 1:2]
    f   = gaussian_source(xy, 0.5, 0.0) - gaussian_source(xy, -0.5, 0.0)
    return (-(uxx + uyy) - f) ** 2

rng3  = np.random.default_rng(13)
xy_col3 = torch.from_numpy(rng3.uniform(-1, 1, (3000, 2)).astype(np.float32))

# BC: u = 0 on boundary
nb3 = 80
x_bc3_np = np.concatenate([
    np.column_stack([np.linspace(-1,1,nb3), -np.ones(nb3)]),
    np.column_stack([np.linspace(-1,1,nb3),  np.ones(nb3)]),
    np.column_stack([-np.ones(nb3), np.linspace(-1,1,nb3)]),
    np.column_stack([ np.ones(nb3), np.linspace(-1,1,nb3)]),
], axis=0).astype(np.float32)
xy_bc3 = torch.from_numpy(x_bc3_np)
u_bc3  = torch.zeros(xy_bc3.shape[0], 1)

print("Training Poisson 2D...")
model_poisson = MLP(2, 1, hidden=64, depth=4).to(DEVICE)
hist_poisson  = train(model_poisson, xy_col3, None, poisson_residual,
                      epochs=3000, lr=1e-3,
                      x_bc=xy_bc3, y_bc=u_bc3, bc_weight=15.0)

# Evaluate
n_e3 = 120
xs3 = np.linspace(-1, 1, n_e3, dtype=np.float32)
XE3, YE3 = np.meshgrid(xs3, xs3)
xy_eval3 = torch.from_numpy(np.column_stack([XE3.ravel(), YE3.ravel()]))
model_poisson.eval()
with torch.no_grad():
    u_pred_poisson = model_poisson(xy_eval3).numpy().reshape(n_e3, n_e3)

# Source field for reference
f_src = (6.0 * np.exp(-((XE3 - 0.5)**2 + YE3**2) / (2*0.15**2))
       - 6.0 * np.exp(-((XE3 + 0.5)**2 + YE3**2) / (2*0.15**2)))

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="#0d1117")
fig.suptitle("Poisson 2D PINN — Two Gaussian Sources",
             color=ACCENT, fontsize=13, fontweight="bold")

for i, (title, field, cmap_i) in enumerate([
    ("Source f(x,y)",       f_src,         "RdBu_r"),
    ("PINN solution u",     u_pred_poisson, CMAP_MAIN),
    ("PDE residual |−Δu−f|", None, CMAP_ERR),
]):
    ax = axes[i]; dark_ax(ax)
    if i < 2:
        cf = ax.contourf(XE3, YE3, field, levels=50, cmap=cmap_i)
    else:
        # Compute residual numerically
        dx = xs3[1] - xs3[0]
        from scipy.ndimage import laplace as sp_laplace
        lap_u = sp_laplace(u_pred_poisson) / dx**2
        resid = np.abs(-lap_u - f_src)
        cf = ax.contourf(XE3, YE3, resid, levels=50, cmap=cmap_i)
        title = f"|−Δu−f|  (max={resid.max():.3f})"
    fig.colorbar(cf, ax=ax, pad=0.02).ax.tick_params(colors="#8b949e", labelsize=7)
    ax.set_title(title, color="#e6edf3", fontsize=10)
    ax.set_xlabel("x", color="#8b949e"); ax.set_ylabel("y", color="#8b949e")
    ax.set_aspect("equal")

plt.tight_layout(pad=0.8)
fig.savefig(os.path.join(OUT, "poisson_2d.png"), dpi=140,
            bbox_inches="tight", facecolor="#0d1117")
plt.close(fig)
print("Saved: poisson_2d.png")


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY FIGURE — all three loss curves
# ═════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0d1117")
dark_ax(ax)
ax.semilogy(hist_burg,    color=ACCENT,  lw=1.6, label="Burgers 1D")
ax.semilogy(hist_heat,    color=GREEN,   lw=1.6, label="Heat 2D (steady)")
ax.semilogy(hist_poisson, color=ORANGE,  lw=1.6, label="Poisson 2D (sources)")
ax.set_xlabel("Epoch", color="#e6edf3", fontsize=11)
ax.set_ylabel("Total loss", color="#e6edf3", fontsize=11)
ax.set_title("PINN training convergence — 3 benchmark problems",
             color=ACCENT, fontsize=12, fontweight="bold")
ax.legend(fontsize=10, framealpha=0.2)
ax.grid(True, color="#30363d", lw=0.5, alpha=0.6)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "common_pinns_losses.png"), dpi=140,
            bbox_inches="tight", facecolor="#0d1117")
plt.close(fig)
print("Saved: common_pinns_losses.png")

print(f"\nAll outputs in: {OUT}")
