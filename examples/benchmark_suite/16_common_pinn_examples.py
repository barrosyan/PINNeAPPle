"""Classic PINN benchmarks — Burgers 1D, Heat 2D, Poisson 2D.

Each problem has an exact reference solution.
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
RED    = "#f78166"

OUT = os.path.join(os.path.dirname(__file__), "_out", "common_pinns")
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
    def __init__(self, in_d, out_d=1, h=80, d=5):
        super().__init__()
        layers = [nn.Linear(in_d, h), nn.Tanh()]
        for _ in range(d - 1):
            layers += [nn.Linear(h, h), nn.Tanh()]
        layers.append(nn.Linear(h, out_d))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


# ═════════════════════════════════════════════════════════════════════════════
# BURGERS 1D — u_t + u·u_x = ν u_xx
# u(x,0) = −sin(πx),  u(±1,t) = 0,  ν = 0.01/π
# ═════════════════════════════════════════════════════════════════════════════

NU   = 0.01 / np.pi
rng  = np.random.default_rng(42)

xt_col = torch.from_numpy(rng.uniform([-1, 0], [1, 1], (4000, 2)).astype(np.float32))

x_ic_np = rng.uniform(-1, 1, (400, 1)).astype(np.float32)
xt_ic   = torch.from_numpy(np.hstack([x_ic_np, np.zeros_like(x_ic_np)]))
u_ic    = torch.from_numpy(-np.sin(np.pi * x_ic_np))

t_bc_np = rng.uniform(0, 1, (200, 1)).astype(np.float32)
xt_bc   = torch.from_numpy(np.vstack([
    np.hstack([-np.ones_like(t_bc_np), t_bc_np]),
    np.hstack([ np.ones_like(t_bc_np), t_bc_np]),
]))
u_bc    = torch.zeros(xt_bc.shape[0], 1)

x_all_bc = torch.cat([xt_ic, xt_bc])
y_all_bc = torch.cat([u_ic, u_bc])

def burgers_res(model, xt):
    xt = xt.requires_grad_(True)
    u  = model(xt)
    g  = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0]
    u_t, u_x = g[:, 1:2], g[:, 0:1]
    u_xx = torch.autograd.grad(u_x, xt, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    return (u_t + u * u_x - NU * u_xx) ** 2

print("Training Burgers 1D…")
model_b = MLP(2, 1, h=80, d=5)
opt_b   = torch.optim.Adam(model_b.parameters(), lr=1e-3)
sch_b   = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, 4000, eta_min=1e-4)
hist_b  = []

for ep in range(4000):
    opt_b.zero_grad()
    loss = burgers_res(model_b, xt_col).mean() + \
           10.0 * nn.functional.mse_loss(model_b(x_all_bc), y_all_bc)
    loss.backward()
    nn.utils.clip_grad_norm_(model_b.parameters(), 1.0)
    opt_b.step(); sch_b.step()
    hist_b.append(float(loss.detach()))

print(f"  Burgers final loss: {hist_b[-1]:.4e}")

NX, NT = 200, 200
x_g = np.linspace(-1, 1, NX, dtype=np.float32)
t_g = np.linspace(0,  1, NT, dtype=np.float32)
XX_b, TT = np.meshgrid(x_g, t_g)
model_b.eval()
with torch.no_grad():
    u_burg = model_b(torch.from_numpy(
        np.column_stack([XX_b.ravel(), TT.ravel()])
    )).numpy().reshape(NT, NX)

# ── Shock front x-position estimate per time
shock_x = []
for ti in range(NT):
    grad = np.gradient(u_burg[ti], x_g)
    idx  = np.argmin(grad)
    shock_x.append(x_g[idx])
shock_x = np.array(shock_x)

fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG,
                          gridspec_kw={"width_ratios": [1.1, 1]})
fig.suptitle("Burgers Equation 1D — u_t + u·u_x = ν u_xx  (ν = 0.01/π)",
             color=ACCENT, fontsize=15, fontweight="bold", y=1.01)

# Heatmap
ax = axes[0]; dark_ax(ax, fc=SURF)
cf = ax.contourf(XX_b, TT, u_burg, levels=80, cmap="RdBu_r",
                 vmin=-1.0, vmax=1.0)
cs = ax.contour(XX_b, TT, u_burg, levels=[-0.5, 0, 0.5],
                colors=[MUTED], linewidths=0.7, alpha=0.7)
ax.clabel(cs, fmt="%.1f", colors=MUTED, fontsize=8)
ax.plot(shock_x, t_g, color=ORANGE, lw=2.0, linestyle="--",
        label="Shock locus")
ax.set_xlabel("x"); ax.set_ylabel("t")
ax.set_title("u(x, t) — space-time heatmap", fontsize=13, pad=8)
ax.legend(fontsize=10, framealpha=0.3, loc="upper left")
cb = fig.colorbar(cf, ax=ax, pad=0.015, fraction=0.04, shrink=0.9)
cb.set_label("u(x, t)", color=MUTED, fontsize=10)
cb.ax.tick_params(colors=MUTED, labelsize=9)
ax.annotate("Shock\nformation", xy=(shock_x[NT // 2], t_g[NT // 2]),
            xytext=(-0.7, 0.6), color=ORANGE, fontsize=10,
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))

# Time slices
ax = axes[1]; dark_ax(ax, fc=SURF)
colors_s = [ACCENT, GREEN, ORANGE, RED]
t_slices = [0.0, 0.25, 0.50, 0.75]
for t_val, col in zip(t_slices, colors_s):
    idx = np.argmin(np.abs(t_g - t_val))
    lw  = 2.2 if t_val > 0 else 1.5
    ls  = "-" if t_val > 0 else "--"
    ax.plot(x_g, u_burg[idx], color=col, lw=lw, ls=ls,
            label=f"t = {t_val:.2f}")
ax.axhline(0, color=BORD, lw=0.7)
ax.axvline(0, color=BORD, lw=0.5, ls=":")
ax.set_xlim(-1, 1); ax.set_ylim(-1.1, 1.1)
ax.set_xlabel("x"); ax.set_ylabel("u(x, t)")
ax.set_title("Solution profiles at t = 0, 0.25, 0.50, 0.75", fontsize=13, pad=8)
ax.legend(fontsize=11, framealpha=0.3)
ax.grid(True, color=BORD, lw=0.5, alpha=0.7)
ax.annotate("Shock develops\n& steepens", xy=(0.05, 0.2),
            xytext=(0.3, 0.65), color=ORANGE, fontsize=10,
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1))

fin_loss_b = hist_b[-1]
fig.text(0.99, 0.01, f"Final training loss: {fin_loss_b:.2e}",
         ha="right", va="bottom", color=MUTED, fontsize=10)
plt.tight_layout(pad=1.2)
fig.savefig(os.path.join(OUT, "burgers_1d.png"), dpi=160,
            bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved burgers_1d.png")


# ═════════════════════════════════════════════════════════════════════════════
# HEAT EQUATION 2D  −Δu = f,  u|∂Ω = 0,  exact u* = sin(πx)sin(πy)
# ═════════════════════════════════════════════════════════════════════════════

rng2    = np.random.default_rng(7)
xy_col2 = torch.from_numpy(rng2.uniform(0, 1, (5000, 2)).astype(np.float32))
nb2     = 150
bc_pts  = np.vstack([
    np.column_stack([np.linspace(0,1,nb2), np.zeros(nb2)]),
    np.column_stack([np.linspace(0,1,nb2), np.ones(nb2)]),
    np.column_stack([np.zeros(nb2), np.linspace(0,1,nb2)]),
    np.column_stack([np.ones(nb2),  np.linspace(0,1,nb2)]),
]).astype(np.float32)
xy_bc2 = torch.from_numpy(bc_pts)
u_bc2  = torch.zeros(xy_bc2.shape[0], 1)

def heat_res(model, xy):
    xy = xy.requires_grad_(True)
    u  = model(xy)
    g  = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0]
    ux, uy = g[:, 0:1], g[:, 1:2]
    uxx = torch.autograd.grad(ux, xy, torch.ones_like(ux), create_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(uy, xy, torch.ones_like(uy), create_graph=True)[0][:, 1:2]
    f   = 2 * np.pi**2 * torch.sin(np.pi * xy[:, 0:1]) * torch.sin(np.pi * xy[:, 1:2])
    return (-(uxx + uyy) - f) ** 2

print("Training Heat 2D…")
model_h = MLP(2, 1, h=80, d=5)
opt_h   = torch.optim.Adam(model_h.parameters(), lr=1e-3)
sch_h   = torch.optim.lr_scheduler.CosineAnnealingLR(opt_h, 4000, eta_min=1e-4)
hist_h  = []

for ep in range(4000):
    opt_h.zero_grad()
    loss = heat_res(model_h, xy_col2).mean() + \
           20.0 * nn.functional.mse_loss(model_h(xy_bc2), u_bc2)
    loss.backward()
    nn.utils.clip_grad_norm_(model_h.parameters(), 1.0)
    opt_h.step(); sch_h.step()
    hist_h.append(float(loss.detach()))

print(f"  Heat 2D final loss: {hist_h[-1]:.4e}")

NE = 120
xs_h = np.linspace(0, 1, NE, dtype=np.float32)
XE, YE = np.meshgrid(xs_h, xs_h)
model_h.eval()
with torch.no_grad():
    u_heat = model_h(torch.from_numpy(
        np.column_stack([XE.ravel(), YE.ravel()])
    )).numpy().reshape(NE, NE)

u_true_h  = np.sin(np.pi * XE) * np.sin(np.pi * YE)
err_h     = np.abs(u_heat - u_true_h)
rel_l2_h  = np.linalg.norm(err_h) / np.linalg.norm(u_true_h)
max_err_h = err_h.max()

fig, axes = plt.subplots(1, 3, figsize=(17, 6.5), facecolor=BG)
fig.suptitle("Heat Equation 2D — −Δu = 2π²sin(πx)sin(πy),  u|∂Ω = 0\n"
             "Exact solution: u*(x,y) = sin(πx) · sin(πy)",
             color=ACCENT, fontsize=14, fontweight="bold")

v_shared = dict(vmin=0, vmax=1.0)
for i, (title, field, cmap_i, extra) in enumerate([
    ("Exact solution u*", u_true_h,  "plasma", v_shared),
    ("PINN prediction",   u_heat,    "plasma", v_shared),
    (f"|Error|   rel-L² = {rel_l2_h:.4f}",  err_h, "hot_r",
     dict(vmin=0, vmax=max_err_h)),
]):
    ax = axes[i]; dark_ax(ax, fc=SURF)
    cf = ax.contourf(XE, YE, field, levels=60, cmap=cmap_i, **extra)
    # Isocurves on true and predicted
    if i < 2:
        ax.contour(XE, YE, field, levels=8,
                   colors=[BORD], linewidths=0.6, alpha=0.8)
    cb = fig.colorbar(cf, ax=ax, pad=0.015, fraction=0.05, shrink=0.9)
    cb.ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")

axes[2].text(0.97, 0.04,
             f"max |err| = {max_err_h:.4f}",
             transform=axes[2].transAxes,
             ha="right", va="bottom", color=ORANGE, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=SURF, edgecolor=BORD, alpha=0.85))

plt.tight_layout(pad=1.2)
fig.savefig(os.path.join(OUT, "heat_2d.png"), dpi=160,
            bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Saved heat_2d.png  (rel-L² = {rel_l2_h:.4f})")


# ═════════════════════════════════════════════════════════════════════════════
# POISSON 2D — two Gaussian sources, u|∂Ω = 0
# No exact solution → show source / PINN solution / gradient field
# ═════════════════════════════════════════════════════════════════════════════

rng3    = np.random.default_rng(13)
xy_col3 = torch.from_numpy(rng3.uniform(-1, 1, (5000, 2)).astype(np.float32))
nb3     = 120
bc3_pts = np.vstack([
    np.column_stack([np.linspace(-1,1,nb3), -np.ones(nb3)]),
    np.column_stack([np.linspace(-1,1,nb3),  np.ones(nb3)]),
    np.column_stack([-np.ones(nb3), np.linspace(-1,1,nb3)]),
    np.column_stack([ np.ones(nb3), np.linspace(-1,1,nb3)]),
]).astype(np.float32)
xy_bc3 = torch.from_numpy(bc3_pts)
u_bc3  = torch.zeros(xy_bc3.shape[0], 1)

AMP, SIG = 8.0, 0.15

def gaussian_src_t(xy):
    r1 = (xy[:, 0:1] - 0.5)**2 + xy[:, 1:2]**2
    r2 = (xy[:, 0:1] + 0.5)**2 + xy[:, 1:2]**2
    return AMP * torch.exp(-r1 / (2*SIG**2)) - AMP * torch.exp(-r2 / (2*SIG**2))

def poisson_res(model, xy):
    xy = xy.requires_grad_(True)
    u  = model(xy)
    g  = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0]
    ux, uy = g[:, 0:1], g[:, 1:2]
    uxx = torch.autograd.grad(ux, xy, torch.ones_like(ux), create_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(uy, xy, torch.ones_like(uy), create_graph=True)[0][:, 1:2]
    return (-(uxx + uyy) - gaussian_src_t(xy)) ** 2

print("Training Poisson 2D…")
model_p = MLP(2, 1, h=80, d=5)
opt_p   = torch.optim.Adam(model_p.parameters(), lr=1e-3)
sch_p   = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p, 4000, eta_min=1e-4)
hist_p  = []

for ep in range(4000):
    opt_p.zero_grad()
    loss = poisson_res(model_p, xy_col3).mean() + \
           15.0 * nn.functional.mse_loss(model_p(xy_bc3), u_bc3)
    loss.backward()
    nn.utils.clip_grad_norm_(model_p.parameters(), 1.0)
    opt_p.step(); sch_p.step()
    hist_p.append(float(loss.detach()))

print(f"  Poisson 2D final loss: {hist_p[-1]:.4e}")

NE3 = 120
xs3 = np.linspace(-1, 1, NE3, dtype=np.float32)
XE3, YE3 = np.meshgrid(xs3, xs3)
model_p.eval()
with torch.no_grad():
    u_pois = model_p(torch.from_numpy(
        np.column_stack([XE3.ravel(), YE3.ravel()])
    )).numpy().reshape(NE3, NE3)

f_src = (AMP * np.exp(-((XE3-0.5)**2+YE3**2)/(2*SIG**2))
       - AMP * np.exp(-((XE3+0.5)**2+YE3**2)/(2*SIG**2)))

# Gradient field of u (autograd on a coarser grid for quiver)
NQ = 24
xq = np.linspace(-1, 1, NQ, dtype=np.float32)
XQ, YQ = np.meshgrid(xq, xq)
xyq = torch.from_numpy(np.column_stack([XQ.ravel(), YQ.ravel()])).requires_grad_(True)
u_q = model_p(xyq)
g_q = torch.autograd.grad(u_q, xyq, torch.ones_like(u_q))[0].detach().numpy()
UQ  = g_q[:, 0].reshape(NQ, NQ)
VQ  = g_q[:, 1].reshape(NQ, NQ)
speed_q = np.sqrt(UQ**2 + VQ**2) + 1e-8

fig, axes = plt.subplots(1, 3, figsize=(17, 6.5), facecolor=BG)
fig.suptitle("Poisson Equation 2D — −Δu = f(x,y),  u|∂Ω = 0\n"
             "Two Gaussian sources:  f = A·e^{−r₊²/2σ²} − A·e^{−r₋²/2σ²}",
             color=ACCENT, fontsize=14, fontweight="bold")

# Source field
ax = axes[0]; dark_ax(ax, fc=SURF)
vlim = np.abs(f_src).max()
cf = ax.contourf(XE3, YE3, f_src, levels=60, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
ax.contour(XE3, YE3, f_src, levels=8, colors=[BORD], linewidths=0.6, alpha=0.7)
ax.plot( 0.5, 0.0, "*", color=RED,   ms=14, zorder=5, label="Source (+)")
ax.plot(-0.5, 0.0, "*", color=ACCENT,ms=14, zorder=5, label="Sink (−)")
cb = fig.colorbar(cf, ax=ax, pad=0.015, fraction=0.05, shrink=0.9)
cb.ax.tick_params(colors=MUTED, labelsize=9)
ax.set_title("Source field f(x,y)", fontsize=12, pad=8)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
ax.legend(fontsize=10, framealpha=0.3, loc="upper right")

# PINN solution
ax = axes[1]; dark_ax(ax, fc=SURF)
cf = ax.contourf(XE3, YE3, u_pois, levels=60, cmap="plasma")
ax.contour(XE3, YE3, u_pois, levels=10, colors=[BORD], linewidths=0.6, alpha=0.7)
cb = fig.colorbar(cf, ax=ax, pad=0.015, fraction=0.05, shrink=0.9)
cb.ax.tick_params(colors=MUTED, labelsize=9)
ax.set_title("PINN solution u(x,y)", fontsize=12, pad=8)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")

# Gradient ∇u field (quiver over contour)
ax = axes[2]; dark_ax(ax, fc=SURF)
cf = ax.contourf(XE3, YE3, u_pois, levels=50, cmap="plasma", alpha=0.65)
ax.quiver(XQ, YQ, UQ / speed_q, VQ / speed_q,
          speed_q, cmap="hot_r", pivot="mid",
          scale=NQ * 1.6, headwidth=3.5, headlength=4.0, alpha=0.9)
cb = fig.colorbar(cf, ax=ax, pad=0.015, fraction=0.05, shrink=0.9)
cb.ax.tick_params(colors=MUTED, labelsize=9)
ax.set_title("PINN gradient field ∇u(x,y)", fontsize=12, pad=8)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
ax.plot( 0.5, 0.0, "*", color=RED,   ms=12, zorder=5)
ax.plot(-0.5, 0.0, "*", color=ACCENT,ms=12, zorder=5)

plt.tight_layout(pad=1.2)
fig.savefig(os.path.join(OUT, "poisson_2d.png"), dpi=160,
            bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved poisson_2d.png")

print(f"\nAll outputs in: {OUT}")
