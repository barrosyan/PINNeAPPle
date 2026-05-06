"""NACA 0012 aerodynamic surrogate — Joukowski mapping + PINN.

Reference solution: Joukowski conformal mapping (exact potential flow).
PINN learns u, v, Cp over the full 2D domain.

Three output figures:
  naca0012_flow.png        — velocity field + streamlines / Cp field
  naca0012_pinn.png        — true vs PINN u-field + % relative error
  naca0012_surface_cp.png  — Cp distribution on airfoil surface
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.path import Path as MplPath
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
SURF2= "#1c2330"
BORD = "#30363d"
MUTED= "#8b949e"
TEXT = "#e6edf3"
ACCENT = "#58a6ff"
GREEN  = "#3fb950"
ORANGE = "#ffa657"
PURPLE = "#d2a8ff"

OUT = os.path.join(os.path.dirname(__file__), "_out", "naca0012")
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Joukowski geometry & exact potential flow
# ─────────────────────────────────────────────────────────────────────────────

ALPHA  = 5.0
U_INF  = 1.0
C_J    = 1.0

def joukowski(zeta, c=C_J):
    return zeta + c**2 / zeta

def make_airfoil(n=600):
    eps   = 0.04
    alpha = np.deg2rad(ALPHA)
    ctr   = complex(-eps, eps)
    R     = abs(1.0 - ctr) + 0.008
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    zeta  = ctr + R * np.exp(1j * theta)
    return joukowski(zeta), ctr, R

def z_to_zeta(z):
    disc = np.sqrt(z**2 - 4 * C_J**2 + 0j)
    z1, z2 = (z + disc) / 2, (z - disc) / 2
    return np.where(np.abs(z1) >= np.abs(z2), z1, z2)

def potential_flow(zeta, ctr, R):
    alpha = np.deg2rad(ALPHA)
    Gamma = 4 * np.pi * U_INF * R * np.sin(alpha)
    s     = zeta - ctr
    df    = U_INF * (np.exp(-1j * alpha) - np.exp(1j * alpha) * R**2 / s**2) \
            + 1j * Gamma / (2 * np.pi * s)
    dz    = 1.0 - C_J**2 / zeta**2
    w     = df / dz
    return np.real(w), -np.imag(w)

z_foil, ctr, R_circ = make_airfoil()

# Evaluation grid
N = 100
XX, YY = np.meshgrid(np.linspace(-2.5, 2.5, N), np.linspace(-2.0, 2.0, N))
ZZ = XX + 1j * YY
ZETA = z_to_zeta(ZZ)
U_REF, V_REF = potential_flow(ZETA, ctr, R_circ)
SPEED = np.sqrt(U_REF**2 + V_REF**2)
CP_REF = 1.0 - SPEED**2 / U_INF**2

foil_path = MplPath(np.column_stack([z_foil.real, z_foil.imag]))
inside = foil_path.contains_points(
    np.column_stack([ZZ.real.ravel(), ZZ.imag.ravel()])
).reshape(N, N)

U_REF[inside] = np.nan
V_REF[inside] = np.nan
CP_REF[inside] = np.nan

# ─────────────────────────────────────────────────────────────────────────────
# 2. FIGURE A — Velocity field + Cp field
# ─────────────────────────────────────────────────────────────────────────────

def dark_ax(ax, facecolor=BG):
    ax.set_facecolor(facecolor)
    for sp in ax.spines.values():
        sp.set_color(BORD)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)

# — Velocity magnitude
spd = np.where(inside, np.nan, SPEED)
vmax_spd = np.nanpercentile(spd, 97)
cf0 = ax0.contourf(XX, YY, spd, levels=60, cmap="plasma", vmin=0, vmax=vmax_spd)
ax0.streamplot(XX, YY,
               np.where(inside, 0.0, np.nan_to_num(U_REF, nan=0)),
               np.where(inside, 0.0, np.nan_to_num(V_REF, nan=0)),
               color="white", linewidth=0.8, density=1.4,
               arrowsize=1.0, arrowstyle="-|>",
               integration_direction="forward")
ax0.fill(z_foil.real, z_foil.imag, color=SURF, zorder=5, linewidth=0)
ax0.plot(z_foil.real, z_foil.imag, color=ACCENT, linewidth=1.4, zorder=6)
cb0 = fig.colorbar(cf0, ax=ax0, pad=0.015, fraction=0.04, shrink=0.85)
cb0.set_label("|V| / U∞", color=MUTED, fontsize=11)
cb0.ax.tick_params(colors=MUTED, labelsize=9)
ax0.set_title("Velocity magnitude + streamlines", fontsize=14, fontweight="bold", pad=10)
ax0.set_xlabel("x / c"); ax0.set_ylabel("y / c")
ax0.set_aspect("equal"); ax0.set_xlim(-2.5, 2.5); ax0.set_ylim(-2.0, 2.0)
# Annotate key regions
ax0.annotate("Accelerated\nflow", xy=(0.3, 0.35), color=GREEN, fontsize=9, ha="center")
ax0.annotate("Stagnation\npoint", xy=(-1.05, 0.04), color=ORANGE,
             fontsize=9, xytext=(-1.9, 0.5),
             arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.9))
ax0.annotate("Wake", xy=(1.5, 0.0), color=PURPLE, fontsize=9, ha="center")
dark_ax(ax0)

# — Cp field
cp_plot = np.clip(np.where(inside, np.nan, CP_REF), -3, 1.5)
cf1 = ax1.contourf(XX, YY, cp_plot, levels=60, cmap="RdBu_r",
                   norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-3, vmax=1.5))
ax1.fill(z_foil.real, z_foil.imag, color=SURF, zorder=5, linewidth=0)
ax1.plot(z_foil.real, z_foil.imag, color=ORANGE, linewidth=1.4, zorder=6)
cb1 = fig.colorbar(cf1, ax=ax1, pad=0.015, fraction=0.04, shrink=0.85)
cb1.set_label("Cp", color=MUTED, fontsize=11)
cb1.ax.tick_params(colors=MUTED, labelsize=9)
ax1.set_title("Pressure coefficient Cp  (α = 5°)", fontsize=14, fontweight="bold", pad=10)
ax1.set_xlabel("x / c"); ax1.set_ylabel("y / c")
ax1.set_aspect("equal"); ax1.set_xlim(-2.5, 2.5); ax1.set_ylim(-2.0, 2.0)
ax1.annotate("Suction\npeak", xy=(-0.8, 0.12), color=TEXT, fontsize=9,
             xytext=(-1.8, 0.8),
             arrowprops=dict(arrowstyle="->", color=TEXT, lw=0.9))
dark_ax(ax1)

fig.suptitle("NACA 0012 — Joukowski Potential Flow  (inviscid, incompressible)",
             color=ACCENT, fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout(pad=1.5)
fig.savefig(os.path.join(OUT, "naca0012_flow.png"), dpi=160,
            bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved naca0012_flow.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PINN surrogate — (x, y) → (u, v, Cp)
# ─────────────────────────────────────────────────────────────────────────────

class PINN(nn.Module):
    def __init__(self, h=128, d=6):
        super().__init__()
        layers = [nn.Linear(2, h), nn.Tanh()]
        for _ in range(d - 1):
            layers += [nn.Linear(h, h), nn.Tanh()]
        layers.append(nn.Linear(h, 3))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# Training data (exterior points only)
mask = ~inside
xy = np.column_stack([XX[mask].astype(np.float32),
                      YY[mask].astype(np.float32)])
uv = np.column_stack([U_REF[mask].astype(np.float32),
                      V_REF[mask].astype(np.float32),
                      CP_REF[mask].astype(np.float32)])

# remove NaN rows (boundaries of singularity)
valid = np.isfinite(uv).all(axis=1)
xy, uv = xy[valid], uv[valid]

xy_t = torch.from_numpy(xy)
uv_t = torch.from_numpy(uv)
xy_mu, xy_s = xy_t.mean(0), xy_t.std(0) + 1e-8
uv_mu, uv_s = uv_t.mean(0), uv_t.std(0) + 1e-8
xy_n = (xy_t - xy_mu) / xy_s
uv_n = (uv_t - uv_mu) / uv_s

DEVICE = "cpu"
model = PINN(128, 6).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4000, eta_min=5e-5)
BATCH = 1024
hist  = []

print("Training PINN…")
for ep in range(4000):
    idx  = torch.randperm(xy_n.shape[0])[:BATCH]
    xb, yb = xy_n[idx], uv_n[idx]
    opt.zero_grad()
    loss = nn.functional.mse_loss(model(xb), yb)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    hist.append(float(loss))

print(f"  Final loss: {hist[-1]:.4e}")

# Evaluate
model.eval()
with torch.no_grad():
    all_xy = torch.from_numpy(
        np.column_stack([XX.ravel().astype(np.float32),
                         YY.ravel().astype(np.float32)])
    )
    pred_n = model((all_xy - xy_mu) / xy_s).numpy()

pred   = pred_n * uv_s.numpy() + uv_mu.numpy()
U_pred = pred[:, 0].reshape(N, N)
V_pred = pred[:, 1].reshape(N, N)
CP_pred = pred[:, 2].reshape(N, N)
U_pred[inside] = np.nan
CP_pred[inside] = np.nan

# ─────────────────────────────────────────────────────────────────────────────
# 4. FIGURE B — u-field: true / PINN / % relative error
# ─────────────────────────────────────────────────────────────────────────────

# Relative % error: |pred - true| / max(|true|, eps) * 100
eps_rel = 0.05
U_rel_err = np.abs(U_pred - U_REF) / (np.abs(U_REF) + eps_rel) * 100
U_rel_err[inside] = np.nan
U_rel_err = np.clip(U_rel_err, 0, 30)   # cap at 30% for readability

fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
fig.suptitle("NACA 0012 PINN — Streamwise velocity u",
             color=ACCENT, fontsize=15, fontweight="bold")

u_ref_plot = np.where(inside, np.nan, U_REF)
vmin_u = np.nanpercentile(u_ref_plot, 2)
vmax_u = np.nanpercentile(u_ref_plot, 98)

for ax, field, title in zip(axes,
    [u_ref_plot, U_pred, U_rel_err],
    ["Reference (Joukowski)", "PINN prediction", "Relative error (%, capped 30%)"]):
    dark_ax(ax)
    cmap_i = "plasma" if "error" not in title.lower() else "hot_r"
    if "error" in title.lower():
        cf = ax.contourf(XX, YY, field, levels=50, cmap=cmap_i, vmin=0, vmax=30)
    else:
        cf = ax.contourf(XX, YY, field, levels=50, cmap=cmap_i, vmin=vmin_u, vmax=vmax_u)
    ax.fill(z_foil.real, z_foil.imag, color=SURF2, zorder=5, linewidth=0)
    ax.plot(z_foil.real, z_foil.imag, color=ACCENT, linewidth=1.2, zorder=6)
    cb = fig.colorbar(cf, ax=ax, pad=0.015, fraction=0.046, shrink=0.9)
    cb.ax.tick_params(colors=MUTED, labelsize=9)
    cb.set_label("u / U∞" if "error" not in title.lower() else "% error",
                 color=MUTED, fontsize=10)
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel("x / c"); ax.set_ylabel("y / c")
    ax.set_aspect("equal"); ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.0, 2.0)

# Compute and show summary metrics
u_ref_flat = U_REF[~inside & np.isfinite(U_REF)]
u_pred_flat = U_pred[~inside & np.isfinite(U_pred) & np.isfinite(U_REF)]
# align
fin = ~inside & np.isfinite(U_REF) & np.isfinite(U_pred)
rel_l2 = np.linalg.norm(U_pred[fin] - U_REF[fin]) / np.linalg.norm(U_REF[fin])
axes[2].text(0.02, 0.04, f"rel-L² = {rel_l2:.3f}", transform=axes[2].transAxes,
             color=GREEN, fontsize=11, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=SURF, edgecolor=BORD, alpha=0.9))

plt.tight_layout(pad=1.2)
fig.savefig(os.path.join(OUT, "naca0012_pinn.png"), dpi=160,
            bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved naca0012_pinn.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. FIGURE C — Surface Cp distribution, upper vs lower surface
# ─────────────────────────────────────────────────────────────────────────────

# Sample along parametric circle, separate upper (θ: π→0) and lower (θ: π→2π)
n_surf = 300
alpha_r = np.deg2rad(ALPHA)
Gamma   = 4 * np.pi * U_INF * R_circ * np.sin(alpha_r)

def surface_cp(theta_arr):
    zeta_s = ctr + R_circ * np.exp(1j * theta_arr)
    z_s    = joukowski(zeta_s)
    s      = zeta_s - ctr
    df     = U_INF * (np.exp(-1j * alpha_r) - np.exp(1j * alpha_r) * R_circ**2 / s**2) \
             + 1j * Gamma / (2 * np.pi * s)
    dz     = 1.0 - C_J**2 / zeta_s**2
    w      = df / dz
    speed  = np.abs(w)
    cp     = 1.0 - speed**2 / U_INF**2
    x_c    = (z_s.real - z_s.real.min()) / (z_s.real.max() - z_s.real.min() + 1e-10)
    return x_c, cp

# Upper surface: θ from π down to 0.12 (avoid leading-edge singularity)
# Lower surface: θ from π up to 2π-0.12 (avoid trailing-edge singularity)
# The Joukowski map has a branch point near θ=0 and θ=2π (trailing edge)
theta_upper = np.linspace(np.pi, 0.12, n_surf)
x_up, cp_up = surface_cp(theta_upper)

theta_lower = np.linspace(np.pi, 2 * np.pi - 0.12, n_surf)
x_lo, cp_lo = surface_cp(theta_lower)

# Display limits: remove only the immediate singularity at leading/trailing edge
CHORD_MIN, CHORD_MAX = 0.01, 0.98
CP_DISPLAY_MIN, CP_DISPLAY_MAX = -3.0, 1.5

def trim(x_c, cp_vals):
    mask = (x_c >= CHORD_MIN) & (x_c <= CHORD_MAX) & (np.abs(cp_vals) < 10.0)
    return x_c[mask], np.clip(cp_vals[mask], CP_DISPLAY_MIN, CP_DISPLAY_MAX)

x_up_t, cp_up_c  = trim(x_up, cp_up)
x_lo_t, cp_lo_c  = trim(x_lo, cp_lo)

# PINN Cp on surface
def pinn_cp_on_surface(x_np, y_np):
    xy_srf = torch.from_numpy(
        np.column_stack([x_np.astype(np.float32), y_np.astype(np.float32)])
    )
    model.eval()
    with torch.no_grad():
        out_n = model((xy_srf - xy_mu) / xy_s).numpy()
    return out_n[:, 2] * uv_s[2].item() + uv_mu[2].item()

z_up = joukowski(ctr + R_circ * np.exp(1j * theta_upper))
z_lo = joukowski(ctr + R_circ * np.exp(1j * theta_lower))
cp_up_pinn_raw = pinn_cp_on_surface(z_up.real, z_up.imag)
cp_lo_pinn_raw = pinn_cp_on_surface(z_lo.real, z_lo.imag)

# Apply same mask as trim
mask_up = (x_up >= CHORD_MIN) & (x_up <= CHORD_MAX) & (np.abs(cp_up) < 10.0)
mask_lo = (x_lo >= CHORD_MIN) & (x_lo <= CHORD_MAX) & (np.abs(cp_lo) < 10.0)
x_up_p = x_up[mask_up]
x_lo_p = x_lo[mask_lo]
cp_up_pinn_c = np.clip(cp_up_pinn_raw[mask_up], CP_DISPLAY_MIN, CP_DISPLAY_MAX)
cp_lo_pinn_c = np.clip(cp_lo_pinn_raw[mask_lo], CP_DISPLAY_MIN, CP_DISPLAY_MAX)

fig, ax = plt.subplots(figsize=(13, 7), facecolor=BG)
dark_ax(ax, facecolor=SURF)

ax.fill_between(x_up_t, cp_up_c, 0, alpha=0.15, color=ACCENT, label="_nolegend_")
ax.fill_between(x_lo_t, cp_lo_c, 0, alpha=0.10, color=ORANGE, label="_nolegend_")
ax.plot(x_up_t, cp_up_c, color=ACCENT,  lw=2.5, label="Upper surface — Joukowski ref")
ax.plot(x_lo_t, cp_lo_c, color=ORANGE,  lw=2.5, label="Lower surface — Joukowski ref")
ax.plot(x_up_p, cp_up_pinn_c, color=ACCENT,  lw=1.6, linestyle="--", alpha=0.85,
        label="Upper surface — PINN")
ax.plot(x_lo_p, cp_lo_pinn_c, color=ORANGE,  lw=1.6, linestyle="--", alpha=0.85,
        label="Lower surface — PINN")

ax.axhline(0, color=MUTED, lw=0.8, linestyle="--", alpha=0.5)
ax.axhline(1, color=MUTED, lw=0.6, linestyle=":", alpha=0.5, label="Cp = 1 (stagnation)")
ax.set_xlim(0, 1)
ax.set_ylim(CP_DISPLAY_MAX + 0.2, CP_DISPLAY_MIN - 0.3)  # inverted aeronautical convention
ax.set_xlabel("x / c  (chord fraction)", fontsize=13)
ax.set_ylabel("Cp  (y-axis inverted — aeronautical convention)", fontsize=12)
ax.set_title("NACA 0012 — Surface pressure coefficient  α = 5°\nSolid: Joukowski reference  |  Dashed: PINN surrogate",
             fontsize=13, fontweight="bold", color=ACCENT)
ax.legend(fontsize=11, framealpha=0.3, loc="lower right")
ax.grid(True, color=BORD, lw=0.5, alpha=0.7)
# Annotate suction peak on upper (suction) surface
idx_peak = np.argmin(cp_up_c)
x_pk, y_pk = x_up_t[idx_peak], cp_up_c[idx_peak]
ax.annotate(f"Suction peak\n−Cp = {-y_pk:.2f}  (x/c ≈ {x_pk:.2f})",
            xy=(x_pk, y_pk),
            xytext=(x_pk + 0.20, y_pk + 0.8),
            color=ACCENT, fontsize=10,
            arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.1))
# Annotate stagnation on lower surface
idx_stag = np.argmax(cp_lo_c)
ax.annotate(f"Stagnation  Cp ≈ {cp_lo_c[idx_stag]:.2f}",
            xy=(x_lo_t[idx_stag], cp_lo_c[idx_stag]),
            xytext=(x_lo_t[idx_stag] + 0.10, cp_lo_c[idx_stag] - 0.3),
            color=ORANGE, fontsize=10,
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1))

plt.tight_layout()
fig.savefig(os.path.join(OUT, "naca0012_surface_cp.png"), dpi=160,
            bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved naca0012_surface_cp.png")
print(f"\nAll outputs in: {OUT}")
