"""Visualization 01 — Potential Flow Past a Circular Cylinder.

Physics
-------
Stream-function formulation of 2D potential flow:
    ψ_xx + ψ_yy = 0    (Laplace equation)

Exact (analytical) solution — uniform flow U=1, cylinder R=0.5 at origin:
    ψ(x,y) = U · y · (1 − R²/r²),   r = √(x²+y²)

Velocity field:
    u = ∂ψ/∂y = U(1 − R²(x²−y²)/r⁴)
    v = −∂ψ/∂x = −2U·R²·xy/r⁴

PINN formulation
----------------
  Domain  : Rectangle [−2, 4] × [−1.5, 1.5] minus circle (0,0,R=0.5)
  PDE     : ψ_xx + ψ_yy = 0
  BCs     : ψ = U·y  on inlet (x=−2), outlet (x=4), top/bottom walls
            ψ = 0    on cylinder surface

Run
---
    python -m examples.visualizations.viz_01_flow_cylinder
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pinneaple_geom     import CSGRectangle, CSGCircle, CSGDifference
from pinneaple_validate import compare_to_analytical

U = 1.0
R = 0.5
XMIN, XMAX = -2.0, 4.0
YMIN, YMAX = -1.5, 1.5
N_COL    = 4_000
N_BC     = 600
EPOCHS   = 6_000
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


# ── Domain ────────────────────────────────────────────────────────────────────
domain = CSGRectangle(XMIN, YMIN, XMAX, YMAX) - CSGCircle(0.0, 0.0, R)


# ── Exact solution ────────────────────────────────────────────────────────────
def psi_exact(xy: np.ndarray) -> np.ndarray:
    x, y = xy[:, 0], xy[:, 1]
    r2 = x**2 + y**2
    r2 = np.where(r2 < R**2, np.nan, r2)
    return U * y * (1.0 - R**2 / r2)


def speed_exact(xy: np.ndarray) -> np.ndarray:
    x, y = xy[:, 0], xy[:, 1]
    r2 = x**2 + y**2
    r4 = r2**2
    r2 = np.where(r2 < R**2, np.nan, r2)
    r4 = np.where(r4 < R**4, np.nan, r4)
    ux = U * (1.0 - R**2 * (x**2 - y**2) / r4)
    vy = -2.0 * U * R**2 * x * y / r4
    return np.sqrt(ux**2 + vy**2)


# ── Network ───────────────────────────────────────────────────────────────────
def make_net() -> nn.Module:
    return nn.Sequential(
        nn.Linear(2, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    ).to(DEVICE)


# ── Training ──────────────────────────────────────────────────────────────────
def train() -> nn.Module:
    torch.manual_seed(0)
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    for ep in range(EPOCHS):
        opt.zero_grad()

        # Interior: Laplace residual ψ_xx + ψ_yy = 0
        pts_np = domain.sample_interior(N_COL, seed=ep % 20)
        pts = torch.tensor(pts_np, dtype=torch.float32, device=DEVICE,
                           requires_grad=True)
        psi = net(pts)
        g   = torch.autograd.grad(psi.sum(), pts, create_graph=True)[0]
        psi_x = g[:, 0:1]
        psi_y = g[:, 1:2]
        psi_xx = torch.autograd.grad(psi_x.sum(), pts, create_graph=True)[0][:, 0:1]
        psi_yy = torch.autograd.grad(psi_y.sum(), pts, create_graph=True)[0][:, 1:2]
        l_pde  = (psi_xx + psi_yy).pow(2).mean()

        # BC: far-field ψ = U·y
        n_ff = N_BC
        x_ff  = torch.cat([
            torch.full((n_ff//4, 1), XMIN), torch.full((n_ff//4, 1), XMAX),
        ]).to(DEVICE)
        y_ff  = (torch.rand(n_ff//2, 1, device=DEVICE) * (YMAX - YMIN) + YMIN)
        xy_ff = torch.cat([x_ff, y_ff], dim=1)
        l_ff  = (net(xy_ff) - U * y_ff).pow(2).mean()

        # Top / bottom walls
        x_tb  = (torch.rand(n_ff//2, 1, device=DEVICE) * (XMAX - XMIN) + XMIN)
        y_tb  = torch.cat([
            torch.full((n_ff//4, 1), YMIN), torch.full((n_ff//4, 1), YMAX),
        ]).to(DEVICE)
        xy_tb = torch.cat([x_tb, y_tb], dim=1)
        l_tb  = (net(xy_tb) - U * y_tb).pow(2).mean()

        # Cylinder surface: ψ = 0
        theta = torch.rand(N_BC, device=DEVICE) * 2 * math.pi
        xc = (R * torch.cos(theta)).unsqueeze(1).to(DEVICE)
        yc = (R * torch.sin(theta)).unsqueeze(1).to(DEVICE)
        xy_cyl = torch.cat([xc, yc], dim=1)
        l_cyl  = net(xy_cyl).pow(2).mean()

        loss = l_pde + 10.0 * (l_ff + l_tb + l_cyl)
        loss.backward()
        opt.step(); sch.step()

        if ep % 1000 == 0:
            print(f"  ep {ep:5d}  loss={float(loss):.4e}")

    return net


# ── Visualization ─────────────────────────────────────────────────────────────
def visualize(net: nn.Module) -> None:
    NX, NY = 400, 200
    xs = np.linspace(XMIN, XMAX, NX, dtype=np.float32)
    ys = np.linspace(YMIN, YMAX, NY, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    xy_flat = np.stack([xg.ravel(), yg.ravel()], axis=1)

    # Mask inside cylinder
    mask_cyl = xg**2 + yg**2 < R**2 * 0.99

    # PINN prediction
    with torch.no_grad():
        xt = torch.tensor(xy_flat, device=DEVICE)
        psi_pinn = net(xt).cpu().numpy().reshape(NY, NX)

    psi_an = psi_exact(xy_flat).reshape(NY, NX)
    spd_an = speed_exact(xy_flat).reshape(NY, NX)

    psi_pinn[mask_cyl] = np.nan
    psi_an[mask_cyl]   = np.nan
    spd_an[mask_cyl]   = np.nan

    # Validation metric
    pts_val = domain.sample_interior(20_000)

    def pinn_fn(xy):
        with torch.no_grad():
            return net(torch.tensor(xy, dtype=torch.float32,
                                    device=DEVICE)).cpu().numpy()

    def exact_fn(xy):
        return psi_exact(xy).reshape(-1, 1)

    from pinneaple_validate import compare_to_analytical
    met = compare_to_analytical(
        model=net,
        analytical_fn=lambda xy: psi_exact(xy).reshape(-1, 1),
        coord_names=["x", "y"],
        domain_bounds={"x": (XMIN, XMAX), "y": (YMIN, YMAX)},
        n_points=20_000, device=DEVICE,
    )
    print(f"\n  Relative L2 = {met['rel_l2']:.4e}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor="#0d1117")
    fig.patch.set_facecolor("#0d1117")

    cyl_patch_kw = dict(color="silver", zorder=5)

    for ax in axes:
        ax.set_facecolor("#0d1117")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=8)

    # Panel 1: Velocity magnitude (exact)
    ax = axes[0]
    spd_plot = np.where(mask_cyl, np.nan, spd_an)
    im = ax.imshow(spd_plot, origin="lower",
                   extent=[XMIN, XMAX, YMIN, YMAX],
                   cmap="viridis", aspect="auto",
                   vmin=0, vmax=2.0)
    # Streamlines from ψ contours
    levels = np.linspace(YMIN * U, YMAX * U, 24)
    ax.contour(xg, yg, psi_an, levels=levels,
               colors="white", linewidths=0.5, alpha=0.55)
    cyl_circle = plt.Circle((0, 0), R, color="silver", zorder=5)
    ax.add_patch(cyl_circle)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors="#8b949e", labelsize=7)
    ax.set_title("Speed |∇ψ| — exact", color="#e6edf3", fontsize=10)
    ax.set_xlabel("x", color="#8b949e"); ax.set_ylabel("y", color="#8b949e")

    # Panel 2: PINN stream function
    ax = axes[1]
    im2 = ax.imshow(psi_pinn, origin="lower",
                    extent=[XMIN, XMAX, YMIN, YMAX],
                    cmap="RdBu_r", aspect="auto",
                    vmin=-1.6, vmax=1.6)
    ax.contour(xg, yg, psi_pinn, levels=levels,
               colors="white", linewidths=0.5, alpha=0.55)
    cyl2 = plt.Circle((0, 0), R, color="silver", zorder=5)
    ax.add_patch(cyl2)
    cb2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(colors="#8b949e", labelsize=7)
    ax.set_title(f"PINN ψ(x,y)  (L2={met['rel_l2']:.2e})",
                 color="#e6edf3", fontsize=10)
    ax.set_xlabel("x", color="#8b949e"); ax.set_ylabel("y", color="#8b949e")

    # Panel 3: Absolute error
    ax = axes[2]
    err = np.abs(psi_pinn - psi_an)
    im3 = ax.imshow(err, origin="lower",
                    extent=[XMIN, XMAX, YMIN, YMAX],
                    cmap="hot", aspect="auto", vmin=0)
    cyl3 = plt.Circle((0, 0), R, color="silver", zorder=5)
    ax.add_patch(cyl3)
    cb3 = plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
    cb3.ax.tick_params(colors="#8b949e", labelsize=7)
    ax.set_title("|ψ_PINN − ψ_exact|", color="#e6edf3", fontsize=10)
    ax.set_xlabel("x", color="#8b949e"); ax.set_ylabel("y", color="#8b949e")

    fig.suptitle("Potential Flow Past Circular Cylinder — PINNeAPPle",
                 color="#e6edf3", fontsize=12, y=1.01)
    plt.tight_layout()
    out = "viz_01_flow_cylinder.png"
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved {out}")


def main() -> None:
    print("─" * 55)
    print("  Viz 01 — Potential Flow Past Cylinder")
    print("─" * 55)
    print(f"  Training on {DEVICE}...")
    net = train()
    visualize(net)


if __name__ == "__main__":
    main()
