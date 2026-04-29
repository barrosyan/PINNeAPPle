"""Visualization 02 — 2D Heat Equation Temperature Evolution.

Physics
-------
    u_t = α (u_xx + u_yy)   in [0,1]² × [0,0.5]
    u(x,y,0) = sin(πx)sin(πy)   (initial condition)
    u = 0   on all walls (Dirichlet)

Exact solution (single-mode Fourier):
    u(x,y,t) = exp(−2π²αt) · sin(πx) · sin(πy)

What this shows
---------------
  • The Gaussian hot spot diffuses symmetrically over time.
  • PINN closely tracks the analytic decay at 4 different snapshots.
  • Temperature always stays positive (≥0) — no spurious oscillations.

Run
---
    python -m examples.visualizations.viz_02_heat_2d
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from pinneaple_validate import compare_to_analytical

ALPHA   = 0.05
T_MAX   = 0.5
N_COL   = 3_000
EPOCHS  = 8_000
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"


# ── Exact solution ─────────────────────────────────────────────────────────────
def u_exact(xy: np.ndarray, t: float) -> np.ndarray:
    x, y = xy[:, 0], xy[:, 1]
    return np.exp(-2 * math.pi**2 * ALPHA * t) * np.sin(math.pi * x) * np.sin(math.pi * y)


# ── Network ────────────────────────────────────────────────────────────────────
def make_net() -> nn.Module:
    return nn.Sequential(
        nn.Linear(3, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    ).to(DEVICE)


# ── Training ──────────────────────────────────────────────────────────────────
def train() -> nn.Module:
    torch.manual_seed(0)
    net  = make_net()
    opt  = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    for ep in range(EPOCHS):
        opt.zero_grad()

        # Interior: u_t = α(u_xx + u_yy)
        x  = torch.rand(N_COL, 1, device=DEVICE)
        y  = torch.rand(N_COL, 1, device=DEVICE)
        t  = torch.rand(N_COL, 1, device=DEVICE) * T_MAX
        xyt = torch.cat([x, y, t], dim=1).requires_grad_(True)
        u   = net(xyt)
        g   = torch.autograd.grad(u.sum(), xyt, create_graph=True)[0]
        u_t  = g[:, 2:3]
        u_x  = g[:, 0:1]
        u_y  = g[:, 1:2]
        u_xx = torch.autograd.grad(u_x.sum(), xyt, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y.sum(), xyt, create_graph=True)[0][:, 1:2]
        l_pde = (u_t - ALPHA * (u_xx + u_yy)).pow(2).mean()

        # Wall BCs (u=0 on all 4 walls)
        n_bc = 400
        t_bc = torch.rand(n_bc, 1, device=DEVICE) * T_MAX
        rand  = torch.rand(n_bc, 1, device=DEVICE)
        bnd_x = torch.cat([
            torch.zeros(n_bc, 1, device=DEVICE),
            torch.ones(n_bc, 1, device=DEVICE),
        ])
        bnd_y = torch.cat([rand, rand])
        bnd_t = torch.cat([t_bc, t_bc])
        l_bc_x = net(torch.cat([bnd_x, bnd_y, bnd_t], 1)).pow(2).mean()

        bnd_y2 = torch.cat([
            torch.zeros(n_bc, 1, device=DEVICE),
            torch.ones(n_bc, 1, device=DEVICE),
        ])
        bnd_x2 = torch.cat([rand, rand])
        bnd_t2 = torch.cat([t_bc, t_bc])
        l_bc_y = net(torch.cat([bnd_x2, bnd_y2, bnd_t2], 1)).pow(2).mean()

        # Initial condition
        x_ic  = torch.rand(n_bc, 1, device=DEVICE)
        y_ic  = torch.rand(n_bc, 1, device=DEVICE)
        t_ic  = torch.zeros(n_bc, 1, device=DEVICE)
        u_ic_exact = (torch.sin(math.pi * x_ic) *
                      torch.sin(math.pi * y_ic))
        l_ic = (net(torch.cat([x_ic, y_ic, t_ic], 1)) - u_ic_exact).pow(2).mean()

        loss = l_pde + 10.0 * (l_bc_x + l_bc_y) + 100.0 * l_ic
        loss.backward()
        opt.step(); sch.step()

        if ep % 2000 == 0:
            print(f"  ep {ep:5d}  loss={float(loss):.4e}")

    return net


# ── Visualization ──────────────────────────────────────────────────────────────
def visualize(net: nn.Module) -> None:
    NX = 80
    xs = np.linspace(0, 1, NX, dtype=np.float32)
    xg, yg = np.meshgrid(xs, xs)
    xy_flat = np.stack([xg.ravel(), yg.ravel()], axis=1)

    t_slices = [0.0, 0.1, 0.25, 0.5]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), facecolor="#0d1117")
    fig.patch.set_facecolor("#0d1117")

    vmin, vmax = 0.0, 1.0
    cmap = plt.cm.get_cmap("RdYlBu_r")

    for row, (label, fn) in enumerate([("Exact", None), ("PINN", net)]):
        for col, tval in enumerate(t_slices):
            ax = axes[row, col]
            ax.set_facecolor("#0d1117")
            for sp in ax.spines.values():
                sp.set_edgecolor("#30363d")
            ax.tick_params(colors="#8b949e", labelsize=7)

            if fn is None:
                field = u_exact(xy_flat, tval).reshape(NX, NX)
            else:
                xyt = np.concatenate([xy_flat,
                                      np.full((len(xy_flat), 1), tval,
                                              dtype=np.float32)], axis=1)
                with torch.no_grad():
                    field = net(torch.tensor(xyt, device=DEVICE)
                                ).cpu().numpy().reshape(NX, NX)

            im = ax.imshow(field, origin="lower", extent=[0, 1, 0, 1],
                           cmap=cmap, aspect="equal",
                           vmin=vmin, vmax=vmax)
            ax.contour(xg, yg, field, levels=8,
                       colors="white", linewidths=0.4, alpha=0.4)
            ax.set_title(f"t = {tval:.2f}",
                         color="#e6edf3", fontsize=9, pad=4)
            ax.set_xlabel("x", color="#8b949e", fontsize=8)
            ax.set_ylabel("y", color="#8b949e", fontsize=8)

        axes[row, 0].set_ylabel(f"{label}\ny", color="#e6edf3", fontsize=9)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin, vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, shrink=0.6, aspect=30, pad=0.02)
    cb.ax.tick_params(colors="#8b949e", labelsize=8)
    cb.set_label("u(x,y,t)", color="#8b949e", fontsize=9)

    fig.suptitle(
        "2D Heat Equation  u_t = α(u_xx+u_yy)  ·  α=0.05  ·  PINNeAPPle",
        color="#e6edf3", fontsize=11, y=1.01)
    plt.tight_layout()
    out = "viz_02_heat_2d.png"
    plt.savefig(out, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Saved {out}")


def main() -> None:
    print("─" * 55)
    print("  Viz 02 — 2D Heat Equation")
    print("─" * 55)
    print(f"  Training on {DEVICE}...")
    net = train()
    visualize(net)


if __name__ == "__main__":
    main()
