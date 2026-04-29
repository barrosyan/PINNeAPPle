"""Visualization 05 — 2D Wave Equation (3D Surface).

Physics
-------
    u_tt = c² (u_xx + u_yy)   in [0,1]² × [0,0.5]

    u(x,y,0)  = sin(2πx)sin(πy)    (initial displacement)
    u_t(x,y,0) = 0                  (at rest initially)
    u = 0   on all walls

Exact solution:
    ω = π√(4+1)·c = π√5·c
    u(x,y,t) = sin(2πx)·sin(πy)·cos(ω·t)

What this shows
---------------
  • Wave oscillating between positive (peaks) and negative (troughs).
  • 3D surface plots at 4 time snapshots.
  • The wave passes through zero and reverses — unlike heat (always ≥0).

Run
---
    python -m examples.visualizations.viz_05_wave_2d
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from pinneaple_validate import compare_to_analytical

C      = 1.0
OMEGA  = math.pi * math.sqrt(5) * C
T_MAX  = 0.5
N_COL  = 2_000
EPOCHS = 8_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Exact solution ─────────────────────────────────────────────────────────────
def u_exact_np(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    return np.sin(2 * math.pi * x) * np.sin(math.pi * y) * np.cos(OMEGA * t)


# ── Network ────────────────────────────────────────────────────────────────────
def make_net() -> nn.Module:
    return nn.Sequential(
        nn.Linear(3, 64), nn.Tanh(),
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

        # Interior: u_tt = C²(u_xx + u_yy)
        x   = torch.rand(N_COL, 1, device=DEVICE)
        y   = torch.rand(N_COL, 1, device=DEVICE)
        t   = torch.rand(N_COL, 1, device=DEVICE) * T_MAX
        xyt = torch.cat([x, y, t], dim=1).requires_grad_(True)
        u   = net(xyt)
        g   = torch.autograd.grad(u.sum(), xyt, create_graph=True)[0]
        u_t  = g[:, 2:3]
        u_x  = g[:, 0:1]
        u_y  = g[:, 1:2]
        u_xx = torch.autograd.grad(u_x.sum(), xyt, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y.sum(), xyt, create_graph=True)[0][:, 1:2]
        u_tt = torch.autograd.grad(u_t.sum(), xyt, create_graph=True)[0][:, 2:3]
        l_pde = (u_tt - C**2 * (u_xx + u_yy)).pow(2).mean()

        # Wall BCs
        n_bc  = 300
        t_bc  = torch.rand(n_bc, 1, device=DEVICE) * T_MAX
        rand  = torch.rand(n_bc, 1, device=DEVICE)
        bnd_x = torch.cat([torch.zeros(n_bc, 1), torch.ones(n_bc, 1)]).to(DEVICE)
        bnd_y = torch.cat([rand, rand])
        bnd_t = torch.cat([t_bc, t_bc])
        l_bc = net(torch.cat([bnd_x, bnd_y, bnd_t], 1)).pow(2).mean()

        bnd_y2 = torch.cat([torch.zeros(n_bc, 1), torch.ones(n_bc, 1)]).to(DEVICE)
        bnd_x2 = torch.cat([rand, rand])
        l_bc += net(torch.cat([bnd_x2, bnd_y2, bnd_t], 1)).pow(2).mean()

        # IC: u(x,y,0)
        x_ic  = torch.rand(n_bc, 1, device=DEVICE)
        y_ic  = torch.rand(n_bc, 1, device=DEVICE)
        t_ic  = torch.zeros(n_bc, 1, device=DEVICE)
        u_ic_ex = (torch.sin(2 * math.pi * x_ic) *
                   torch.sin(math.pi * y_ic))
        xyt_ic  = torch.cat([x_ic, y_ic, t_ic], 1).requires_grad_(True)
        u_ic    = net(xyt_ic)
        l_ic    = (u_ic - u_ic_ex).pow(2).mean()
        # IC velocity: u_t(x,y,0) = 0
        u_t_ic = torch.autograd.grad(u_ic.sum(), xyt_ic,
                                      create_graph=True)[0][:, 2:3]
        l_ic  += u_t_ic.pow(2).mean()

        loss = l_pde + 10.0 * l_bc + 100.0 * l_ic
        loss.backward()
        opt.step(); sch.step()

        if ep % 2000 == 0:
            print(f"  ep {ep:5d}  loss={float(loss):.4e}")

    return net


# ── Visualization ──────────────────────────────────────────────────────────────
def visualize(net: nn.Module) -> None:
    NX = 50
    xs = np.linspace(0, 1, NX, dtype=np.float32)
    xg, yg = np.meshgrid(xs, xs)
    xy_flat = np.stack([xg.ravel(), yg.ravel()], axis=1)

    t_slices = [0.0, 1.0/(4*OMEGA/(2*math.pi)),
                2.0/(4*OMEGA/(2*math.pi)), 3.0/(4*OMEGA/(2*math.pi))]
    t_slices = [0.0, 0.1, 0.25, 0.45]

    fig = plt.figure(figsize=(16, 4.5), facecolor="#0d1117")
    fig.patch.set_facecolor("#0d1117")

    for i, tval in enumerate(t_slices):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#8b949e", labelsize=6)

        # PINN prediction
        xyt = np.concatenate([xy_flat,
                               np.full((len(xy_flat), 1), tval, dtype=np.float32)],
                              axis=1)
        with torch.no_grad():
            u_pinn = net(torch.tensor(xyt, device=DEVICE)
                         ).cpu().numpy().reshape(NX, NX)

        u_ex = u_exact_np(xg, yg, tval)

        # Coloring by height
        from matplotlib import cm
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=-1.0, vmax=1.0)
        colors = cm.RdYlBu_r(norm(u_pinn))

        ax.plot_surface(xg, yg, u_pinn,
                        facecolors=colors,
                        linewidth=0, antialiased=True,
                        shade=False, alpha=0.92)
        ax.contourf(xg, yg, u_pinn, zdir="z", offset=-1.1,
                    levels=12, cmap="RdYlBu_r", alpha=0.35)

        ax.set_zlim(-1.1, 1.1)
        ax.set_title(f"t = {tval:.2f}  (PINN)", color="#e6edf3",
                     fontsize=8, pad=2)
        ax.set_xlabel("x", color="#8b949e", fontsize=7, labelpad=2)
        ax.set_ylabel("y", color="#8b949e", fontsize=7, labelpad=2)
        ax.set_zlabel("u", color="#8b949e", fontsize=7, labelpad=2)

        # Error
        l2 = float(np.sqrt(((u_pinn - u_ex)**2).mean()) /
                   (np.abs(u_ex).mean() + 1e-8))
        ax.text2D(0.05, 0.95, f"L2={l2:.2e}", transform=ax.transAxes,
                  color="#ffa657", fontsize=7)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.set_edgecolor("#30363d")

    fig.suptitle(
        "2D Wave Equation  u_tt = c²(u_xx+u_yy)  ·  c=1  ·  PINNeAPPle",
        color="#e6edf3", fontsize=11, y=0.98)
    plt.tight_layout()
    out = "viz_05_wave_2d.png"
    plt.savefig(out, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Saved {out}")


def main() -> None:
    print("─" * 55)
    print("  Viz 05 — 2D Wave Equation (3D Surface)")
    print("─" * 55)
    print(f"  Training on {DEVICE}...")
    net = train()
    visualize(net)


if __name__ == "__main__":
    main()
