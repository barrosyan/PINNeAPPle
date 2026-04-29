"""04_wave_equation_1d.py — 1D Wave Equation.

Physics problem
---------------
Vibrating string (or acoustic wave in 1D):

    u_tt = c² u_xx,    x ∈ [0,1],  t ∈ [0, T]
    u(0,t) = u(1,t) = 0          (fixed ends)
    u(x,0) = sin(πx)             (initial displacement)
    u_t(x,0) = 0                 (released from rest)

Exact solution (d'Alembert / Fourier):  u(x,t) = sin(πx)·cos(cπt)

Three experiments with different wave speeds c:
  - c = 1.0  →  slow wave
  - c = 2.0  →  medium wave
  - c = 4.0  →  fast wave (more oscillations, harder to learn)

What this example shows
-----------------------
- Second-order time derivative in PINN loss
- Wave propagation: the solution does not decay — it oscillates
- Speed-of-light analogy: larger c → faster information travel

Tier 1 — Explorer.
"""

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T_END    = 1.0
N_COL    = 5000
N_EPOCHS = 10_000

EXPERIMENTS = [
    {"c": 1.0, "label": "c = 1  (slow)"},
    {"c": 2.0, "label": "c = 2  (medium)"},
    {"c": 4.0, "label": "c = 4  (fast)"},
]


def exact(x, t, c):
    return np.sin(math.pi * x) * np.cos(c * math.pi * t)


def build_net():
    return nn.Sequential(
        nn.Linear(2, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 1),
    )


def train(c: float):
    torch.manual_seed(3)
    net = build_net()
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    def model(xt):
        x = xt[:, 0:1]
        return x * (1 - x) * net(xt)   # hard Dirichlet at x=0,1

    # Interior
    x_col = torch.rand(N_COL, 1)
    t_col = torch.rand(N_COL, 1) * T_END
    xt_col = torch.cat([x_col, t_col], dim=1)

    # IC: u(x,0) = sin(πx)
    x_ic  = torch.linspace(0, 1, 300).unsqueeze(1)
    xt_ic = torch.cat([x_ic, torch.zeros_like(x_ic)], dim=1)
    u_ic  = torch.sin(math.pi * x_ic)

    # IC velocity: u_t(x,0) = 0
    x_vel  = torch.linspace(0, 1, 300).unsqueeze(1)
    xt_vel = torch.cat([x_vel, torch.zeros_like(x_vel)], dim=1)

    for _ in range(N_EPOCHS):
        opt.zero_grad()

        # PDE
        xt = xt_col.clone().requires_grad_(True)
        u  = model(xt)
        u_t  = torch.autograd.grad(u.sum(), xt, create_graph=True)[0][:, 1:2]
        u_tt = torch.autograd.grad(u_t.sum(), xt, create_graph=True)[0][:, 1:2]
        u_x  = torch.autograd.grad(u.sum(), xt, create_graph=True)[0][:, 0:1]
        u_xx = torch.autograd.grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]
        loss_pde = (u_tt - c ** 2 * u_xx).pow(2).mean()

        # IC displacement
        loss_u0 = (model(xt_ic) - u_ic).pow(2).mean()

        # IC velocity
        xt_v = xt_vel.clone().requires_grad_(True)
        u_v  = model(xt_v)
        ut_v = torch.autograd.grad(u_v.sum(), xt_v, create_graph=True)[0][:, 1:2]
        loss_v0 = ut_v.pow(2).mean()

        (loss_pde + 100 * loss_u0 + 100 * loss_v0).backward()
        opt.step()
        sch.step()

    return model


def main():
    print("1D Wave Equation — three wave speed experiments\n")

    n_vis = 60
    x_lin = np.linspace(0, 1,    n_vis, dtype=np.float32)
    t_lin = np.linspace(0, T_END, n_vis, dtype=np.float32)
    xx, tt = np.meshgrid(x_lin, t_lin)
    xt_vis = torch.tensor(
        np.stack([xx.ravel(), tt.ravel()], axis=1), dtype=torch.float32
    )

    fig, axes = plt.subplots(len(EXPERIMENTS), 3, figsize=(18, 5 * len(EXPERIMENTS)))

    for row, cfg in zip(axes, EXPERIMENTS):
        print(f"Training: {cfg['label']} ...")
        model = train(cfg["c"])

        with torch.no_grad():
            u_pred = model(xt_vis).numpy().reshape(n_vis, n_vis)
        u_true = exact(xx, tt, cfg["c"])
        err    = np.abs(u_pred - u_true)
        l2 = float(np.sqrt(((u_pred - u_true) ** 2).mean()) /
                   np.sqrt((u_true ** 2).mean() + 1e-8))
        print(f"  → relative L2: {l2:.4e}")

        for ax, field, title, cmap in zip(
            row,
            [u_true, u_pred, err],
            [f"Exact  ({cfg['label']})", f"PINN  (L2={l2:.2e})", "|Exact − PINN|"],
            ["RdBu_r", "RdBu_r", "Reds"],
        ):
            lv = 30
            im = ax.contourf(xx, tt, field, levels=lv, cmap=cmap)
            plt.colorbar(im, ax=ax)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("t")

    plt.suptitle("1D Wave Equation   u_tt = c² u_xx", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("04_wave_equation_1d.png", dpi=130)
    print("\nSaved 04_wave_equation_1d.png")


if __name__ == "__main__":
    main()
