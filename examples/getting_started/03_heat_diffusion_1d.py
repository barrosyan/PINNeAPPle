"""03_heat_diffusion_1d.py — 1D Heat Diffusion Equation.

Physics problem
---------------
Transient heat conduction in a rod:

    u_t = α u_xx,    x ∈ [0,1],  t ∈ [0, T]
    u(x,0) = sin(πx)          (initial condition)
    u(0,t) = u(1,t) = 0       (Dirichlet, rod ends held at 0)

Exact solution:  u(x,t) = sin(πx) · e^{-α π² t}

Three experiments with different diffusivities α:
  - α = 0.01  →  slow diffusion (long memory)
  - α = 0.1   →  moderate
  - α = 0.5   →  fast diffusion (rapid decay)

What this example shows
-----------------------
- Space-time PINN: network input is (x, t), output is u
- How α controls the speed of thermal equilibration
- Contour plots of predicted vs exact temperature field

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
N_COL    = 4096
N_EPOCHS = 8_000

EXPERIMENTS = [
    {"alpha": 0.01, "label": "α = 0.01  (slow)"},
    {"alpha": 0.10, "label": "α = 0.10  (moderate)"},
    {"alpha": 0.50, "label": "α = 0.50  (fast)"},
]


def exact(x, t, alpha):
    return np.sin(math.pi * x) * np.exp(-alpha * math.pi ** 2 * t)


def build_net():
    return nn.Sequential(
        nn.Linear(2, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    )


def train(alpha: float):
    torch.manual_seed(7)
    net = build_net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    def model(xt):
        # Hard BCs: u = x*(1-x) * net(x,t)  enforces u(0)=u(1)=0
        x = xt[:, 0:1]
        return x * (1 - x) * net(xt)

    # Interior collocation
    x_col = torch.rand(N_COL, 1)
    t_col = torch.rand(N_COL, 1) * T_END
    xt_col = torch.cat([x_col, t_col], dim=1)

    # Initial condition collocation
    x_ic = torch.linspace(0, 1, 200).unsqueeze(1)
    t_ic = torch.zeros(200, 1)
    xt_ic = torch.cat([x_ic, t_ic], dim=1)
    u_ic  = torch.sin(math.pi * x_ic)

    for _ in range(N_EPOCHS):
        opt.zero_grad()

        xt = xt_col.clone().requires_grad_(True)
        u  = model(xt)
        u_t = torch.autograd.grad(u.sum(), xt, create_graph=True)[0][:, 1:2]
        u_x = torch.autograd.grad(u.sum(), xt, create_graph=True)[0][:, 0:1]
        u_xx = torch.autograd.grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]
        res  = u_t - alpha * u_xx
        loss_pde = res.pow(2).mean()

        u_pred_ic = model(xt_ic)
        loss_ic   = (u_pred_ic - u_ic).pow(2).mean()

        (loss_pde + 50 * loss_ic).backward()
        opt.step()
        sch.step()

    return model


def main():
    print("1D Heat Diffusion Equation — three diffusivity experiments\n")

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
        model = train(cfg["alpha"])

        with torch.no_grad():
            u_pred = model(xt_vis).numpy().reshape(n_vis, n_vis)
        u_true = exact(xx, tt, cfg["alpha"])
        err    = np.abs(u_pred - u_true)

        l2 = float(np.sqrt(((u_pred - u_true) ** 2).mean()) /
                   np.sqrt((u_true ** 2).mean() + 1e-8))
        print(f"  → relative L2: {l2:.4e}")

        vmin, vmax = u_true.min(), u_true.max()
        for ax, field, title, cmap in zip(
            row,
            [u_true, u_pred, err],
            [f"Exact  ({cfg['label']})", f"PINN  (L2={l2:.2e})", "|Exact − PINN|"],
            ["hot", "hot", "Reds"],
        ):
            vn = vmin if "|" not in title else None
            vx = vmax if "|" not in title else None
            im = ax.contourf(xx, tt, field, levels=30, cmap=cmap, vmin=vn, vmax=vx)
            plt.colorbar(im, ax=ax)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("t")

    plt.suptitle("1D Heat Equation   u_t = α u_xx", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("03_heat_diffusion_1d.png", dpi=130)
    print("\nSaved 03_heat_diffusion_1d.png")


if __name__ == "__main__":
    main()
