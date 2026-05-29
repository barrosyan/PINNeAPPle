"""09_lorenz_system.py — Lorenz System (Chaos and Strange Attractors).

Physics problem
---------------
The Lorenz system is a simplified model of atmospheric convection:

    dx/dt = σ (y − x)
    dy/dt = x (ρ − z) − y
    dz/dt = x y − β z

    (x₀, y₀, z₀) = initial conditions

With σ=10, β=8/3, ρ=28 the system exhibits deterministic chaos:
tiny differences in initial conditions lead to exponentially diverging
trajectories — the "butterfly effect".

Three experiments
-----------------
  1. Classic chaos:      σ=10, β=8/3,  ρ=28  →  fully chaotic
  2. Near transition:    σ=10, β=8/3,  ρ=24  →  periodic orbits near chaos
  3. Periodic:           σ=10, β=8/3,  ρ=100 →  complex but periodic

Key physical insight
--------------------
PINNs can learn short-time Lorenz trajectories well, but long-time
prediction is fundamentally limited by chaos.  This example makes that
explicit: the PINN match degrades as t → T for the chaotic case.

What this example shows
-----------------------
- Multi-output PINN on a 3D system
- The characteristic butterfly attractor in 3D phase space
- Limits of PINN long-time prediction under chaos

Tier 1 — Explorer.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SIGMA = 10.0
BETA  = 8.0 / 3.0
T_END = 3.0        # short window so PINN can track chaos
N_COL = 600
N_EPOCHS = 15_000

EXPERIMENTS = [
    {"rho": 28.0,  "ic": [1.0, 0.0, 0.0], "label": "Chaotic  ρ=28"},
    {"rho": 24.0,  "ic": [1.0, 0.0, 0.0], "label": "Near transition  ρ=24"},
    {"rho": 100.0, "ic": [1.0, 0.0, 0.0], "label": "Periodic  ρ=100"},
]


def reference(rho, ic):
    def rhs(t, z):
        x, y, z_ = z
        return [SIGMA*(y-x), x*(rho-z_)-y, x*y - BETA*z_]
    sol = solve_ivp(rhs, [0, T_END], ic,
                    method="RK45", dense_output=True, rtol=1e-10, atol=1e-12)
    t = np.linspace(0, T_END, 800)
    z = sol.sol(t)
    return t, z[0], z[1], z[2]


def build_net():
    return nn.Sequential(
        nn.Linear(1, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 3),   # outputs: [x, y, z]
    )


def train(rho, ic):
    torch.manual_seed(77)
    net = build_net()
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    # Scale so outputs are O(1)
    scale = torch.tensor([[20.0, 25.0, 30.0]], dtype=torch.float32)
    ic_t  = torch.tensor([ic], dtype=torch.float32)

    t_col = torch.linspace(0, T_END, N_COL).unsqueeze(1)
    t0    = torch.zeros(1, 1)

    for _ in range(N_EPOCHS):
        opt.zero_grad()

        tc = t_col.clone().requires_grad_(True)
        z  = net(tc) * scale   # (N, 3)
        x_, y_, z_ = z[:, 0:1], z[:, 1:2], z[:, 2:3]
        zd = torch.autograd.grad(z.sum(), tc, create_graph=True)[0]
        xd, yd, zd_ = zd[:, 0:1], zd[:, 1:2], zd[:, 2:3]

        res_x = xd - SIGMA * (y_ - x_)
        res_y = yd - (x_ * (rho - z_) - y_)
        res_z = zd_ - (x_ * y_ - BETA * z_)
        loss_pde = (res_x.pow(2).mean() + res_y.pow(2).mean() +
                    res_z.pow(2).mean())

        z0 = net(t0) * scale
        loss_ic = (z0 - ic_t).pow(2).mean()

        (loss_pde + 200 * loss_ic).backward()
        opt.step()
        sch.step()

    t_eval = np.linspace(0, T_END, 800, dtype=np.float32)
    with torch.no_grad():
        z_pred = (net(torch.tensor(t_eval[:, None])) * scale.numpy()).numpy()
    return t_eval, z_pred[:, 0], z_pred[:, 1], z_pred[:, 2]


def main():
    print("Lorenz System — chaos, attractors, and PINN limits\n")

    fig = plt.figure(figsize=(18, 5 * len(EXPERIMENTS)))
    plot_idx = 1

    for cfg in EXPERIMENTS:
        print(f"Computing reference: {cfg['label']} ...")
        t_ref, xr, yr, zr = reference(cfg["rho"], cfg["ic"])
        print(f"Training PINN: {cfg['label']} ...")
        t_p, xp, yp, zp   = train(cfg["rho"], cfg["ic"])

        l2x = float(np.sqrt(((np.interp(t_p, t_ref, xr) - xp)**2).mean()) /
                    (np.abs(xr).mean() + 1e-8))

        # 3D phase portrait
        ax3d = fig.add_subplot(len(EXPERIMENTS), 3, plot_idx, projection="3d")
        ax3d.plot(xr, yr, zr, "k-",  lw=0.8, alpha=0.8, label="Reference")
        ax3d.plot(xp, yp, zp, "r--", lw=0.8, alpha=0.8, label="PINN")
        ax3d.set_title(f"{cfg['label']}\n3D attractor")
        ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
        ax3d.legend(fontsize=7)
        plot_idx += 1

        # x(t) time series
        ax_t = fig.add_subplot(len(EXPERIMENTS), 3, plot_idx)
        ax_t.plot(t_ref, xr, "k-",  lw=1.5, label="Reference")
        ax_t.plot(t_p,   xp, "r--", lw=1,   label=f"PINN (L2={l2x:.2e})")
        ax_t.set_title("x(t) time series")
        ax_t.set_xlabel("t"); ax_t.set_ylabel("x(t)")
        ax_t.legend(fontsize=8); ax_t.grid(True, alpha=0.3)
        plot_idx += 1

        # Error growth
        x_interp = np.interp(t_p, t_ref, xr)
        err = np.abs(xp - x_interp)
        ax_e = fig.add_subplot(len(EXPERIMENTS), 3, plot_idx)
        ax_e.semilogy(t_p, err + 1e-8, "b-", lw=1.2)
        ax_e.set_title("Pointwise error |x_PINN − x_ref|")
        ax_e.set_xlabel("t"); ax_e.set_ylabel("|error|")
        ax_e.grid(True, which="both", alpha=0.3)
        plot_idx += 1

        print(f"  → relative L2 (x): {l2x:.4e}")

    plt.suptitle("Lorenz System   dx/dt=σ(y−x),  dy/dt=x(ρ−z)−y,  dz/dt=xy−βz",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig("09_lorenz_system.png", dpi=120)
    print("\nSaved 09_lorenz_system.png")


if __name__ == "__main__":
    main()
