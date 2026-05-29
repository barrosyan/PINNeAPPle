"""06_lotka_volterra.py — Lotka-Volterra Predator-Prey System.

Physics problem
---------------
Classic ecological model for two interacting species:

    dx/dt =  α x − β x y        (prey grows, dies from predation)
    dy/dt = −γ y + δ x y        (predator dies, grows from predation)
    x(0) = x0,  y(0) = y0

x = prey population,  y = predator population

There is no closed-form solution — the system produces periodic orbits
in phase space, a hallmark of Hamiltonian-like dynamics.

Three experiments
-----------------
  1. Classic cycle:      α=1.0, β=0.1, γ=1.5, δ=0.075,  x0=10, y0=5
  2. High predation:     α=1.0, β=0.3, γ=1.5, δ=0.2,    x0=10, y0=5
  3. Slow dynamics:      α=0.5, β=0.05,γ=0.8, δ=0.04,   x0=15, y0=3

What this example shows
-----------------------
- Multi-output PINN (network predicts two coupled ODEs simultaneously)
- Cyclic / conserved behaviour in a nonlinear system
- How parameters change orbit shape and period

Tier 1 — Explorer.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T_END    = 30.0
N_COL    = 500
N_EPOCHS = 12_000

EXPERIMENTS = [
    {"alpha": 1.0, "beta": 0.10, "gamma": 1.5, "delta": 0.075,
     "x0": 10.0, "y0": 5.0, "label": "Classic cycle"},
    {"alpha": 1.0, "beta": 0.30, "gamma": 1.5, "delta": 0.200,
     "x0": 10.0, "y0": 5.0, "label": "High predation"},
    {"alpha": 0.5, "beta": 0.05, "gamma": 0.8, "delta": 0.040,
     "x0": 15.0, "y0": 3.0, "label": "Slow dynamics"},
]


def reference(cfg) -> tuple:
    """Numerical reference via scipy RK45."""
    def rhs(t, z):
        x, y = z
        return [cfg["alpha"]*x - cfg["beta"]*x*y,
                -cfg["gamma"]*y + cfg["delta"]*x*y]
    sol = solve_ivp(rhs, [0, T_END], [cfg["x0"], cfg["y0"]],
                    method="RK45", dense_output=True,
                    rtol=1e-8, atol=1e-10)
    t = np.linspace(0, T_END, 600)
    z = sol.sol(t)
    return t, z[0], z[1]


def build_net():
    return nn.Sequential(
        nn.Linear(1, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 2),   # outputs: [x, y]
    )


def train(cfg) -> tuple:
    torch.manual_seed(42)
    net = build_net()
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    alpha, beta, gamma, delta = cfg["alpha"], cfg["beta"], cfg["gamma"], cfg["delta"]
    x0, y0 = cfg["x0"], cfg["y0"]

    # Normalise output scale  (helps with large population values)
    scale = torch.tensor([[x0, y0]], dtype=torch.float32)

    t_col = torch.linspace(0, T_END, N_COL).unsqueeze(1)
    t0    = torch.zeros(1, 1)

    for _ in range(N_EPOCHS):
        opt.zero_grad()

        tc = t_col.clone().requires_grad_(True)
        z  = net(tc) * scale   # (N, 2) — [x, y]
        x, y = z[:, 0:1], z[:, 1:2]

        zd = torch.autograd.grad(z.sum(), tc, create_graph=True)[0]
        xd, yd = zd[:, 0:1], zd[:, 1:2]

        res_x = xd - (alpha * x - beta * x * y)
        res_y = yd - (-gamma * y + delta * x * y)
        loss_pde = res_x.pow(2).mean() + res_y.pow(2).mean()

        z0 = net(t0) * scale
        ic = torch.tensor([[x0, y0]], dtype=torch.float32)
        loss_ic = (z0 - ic).pow(2).mean()

        (loss_pde + 100 * loss_ic).backward()
        opt.step()
        sch.step()

    t_eval = np.linspace(0, T_END, 600, dtype=np.float32)
    with torch.no_grad():
        z_pred = (net(torch.tensor(t_eval[:, None])) * scale.numpy()).numpy()
    return t_eval, z_pred[:, 0], z_pred[:, 1]


def main():
    print("Lotka-Volterra Predator-Prey System\n")
    fig, axes = plt.subplots(len(EXPERIMENTS), 2,
                             figsize=(14, 5 * len(EXPERIMENTS)))

    for row, cfg in zip(axes, EXPERIMENTS):
        print(f"Training: {cfg['label']} ...")
        t_ref, x_ref, y_ref = reference(cfg)
        t_p,   x_p,   y_p   = train(cfg)

        l2x = float(np.sqrt(((np.interp(t_p, t_ref, x_ref) - x_p)**2).mean()) /
                    (np.abs(x_ref).mean() + 1e-8))
        l2y = float(np.sqrt(((np.interp(t_p, t_ref, y_ref) - y_p)**2).mean()) /
                    (np.abs(y_ref).mean() + 1e-8))
        print(f"  → L2 prey={l2x:.3e}  predator={l2y:.3e}")

        # Time series
        row[0].plot(t_ref, x_ref, "b-",  lw=1.5, label="Prey (exact)")
        row[0].plot(t_ref, y_ref, "r-",  lw=1.5, label="Predator (exact)")
        row[0].plot(t_p,   x_p,   "b--", lw=1,   label="Prey (PINN)")
        row[0].plot(t_p,   y_p,   "r--", lw=1,   label="Predator (PINN)")
        row[0].set_title(f"{cfg['label']} — time series")
        row[0].set_xlabel("Time  t")
        row[0].set_ylabel("Population")
        row[0].legend(fontsize=8)
        row[0].grid(True, alpha=0.3)

        # Phase portrait
        row[1].plot(x_ref, y_ref, "k-",  lw=1.5, label="Exact orbit")
        row[1].plot(x_p,   y_p,   "r--", lw=1,   label="PINN orbit")
        row[1].set_title(f"{cfg['label']} — phase portrait")
        row[1].set_xlabel("Prey  x")
        row[1].set_ylabel("Predator  y")
        row[1].legend(fontsize=8)
        row[1].grid(True, alpha=0.3)

    plt.suptitle("Lotka-Volterra  dx/dt = αx − βxy,  dy/dt = −γy + δxy",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("06_lotka_volterra.png", dpi=130)
    print("\nSaved 06_lotka_volterra.png")


if __name__ == "__main__":
    main()
