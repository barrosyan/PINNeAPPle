"""07_nonlinear_pendulum.py — Nonlinear Pendulum.

Physics problem
---------------
A simple pendulum without the small-angle approximation:

    θ''(t) + (g/L) sin(θ) = 0,    t ∈ [0, T]
    θ(0) = θ0,  θ'(0) = 0         (released from rest)

The linear approximation gives  θ''+ (g/L)θ = 0  with T = 2π√(L/g).
The nonlinear model has a longer period for large amplitudes — a key
physical effect that the PINN must capture.

Three experiments  (g=9.81, L=1.0)
------------------------------------
  - Small angle:   θ0 = 0.1 rad  (~6°)   → nearly linear
  - Medium angle:  θ0 = 1.0 rad  (~57°)  → nonlinear, period longer
  - Large angle:   θ0 = 2.8 rad  (~160°) → strongly nonlinear, very slow

What this example shows
-----------------------
- sin(θ) nonlinearity in the ODE residual
- Period lengthening for large amplitudes
- Comparison with the linear (small-angle) approximation

Tier 1 — Explorer.
"""

import math
import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = 9.81
L = 1.0
T_END = 10.0
N_COL = 400
N_EPOCHS = 12_000

EXPERIMENTS = [
    {"theta0": 0.1, "label": "Small angle  θ₀=0.1 rad"},
    {"theta0": 1.0, "label": "Medium angle  θ₀=1.0 rad"},
    {"theta0": 2.8, "label": "Large angle  θ₀=2.8 rad"},
]


def reference(theta0: float):
    def rhs(t, z):
        theta, omega = z
        return [omega, -(G / L) * math.sin(theta)]
    sol = solve_ivp(rhs, [0, T_END], [theta0, 0.0],
                    method="RK45", dense_output=True,
                    rtol=1e-10, atol=1e-12)
    t = np.linspace(0, T_END, 800)
    z = sol.sol(t)
    return t, z[0]


def linear_approx(t: np.ndarray, theta0: float) -> np.ndarray:
    omega0 = math.sqrt(G / L)
    return theta0 * np.cos(omega0 * t)


def build_net():
    return nn.Sequential(
        nn.Linear(1, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    )


def train(theta0: float):
    torch.manual_seed(1)
    net = build_net()
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    t_col = torch.linspace(0, T_END, N_COL).unsqueeze(1)
    t0    = torch.zeros(1, 1)

    for _ in range(N_EPOCHS):
        opt.zero_grad()

        tc   = t_col.clone().requires_grad_(True)
        th   = net(tc)
        th_t  = torch.autograd.grad(th.sum(), tc, create_graph=True)[0]
        th_tt = torch.autograd.grad(th_t.sum(), tc, create_graph=True)[0]
        res  = th_tt + (G / L) * torch.sin(th)
        loss_pde = res.pow(2).mean()

        t0g  = t0.clone().requires_grad_(True)
        th0  = net(t0g)
        dth0 = torch.autograd.grad(th0.sum(), t0g, create_graph=True)[0]
        loss_ic = (th0 - theta0).pow(2) + dth0.pow(2)

        (loss_pde + 200 * loss_ic).backward()
        opt.step()
        sch.step()

    t_eval = np.linspace(0, T_END, 800, dtype=np.float32)
    with torch.no_grad():
        th_pred = net(torch.tensor(t_eval[:, None])).numpy().ravel()
    return t_eval, th_pred


def main():
    print("Nonlinear Pendulum — three amplitude experiments\n")
    fig, axes = plt.subplots(len(EXPERIMENTS), 2,
                             figsize=(14, 5 * len(EXPERIMENTS)))

    for row, cfg in zip(axes, EXPERIMENTS):
        th0 = cfg["theta0"]
        print(f"Training: {cfg['label']} ...")
        t_ref, th_ref = reference(th0)
        t_p,   th_p   = train(th0)
        th_lin = linear_approx(t_ref, th0)

        l2 = float(np.sqrt(((np.interp(t_p, t_ref, th_ref) - th_p)**2).mean()) /
                   (np.abs(th_ref).mean() + 1e-8))
        print(f"  → relative L2: {l2:.4e}")

        # Time series
        row[0].plot(t_ref, th_ref, "k-",  lw=2,   label="Exact (nonlinear)")
        row[0].plot(t_p,   th_p,   "r--", lw=1.5, label=f"PINN (L2={l2:.2e})")
        row[0].plot(t_ref, th_lin, "b:",  lw=1.2, label="Linear approx.")
        row[0].set_title(cfg["label"])
        row[0].set_xlabel("Time  t  (s)")
        row[0].set_ylabel("θ (rad)")
        row[0].legend(fontsize=8)
        row[0].grid(True, alpha=0.3)

        # Phase portrait  (θ, θ')
        omega_ref = np.gradient(th_ref, t_ref)
        omega_p   = np.gradient(th_p, t_p)
        row[1].plot(th_ref, omega_ref, "k-",  lw=1.5, label="Exact orbit")
        row[1].plot(th_p,   omega_p,   "r--", lw=1,   label="PINN orbit")
        row[1].set_title("Phase portrait")
        row[1].set_xlabel("θ (rad)")
        row[1].set_ylabel("θ' (rad/s)")
        row[1].legend(fontsize=8)
        row[1].grid(True, alpha=0.3)

    plt.suptitle("Nonlinear Pendulum   θ'' + (g/L)sin(θ) = 0",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("07_nonlinear_pendulum.png", dpi=130)
    print("\nSaved 07_nonlinear_pendulum.png")


if __name__ == "__main__":
    main()
