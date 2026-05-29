"""10_coupled_oscillators.py — Coupled Spring-Mass System.

Physics problem
---------------
Two masses connected by three springs (wall–m1–m2–wall):

    m1 x1'' = −k1 x1 + k2 (x2 − x1)
    m2 x2'' = −k2 (x2 − x1) − k3 x2

For m1=m2=1 and k1=k2=k3=k:
    x1'' = −2k x1 + k x2
    x2'' =  k x1 − 2k x2

This linear system has two normal modes:
  - Mode 1 (in-phase):      ω1 = √k        masses move together
  - Mode 2 (out-of-phase):  ω2 = √(3k)     masses move in opposite directions

Three experiments
-----------------
  1. Pure mode 1:  x1(0)=1, x2(0)=1,  x1'=x2'=0  → in-phase oscillation
  2. Pure mode 2:  x1(0)=1, x2(0)=-1, x1'=x2'=0  → out-of-phase oscillation
  3. Mixed modes:  x1(0)=1, x2(0)=0,  x1'=x2'=0  → beat phenomenon

What this example shows
-----------------------
- Coupled multi-DOF system solved with a two-output PINN
- Normal mode decomposition visible in the predicted trajectories
- Beating: energy exchange between two coupled oscillators

Tier 1 — Explorer.
"""

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K     = 1.0
M     = 1.0
T_END = 4 * math.pi   # two slow periods
N_COL = 400
N_EPOCHS = 10_000

# Normal mode frequencies
OMEGA1 = math.sqrt(K)          # in-phase
OMEGA2 = math.sqrt(3 * K)      # out-of-phase

EXPERIMENTS = [
    {"x1_0": 1.0, "x2_0":  1.0, "dx1_0": 0.0, "dx2_0": 0.0,
     "label": "Pure mode 1 (in-phase)"},
    {"x1_0": 1.0, "x2_0": -1.0, "dx1_0": 0.0, "dx2_0": 0.0,
     "label": "Pure mode 2 (out-of-phase)"},
    {"x1_0": 1.0, "x2_0":  0.0, "dx1_0": 0.0, "dx2_0": 0.0,
     "label": "Mixed modes (beating)"},
]


def exact(t, x1_0, x2_0, dx1_0, dx2_0):
    """
    Decompose ICs into normal mode amplitudes.
    q1 = (x1+x2)/2  →  q1'' + ω1² q1 = 0
    q2 = (x1-x2)/2  →  q2'' + ω2² q2 = 0
    """
    q1_0 = (x1_0 + x2_0) / 2
    q2_0 = (x1_0 - x2_0) / 2
    dq1_0 = (dx1_0 + dx2_0) / 2
    dq2_0 = (dx1_0 - dx2_0) / 2

    q1 = q1_0 * np.cos(OMEGA1 * t) + (dq1_0 / OMEGA1) * np.sin(OMEGA1 * t)
    q2 = q2_0 * np.cos(OMEGA2 * t) + (dq2_0 / OMEGA2) * np.sin(OMEGA2 * t)

    x1 = q1 + q2
    x2 = q1 - q2
    return x1, x2


def build_net():
    return nn.Sequential(
        nn.Linear(1, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 2),   # outputs: [x1, x2]
    )


def train(cfg):
    torch.manual_seed(5)
    net = build_net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    x1_0, x2_0 = cfg["x1_0"], cfg["x2_0"]
    dx1_0, dx2_0 = cfg["dx1_0"], cfg["dx2_0"]
    ic = torch.tensor([[x1_0, x2_0]], dtype=torch.float32)

    t_col = torch.linspace(0, T_END, N_COL).unsqueeze(1)
    t0    = torch.zeros(1, 1)

    for _ in range(N_EPOCHS):
        opt.zero_grad()

        tc  = t_col.clone().requires_grad_(True)
        z   = net(tc)                                         # (N, 2)
        x1, x2 = z[:, 0:1], z[:, 1:2]
        zd  = torch.autograd.grad(z.sum(), tc, create_graph=True)[0]
        x1d, x2d = zd[:, 0:1], zd[:, 1:2]
        zdd = torch.autograd.grad(zd.sum(), tc, create_graph=True)[0]
        x1dd, x2dd = zdd[:, 0:1], zdd[:, 1:2]

        # EOM  (m=1, k1=k2=k3=K)
        res1 = x1dd + 2*K*x1 - K*x2
        res2 = x2dd - K*x1  + 2*K*x2
        loss_pde = res1.pow(2).mean() + res2.pow(2).mean()

        # ICs: position
        z0  = net(t0)
        loss_ic_pos = (z0 - ic).pow(2).mean()

        # ICs: velocity
        t0g = t0.clone().requires_grad_(True)
        z0g = net(t0g)
        dz0 = torch.autograd.grad(z0g.sum(), t0g, create_graph=True)[0]
        vel_ic = torch.tensor([[dx1_0, dx2_0]], dtype=torch.float32)
        loss_ic_vel = (dz0 - vel_ic).pow(2).mean()

        (loss_pde + 200 * (loss_ic_pos + loss_ic_vel)).backward()
        opt.step()
        sch.step()

    t_eval = np.linspace(0, T_END, 600, dtype=np.float32)
    with torch.no_grad():
        z_pred = net(torch.tensor(t_eval[:, None])).numpy()
    return t_eval, z_pred[:, 0], z_pred[:, 1]


def main():
    print("Coupled Spring-Mass Oscillators — three mode experiments\n")
    fig, axes = plt.subplots(len(EXPERIMENTS), 2,
                             figsize=(14, 5 * len(EXPERIMENTS)))

    for row, cfg in zip(axes, EXPERIMENTS):
        print(f"Training: {cfg['label']} ...")
        t_p, x1_p, x2_p = train(cfg)
        x1_ex, x2_ex    = exact(t_p, cfg["x1_0"], cfg["x2_0"],
                                  cfg["dx1_0"], cfg["dx2_0"])

        l2 = float(np.sqrt(((x1_p - x1_ex)**2 + (x2_p - x2_ex)**2).mean()) /
                   (np.abs(x1_ex).mean() + np.abs(x2_ex).mean() + 1e-8))
        print(f"  → relative L2: {l2:.4e}")

        # Time series
        row[0].plot(t_p, x1_ex, "b-",  lw=2,   label="m1 exact")
        row[0].plot(t_p, x2_ex, "g-",  lw=2,   label="m2 exact")
        row[0].plot(t_p, x1_p,  "b--", lw=1.5, label="m1 PINN")
        row[0].plot(t_p, x2_p,  "g--", lw=1.5, label="m2 PINN")
        row[0].set_title(f"{cfg['label']}  (L2={l2:.2e})")
        row[0].set_xlabel("Time  t")
        row[0].set_ylabel("Displacement")
        row[0].legend(fontsize=8)
        row[0].grid(True, alpha=0.3)

        # Configuration space  (x1 vs x2)
        row[1].plot(x1_ex, x2_ex, "k-",  lw=1.5, label="Exact orbit")
        row[1].plot(x1_p,  x2_p,  "r--", lw=1,   label="PINN orbit")
        row[1].set_title("Configuration space  x1 vs x2")
        row[1].set_xlabel("x1"); row[1].set_ylabel("x2")
        row[1].legend(fontsize=8)
        row[1].grid(True, alpha=0.3)

    plt.suptitle("Coupled Oscillators  x1''=−2kx1+kx2,  x2''=kx1−2kx2",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("10_coupled_oscillators.png", dpi=130)
    print("\nSaved 10_coupled_oscillators.png")


if __name__ == "__main__":
    main()
