"""05_logistic_growth.py — Logistic Growth / Population Dynamics.

Physics problem
---------------
Population dynamics with carrying capacity:

    dN/dt = r·N·(1 - N/K),    t ∈ [0, T]
    N(0) = N0

Exact solution:
    N(t) = K / (1 + ((K - N0)/N0)·e^{-rt})

Parameters:
  - r  = intrinsic growth rate
  - K  = carrying capacity (maximum sustainable population)
  - N0 = initial population

Three experiments:
  - Fast growth:  r=2.0, K=1.0, N0=0.1
  - Slow growth:  r=0.5, K=1.0, N0=0.1
  - Above K:      r=1.5, K=1.0, N0=1.5  (population decays to carrying capacity)

What this example shows
-----------------------
- Nonlinear first-order ODE solved with PINN
- Conservation of the carrying capacity equilibrium
- Behaviour above and below K

Tier 1 — Explorer.
"""

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T_END    = 8.0
N_COL    = 300
N_EPOCHS = 8_000

EXPERIMENTS = [
    {"r": 2.0, "K": 1.0, "N0": 0.1, "label": "Fast growth  r=2.0"},
    {"r": 0.5, "K": 1.0, "N0": 0.1, "label": "Slow growth  r=0.5"},
    {"r": 1.5, "K": 1.0, "N0": 1.5, "label": "Above K → decay  N0=1.5"},
]


def exact(t, r, K, N0):
    return K / (1 + ((K - N0) / N0) * np.exp(-r * t))


def build_net():
    # Softplus output to keep N > 0 always
    class LogisticNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, 32), nn.Tanh(),
                nn.Linear(32, 32), nn.Tanh(),
                nn.Linear(32, 1),
            )
        def forward(self, t):
            return torch.nn.functional.softplus(self.net(t))
    return LogisticNet()


def train(r, K, N0):
    torch.manual_seed(0)
    net = build_net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=3000, gamma=0.3)

    t_col = torch.linspace(0, T_END, N_COL).unsqueeze(1)
    t0    = torch.zeros(1, 1)

    for _ in range(N_EPOCHS):
        opt.zero_grad()

        tc = t_col.clone().requires_grad_(True)
        N  = net(tc)
        Nt = torch.autograd.grad(N.sum(), tc, create_graph=True)[0]
        res = Nt - r * N * (1 - N / K)
        loss_pde = res.pow(2).mean()

        loss_ic = (net(t0) - N0).pow(2)
        (loss_pde + 500 * loss_ic).backward()
        opt.step()
        sch.step()

    t_eval = np.linspace(0, T_END, 400, dtype=np.float32)
    with torch.no_grad():
        N_pred = net(torch.tensor(t_eval[:, None])).numpy().ravel()
    return t_eval, N_pred, exact(t_eval, r, K, N0)


def main():
    print("Logistic Growth — three population experiments\n")
    fig, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(16, 5))

    for ax, cfg in zip(axes, EXPERIMENTS):
        print(f"Training: {cfg['label']} ...")
        t, N_pinn, N_true = train(cfg["r"], cfg["K"], cfg["N0"])
        l2 = float(np.sqrt(((N_pinn - N_true) ** 2).mean()) /
                   np.sqrt((N_true ** 2).mean() + 1e-8))

        ax.plot(t, N_true, "k-",  lw=2,   label="Exact")
        ax.plot(t, N_pinn, "r--", lw=1.5, label=f"PINN  L2={l2:.2e}")
        ax.axhline(cfg["K"], color="steelblue", ls="--", lw=1, label=f"K = {cfg['K']}")
        ax.set_title(cfg["label"])
        ax.set_xlabel("Time  t")
        ax.set_ylabel("Population  N(t)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        print(f"  → relative L2: {l2:.4e}")

    plt.suptitle("Logistic Growth   dN/dt = r·N·(1 − N/K)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("05_logistic_growth.png", dpi=130)
    print("\nSaved 05_logistic_growth.png")


if __name__ == "__main__":
    main()
