"""01_harmonic_oscillator.py — Simple Harmonic Oscillator with PINNs.

Physics problem
---------------
A mass on a spring with no damping:

    x''(t) + ω² x(t) = 0,    t ∈ [0, T]
    x(0) = x0,  x'(0) = v0

Exact solution:  x(t) = x0·cos(ωt) + (v0/ω)·sin(ωt)

What this example shows
-----------------------
- How to encode an ODE as a PINN loss (no library needed beyond PyTorch)
- How different spring constants (ω²) change the solution
- How to enforce initial conditions as soft constraints
- Direct comparison between PINN prediction and exact solution

Tier 1 — Explorer: just run this file and look at the plot.
"""

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Experiment configurations  (feel free to change these)
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {"omega": 1.0, "x0": 1.0, "v0": 0.0, "label": "ω=1  (slow)"},
    {"omega": 3.0, "x0": 1.0, "v0": 0.0, "label": "ω=3  (medium)"},
    {"omega": 6.0, "x0": 0.5, "v0": 2.0, "label": "ω=6  (fast, shifted IC)"},
]
T_END   = 2 * math.pi   # simulate one full slow period
N_COL   = 300            # collocation points
N_EPOCHS = 8_000


# ---------------------------------------------------------------------------
# Exact solution
# ---------------------------------------------------------------------------
def exact(t: np.ndarray, omega: float, x0: float, v0: float) -> np.ndarray:
    return x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)


# ---------------------------------------------------------------------------
# Network  (tiny MLP with Tanh activations)
# ---------------------------------------------------------------------------
def build_net() -> nn.Module:
    return nn.Sequential(
        nn.Linear(1, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    )


# ---------------------------------------------------------------------------
# Physics residual  x'' + ω² x = 0
# ---------------------------------------------------------------------------
def pde_residual(net: nn.Module, t: torch.Tensor, omega: float) -> torch.Tensor:
    t = t.requires_grad_(True)
    x  = net(t)
    x_t  = torch.autograd.grad(x.sum(), t, create_graph=True)[0]
    x_tt = torch.autograd.grad(x_t.sum(), t, create_graph=True)[0]
    return x_tt + omega ** 2 * x


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(omega: float, x0: float, v0: float) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(42)
    net = build_net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    t_col = torch.linspace(0, T_END, N_COL).unsqueeze(1)
    t0    = torch.zeros(1, 1)

    for epoch in range(1, N_EPOCHS + 1):
        opt.zero_grad()

        # PDE residual
        res  = pde_residual(net, t_col.clone(), omega)
        loss_pde = res.pow(2).mean()

        # Initial conditions:  x(0)=x0,  x'(0)=v0
        t0_g = t0.clone().requires_grad_(True)
        x_t0 = net(t0_g)
        dx_t0 = torch.autograd.grad(x_t0.sum(), t0_g, create_graph=True)[0]
        loss_ic = (x_t0 - x0).pow(2) + (dx_t0 - v0).pow(2)

        loss = loss_pde + 100 * loss_ic
        loss.backward()
        opt.step()
        sch.step()

    t_eval = np.linspace(0, T_END, 400, dtype=np.float32)
    with torch.no_grad():
        x_pred = net(torch.tensor(t_eval[:, None])).numpy().ravel()
    x_ex = exact(t_eval, omega, x0, v0)
    return t_eval, x_pred, x_ex


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Simple Harmonic Oscillator — PINN experiments\n")

    fig, axes = plt.subplots(len(EXPERIMENTS), 1,
                             figsize=(11, 4 * len(EXPERIMENTS)), sharex=False)

    for ax, cfg in zip(axes, EXPERIMENTS):
        print(f"Training: {cfg['label']} ...")
        t, x_pinn, x_true = train(cfg["omega"], cfg["x0"], cfg["v0"])

        l2 = float(np.sqrt(((x_pinn - x_true) ** 2).mean()) /
                   np.sqrt((x_true ** 2).mean() + 1e-8))

        ax.plot(t, x_true, "k-",  lw=2,   label="Exact solution")
        ax.plot(t, x_pinn, "r--", lw=1.5, label=f"PINN  (rel-L2={l2:.2e})")
        ax.set_title(f"Harmonic oscillator — {cfg['label']}")
        ax.set_xlabel("Time  t")
        ax.set_ylabel("Displacement  x(t)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        print(f"  → relative L2 error: {l2:.4e}")

    plt.tight_layout()
    plt.savefig("01_harmonic_oscillator.png", dpi=130)
    print("\nSaved 01_harmonic_oscillator.png")


if __name__ == "__main__":
    main()
