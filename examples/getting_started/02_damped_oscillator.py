"""02_damped_oscillator.py — Damped Harmonic Oscillator.

Physics problem
---------------
A mass-spring-damper system:

    x''(t) + 2ζω x'(t) + ω² x(t) = 0,    t ∈ [0, T]
    x(0) = 1,  x'(0) = 0

Three physical regimes controlled by the damping ratio ζ:
  - ζ < 1  →  underdamped  (oscillates and decays)
  - ζ = 1  →  critically damped  (fastest non-oscillatory decay)
  - ζ > 1  →  overdamped  (slow exponential decay)

Exact solutions
---------------
  Underdamped (ζ<1):     x = e^{-ζωt} [ cos(ω_d t) + (ζ/√(1-ζ²)) sin(ω_d t) ]
                         ω_d = ω √(1-ζ²)
  Critically damped:     x = e^{-ωt} (1 + ωt)
  Overdamped (ζ>1):      x = e^{-ζωt} [ cosh(ω_d t) + (ζ/√(ζ²-1)) sinh(ω_d t) ]
                         ω_d = ω √(ζ²-1)

What this example shows
-----------------------
- A single PINN architecture solves all three regimes
- How damping ratio changes the qualitative physics
- Soft IC constraints with gradient enforcement

Tier 1 — Explorer.
"""

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OMEGA   = 2.0   # natural frequency (rad/s)
T_END   = 6.0   # simulation time (s)
N_COL   = 300
N_EPOCHS = 8_000

REGIMES = [
    {"zeta": 0.1, "label": "Underdamped  ζ=0.1"},
    {"zeta": 1.0, "label": "Critically damped  ζ=1.0"},
    {"zeta": 2.0, "label": "Overdamped  ζ=2.0"},
]


# ---------------------------------------------------------------------------
# Exact solutions
# ---------------------------------------------------------------------------
def exact(t: np.ndarray, zeta: float, omega: float = OMEGA) -> np.ndarray:
    if zeta < 1.0:
        wd = omega * math.sqrt(1 - zeta ** 2)
        return np.exp(-zeta * omega * t) * (
            np.cos(wd * t) + (zeta / math.sqrt(1 - zeta ** 2)) * np.sin(wd * t)
        )
    elif abs(zeta - 1.0) < 1e-9:
        return np.exp(-omega * t) * (1 + omega * t)
    else:
        wd = omega * math.sqrt(zeta ** 2 - 1)
        return np.exp(-zeta * omega * t) * (
            np.cosh(wd * t) + (zeta / math.sqrt(zeta ** 2 - 1)) * np.sinh(wd * t)
        )


# ---------------------------------------------------------------------------
# PINN
# ---------------------------------------------------------------------------
def build_net() -> nn.Module:
    return nn.Sequential(
        nn.Linear(1, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    )


def train(zeta: float) -> tuple:
    torch.manual_seed(0)
    net = build_net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=3000, gamma=0.3)

    t_col = torch.linspace(0, T_END, N_COL).unsqueeze(1)
    t0    = torch.zeros(1, 1)

    for _ in range(N_EPOCHS):
        opt.zero_grad()

        tc = t_col.clone().requires_grad_(True)
        x  = net(tc)
        xt  = torch.autograd.grad(x.sum(), tc, create_graph=True)[0]
        xtt = torch.autograd.grad(xt.sum(), tc, create_graph=True)[0]
        res = xtt + 2 * zeta * OMEGA * xt + OMEGA ** 2 * x
        loss_pde = res.pow(2).mean()

        t0g  = t0.clone().requires_grad_(True)
        x0   = net(t0g)
        dx0  = torch.autograd.grad(x0.sum(), t0g, create_graph=True)[0]
        loss_ic = x0.pow(2) + (x0 - 1).pow(2) + dx0.pow(2)

        (loss_pde + 200 * loss_ic).backward()
        opt.step()
        sch.step()

    t_eval = np.linspace(0, T_END, 400, dtype=np.float32)
    with torch.no_grad():
        x_pred = net(torch.tensor(t_eval[:, None])).numpy().ravel()
    return t_eval, x_pred, exact(t_eval, zeta)


def main():
    print("Damped Harmonic Oscillator — three physical regimes\n")
    fig, axes = plt.subplots(1, len(REGIMES), figsize=(16, 5))

    for ax, cfg in zip(axes, REGIMES):
        print(f"Training: {cfg['label']} ...")
        t, x_pinn, x_true = train(cfg["zeta"])
        l2 = float(np.sqrt(((x_pinn - x_true) ** 2).mean()) /
                   np.sqrt((x_true ** 2).mean() + 1e-8))

        ax.plot(t, x_true, "k-",  lw=2,   label="Exact")
        ax.plot(t, x_pinn, "r--", lw=1.5, label=f"PINN  L2={l2:.2e}")
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.set_title(cfg["label"])
        ax.set_xlabel("Time  t  (s)")
        ax.set_ylabel("x(t)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        print(f"  → relative L2: {l2:.4e}")

    plt.suptitle("Damped Harmonic Oscillator  —  x'' + 2ζω x' + ω² x = 0",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("02_damped_oscillator.png", dpi=130)
    print("\nSaved 02_damped_oscillator.png")


if __name__ == "__main__":
    main()
