"""08_van_der_pol.py — Van der Pol Oscillator.

Physics problem
---------------
A nonlinear oscillator with amplitude-dependent damping:

    x'' − μ(1 − x²)x' + x = 0,    t ∈ [0, T]
    x(0) = 2,  x'(0) = 0

The nonlinear term μ(1−x²) acts as:
  - Negative damping when |x| < 1  → pumps energy in
  - Positive damping when |x| > 1  → dissipates energy
This self-regulation drives the system to a stable limit cycle.

Three experiments controlled by μ (stiffness / nonlinearity):
  - μ = 0.5  →  weakly nonlinear, nearly sinusoidal
  - μ = 2.0  →  moderately nonlinear
  - μ = 5.0  →  strongly nonlinear (stiff), sharp pulse-like waveform

What this example shows
-----------------------
- Self-excited oscillation and limit cycles
- How μ shapes the waveform from sinusoidal to pulse-like
- Stiff ODEs are harder for PINNs — observe the L2 increase with μ

Tier 1 — Explorer.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T_END    = 20.0
N_COL    = 600
N_EPOCHS = 12_000

EXPERIMENTS = [
    {"mu": 0.5, "label": "μ = 0.5  (weakly nonlinear)"},
    {"mu": 2.0, "label": "μ = 2.0  (moderate)"},
    {"mu": 5.0, "label": "μ = 5.0  (strongly nonlinear)"},
]


def reference(mu: float):
    def rhs(t, z):
        x, xd = z
        return [xd, mu * (1 - x**2) * xd - x]
    sol = solve_ivp(rhs, [0, T_END], [2.0, 0.0],
                    method="RK45", dense_output=True,
                    rtol=1e-9, atol=1e-11)
    t = np.linspace(0, T_END, 800)
    z = sol.sol(t)
    return t, z[0], z[1]


def build_net():
    return nn.Sequential(
        nn.Linear(1, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 1),
    )


def train(mu: float):
    torch.manual_seed(9)
    net = build_net()
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    t_col = torch.linspace(0, T_END, N_COL).unsqueeze(1)
    t0    = torch.zeros(1, 1)

    for _ in range(N_EPOCHS):
        opt.zero_grad()

        tc  = t_col.clone().requires_grad_(True)
        x   = net(tc)
        xd  = torch.autograd.grad(x.sum(), tc, create_graph=True)[0]
        xdd = torch.autograd.grad(xd.sum(), tc, create_graph=True)[0]
        res = xdd - mu * (1 - x**2) * xd + x
        loss_pde = res.pow(2).mean()

        t0g  = t0.clone().requires_grad_(True)
        x0   = net(t0g)
        dx0  = torch.autograd.grad(x0.sum(), t0g, create_graph=True)[0]
        loss_ic = (x0 - 2.0).pow(2) + dx0.pow(2)

        (loss_pde + 200 * loss_ic).backward()
        opt.step()
        sch.step()

    t_eval = np.linspace(0, T_END, 800, dtype=np.float32)
    with torch.no_grad():
        x_pred = net(torch.tensor(t_eval[:, None])).numpy().ravel()
    return t_eval, x_pred


def main():
    print("Van der Pol Oscillator — three μ experiments\n")
    fig, axes = plt.subplots(len(EXPERIMENTS), 2,
                             figsize=(14, 5 * len(EXPERIMENTS)))

    for row, cfg in zip(axes, EXPERIMENTS):
        mu = cfg["mu"]
        print(f"Training: {cfg['label']} ...")
        t_ref, x_ref, xd_ref = reference(mu)
        t_p, x_p              = train(mu)

        l2 = float(np.sqrt(((np.interp(t_p, t_ref, x_ref) - x_p)**2).mean()) /
                   (np.abs(x_ref).mean() + 1e-8))
        print(f"  → relative L2: {l2:.4e}")

        # Time series
        row[0].plot(t_ref, x_ref, "k-",  lw=2,   label="Reference (RK45)")
        row[0].plot(t_p,   x_p,   "r--", lw=1.5, label=f"PINN (L2={l2:.2e})")
        row[0].set_title(cfg["label"])
        row[0].set_xlabel("Time  t")
        row[0].set_ylabel("x(t)")
        row[0].legend(fontsize=9)
        row[0].grid(True, alpha=0.3)

        # Phase portrait (limit cycle)
        xd_p = np.gradient(x_p, t_p)
        row[1].plot(x_ref, xd_ref, "k-",  lw=1.5, label="Exact orbit")
        row[1].plot(x_p,   xd_p,   "r--", lw=1,   label="PINN orbit")
        row[1].set_title("Phase portrait (limit cycle)")
        row[1].set_xlabel("x")
        row[1].set_ylabel("ẋ")
        row[1].legend(fontsize=9)
        row[1].grid(True, alpha=0.3)

    plt.suptitle("Van der Pol Oscillator   x'' − μ(1−x²)x' + x = 0",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("08_van_der_pol.png", dpi=130)
    print("\nSaved 08_van_der_pol.png")


if __name__ == "__main__":
    main()
