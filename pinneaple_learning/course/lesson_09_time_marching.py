"""Lesson 09 — Time-Marching PINNs for Stiff and Long-Time Problems.

Physics problem
---------------
Van der Pol oscillator (stiff when μ is large):
    x'' − μ(1 − x²)x' + x = 0
    x(0) = 2,   x'(0) = 0

For μ=3: stiff, self-excited oscillation with limit cycle.
For μ=0.1: near-harmonic, easy for standard PINN.

The challenge
-------------
Standard PINNs solve the ENTIRE time domain at once. For stiff problems
(large μ) or long times, the loss landscape becomes very ill-conditioned
and training fails.

Solution: Time-marching — divide [0, T] into windows and solve sequentially.
The solution at the end of window k becomes the IC for window k+1.

PINNeAPPle's TimeMarchingTrainer does this automatically.

Key classes
-----------
  from pinneaple_train import TimeMarchingTrainer
  from pinneaple_train import CausalPINNTrainer  (alternative: causal weighting)

Run this lesson
---------------
    python -m pinneaple_learning.course.lesson_09_time_marching
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
import scipy.integrate as sci
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PINNeAPPle training imports
from pinneaple_train import TimeMarchingTrainer, CausalPINNTrainer

MU     = 3.0      # stiffness parameter — high = stiff
T_END  = 15.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 0


# ── Reference solution (scipy RK45) ──────────────────────────────────────
def vdp_reference(mu: float, t_end: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def vdp(t, y):
        x, xd = y
        return [xd, mu * (1 - x**2) * xd - x]

    sol = sci.solve_ivp(vdp, [0, t_end], [2.0, 0.0],
                        method="RK45", max_step=0.01,
                        dense_output=True)
    t_np = np.linspace(0, t_end, 1000)
    y    = sol.sol(t_np)
    return t_np, y[0], y[1]


# ── Network ────────────────────────────────────────────────────────────────
def make_net() -> nn.Module:
    return nn.Sequential(
        nn.Linear(1, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 2),   # outputs: [x, x']
    )


# ── Van der Pol residuals ─────────────────────────────────────────────────
def vdp_residual(net: nn.Module, t: torch.Tensor, mu: float) -> torch.Tensor:
    t   = t.requires_grad_(True)
    z   = net(t)          # (N, 2)  [x, x']
    x, xd = z[:, 0:1], z[:, 1:2]
    zd   = torch.autograd.grad(z.sum(), t, create_graph=True)[0]
    xdd  = zd[:, 0:1]    # x'' = (dz/dt)[0]
    # ODE system:  x'' - μ(1-x²)x' + x = 0
    res_xdd = xdd - mu * (1 - x**2) * xd + x
    # Consistency:  x' = dz[1]/dt  (automatically satisfied by the 2-output net)
    return res_xdd


# ── Approach 1: Standard PINN (global time domain) ───────────────────────
def train_standard(mu: float) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(SEED)
    net = make_net().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=8_000)

    t_col = torch.linspace(0, T_END, 600, device=DEVICE).unsqueeze(1)
    t0    = torch.zeros(1, 1, device=DEVICE)

    for _ in range(8_000):
        opt.zero_grad()
        r     = vdp_residual(net, t_col.clone(), mu)
        l_pde = r.pow(2).mean()

        z0 = net(t0)
        l_ic_x  = (z0[0, 0] - 2.0).pow(2)
        l_ic_xd = (z0[0, 1] - 0.0).pow(2)

        (l_pde + 1000.0 * (l_ic_x + l_ic_xd)).backward()
        opt.step(); sch.step()

    t_np = np.linspace(0, T_END, 1000, dtype=np.float32)
    with torch.no_grad():
        z = net(torch.tensor(t_np[:, None], device=DEVICE))
    return t_np, z[:, 0].cpu().numpy()


# ── Approach 2: TimeMarchingTrainer ───────────────────────────────────────
def train_time_marching(mu: float) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(SEED)
    net = make_net().to(DEVICE)

    def make_loss(ic_state: torch.Tensor, t_start: float, t_end: float):
        """Loss factory for one time window. ic_state = [x0, xd0]."""
        def loss_fn(model: nn.Module) -> dict:
            t_col = torch.rand(300, 1, device=DEVICE) * (t_end - t_start) + t_start
            r     = vdp_residual(model, t_col.clone(), mu)
            l_pde = r.pow(2).mean()

            t0s   = torch.full((1, 1), t_start, device=DEVICE)
            z0    = model(t0s)
            l_ic  = (z0 - ic_state.unsqueeze(0)).pow(2).sum()
            return {"total": l_pde + 1000.0 * l_ic, "pde": l_pde, "ic": l_ic}
        return loss_fn

    # TimeMarchingTrainer divides [0, T_END] into windows and solves each
    trainer = TimeMarchingTrainer(
        model_factory = make_net,                  # fresh net per window
        loss_factory  = make_loss,
        t_start       = 0.0,
        t_end         = T_END,
        n_windows     = 15,                        # 15 windows of 1 time unit each
        epochs_per_window = 3_000,
        lr            = 1e-3,
        device        = DEVICE,
        ic_state      = torch.tensor([2.0, 0.0]),  # initial state
    )
    t_all, x_all = trainer.integrate()
    return t_all, x_all


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    print("─" * 60)
    print("  Lesson 09 — Time-Marching PINNs")
    print("─" * 60)
    print(f"\n  Van der Pol oscillator:  μ={MU} (stiff)")
    print(f"  x'' − μ(1−x²)x' + x = 0,  x(0)=2, x'(0)=0")
    print(f"  T_end = {T_END}\n")

    # Reference
    print("  Computing scipy reference solution...")
    t_ref, x_ref, _ = vdp_reference(MU, T_END)

    # Standard PINN
    print("  [1/2]  Standard PINN (full domain)...")
    t_std, x_std = train_standard(MU)

    # Time-marching
    print("  [2/2]  TimeMarchingTrainer (15 windows)...")
    t_tm, x_tm = train_time_marching(MU)

    # Errors
    def rel_l2_interp(t_pred, x_pred):
        x_ref_interp = np.interp(t_pred, t_ref, x_ref)
        return float(np.sqrt(((x_pred - x_ref_interp)**2).mean()) /
                     (np.abs(x_ref_interp).mean() + 1e-8))

    e_std = rel_l2_interp(t_std, x_std)
    e_tm  = rel_l2_interp(t_tm,  x_tm)
    print(f"\n  Standard PINN   relative L2: {e_std:.4e}")
    print(f"  Time-marching   relative L2: {e_tm:.4e}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.plot(t_ref, x_ref, "k-",  lw=2.5, label="scipy reference")
    ax.plot(t_std, x_std, "r--", lw=1.5, label=f"Standard PINN (L2={e_std:.2e})")
    ax.set_title(f"Standard PINN  (μ={MU}, T={T_END})", fontsize=10)
    ax.set_xlabel("t"); ax.set_ylabel("x(t)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_ref, x_ref, "k-",  lw=2.5, label="scipy reference")
    ax.plot(t_tm,  x_tm,  "b--", lw=1.5, label=f"TimeMarchingTrainer (L2={e_tm:.2e})")
    ax.set_title(f"TimeMarchingTrainer — 15 windows (μ={MU})", fontsize=10)
    ax.set_xlabel("t"); ax.set_ylabel("x(t)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle(f"Lesson 09 — Van der Pol  μ={MU}  (stiff limit cycle)", fontsize=12)
    plt.tight_layout()
    out = "lesson_09_time_marching.png"
    plt.savefig(out, dpi=130)
    print(f"\n  Saved {out}")

    print(f"""
  Key takeaways:
    1. Standard PINN on a stiff long-time problem typically fails (L2~{e_std:.0e}).
       The loss landscape is very ill-conditioned for large μ.
    2. TimeMarchingTrainer splits the domain into windows and chains them.
       Each window is small and well-conditioned (L2~{e_tm:.0e}).
    3. The IC for window k+1 comes from the network's output at t=end of k.
    4. PINNeAPPle's CausalPINNTrainer is an alternative: it uses causal
       weighting so early times are solved before late times.

  When to use time-marching:
    • Stiff ODEs (large μ in Van der Pol, stiff reaction networks)
    • Long-time integration (many oscillation periods)
    • Problems where the solution changes character over time

  Next lesson:
    python -m pinneaple_learning.course.lesson_10_validation
""")


if __name__ == "__main__":
    main()
