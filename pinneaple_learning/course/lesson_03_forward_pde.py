"""Lesson 03 — Forward PDE: Heat Equation with TwoPhaseTrainer.

Physics problem
---------------
1D heat equation:
    u_t = α u_xx,    (x,t) ∈ [0,1] × [0,1]
    u(x,0) = sin(πx)
    u(0,t) = u(1,t) = 0

Exact solution:  u(x,t) = exp(−α π² t) sin(πx)

What you will learn
-------------------
  Step 1 — Define the PDE symbolically with PINNeAPPle's SymbolicPDE
  Step 2 — Build boundary and initial conditions
  Step 3 — Use TwoPhaseTrainer (Adam warm-up → L-BFGS refinement)
  Step 4 — Use compare_to_analytical to measure accuracy

TwoPhaseTrainer strategy
-------------------------
  Phase 1 (Adam): fast convergence to a good basin
  Phase 2 (L-BFGS): high-accuracy refinement
  This combination is the standard approach for research-grade PINNs.

Run this lesson
---------------
    python -m pinneaple_learning.course.lesson_03_forward_pde
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sympy as sp

# PINNeAPPle imports for this lesson
from pinneaple_symbolic import SymbolicPDE, DirichletBC
from pinneaple_train    import TwoPhaseTrainer, TwoPhaseConfig
from pinneaple_validate import compare_to_analytical

ALPHA   = 0.05
T_MAX   = 1.0
N_COL   = 3_000
N_BC    = 300
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"


# ── Network ────────────────────────────────────────────────────────────────
def make_net() -> nn.Module:
    return nn.Sequential(
        nn.Linear(2, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    )


# ── Exact solution for validation ─────────────────────────────────────────
def exact(xt: np.ndarray) -> np.ndarray:
    x, t = xt[:, 0], xt[:, 1]
    return np.exp(-ALPHA * math.pi**2 * t) * np.sin(math.pi * x)


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    print("─" * 60)
    print("  Lesson 03 — Forward PDE: Heat Equation")
    print("─" * 60)
    print(f"\n  u_t = α u_xx,  α={ALPHA},  (x,t)∈[0,1]×[0,{T_MAX}]")
    print("  BCs: u(0,t) = u(1,t) = 0")
    print("  IC:  u(x,0) = sin(πx)")
    print("  Exact: exp(−α π² t) sin(πx)\n")

    torch.manual_seed(0)
    net = make_net().to(DEVICE)

    # ── Step 1: Symbolic PDE ───────────────────────────────────────────────
    print("  [Step 1] Defining PDE symbolically with SymbolicPDE...")
    x_sym = sp.Symbol("x")
    t_sym = sp.Symbol("t")
    u_sym = sp.Function("u")
    a_sym = sp.Symbol("alpha")

    heat_pde = SymbolicPDE(
        expr       = u_sym(x_sym, t_sym).diff(t_sym)
                     - a_sym * u_sym(x_sym, t_sym).diff(x_sym, 2),
        coord_syms = [x_sym, t_sym],
        field_syms = [u_sym],
        param_syms = [a_sym],
    )
    residual_fn = heat_pde.to_residual_fn(net)

    # ── Step 2: Boundary and initial conditions ────────────────────────────
    print("  [Step 2] Setting up BCs and IC...")
    bc_left  = DirichletBC(value_fn=lambda xt: torch.zeros(len(xt), 1, device=DEVICE))
    bc_right = DirichletBC(value_fn=lambda xt: torch.zeros(len(xt), 1, device=DEVICE))

    def sample_col():
        x = torch.rand(N_COL, 1)
        t = torch.rand(N_COL, 1) * T_MAX
        return torch.cat([x, t], dim=1).to(DEVICE)

    def sample_bc_left():
        x = torch.zeros(N_BC, 1)
        t = torch.rand(N_BC, 1) * T_MAX
        return torch.cat([x, t], dim=1).to(DEVICE)

    def sample_bc_right():
        x = torch.ones(N_BC, 1)
        t = torch.rand(N_BC, 1) * T_MAX
        return torch.cat([x, t], dim=1).to(DEVICE)

    def sample_ic():
        x = torch.rand(N_BC, 1)
        t = torch.zeros(N_BC, 1)
        xt = torch.cat([x, t], dim=1).to(DEVICE)
        u0 = torch.sin(math.pi * x).to(DEVICE)
        return xt, u0

    xt_ic, u_ic = sample_ic()

    # ── Step 3: TwoPhaseTrainer ────────────────────────────────────────────
    print("  [Step 3] Training with TwoPhaseTrainer (Adam → L-BFGS)...")

    def pinn_loss(model: nn.Module) -> dict[str, torch.Tensor]:
        xt_col = sample_col()
        r      = residual_fn(xt_col.requires_grad_(True))
        l_pde  = (r**2).mean()
        l_bc   = (bc_left.loss(model, sample_bc_left()) +
                  bc_right.loss(model, sample_bc_right()))
        l_ic   = (model(xt_ic) - u_ic).pow(2).mean()
        total  = l_pde + 10.0 * l_bc + 100.0 * l_ic
        return {"total": total, "pde": l_pde, "bc": l_bc, "ic": l_ic}

    cfg = TwoPhaseConfig(
        phase1_epochs   = 5_000,   # Adam warm-up
        phase2_epochs   = 1_000,   # L-BFGS refinement
        phase1_lr       = 1e-3,
        phase2_lr       = 1.0,
        device          = DEVICE,
    )
    trainer = TwoPhaseTrainer(net, pinn_loss, cfg)
    history = trainer.train()
    print("    Training complete.")

    # ── Step 4: Validate against exact solution ────────────────────────────
    print("  [Step 4] Validating with compare_to_analytical...")
    metrics = compare_to_analytical(
        model        = net,
        analytical_fn = exact,
        coord_names  = ["x", "t"],
        domain_bounds = {"x": (0.0, 1.0), "t": (0.0, T_MAX)},
        n_points     = 10_000,
        device       = DEVICE,
    )
    print(f"    Relative L2  : {metrics['rel_l2']:.4e}")
    print(f"    RMSE         : {metrics['rmse']:.4e}")
    print(f"    Max error    : {metrics['max_error']:.4e}")

    # ── Plot ───────────────────────────────────────────────────────────────
    x_np  = np.linspace(0, 1, 80, dtype=np.float32)
    t_slices = [0.0, 0.1, 0.3, 0.5, 1.0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(t_slices)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    for col, t_val in zip(colors, t_slices):
        t_np = np.full_like(x_np, t_val)
        xt   = torch.tensor(np.stack([x_np, t_np], axis=1), device=DEVICE)
        with torch.no_grad():
            u_p = net(xt).cpu().numpy().ravel()
        u_e = np.exp(-ALPHA * math.pi**2 * t_val) * np.sin(math.pi * x_np)
        ax.plot(x_np, u_e, "-",  color=col, lw=2,   label=f"t={t_val:.1f} exact")
        ax.plot(x_np, u_p, "--", color=col, lw=1.5, label=f"t={t_val:.1f} PINN")
    ax.set_title(f"Heat equation — multiple time slices (α={ALPHA})")
    ax.set_xlabel("x"); ax.set_ylabel("u(x,t)")
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    ax = axes[1]
    loss_names = [k for k in history[0] if k != "total"]
    for name in loss_names:
        vals = [h[name] for h in history if name in h]
        ax.semilogy(vals, label=name, lw=1.5)
    ax.axvline(cfg.phase1_epochs, color="k", ls=":", lw=1, label="Adam→L-BFGS")
    ax.set_title("Loss history (phase 1: Adam, phase 2: L-BFGS)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle(f"Lesson 03 — Heat Equation Forward Solve  α={ALPHA}", fontsize=12)
    plt.tight_layout()
    out = "lesson_03_forward_pde.png"
    plt.savefig(out, dpi=130)
    print(f"\n  Saved {out}")

    print(f"""
  Key takeaways:
    1. SymbolicPDE: one SymPy expression → autograd residual. Works for any PDE.
    2. TwoPhaseTrainer: Adam finds a good basin, L-BFGS achieves high accuracy.
       Final L2 = {metrics['rel_l2']:.2e} — typical for a 1D heat equation.
    3. compare_to_analytical: your validation shortcut. Always call this.
    4. Note how the L_pde curve drops faster in Phase 2 (L-BFGS).

  Next lesson:
    python -m pinneaple_learning.course.lesson_04_geometry
""")


if __name__ == "__main__":
    main()
