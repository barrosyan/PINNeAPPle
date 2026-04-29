"""Lesson 01 — Your First PINN with PINNeAPPle.

Physics problem
---------------
Harmonic oscillator:
    x''(t) + ω² x(t) = 0,    t ∈ [0, 4π]
    x(0)  = 1   (initial position)
    x'(0) = 0   (initial velocity)

Exact solution:  x(t) = cos(ωt)

What you will learn
-------------------
  Step 1 — Build a PINN manually (raw PyTorch) to understand what's happening
  Step 2 — Rewrite it with PINNeAPPle's SymbolicPDE for the residual
  Step 3 — Use HardBC to enforce x(0) = 1 exactly (no penalty needed)
  Step 4 — Compare accuracy and plot results

Run this lesson
---------------
    python -m pinneaple_learning.course.lesson_01_first_pinn
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

# PINNeAPPle imports — the key actors in this lesson
from pinneaple_symbolic import SymbolicPDE, HardBC

# ── Constants ──────────────────────────────────────────────────────────────
OMEGA   = 2.0          # angular frequency
T_END   = 4 * math.pi  # two full periods
N_COL   = 300          # collocation points
EPOCHS  = 8_000
LR      = 1e-3
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"


# ── Helper: a plain Tanh MLP ───────────────────────────────────────────────
def make_mlp(in_dim: int = 1, out_dim: int = 1, width: int = 64, depth: int = 4) -> nn.Module:
    layers = [nn.Linear(in_dim, width), nn.Tanh()]
    for _ in range(depth - 2):
        layers += [nn.Linear(width, width), nn.Tanh()]
    layers.append(nn.Linear(width, out_dim))
    return nn.Sequential(*layers)


# ── Step 1: Manual PINN (raw PyTorch) ─────────────────────────────────────
def _train_manual(omega: float) -> tuple[np.ndarray, np.ndarray]:
    """The classic PINN loop — every operation spelled out for clarity."""
    torch.manual_seed(42)
    net = make_mlp().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    t_col = torch.linspace(0, T_END, N_COL, device=DEVICE).unsqueeze(1)
    t0    = torch.zeros(1, 1, device=DEVICE)

    for epoch in range(EPOCHS):
        opt.zero_grad()

        # ── PDE residual  (autograd computes x'' exactly) ──────────────
        tc  = t_col.clone().requires_grad_(True)
        x   = net(tc)
        xd  = torch.autograd.grad(x.sum(), tc, create_graph=True)[0]
        xdd = torch.autograd.grad(xd.sum(), tc, create_graph=True)[0]
        loss_pde = (xdd + omega**2 * x).pow(2).mean()

        # ── Initial condition losses ────────────────────────────────────
        x0  = net(t0)
        loss_pos = (x0 - 1.0).pow(2)

        t0g = t0.requires_grad_(True)
        xd0 = torch.autograd.grad(net(t0g).sum(), t0g, create_graph=True)[0]
        loss_vel = xd0.pow(2)

        loss = loss_pde + 200.0 * (loss_pos + loss_vel)
        loss.backward()
        opt.step()

    t_np = np.linspace(0, T_END, 500, dtype=np.float32)
    with torch.no_grad():
        x_pred = net(torch.tensor(t_np[:, None], device=DEVICE)).cpu().numpy().ravel()
    return t_np, x_pred


# ── Step 2 & 3: PINNeAPPle SymbolicPDE + HardBC ───────────────────────────
def _train_pinneaple(omega: float) -> tuple[np.ndarray, np.ndarray]:
    """
    PINNeAPPle approach:
      • SymbolicPDE converts a SymPy expression into an autograd residual function
      • HardBC wraps the network so x(0) = 1 is enforced exactly via ansatz:
            x_wrapped(t) = 1 + t * net(t)     →  x_wrapped(0) = 1, always
    """
    torch.manual_seed(42)

    # ── Step 2: Define the PDE symbolically ────────────────────────────────
    t_sym  = sp.Symbol("t")
    x_sym  = sp.Function("x")
    om_sym = sp.Symbol("omega")

    pde = SymbolicPDE(
        expr       = x_sym(t_sym).diff(t_sym, 2) + om_sym**2 * x_sym(t_sym),
        coord_syms = [t_sym],
        field_syms = [x_sym],
        param_syms = [om_sym],
    )

    # ── Step 3: Hard-enforce x(0) = 1 with ansatz  x(t) = 1 + t·net(t) ───
    net = make_mlp().to(DEVICE)
    hard_ic = HardBC(
        distance_fn  = lambda t: t,                           # φ(t) = t
        bc_value_fn  = lambda t: torch.ones_like(t),          # g(t) = 1.0
    )
    net_wrapped = hard_ic.wrap_model(net).to(DEVICE)

    # Bind omega into the residual function
    residual_fn = pde.to_residual_fn(net_wrapped)

    opt = torch.optim.Adam(net_wrapped.parameters(), lr=LR)
    t_col = torch.linspace(0, T_END, N_COL, device=DEVICE).unsqueeze(1)
    t0    = torch.zeros(1, 1, device=DEVICE)

    for epoch in range(EPOCHS):
        opt.zero_grad()

        # PDE residual — one line, courtesy of SymbolicPDE
        r = residual_fn(t_col.requires_grad_(True))
        loss_pde = (r ** 2).mean()

        # Velocity IC (x'(0) = 0) still needs a penalty
        t0g  = t0.clone().requires_grad_(True)
        x0g  = net_wrapped(t0g)
        xd0  = torch.autograd.grad(x0g.sum(), t0g, create_graph=True)[0]
        loss_vel = xd0.pow(2)

        # Position IC is zero by construction → no L_pos term needed!
        loss = loss_pde + 200.0 * loss_vel
        loss.backward()
        opt.step()

    t_np = np.linspace(0, T_END, 500, dtype=np.float32)
    with torch.no_grad():
        x_pred = net_wrapped(torch.tensor(t_np[:, None], device=DEVICE)).cpu().numpy().ravel()
    return t_np, x_pred


# ── Main lesson runner ─────────────────────────────────────────────────────
def main() -> None:
    print("─" * 60)
    print("  Lesson 01 — Your First PINN with PINNeAPPle")
    print("─" * 60)
    print(f"\n  Problem : x''(t) + ω²x(t) = 0,  ω = {OMEGA},  t ∈ [0, {T_END:.1f}]")
    print("  Exact   : x(t) = cos(ωt)\n")

    # ── Train both approaches ─────────────────────────────────────────────
    print("  [1/2]  Training manual PINN (raw PyTorch)...")
    t_np, x_manual = _train_manual(OMEGA)

    print("  [2/2]  Training PINNeAPPle PINN (SymbolicPDE + HardBC)...")
    t_np, x_pinneaple = _train_pinneaple(OMEGA)

    # ── Compute exact solution ────────────────────────────────────────────
    x_exact = np.cos(OMEGA * t_np)

    # ── Metrics ───────────────────────────────────────────────────────────
    def rel_l2(pred):
        return float(np.sqrt(((pred - x_exact)**2).mean()) /
                     (np.abs(x_exact).mean() + 1e-8))

    e_manual    = rel_l2(x_manual)
    e_pinneaple = rel_l2(x_pinneaple)
    print(f"\n  Relative L2 error:")
    print(f"    Manual PINN      : {e_manual:.4e}")
    print(f"    PINNeAPPle PINN  : {e_pinneaple:.4e}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, pred, label in zip(axes,
                                [x_manual, x_pinneaple],
                                ["Manual PINN (raw PyTorch)",
                                 "PINNeAPPle PINN\n(SymbolicPDE + HardBC)"]):
        err = rel_l2(pred)
        ax.plot(t_np, x_exact, "k-",  lw=2.5, label="Exact  cos(ωt)")
        ax.plot(t_np, pred,    "r--", lw=1.5, label=f"PINN  (L2={err:.2e})")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("time  t")
        ax.set_ylabel("x(t)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Lesson 01 — Harmonic Oscillator  ω={OMEGA}", fontsize=12)
    plt.tight_layout()
    out = "lesson_01_first_pinn.png"
    plt.savefig(out, dpi=130)
    print(f"\n  Saved {out}")

    # ── Key takeaways ─────────────────────────────────────────────────────
    print("""
  Key takeaways:
    1. Both approaches solve the same ODE — PINNeAPPle is less boilerplate.
    2. SymbolicPDE: write the PDE once in SymPy, get autograd residuals free.
    3. HardBC: enforce x(0)=1 EXACTLY via ansatz — no lambda tuning needed.
    4. The velocity IC (x'=0) still uses a penalty — that's normal for ICs
       that involve derivatives (harder to enforce via ansatz).

  Next lesson:
    python -m pinneaple_learning.course.lesson_02_loss_functions
""")


if __name__ == "__main__":
    main()
