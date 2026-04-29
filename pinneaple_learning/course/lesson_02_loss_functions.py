"""Lesson 02 — The PINN Loss Function: Anatomy and Tuning.

Physics problem
---------------
1D heat equation (spacetime):
    u_t = α u_xx,    (x,t) ∈ [0,1] × [0,0.5]
    u(x, 0) = sin(πx)    (initial condition)
    u(0, t) = u(1, t) = 0 (Dirichlet BCs)

Exact solution:  u(x,t) = exp(−α π² t) · sin(πx)

What you will learn
-------------------
  Step 1 — What each loss term does physically
  Step 2 — How to build loss terms with PINNeAPPle's DirichletBC, NeumannBC
  Step 3 — How loss weight λ affects convergence (compare 3 settings)
  Step 4 — How to use PINNeAPPle's SelfAdaptiveWeights (auto-balancing)

Run this lesson
---------------
    python -m pinneaple_learning.course.lesson_02_loss_functions
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PINNeAPPle imports for this lesson
from pinneaple_symbolic import DirichletBC
from pinneaple_train import SelfAdaptiveWeights

ALPHA  = 0.1                  # thermal diffusivity
EPOCHS = 6_000
LR     = 1e-3
N_COL  = 2_000                # interior collocation points
N_BC   = 200                  # boundary points per edge
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Network ────────────────────────────────────────────────────────────────
def make_net() -> nn.Module:
    return nn.Sequential(
        nn.Linear(2, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    )


# ── Sampling helpers ───────────────────────────────────────────────────────
def sample_interior(n: int) -> torch.Tensor:
    x = torch.rand(n, 1)
    t = torch.rand(n, 1) * 0.5
    return torch.cat([x, t], dim=1).to(DEVICE)


def sample_ic(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.rand(n, 1)
    t = torch.zeros(n, 1)
    xt = torch.cat([x, t], dim=1).to(DEVICE)
    u0 = torch.sin(math.pi * x).to(DEVICE)
    return xt, u0


def sample_bc_left(n: int) -> torch.Tensor:
    x = torch.zeros(n, 1)
    t = torch.rand(n, 1) * 0.5
    return torch.cat([x, t], dim=1).to(DEVICE)


def sample_bc_right(n: int) -> torch.Tensor:
    x = torch.ones(n, 1)
    t = torch.rand(n, 1) * 0.5
    return torch.cat([x, t], dim=1).to(DEVICE)


# ── PDE residual (manual autograd) ────────────────────────────────────────
def heat_residual(net: nn.Module, xt: torch.Tensor) -> torch.Tensor:
    """Returns (u_t - α u_xx) at each collocation point."""
    xt = xt.requires_grad_(True)
    u  = net(xt)
    grads = torch.autograd.grad(u.sum(), xt, create_graph=True)[0]
    u_t   = grads[:, 1:2]
    u_x   = grads[:, 0:1]
    u_xx  = torch.autograd.grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]
    return u_t - ALPHA * u_xx


# ── Training with manual weights ───────────────────────────────────────────
def train_manual(lam_bc: float, lam_ic: float) -> tuple[list, float]:
    torch.manual_seed(0)
    net = make_net().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    # PINNeAPPle DirichletBC: value_fn returns 0 (homogeneous BC)
    bc_left  = DirichletBC(value_fn=lambda xt: torch.zeros(xt.shape[0], 1, device=DEVICE))
    bc_right = DirichletBC(value_fn=lambda xt: torch.zeros(xt.shape[0], 1, device=DEVICE))

    history = []
    xt_ic, u_ic = sample_ic(N_BC)

    for epoch in range(EPOCHS):
        opt.zero_grad()

        # ── L_pde: physics residual at random interior points ──────────
        xt_col = sample_interior(N_COL)
        r      = heat_residual(net, xt_col)
        l_pde  = r.pow(2).mean()

        # ── L_bc: Dirichlet BCs (using PINNeAPPle DirichletBC) ─────────
        l_bc = (bc_left.loss(net, sample_bc_left(N_BC)) +
                bc_right.loss(net, sample_bc_right(N_BC)))

        # ── L_ic: initial condition (penalise deviation at t=0) ─────────
        l_ic = (net(xt_ic) - u_ic).pow(2).mean()

        loss = l_pde + lam_bc * l_bc + lam_ic * l_ic
        loss.backward()
        opt.step()
        history.append(float(loss))

    # ── Evaluate final relative L2 error ──────────────────────────────
    x_t = np.mgrid[0:1:50j, 0:0.5:50j]
    x_np = x_t[0].ravel().astype(np.float32)
    t_np = x_t[1].ravel().astype(np.float32)
    xt_test = torch.tensor(np.stack([x_np, t_np], axis=1), device=DEVICE)
    with torch.no_grad():
        u_pred = net(xt_test).cpu().numpy().ravel()
    u_exact = np.exp(-ALPHA * math.pi**2 * t_np) * np.sin(math.pi * x_np)
    rel_l2 = float(np.sqrt(((u_pred - u_exact)**2).mean()) /
                   (np.abs(u_exact).mean() + 1e-8))
    return history, rel_l2


# ── Training with SelfAdaptiveWeights ─────────────────────────────────────
def train_adaptive() -> tuple[list, float]:
    """Let PINNeAPPle balance the loss terms automatically."""
    torch.manual_seed(0)
    net = make_net().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    bc_left  = DirichletBC(value_fn=lambda xt: torch.zeros(xt.shape[0], 1, device=DEVICE))
    bc_right = DirichletBC(value_fn=lambda xt: torch.zeros(xt.shape[0], 1, device=DEVICE))

    # SelfAdaptiveWeights: learns weights for each named loss term
    saw = SelfAdaptiveWeights(loss_names=["pde", "bc", "ic"], alpha=0.9)
    xt_ic, u_ic = sample_ic(N_BC)

    history = []
    for epoch in range(EPOCHS):
        opt.zero_grad()

        xt_col = sample_interior(N_COL)
        l_pde  = heat_residual(net, xt_col).pow(2).mean()
        l_bc   = (bc_left.loss(net, sample_bc_left(N_BC)) +
                  bc_right.loss(net, sample_bc_right(N_BC)))
        l_ic   = (net(xt_ic) - u_ic).pow(2).mean()

        # SelfAdaptiveWeights automatically balances the three terms
        loss_dict = {"pde": l_pde, "bc": l_bc, "ic": l_ic}
        loss = saw(loss_dict)
        loss.backward()
        opt.step()
        history.append(float(loss))

    x_t = np.mgrid[0:1:50j, 0:0.5:50j]
    x_np = x_t[0].ravel().astype(np.float32)
    t_np = x_t[1].ravel().astype(np.float32)
    xt_test = torch.tensor(np.stack([x_np, t_np], axis=1), device=DEVICE)
    with torch.no_grad():
        u_pred = net(xt_test).cpu().numpy().ravel()
    u_exact = np.exp(-ALPHA * math.pi**2 * t_np) * np.sin(math.pi * x_np)
    rel_l2 = float(np.sqrt(((u_pred - u_exact)**2).mean()) /
                   (np.abs(u_exact).mean() + 1e-8))
    return history, rel_l2


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    print("─" * 60)
    print("  Lesson 02 — PINN Loss Functions: Anatomy and Tuning")
    print("─" * 60)
    print(f"\n  Problem: u_t = α u_xx,  α={ALPHA},  u(x,0)=sin(πx),  u(0,t)=u(1,t)=0")
    print("  Exact:   u(x,t) = exp(−α π² t) · sin(πx)\n")

    configs = [
        ("λ_bc=1, λ_ic=1   (too small)", 1.0,   1.0),
        ("λ_bc=10, λ_ic=100  (balanced)", 10.0, 100.0),
        ("λ_bc=1000, λ_ic=1000 (too large)", 1000.0, 1000.0),
    ]

    results = {}
    for label, lam_bc, lam_ic in configs:
        print(f"  Training  {label} ...")
        h, err = train_manual(lam_bc, lam_ic)
        results[label] = (h, err)
        print(f"    → L2 = {err:.4e}")

    print("  Training  SelfAdaptiveWeights (auto-balanced) ...")
    h_saw, err_saw = train_adaptive()
    results["SelfAdaptiveWeights"] = (h_saw, err_saw)
    print(f"    → L2 = {err_saw:.4e}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    colors = ["#d62728", "#2ca02c", "#ff7f0e", "#1f77b4"]
    for (label, (h, err)), c in zip(results.items(), colors):
        ax.semilogy(h, color=c, lw=1.5, label=f"{label}  (L2={err:.1e})")
    ax.set_title("Loss curves — effect of weight choice")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Total loss (log)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1]
    labels = list(results.keys())
    errors = [v[1] for v in results.values()]
    bars = ax.bar(range(len(labels)), errors,
                  color=colors[:len(labels)], edgecolor="k", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.split("(")[0].strip() for l in labels], fontsize=7, rotation=10)
    ax.set_title("Final relative L2 error by weight strategy")
    ax.set_ylabel("Relative L2 error")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, err in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width()/2, err * 1.2,
                f"{err:.1e}", ha="center", fontsize=7)

    plt.suptitle("Lesson 02 — Loss Function Anatomy and Weight Tuning", fontsize=12)
    plt.tight_layout()
    out = "lesson_02_loss_functions.png"
    plt.savefig(out, dpi=130)
    print(f"\n  Saved {out}")

    print("""
  Key takeaways:
    1. L_pde alone is not enough — BCs and ICs need explicit terms.
    2. Small λ: BCs/ICs dominate training early but stay inaccurate.
       Large λ: BCs/ICs dominate but L_pde is neglected.
    3. PINNeAPPle's DirichletBC.loss() builds the BC term for you.
    4. SelfAdaptiveWeights automatically tracks the gradient magnitudes
       and adjusts weights every step — reduces hyperparameter search.

  Next lesson:
    python -m pinneaple_learning.course.lesson_03_forward_pde
""")


if __name__ == "__main__":
    main()
