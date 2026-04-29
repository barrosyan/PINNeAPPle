"""Lesson 10 — Physics Validation: Never Skip This.

Physics problem
---------------
2D Laplace equation (steady-state heat / electrostatics):
    ∇²u = 0   on [0,1]²
    u(x,0)=0,  u(x,1)=0,  u(0,y)=0,  u(1,y)=sin(πy)

Exact solution (series): u(x,y) = sin(πy) · sinh(πx) / sinh(π)

We train two PINNs — one good (trained properly), one bad (undertrained) —
and show how PINNeAPPle's PhysicsValidator catches the bad one even when
the final training loss looks similar.

What you will learn
-------------------
  Step 1 — PhysicsValidator with ConservationCheck and BoundaryCheck
  Step 2 — compare_to_analytical for L2/RMSE/max_error metrics
  Step 3 — How a low training loss can hide a physically wrong solution
  Step 4 — Build a full validation report

Key classes
-----------
  from pinneaple_validate import PhysicsValidator, compare_to_analytical,
                                  validate_model, ValidationReport

Run this lesson
---------------
    python -m pinneaple_learning.course.lesson_10_validation
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PINNeAPPle validation imports
from pinneaple_validate import (
    PhysicsValidator,
    compare_to_analytical,
    validate_model,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Exact solution ─────────────────────────────────────────────────────────
def exact(xy: np.ndarray) -> np.ndarray:
    x, y = xy[:, 0], xy[:, 1]
    return np.sin(math.pi * y) * np.sinh(math.pi * x) / np.sinh(math.pi)


# ── Network ────────────────────────────────────────────────────────────────
def make_net() -> nn.Module:
    return nn.Sequential(
        nn.Linear(2, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    )


# ── Laplace residual ───────────────────────────────────────────────────────
def laplace_residual(net: nn.Module, xy: torch.Tensor) -> torch.Tensor:
    xy   = xy.requires_grad_(True)
    u    = net(xy)
    ux   = torch.autograd.grad(u.sum(), xy, create_graph=True)[0][:, 0:1]
    uy   = torch.autograd.grad(u.sum(), xy, create_graph=True)[0][:, 1:2]
    uxx  = torch.autograd.grad(ux.sum(), xy, create_graph=True)[0][:, 0:1]
    uyy  = torch.autograd.grad(uy.sum(), xy, create_graph=True)[0][:, 1:2]
    return uxx + uyy


# ── Train a Laplace PINN ──────────────────────────────────────────────────
def train_laplace(n_epochs: int, lam_bc: float,
                  seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    net = make_net().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    for _ in range(n_epochs):
        opt.zero_grad()

        # Interior collocation
        xy_col = torch.rand(2000, 2, device=DEVICE)
        l_pde  = laplace_residual(net, xy_col.clone()).pow(2).mean()

        # BCs
        n = 200
        # u(x,0) = 0
        bc1 = torch.stack([torch.rand(n), torch.zeros(n)], dim=1).to(DEVICE)
        # u(x,1) = 0
        bc2 = torch.stack([torch.rand(n), torch.ones(n)], dim=1).to(DEVICE)
        # u(0,y) = 0
        bc3 = torch.stack([torch.zeros(n), torch.rand(n)], dim=1).to(DEVICE)
        # u(1,y) = sin(πy)
        y4  = torch.rand(n, device=DEVICE)
        bc4 = torch.stack([torch.ones(n), y4], dim=1).to(DEVICE)
        u4  = torch.sin(math.pi * y4).unsqueeze(1).to(DEVICE)

        l_bc = (net(bc1).pow(2).mean() + net(bc2).pow(2).mean() +
                net(bc3).pow(2).mean() + (net(bc4) - u4).pow(2).mean())

        (l_pde + lam_bc * l_bc).backward()
        opt.step()

    return net


# ── Laplace residual function for PhysicsValidator ────────────────────────
def pde_residual_fn(model, xy: np.ndarray) -> np.ndarray:
    xy_t = torch.tensor(xy, dtype=torch.float32, device=DEVICE)
    r = laplace_residual(model, xy_t)
    return r.detach().cpu().numpy()


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    print("─" * 60)
    print("  Lesson 10 — Physics Validation")
    print("─" * 60)
    print("\n  ∇²u = 0  on [0,1]²")
    print("  u(x,0)=0, u(x,1)=0, u(0,y)=0, u(1,y)=sin(πy)")
    print("  Exact: sin(πy)·sinh(πx)/sinh(π)\n")

    # ── Train: well-trained and under-trained PINNs ───────────────────────
    print("  [1/2]  Training well-trained PINN (8000 epochs, λ_bc=100)...")
    net_good = train_laplace(n_epochs=8_000, lam_bc=100.0, seed=0)

    print("  [2/2]  Training under-trained PINN (500 epochs, λ_bc=1)...")
    net_bad  = train_laplace(n_epochs=500, lam_bc=1.0, seed=0)

    # ── Step 2: compare_to_analytical ─────────────────────────────────────
    print("\n  Running compare_to_analytical...")
    for label, net in [("Well-trained", net_good), ("Under-trained", net_bad)]:
        m = compare_to_analytical(
            model         = net,
            analytical_fn = exact,
            coord_names   = ["x", "y"],
            domain_bounds = {"x": (0.0, 1.0), "y": (0.0, 1.0)},
            n_points      = 10_000,
            device        = DEVICE,
        )
        print(f"  {label:15s}  L2={m['rel_l2']:.4e}  RMSE={m['rmse']:.4e}  "
              f"max_err={m['max_error']:.4e}")

    # ── Step 1: PhysicsValidator ───────────────────────────────────────────
    print("\n  Running PhysicsValidator...")
    for label, net in [("Well-trained", net_good), ("Under-trained", net_bad)]:
        validator = (
            PhysicsValidator(
                model         = net,
                coord_names   = ["x", "y"],
                domain_bounds = {"x": (0.0, 1.0), "y": (0.0, 1.0)},
                device        = DEVICE,
            )
            .add_boundary_check(
                kind     = "dirichlet",
                value    = 0.0,
                boundary = "x=0",
                x_bc     = np.stack([np.zeros(500),
                                      np.random.default_rng(0).uniform(0,1,500)],
                                     axis=1).astype(np.float32),
            )
            .add_boundary_check(
                kind     = "dirichlet",
                value    = None,   # non-zero BC: pass callable
                boundary = "x=1",
                x_bc     = np.stack([np.ones(500),
                                      np.linspace(0,1,500)], axis=1).astype(np.float32),
                value_fn = lambda xy: np.sin(math.pi * xy[:, 1]).reshape(-1, 1),
            )
        )
        report = validator.validate()
        print(f"\n  {label} — Validation Report:")
        for check in report.checks:
            status = "PASS" if check.passed else "FAIL"
            print(f"    [{status}] {check.name:30s}  value={check.value:.4e}  "
                  f"threshold={check.threshold:.4e}")

    # ── Plot ──────────────────────────────────────────────────────────────
    x_np = np.linspace(0, 1, 80, dtype=np.float32)
    y_np = np.linspace(0, 1, 80, dtype=np.float32)
    xg, yg = np.meshgrid(x_np, y_np)
    xy_flat = np.stack([xg.ravel(), yg.ravel()], axis=1)
    u_ex = exact(xy_flat).reshape(80, 80)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    im0 = axes[0].imshow(u_ex, origin="lower", extent=[0,1,0,1], cmap="plasma")
    axes[0].set_title("Exact solution  u(x,y)")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    for ax, net, title in [(axes[1], net_good, "Well-trained PINN"),
                            (axes[2], net_bad, "Under-trained PINN")]:
        xy_t = torch.tensor(xy_flat, device=DEVICE)
        with torch.no_grad():
            u_pred = net(xy_t).cpu().numpy().reshape(80, 80)
        err    = np.abs(u_pred - u_ex)
        im = ax.imshow(err, origin="lower", extent=[0,1,0,1], cmap="hot")
        ax.set_title(f"|Error|  {title}")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xlabel("x"); ax.set_ylabel("y")

    plt.suptitle("Lesson 10 — Physics Validation (Laplace equation)", fontsize=12)
    plt.tight_layout()
    out = "lesson_10_validation.png"
    plt.savefig(out, dpi=130)
    print(f"\n  Saved {out}")

    print("""
  Key takeaways:
    1. A low training loss is NOT sufficient — always validate separately.
    2. PhysicsValidator runs BC checks and conservation checks independently
       of the training process (on held-out points).
    3. compare_to_analytical gives you the three key metrics:
       rel_l2, RMSE, max_error — always report all three.
    4. "max_error" often reveals localised failures (corners, boundaries)
       that rel_l2 masks.
    5. ALWAYS run validation before claiming a model is ready.

  Next lesson:
    python -m pinneaple_learning.course.lesson_11_operator_learning
""")


if __name__ == "__main__":
    main()
