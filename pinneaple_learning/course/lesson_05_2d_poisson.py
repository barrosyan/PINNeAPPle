"""Lesson 05 — 2D Poisson on an L-shape Domain.

Physics problem
---------------
Poisson equation on an L-shaped domain Ω:
    −∇²u = f,   where f = 1 (uniform source)
    u = 0        on ∂Ω  (homogeneous Dirichlet)

The L-shape: [0,2]×[0,2] minus the top-right quadrant [1,2]×[1,2].
This is a classic benchmark — the re-entrant corner creates a singularity
in the gradient that challenges both classical FEM and PINNs.

What you will learn
-------------------
  Step 1 — Build the L-shape domain with CSG
  Step 2 — Hard-enforce u=0 on all boundaries using HardBC + SDF
  Step 3 — Define the 2D Poisson residual with SymbolicPDE
  Step 4 — Train and validate
  Step 5 — Compare SDF-based hard BC vs penalty BC

Run this lesson
---------------
    python -m pinneaple_learning.course.lesson_05_2d_poisson
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sympy as sp

# PINNeAPPle imports
from pinneaple_geom     import CSGRectangle, lshape
from pinneaple_symbolic import SymbolicPDE, DirichletBC, HardBC
from pinneaple_train    import TwoPhaseTrainer, TwoPhaseConfig

EPOCHS_P1  = 6_000
EPOCHS_P2  = 1_000
N_COL      = 4_000
N_BC       = 600
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ── Network factory ────────────────────────────────────────────────────────
def make_net() -> nn.Module:
    return nn.Sequential(
        nn.Linear(2, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    )


# ── Build the L-shape domain with PINNeAPPle CSG ──────────────────────────
def build_lshape():
    """[0,2]×[0,2] minus the top-right [1,2]×[1,2]."""
    full  = CSGRectangle(0, 0, 2, 2)
    notch = CSGRectangle(1, 1, 2, 2)
    return full - notch


# ── 2D Poisson residual ────────────────────────────────────────────────────
def make_poisson_residual(net: nn.Module):
    """Return a function  r(xy) = −(u_xx + u_yy) − 1."""
    def residual(xy: torch.Tensor) -> torch.Tensor:
        xy = xy.requires_grad_(True)
        u  = net(xy)
        # u_x and u_y
        grad1 = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
        u_x, u_y = grad1[:, 0:1], grad1[:, 1:2]
        # u_xx and u_yy
        u_xx = torch.autograd.grad(u_x.sum(), xy, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y.sum(), xy, create_graph=True)[0][:, 1:2]
        return -(u_xx + u_yy) - 1.0  # residual = 0 means −∇²u = 1
    return residual


# ── Train with penalty BC (soft Dirichlet) ────────────────────────────────
def train_penalty() -> tuple[nn.Module, list]:
    torch.manual_seed(0)
    domain = build_lshape()
    net    = make_net().to(DEVICE)
    resid  = make_poisson_residual(net)

    # DirichletBC: penalise u≠0 on boundary
    bc = DirichletBC(value_fn=lambda xy: torch.zeros(len(xy), 1, device=DEVICE))

    def loss_fn(model: nn.Module) -> dict:
        xy_col = torch.tensor(domain.sample_interior(N_COL, seed=np.random.randint(0, 1000)),
                              device=DEVICE)
        xy_bnd = torch.tensor(domain.sample_boundary(N_BC,  seed=np.random.randint(0, 1000)),
                              device=DEVICE)
        l_pde = resid(xy_col).pow(2).mean()
        l_bc  = bc.loss(model, xy_bnd)
        return {"total": l_pde + 100.0 * l_bc, "pde": l_pde, "bc": l_bc}

    cfg     = TwoPhaseConfig(EPOCHS_P1, EPOCHS_P2, phase1_lr=1e-3, device=DEVICE)
    trainer = TwoPhaseTrainer(net, loss_fn, cfg)
    history = trainer.train()
    return net, history


# ── Train with HardBC (SDF ansatz) ────────────────────────────────────────
def train_hard_bc() -> tuple[nn.Module, list]:
    torch.manual_seed(0)
    domain     = build_lshape()
    net_inner  = make_net()

    # SDF-based HardBC: u(x) = SDF(x) * net(x) → u=0 on boundary exactly
    sdf_fn = domain.sdf  # np.ndarray → np.ndarray

    def distance_fn(xy: torch.Tensor) -> torch.Tensor:
        # SDF is negative inside the domain → negate so it's positive inside
        sdf = torch.tensor(
            -sdf_fn(xy.detach().cpu().numpy()).astype(np.float32),
            device=xy.device
        ).clamp(min=0).unsqueeze(1)
        return sdf

    hard_bc = HardBC(
        distance_fn  = distance_fn,
        bc_value_fn  = lambda xy: torch.zeros(len(xy), 1, device=xy.device),
    )
    net = hard_bc.wrap_model(net_inner).to(DEVICE)
    resid = make_poisson_residual(net)

    def loss_fn(model: nn.Module) -> dict:
        xy_col = torch.tensor(domain.sample_interior(N_COL, seed=np.random.randint(0, 1000)),
                              device=DEVICE)
        l_pde = resid(xy_col).pow(2).mean()
        return {"total": l_pde, "pde": l_pde}

    cfg     = TwoPhaseConfig(EPOCHS_P1, EPOCHS_P2, phase1_lr=1e-3, device=DEVICE)
    trainer = TwoPhaseTrainer(net, loss_fn, cfg)
    history = trainer.train()
    return net, history


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    print("─" * 60)
    print("  Lesson 05 — 2D Poisson on L-shape Domain")
    print("─" * 60)
    print("\n  −∇²u = 1  on L-shape,   u = 0 on ∂Ω\n")

    domain = build_lshape()

    print("  [1/2]  Training with penalty Dirichlet BC...")
    net_pen, hist_pen = train_penalty()
    print("  [2/2]  Training with SDF-based HardBC (exact enforcement)...")
    net_hard, hist_hard = train_hard_bc()

    # ── Evaluate on a regular grid (mask outside domain) ─────────────────
    xg, yg = np.meshgrid(np.linspace(0, 2, 120), np.linspace(0, 2, 120))
    xy_flat = np.stack([xg.ravel(), yg.ravel()], axis=1).astype(np.float32)
    inside  = domain.contains(xy_flat)

    results = {}
    for label, net in [("Penalty BC", net_pen), ("HardBC (SDF)", net_hard)]:
        xy_t = torch.tensor(xy_flat[inside], device=DEVICE)
        with torch.no_grad():
            u_pred = net(xy_t).cpu().numpy().ravel()
        field = np.full(len(xy_flat), np.nan)
        field[inside] = u_pred
        results[label] = field.reshape(120, 120)

        # Measure BC residual (how much u deviates from 0 on boundary)
        bnd_pts = torch.tensor(domain.sample_boundary(1000, seed=7), device=DEVICE)
        with torch.no_grad():
            u_bnd = net(bnd_pts).cpu().numpy()
        bc_err = float(np.abs(u_bnd).mean())
        print(f"  {label:20s}  mean |u_boundary| = {bc_err:.4e}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (label, field) in zip(axes[:2], results.items()):
        im = ax.imshow(field, origin="lower", extent=[0, 2, 0, 2],
                       cmap="plasma", interpolation="bilinear")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("y")

    ax = axes[2]
    for (label, hist), color in zip([("Penalty BC", hist_pen),
                                     ("HardBC", hist_hard)],
                                    ["steelblue", "crimson"]):
        vals = [h.get("pde", np.nan) for h in hist]
        ax.semilogy(vals, label=label, color=color, lw=1.5)
    ax.set_title("PDE residual loss comparison")
    ax.set_xlabel("Epoch"); ax.set_ylabel("L_pde (log)")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle("Lesson 05 — 2D Poisson on L-shape  (−∇²u = 1)", fontsize=12)
    plt.tight_layout()
    out = "lesson_05_2d_poisson.png"
    plt.savefig(out, dpi=130)
    print(f"\n  Saved {out}")

    print("""
  Key takeaways:
    1. CSG makes it trivial to define complex domains — one expression.
    2. HardBC (SDF ansatz) enforces u=0 EXACTLY on every boundary point,
       while penalty BC has residual errors ~ 1e-3 near corners.
    3. HardBC removes one hyperparameter (λ_bc) entirely.
    4. The re-entrant corner (L-shape) creates a stress concentration —
       PINNs need more collocation points near singularities.

  Next lesson:
    python -m pinneaple_learning.course.lesson_06_model_architectures
""")


if __name__ == "__main__":
    main()
