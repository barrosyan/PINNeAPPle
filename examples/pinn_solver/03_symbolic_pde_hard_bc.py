"""Poisson equation with symbolic PDE compiler + hard boundary conditions.

Demonstrates:
  1. Defining the Poisson equation symbolically via SymPy.
  2. Using HardBC to enforce u = 0 on the unit-square boundary *exactly* (by
     construction), without any BC penalty term in the loss.
  3. A minimal PINN training loop.
  4. Plotting the predicted solution, exact solution, and point-wise error.

Problem (unit square [0,1]^2):

    u_xx + u_yy + 2*pi^2 * sin(pi*x) * sin(pi*y) = 0

Exact solution:

    u(x, y) = sin(pi * x) * sin(pi * y)

    => Laplacian(u) = -2*pi^2 * sin(pi*x) * sin(pi*y)
    => LHS residual = -2*pi^2*u + 2*pi^2*u = 0   ✓

Boundary condition:

    u = 0  on  partial([0,1]^2)

Because sin(pi*0) = sin(pi*1) = 0, the exact solution satisfies homogeneous
Dirichlet BCs on all four sides.

Hard BC ansatz:

    u_net(x, y) = phi(x, y) * net(x, y)

where

    phi(x, y) = x*(1-x) * y*(1-y)

This is 0 on all four sides and strictly positive inside, so u_net
satisfies u=0 on the boundary for *any* network weights.

Run:
    python examples/pinneaple_pinn/03_symbolic_pde_hard_bc.py
"""

from __future__ import annotations

import math

import numpy as np
import sympy as sp
import torch
import torch.nn as nn

# Optional matplotlib import — plotting is skipped gracefully if unavailable.
try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend (works everywhere)
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from pinneaple_symbolic import HardBC, SymbolicPDE


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Simple fully-connected network with Tanh activations."""

    def __init__(self, in_dim: int, out_dim: int, width: int = 64, depth: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_interior(n: int, seed: int = 0) -> torch.Tensor:
    """Sample n points uniformly inside (0,1)^2 (strictly interior)."""
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2), dtype=np.float32)
    # Exclude boundary by clamping slightly — not strictly necessary with hard BC
    # but keeps the problem well-conditioned.
    pts = np.clip(pts, 1e-4, 1.0 - 1e-4).astype(np.float32)
    return torch.from_numpy(pts)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def u_exact(pts: torch.Tensor) -> torch.Tensor:
    """Exact solution: sin(pi*x) * sin(pi*y), shape (N,1)."""
    x = pts[:, 0]
    y = pts[:, 1]
    return (torch.sin(math.pi * x) * torch.sin(math.pi * y)).unsqueeze(1)


def u_exact_np(X: np.ndarray) -> np.ndarray:
    return (np.sin(math.pi * X[:, 0]) * np.sin(math.pi * X[:, 1]))[:, None]


# ---------------------------------------------------------------------------
# Build the symbolic PDE
# ---------------------------------------------------------------------------

def build_pde() -> SymbolicPDE:
    """Construct the Poisson equation using SymPy."""
    x_sym, y_sym = sp.symbols("x y")
    u_sym = sp.Function("u")

    # Residual: u_xx + u_yy + 2*pi^2 * sin(pi*x) * sin(pi*y) = 0
    expr = (
        u_sym(x_sym, y_sym).diff(x_sym, 2)
        + u_sym(x_sym, y_sym).diff(y_sym, 2)
        + 2 * sp.pi**2 * sp.sin(sp.pi * x_sym) * sp.sin(sp.pi * y_sym)
    )
    print("Symbolic PDE residual:")
    print("  R =", expr)
    print()
    return SymbolicPDE(expr, coord_syms=[x_sym, y_sym], field_syms=[u_sym])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    n_col: int = 4096,
    n_steps: int = 3000,
    lr: float = 2e-3,
    width: int = 64,
    depth: int = 4,
    seed: int = 42,
) -> tuple[nn.Module, list[float]]:
    """Full training loop.  Returns (wrapped_model, loss_history)."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # ---- Build symbolic PDE ----
    pde = build_pde()

    # ---- Hard BC: phi(x,y) = x*(1-x)*y*(1-y) ----
    def phi(pts: torch.Tensor) -> torch.Tensor:
        x = pts[:, 0:1]
        y = pts[:, 1:2]
        return x * (1.0 - x) * y * (1.0 - y)

    def g_bc(pts: torch.Tensor) -> torch.Tensor:
        return torch.zeros(pts.shape[0], 1, device=pts.device, dtype=pts.dtype)

    hard_bc = HardBC(distance_fn=phi, bc_value_fn=g_bc)

    # ---- Base network (input: 2, output: 1) ----
    base_net = MLP(in_dim=2, out_dim=1, width=width, depth=depth).to(device=device, dtype=dtype)

    # ---- Wrapped model: satisfies u=0 on boundary by construction ----
    model = hard_bc.wrap_model(base_net)

    # ---- Residual function from SymPy expression ----
    residual_fn = pde.to_residual_fn(model)

    # ---- Collocation points (interior) ----
    x_col = sample_interior(n_col, seed=seed).to(device=device, dtype=dtype)

    # ---- Optimizer ----
    opt = torch.optim.Adam(base_net.parameters(), lr=lr)

    history: list[float] = []

    print(f"Training on {device} for {n_steps} steps ...")
    for step in range(1, n_steps + 1):
        opt.zero_grad(set_to_none=True)

        # PDE residual loss (no BC loss needed — hard BC enforces BCs exactly)
        res = residual_fn(x_col)          # (N, 1)
        loss = torch.mean(res ** 2)

        loss.backward()
        opt.step()

        history.append(float(loss.item()))

        if step % 300 == 0 or step == 1:
            print(f"  step={step:04d}  pde_loss={float(loss):.4e}")

    return model, history


# ---------------------------------------------------------------------------
# Evaluation and plotting
# ---------------------------------------------------------------------------

def evaluate_and_plot(model: nn.Module, history: list[float], grid_size: int = 64) -> None:
    """Evaluate on a grid, compute error metrics, and optionally plot."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    g = grid_size
    xs = torch.linspace(0.0, 1.0, g, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    Xg = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    with torch.no_grad():
        pred = model(Xg).cpu().numpy().reshape(g, g)

    exact = u_exact_np(Xg.cpu().numpy()).reshape(g, g)
    err = np.abs(pred - exact)

    rel_l2 = float(np.linalg.norm(pred - exact) / (np.linalg.norm(exact) + 1e-12))
    max_abs = float(np.max(err))
    print(f"\nEvaluation on {g}x{g} grid:")
    print(f"  Relative L2 error : {rel_l2:.3e}")
    print(f"  Max absolute error : {max_abs:.3e}")

    if not HAS_MPL:
        print("  (matplotlib not available — skipping plots)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    im0 = axes[0].imshow(
        pred, origin="lower", extent=[0, 1, 0, 1], cmap="RdBu_r",
        vmin=pred.min(), vmax=pred.max(),
    )
    axes[0].set_title("Predicted u(x,y)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        exact, origin="lower", extent=[0, 1, 0, 1], cmap="RdBu_r",
        vmin=exact.min(), vmax=exact.max(),
    )
    axes[1].set_title("Exact u(x,y) = sin(pi*x)*sin(pi*y)")
    axes[1].set_xlabel("x")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        err, origin="lower", extent=[0, 1, 0, 1], cmap="hot_r",
    )
    axes[2].set_title(f"Absolute error  (rel L2 = {rel_l2:.2e})")
    axes[2].set_xlabel("x")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    out_path = "03_symbolic_hard_bc_result.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to: {out_path}")
    plt.close(fig)

    # Loss curve
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.semilogy(history, lw=1.2, color="steelblue")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("PDE loss (MSE)")
    ax2.set_title("Training loss — Poisson PINN with hard BC")
    ax2.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    loss_path = "03_symbolic_hard_bc_loss.png"
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    print(f"  Loss curve saved to: {loss_path}")
    plt.close(fig2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    model, history = train(
        n_col=4096,
        n_steps=3000,
        lr=2e-3,
        width=64,
        depth=4,
        seed=42,
    )
    evaluate_and_plot(model, history, grid_size=64)


if __name__ == "__main__":
    main()
