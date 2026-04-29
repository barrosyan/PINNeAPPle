"""02_symbolic_pde.py — Symbolic PDE compilation workflow.

Demonstrates:
- Defining a PDE with SymPy expressions
- Compiling residuals with SymbolicPDE
- Enforcing BCs with HardBC (exact) and NeumannBC (soft penalty)
- Training a standard PINN on the compiled residual
"""

import math
import sympy as sp
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_symbolic.compiler import SymbolicPDE
from pinneaple_symbolic.bc import HardBC, NeumannBC
from pinneaple_models.modified_mlp import ModifiedMLP


# ---------------------------------------------------------------------------
# Problem: 2D Poisson  Δu + 2π² sin(πx)sin(πy) = 0  on [0,1]²
# Exact solution: u = sin(πx)sin(πy)
# ---------------------------------------------------------------------------

def build_symbolic_pde():
    """Define and compile the Poisson PDE using SymPy."""
    x, y = sp.symbols("x y")
    u = sp.Function("u")
    pi = sp.pi

    # Residual expression: u_xx + u_yy + 2π²sin(πx)sin(πy) = 0
    expr = (u(x, y).diff(x, 2)
            + u(x, y).diff(y, 2)
            + 2 * pi**2 * sp.sin(pi * x) * sp.sin(pi * y))

    pde = SymbolicPDE(expr, coord_syms=[x, y], field_syms=[u])
    return pde


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Setup: symbolic PDE compiler --------------------------------------
    print("Compiling symbolic PDE...")
    pde = build_symbolic_pde()

    # --- Model -------------------------------------------------------------
    net = ModifiedMLP(in_dim=2, out_dim=1, hidden_dim=64, n_layers=4,
                      n_fourier=32).to(device)

    # Hard BC: multiply by distance function to enforce u=0 on all four sides
    phi = lambda x: (x[:, 0:1] * (1.0 - x[:, 0:1]) *
                     x[:, 1:2] * (1.0 - x[:, 1:2]))
    hard_bc = HardBC(
        distance_fn=phi,
        bc_value_fn=lambda x: torch.zeros(x.shape[0], 1, device=x.device),
    )
    model = hard_bc.wrap_model(net)

    # Compile the residual function from the symbolic PDE
    residual_fn = pde.to_residual_fn(model)

    # Soft Neumann BC on right wall (x=1): du/dn = π·cos(π)·sin(πy) ≈ -π·sin(πy)
    def neumann_target(xy):
        """Expected normal derivative at x=1: du/dn = -π sin(πy)."""
        return -math.pi * torch.sin(math.pi * xy[:, 1:2])

    neumann_bc = NeumannBC(
        normal_dir=0,        # derivative w.r.t. x
        target_fn=neumann_target,
    )

    # --- Collocation points ------------------------------------------------
    n_col = 3000
    xy_int = torch.rand(n_col, 2, device=device, requires_grad=True)

    # Neumann points on right boundary x=1
    y_neu = torch.rand(200, 1, device=device)
    xy_neu = torch.cat([torch.ones(200, 1, device=device), y_neu], dim=1)
    xy_neu.requires_grad_(True)

    # --- Training ----------------------------------------------------------
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    history_pde = []
    history_neu = []
    n_epochs = 4000
    print_every = 500

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad(set_to_none=True)

        # Physics (compiled symbolic residual)
        res_pde = residual_fn(xy_int)              # (N, 1)
        loss_pde = res_pde.pow(2).mean()

        # Soft Neumann penalty
        loss_neu = neumann_bc.loss(model, xy_neu)

        loss = loss_pde + 0.1 * loss_neu
        loss.backward()
        optimizer.step()

        history_pde.append(float(loss_pde.item()))
        history_neu.append(float(loss_neu.item()))

        if epoch % print_every == 0:
            print(f"  epoch {epoch:4d} | pde={loss_pde.item():.3e}"
                  f"  neu={loss_neu.item():.3e}")

    print("\nTraining complete.")

    # --- Evaluation --------------------------------------------------------
    x_lin = np.linspace(0, 1, 64)
    xx, yy = np.meshgrid(x_lin, x_lin)
    xy_test = torch.tensor(
        np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        u_pred = model(xy_test).cpu().numpy().reshape(64, 64)

    u_exact = np.sin(math.pi * xx) * np.sin(math.pi * yy)
    l2_err = (np.sqrt(((u_pred - u_exact) ** 2).mean())
              / np.sqrt((u_exact ** 2).mean() + 1e-14))
    print(f"Relative L2 error: {l2_err:.4e}")

    # --- Loss plot ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(history_pde, label="PDE residual")
    ax.semilogy(history_neu, label="Neumann BC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Symbolic PDE training losses")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("02_symbolic_pde_result.png", dpi=120)
    print("Saved 02_symbolic_pde_result.png")


if __name__ == "__main__":
    main()
