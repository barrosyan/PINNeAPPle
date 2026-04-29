"""01_basic_pinn.py — Standard PINN for the 2D Poisson equation.

Demonstrates:
- Domain setup with CSGRectangle
- SIREN network
- PDE residual (Laplacian u = f) and Dirichlet BCs via HardBC
- A simple Adam training loop
- Loss curve plotting with matplotlib
"""

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_geom.csg import CSGRectangle
from pinneaple_models.siren import SIREN
from pinneaple_symbolic.bc import HardBC


# ---------------------------------------------------------------------------
# Problem definition  —  Poisson: Δu = -2π² sin(πx)sin(πy)
# Exact solution u(x,y) = sin(πx)sin(πy)  on  [0,1]²
# ---------------------------------------------------------------------------

def f_source(xy: torch.Tensor) -> torch.Tensor:
    """Forcing term f = -2π² sin(πx)sin(πy)."""
    x, y = xy[:, 0:1], xy[:, 1:2]
    return -2.0 * math.pi ** 2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)


def poisson_residual(model_wrapped, xy: torch.Tensor) -> torch.Tensor:
    """Physics residual: Δu - f = 0.

    Args:
        model_wrapped: HardBC-wrapped callable returning (N,1) u values.
        xy: collocation points (N,2) with requires_grad=True.
    Returns:
        Residual tensor (N,1).
    """
    u = model_wrapped(xy)                                   # (N, 1)
    grads = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
    u_x, u_y = grads[:, 0:1], grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x.sum(), xy, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y.sum(), xy, create_graph=True)[0][:, 1:2]
    laplacian = u_xx + u_yy
    return laplacian - f_source(xy)


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Setup: geometry ---------------------------------------------------
    rect = CSGRectangle(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)
    xy_int = rect.sample_interior(n=4096, seed=0)          # (N, 2) numpy
    xy_bnd = rect.sample_boundary(n=512,  seed=1)

    xy_col = torch.tensor(xy_int, dtype=torch.float32, device=device,
                          requires_grad=True)
    xy_bc  = torch.tensor(xy_bnd, dtype=torch.float32, device=device)
    u_bc   = torch.zeros(xy_bc.shape[0], 1, device=device)  # homogeneous Dirichlet

    # --- Model: SIREN ------------------------------------------------------
    net = SIREN(in_dim=2, out_dim=1, hidden_dim=128, n_layers=4, omega_0=30.0,
                outermost_linear=True).to(device)

    # Hard BC: u = phi(x) * net(x)  where  phi = x(1-x)y(1-y)
    phi = lambda x: (x[:, 0:1] * (1 - x[:, 0:1]) *
                     x[:, 1:2] * (1 - x[:, 1:2]))
    hard_bc = HardBC(distance_fn=phi,
                     bc_value_fn=lambda x: torch.zeros(x.shape[0], 1, device=x.device))
    model = hard_bc.wrap_model(net)

    # --- Training ----------------------------------------------------------
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    history = []
    n_epochs = 5000
    print_every = 500

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad(set_to_none=True)

        # PDE residual loss (auto-diff through HardBC wrapper)
        res = poisson_residual(model, xy_col)
        loss_pde = res.pow(2).mean()

        loss = loss_pde
        loss.backward()
        optimizer.step()
        scheduler.step()

        history.append(float(loss.item()))

        if epoch % print_every == 0:
            print(f"  epoch {epoch:5d}  loss_pde = {loss_pde.item():.4e}")

    print("\nTraining complete.")

    # --- Evaluation --------------------------------------------------------
    x_lin = np.linspace(0, 1, 64)
    xx, yy = np.meshgrid(x_lin, x_lin)
    xy_test = torch.tensor(
        np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        u_pred = model(xy_test).cpu().numpy().reshape(64, 64)

    u_exact = (np.sin(math.pi * xx) * np.sin(math.pi * yy))
    l2_err = np.sqrt(((u_pred - u_exact) ** 2).mean()) / np.sqrt((u_exact ** 2).mean())
    print(f"Relative L2 error: {l2_err:.4e}")

    # --- Loss plotting -----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].semilogy(history)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total loss")
    axes[0].set_title("Training loss")
    axes[0].grid(True, which="both", alpha=0.3)

    im = axes[1].contourf(xx, yy, u_pred, levels=32, cmap="viridis")
    plt.colorbar(im, ax=axes[1])
    axes[1].set_title(f"Predicted u  (L2 err = {l2_err:.2e})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    plt.tight_layout()
    plt.savefig("01_basic_pinn_result.png", dpi=120)
    print("Saved 01_basic_pinn_result.png")


if __name__ == "__main__":
    main()
