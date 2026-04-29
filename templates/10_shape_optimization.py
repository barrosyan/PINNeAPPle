"""10_shape_optimization.py — Shape optimization with continuous adjoint.

Demonstrates:
- ContinuousAdjointSolver for computing shape sensitivities
- naca_parametric to generate NACA 4-digit airfoil control points
- DragAdjointObjective
- Gradient-based optimization loop with ShapeParametrization
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_design_opt.adjoint import (
    ContinuousAdjointSolver,
    ShapeParametrization,
    naca_parametric,
)

try:
    from pinneaple_design_opt.adjoint import DragAdjointObjective
    _DRAG_OBJ_AVAILABLE = True
except ImportError:
    _DRAG_OBJ_AVAILABLE = False


# ---------------------------------------------------------------------------
# Simple PINN surrogate for 2D potential flow around a thin airfoil
# Trained model approximates the pressure coefficient: Cp(x,y)
# ---------------------------------------------------------------------------

class PotentialFlowPINN(nn.Module):
    """Minimal surrogate for potential flow pressure coefficient."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(xy)


# ---------------------------------------------------------------------------
# PDE residual: Laplace (potential flow)  Δφ = 0
# ---------------------------------------------------------------------------

def laplace_residual(model: nn.Module, xy: torch.Tensor) -> torch.Tensor:
    xy = xy.requires_grad_(True)
    phi = model(xy)
    if hasattr(phi, "y"):
        phi = phi.y
    g1 = torch.autograd.grad(phi.sum(), xy, create_graph=True)[0]
    phi_x = g1[:, 0:1]
    phi_y = g1[:, 1:2]
    phi_xx = torch.autograd.grad(phi_x.sum(), xy, create_graph=True)[0][:, 0:1]
    phi_yy = torch.autograd.grad(phi_y.sum(), xy, create_graph=True)[0][:, 1:2]
    return phi_xx + phi_yy


# ---------------------------------------------------------------------------
# Drag objective: minimize the net x-momentum flux (proxy for drag)
# J = mean( -∂φ/∂x ) on the near-field boundary
# ---------------------------------------------------------------------------

def drag_objective(model: nn.Module, xy: torch.Tensor) -> torch.Tensor:
    """Proxy drag: mean negative velocity in x-direction."""
    xy_g = xy.requires_grad_(True)
    phi = model(xy_g)
    if hasattr(phi, "y"):
        phi = phi.y
    u = torch.autograd.grad(phi.sum(), xy_g, create_graph=True)[0][:, 0:1]
    # Drag ~ ∫ u² ds  (simplified)
    return u.pow(2).mean()


def train_surrogate(model: nn.Module, xy_col: torch.Tensor, n_epochs: int = 2000):
    """Quick training pass to initialise the PINN surrogate."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(1, n_epochs + 1):
        optimizer.zero_grad()
        xy_c = xy_col.clone().requires_grad_(True)
        res = laplace_residual(model, xy_c)
        loss = res.pow(2).mean()
        loss.backward()
        optimizer.step()
        if ep % 500 == 0:
            print(f"  [surrogate] ep={ep:4d}  loss={loss.item():.3e}")


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- NACA airfoil control points -------------------------------------
    # naca_parametric(t_c) returns a (n_ctrl, 2) tensor of surface points
    t_c = 0.12   # NACA 0012
    ctrl_pts_init = naca_parametric(t_c=t_c)          # (n_ctrl, 2) tensor
    print(f"NACA 00{int(t_c*100):02d} control points: {ctrl_pts_init.shape}")

    shape_params = ShapeParametrization(
        control_points=ctrl_pts_init,
        device=str(device),
    )

    # --- Collocation points around airfoil --------------------------------
    n_col = 2000
    # Sample in a box [−0.5, 1.5] × [−1, 1] excluding the airfoil interior
    xy_col_np = np.random.uniform(
        low=[- 0.5, -1.0], high=[1.5, 1.0], size=(n_col, 2)
    ).astype(np.float32)
    xy_col = torch.tensor(xy_col_np, device=device)

    # --- Train PINN surrogate --------------------------------------------
    model = PotentialFlowPINN().to(device)
    print("Training PINN surrogate...")
    train_surrogate(model, xy_col, n_epochs=2000)

    # --- Continuous adjoint optimizer ------------------------------------
    if _DRAG_OBJ_AVAILABLE:
        objective_fn = DragAdjointObjective(velocity_channel=0)
        obj_callable = lambda m, x: objective_fn(m, x)
    else:
        obj_callable = drag_objective

    adjoint_solver = ContinuousAdjointSolver(
        primal_model=model,
        pde_residual_fn=laplace_residual,
        objective_fn=obj_callable,
    )

    # --- Shape optimization loop ------------------------------------------
    opt_history = []
    n_steps = 50
    lr = 5e-3
    shape_optimizer = torch.optim.Adam(shape_params.parameters(), lr=lr)

    print(f"\nRunning {n_steps} adjoint optimization steps...")
    for step in range(1, n_steps + 1):
        shape_optimizer.zero_grad()

        # Deform collocation mesh with current shape
        xy_deformed = shape_params.deform_mesh(xy_col.detach()).to(device)
        xy_deformed.requires_grad_(True)

        # Compute objective
        J = obj_callable(model, xy_deformed)
        J.backward()

        # Compute adjoint shape sensitivity
        with torch.no_grad():
            sens = adjoint_solver.shape_sensitivity(xy_col.detach(), shape_params)
            if shape_params.control_points.grad is not None:
                shape_params.control_points.grad.add_(sens)
            else:
                shape_params.control_points.grad = sens.clone()

        shape_optimizer.step()
        opt_history.append(float(J.item()))

        if step % 10 == 0:
            print(f"  step {step:3d} | J = {J.item():.4e}")

    best_J = min(opt_history)
    print(f"\nBest objective J = {best_J:.4e}")

    # --- Plot results -----------------------------------------------------
    ctrl_final = shape_params.control_points.detach().cpu().numpy()
    ctrl_init  = ctrl_pts_init.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Airfoil shape comparison
    axes[0].plot(ctrl_init[:, 0],  ctrl_init[:, 1],  "b-o", ms=4,
                 label="Initial NACA")
    axes[0].plot(ctrl_final[:, 0], ctrl_final[:, 1], "r-o", ms=4,
                 label="Optimized")
    axes[0].set_aspect("equal")
    axes[0].legend()
    axes[0].set_title("Airfoil control points: initial vs optimized")
    axes[0].set_xlabel("x/c")
    axes[0].set_ylabel("y/c")
    axes[0].grid(True, alpha=0.3)

    # Optimization history
    axes[1].plot(opt_history, "k-")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Objective J (drag proxy)")
    axes[1].set_title("Shape optimization convergence")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("10_shape_optimization_result.png", dpi=120)
    print("Saved 10_shape_optimization_result.png")


if __name__ == "__main__":
    main()
