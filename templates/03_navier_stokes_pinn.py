"""03_navier_stokes_pinn.py — 2D Navier-Stokes PINN.

Demonstrates:
- ModifiedMLP architecture (improved convergence for NS problems)
- NS residual loss (momentum + continuity)
- RANS turbulence preset (Spalart-Allmaras)
- Velocity/pressure prediction and visualization
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_models.modified_mlp import ModifiedMLP

try:
    from pinneaple_environment import SpalartAllmarasResiduals
    _SA_AVAILABLE = True
except Exception:
    _SA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Problem: lid-driven cavity, Re=100
# Domain: [0,1]², lid velocity u_top = 1
# Outputs: (u, v, p)
# ---------------------------------------------------------------------------

NU = 0.01   # kinematic viscosity  (Re ≈ 100)


def ns_residuals(model: nn.Module, xy: torch.Tensor):
    """Compute 2D incompressible NS residuals at collocation points.

    Returns
    -------
    tuple (r_mom_u, r_mom_v, r_cont) — each (N, 1)
    """
    xy.requires_grad_(True)
    out = model(xy)
    if hasattr(out, "y"):
        out = out.y
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    # First-order gradients
    def grad(f, x=xy):
        return torch.autograd.grad(f.sum(), x, create_graph=True)[0]

    u_xy = grad(u)
    u_x, u_y = u_xy[:, 0:1], u_xy[:, 1:2]

    v_xy = grad(v)
    v_x, v_y = v_xy[:, 0:1], v_xy[:, 1:2]

    p_xy = grad(p)
    p_x, p_y = p_xy[:, 0:1], p_xy[:, 1:2]

    # Second-order
    u_xx = grad(u_x)[:, 0:1]
    u_yy = grad(u_y)[:, 1:2]
    v_xx = grad(v_x)[:, 0:1]
    v_yy = grad(v_y)[:, 1:2]

    # NS residuals
    r_u = u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
    r_v = u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)
    r_c = u_x + v_y                                         # continuity
    return r_u, r_v, r_c


def sample_boundary_conditions(device):
    """Return (xy_bc, u_bc, v_bc) for the lid-driven cavity."""
    n_bc = 200

    # Bottom wall y=0: u=v=0
    xy_bot = torch.zeros(n_bc, 2, device=device)
    xy_bot[:, 0] = torch.linspace(0, 1, n_bc)
    u_bot = torch.zeros(n_bc, 1, device=device)
    v_bot = torch.zeros(n_bc, 1, device=device)

    # Top lid y=1: u=1, v=0
    xy_top = torch.zeros(n_bc, 2, device=device)
    xy_top[:, 0] = torch.linspace(0, 1, n_bc)
    xy_top[:, 1] = 1.0
    u_top = torch.ones(n_bc, 1, device=device)
    v_top = torch.zeros(n_bc, 1, device=device)

    # Left wall x=0: u=v=0
    xy_left = torch.zeros(n_bc, 2, device=device)
    xy_left[:, 1] = torch.linspace(0, 1, n_bc)
    u_left = torch.zeros(n_bc, 1, device=device)
    v_left = torch.zeros(n_bc, 1, device=device)

    # Right wall x=1: u=v=0
    xy_right = torch.zeros(n_bc, 2, device=device)
    xy_right[:, 0] = 1.0
    xy_right[:, 1] = torch.linspace(0, 1, n_bc)
    u_right = torch.zeros(n_bc, 1, device=device)
    v_right = torch.zeros(n_bc, 1, device=device)

    xy_bc = torch.cat([xy_bot, xy_top, xy_left, xy_right], dim=0)
    u_bc  = torch.cat([u_bot,  u_top,  u_left,  u_right],  dim=0)
    v_bc  = torch.cat([v_bot,  v_top,  v_left,  v_right],  dim=0)
    return xy_bc, u_bc, v_bc


def main():
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if _SA_AVAILABLE:
        print("Spalart-Allmaras preset available (using plain NS for this demo).")

    # --- Model: ModifiedMLP -----------------------------------------------
    # 3 outputs: u, v, p
    model = ModifiedMLP(
        in_dim=2, out_dim=3,
        hidden_dim=128, n_layers=6,
        n_fourier=32, sigma=1.0,
    ).to(device)

    # --- Collocation points -----------------------------------------------
    n_col = 5000
    xy_col = torch.rand(n_col, 2, device=device)

    # --- Boundary conditions ----------------------------------------------
    xy_bc, u_bc, v_bc = sample_boundary_conditions(device)

    # --- Training ----------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6000)

    history = {"pde": [], "bc": [], "total": []}
    n_epochs = 6000
    print_every = 1000

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad(set_to_none=True)

        # PDE residuals
        xy_c = xy_col.clone().requires_grad_(True)
        r_u, r_v, r_c = ns_residuals(model, xy_c)
        loss_pde = r_u.pow(2).mean() + r_v.pow(2).mean() + r_c.pow(2).mean()

        # Boundary condition loss
        out_bc = model(xy_bc)
        if hasattr(out_bc, "y"):
            out_bc = out_bc.y
        u_pred_bc = out_bc[:, 0:1]
        v_pred_bc = out_bc[:, 1:2]
        loss_bc = (u_pred_bc - u_bc).pow(2).mean() + (v_pred_bc - v_bc).pow(2).mean()

        loss = loss_pde + 10.0 * loss_bc
        loss.backward()
        optimizer.step()
        scheduler.step()

        history["pde"].append(float(loss_pde.item()))
        history["bc"].append(float(loss_bc.item()))
        history["total"].append(float(loss.item()))

        if epoch % print_every == 0:
            print(f"  epoch {epoch:5d} | pde={loss_pde.item():.3e}"
                  f"  bc={loss_bc.item():.3e}")

    print("\nTraining complete.")

    # --- Prediction -------------------------------------------------------
    n_vis = 50
    x_lin = np.linspace(0, 1, n_vis)
    xx, yy = np.meshgrid(x_lin, x_lin)
    xy_vis = torch.tensor(
        np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        out_vis = model(xy_vis)
        if hasattr(out_vis, "y"):
            out_vis = out_vis.y
        out_vis = out_vis.cpu().numpy()

    u_vis = out_vis[:, 0].reshape(n_vis, n_vis)
    v_vis = out_vis[:, 1].reshape(n_vis, n_vis)
    p_vis = out_vis[:, 2].reshape(n_vis, n_vis)
    speed = np.sqrt(u_vis**2 + v_vis**2)

    # --- Visualization ----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, field, title in zip(
        axes,
        [speed, p_vis, np.array(history["pde"])],
        ["Speed |u|", "Pressure p", "PDE loss"],
    ):
        if title == "PDE loss":
            ax.semilogy(field)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True, which="both", alpha=0.3)
        else:
            im = ax.contourf(xx, yy, field, levels=30, cmap="RdBu_r")
            plt.colorbar(im, ax=ax)
            ax.quiver(xx[::5, ::5], yy[::5, ::5],
                      u_vis[::5, ::5], v_vis[::5, ::5],
                      alpha=0.5, scale=10)
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig("03_navier_stokes_result.png", dpi=120)
    print("Saved 03_navier_stokes_result.png")


if __name__ == "__main__":
    main()
