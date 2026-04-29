"""04_domain_decomposition.py — DoMINO domain decomposition PINN.

Demonstrates:
- Splitting a 2D domain into overlapping subdomains with DoMINO.partition
- Building a DoMINO trainer with per-subdomain networks
- Interface continuity enforcement
- Training loop via domino.train
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_pinn.domino import DoMINO, Subdomain


# ---------------------------------------------------------------------------
# Problem: 2D Laplace equation  Δu = 0  on [0,2]×[0,1]
# BCs: u(0,y)=0, u(2,y)=1, u(x,0)=u(x,1)=0
# Exact (approx): u ≈ x / 2  (linear in x)
# ---------------------------------------------------------------------------

def laplace_residual(model, xy: torch.Tensor) -> torch.Tensor:
    """Laplace residual: Δu = 0."""
    xy = xy.requires_grad_(True)
    out = model(xy)
    if hasattr(out, "y"):
        out = out.y

    grad1 = torch.autograd.grad(out.sum(), xy, create_graph=True)[0]
    u_x, u_y = grad1[:, 0:1], grad1[:, 1:2]
    u_xx = torch.autograd.grad(u_x.sum(), xy, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y.sum(), xy, create_graph=True)[0][:, 1:2]
    return u_xx + u_yy                                   # (N, 1)


def bc_loss(model, device) -> torch.Tensor:
    """Soft boundary condition losses for all four walls."""
    n = 200

    # Left wall x=0: u=0
    y_l = torch.linspace(0, 1, n, device=device)
    xy_l = torch.stack([torch.zeros(n, device=device), y_l], dim=1)
    u_l = model(xy_l)
    if hasattr(u_l, "y"):
        u_l = u_l.y
    loss = u_l.pow(2).mean()

    # Right wall x=2: u=1
    xy_r = torch.stack([2.0 * torch.ones(n, device=device), y_l], dim=1)
    u_r = model(xy_r)
    if hasattr(u_r, "y"):
        u_r = u_r.y
    loss += (u_r - 1.0).pow(2).mean()

    # Bottom y=0 and top y=1: u=0
    x_tb = torch.linspace(0, 2, n, device=device)
    for y_val in [0.0, 1.0]:
        xy_tb = torch.stack([x_tb, y_val * torch.ones(n, device=device)], dim=1)
        u_tb = model(xy_tb)
        if hasattr(u_tb, "y"):
            u_tb = u_tb.y
        loss += u_tb.pow(2).mean()

    return loss / 4.0


def main():
    torch.manual_seed(99)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Setup: DoMINO partition ------------------------------------------
    # Split [0,2]×[0,1] into 2×2 = 4 subdomains with 10 % overlap
    subdomains = DoMINO.partition(
        bounds=[(0.0, 2.0), (0.0, 1.0)],
        n_splits=(2, 2),
        overlap=0.10,
    )
    print(f"Created {len(subdomains)} subdomains:")
    for i, sd in enumerate(subdomains):
        print(f"  [{i}] {sd}")

    # --- Model factory: small MLP per subdomain ---------------------------
    def model_factory():
        return nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    domino = DoMINO(
        subdomains=subdomains,
        model_factory=model_factory,
        interface_weight=20.0,
    ).to(device)

    # --- Collocation points -----------------------------------------------
    n_col = 4096
    xy_col_np = np.column_stack([
        np.random.uniform(0, 2, n_col),
        np.random.uniform(0, 1, n_col),
    ]).astype(np.float32)
    xy_col = torch.tensor(xy_col_np, device=device)

    # --- Training ----------------------------------------------------------
    optimizer = torch.optim.Adam(domino.parameters(), lr=1e-3)
    history = {"pde": [], "bc": [], "interface": [], "total": []}
    n_epochs = 4000
    print_every = 500

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad(set_to_none=True)

        # PDE residual via domino forward (blended across subdomains)
        xy_c = xy_col.clone().requires_grad_(True)
        res = laplace_residual(domino, xy_c)
        loss_pde = res.pow(2).mean()

        # BC loss
        loss_bc = bc_loss(domino, device)

        # Interface continuity (built into DoMINO.interface_loss)
        loss_iface = domino.interface_loss(xy_c.detach())

        loss = loss_pde + 10.0 * loss_bc + domino.interface_weight * loss_iface
        loss.backward()
        nn.utils.clip_grad_norm_(domino.parameters(), 1.0)
        optimizer.step()

        history["pde"].append(float(loss_pde.item()))
        history["bc"].append(float(loss_bc.item()))
        history["interface"].append(float(loss_iface.item()))
        history["total"].append(float(loss.item()))

        if epoch % print_every == 0:
            print(f"  epoch {epoch:4d} | pde={loss_pde.item():.3e}"
                  f"  bc={loss_bc.item():.3e}"
                  f"  iface={loss_iface.item():.3e}")

    print("\nTraining complete.")

    # --- Prediction -------------------------------------------------------
    nx, ny = 80, 40
    x_lin = np.linspace(0, 2, nx)
    y_lin = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x_lin, y_lin)
    xy_vis = torch.tensor(
        np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        u_pred = domino(xy_vis).cpu().numpy().reshape(ny, nx)

    # --- Plot -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    im = axes[0].contourf(xx, yy, u_pred, levels=30, cmap="viridis")
    plt.colorbar(im, ax=axes[0])
    axes[0].set_title("DoMINO solution u(x,y)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    axes[1].semilogy(history["pde"],   label="PDE")
    axes[1].semilogy(history["bc"],    label="BC")
    axes[1].semilogy(history["interface"], label="Interface")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].set_title("DoMINO training losses")
    axes[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("04_domain_decomposition_result.png", dpi=120)
    print("Saved 04_domain_decomposition_result.png")


if __name__ == "__main__":
    main()
