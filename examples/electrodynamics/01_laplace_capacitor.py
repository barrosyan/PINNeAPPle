"""
Electrostatic potential in a parallel-plate capacitor.

PDE : nabla^2 phi = 0   on  [0,1]^2
BC  : phi = 0  on y=0  (bottom plate, ground)
      phi = 1  on y=1  (top plate, +V)
      dphi/dn = 0  on x=0, x=1  (Neumann, no flux through sides)

Exact: phi(x, y) = y
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
torch.manual_seed(42)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def laplacian(net, xy):
    """Returns (u, nabla^2 u) with shared forward pass."""
    u = net(xy)
    du = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(du[:, 0].sum(), xy, create_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(du[:, 1].sum(), xy, create_graph=True)[0][:, 1:2]
    return u, uxx + uyy


def du_dx(net, xy):
    """Returns dphi/dx for Neumann BC on x=0 and x=1."""
    u = net(xy)
    du = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0]
    return du[:, 0:1]


def train():
    net = MLP().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000, eta_min=1e-5)

    N_COL = 2048
    N_BND = 256

    t = torch.rand(N_BND, 1, device=DEVICE)
    bot   = torch.cat([t, torch.zeros_like(t)], dim=1)
    top   = torch.cat([t, torch.ones_like(t)],  dim=1)
    left  = torch.cat([torch.zeros_like(t), t], dim=1)
    right = torch.cat([torch.ones_like(t),  t], dim=1)

    losses = []
    for epoch in range(5000):
        xy = torch.rand(N_COL, 2, device=DEVICE).requires_grad_(True)

        _, lap = laplacian(net, xy)
        loss_pde = (lap ** 2).mean()

        loss_bot = (net(bot) ** 2).mean()
        loss_top = ((net(top) - 1.0) ** 2).mean()

        xl = left.clone().detach().requires_grad_(True)
        loss_neu = (du_dx(net, xl) ** 2).mean()
        xr = right.clone().detach().requires_grad_(True)
        loss_neu = loss_neu + (du_dx(net, xr) ** 2).mean()

        loss = loss_pde + 20.0 * (loss_bot + loss_top) + loss_neu

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        losses.append(float(loss.item()))
        if epoch % 1000 == 0:
            print(f"  epoch {epoch:5d}  loss={loss.item():.3e}")

    return net, losses


def plot_results(net, losses):
    N = 100
    xv = torch.linspace(0, 1, N, device=DEVICE)
    yv = torch.linspace(0, 1, N, device=DEVICE)
    X, Y = torch.meshgrid(xv, yv, indexing="ij")
    xy_grid = torch.stack([X.flatten(), Y.flatten()], dim=1)

    with torch.no_grad():
        phi_pred = net(xy_grid).reshape(N, N).cpu().numpy()

    phi_exact = Y.cpu().numpy()
    err = np.abs(phi_pred - phi_exact)

    # Electric field E = -grad(phi)
    xy_g = xy_grid.requires_grad_(True)
    phi_g = net(xy_g)
    dphi = torch.autograd.grad(phi_g, xy_g, torch.ones_like(phi_g))[0]
    Ex = -dphi[:, 0].reshape(N, N).detach().cpu().numpy()
    Ey = -dphi[:, 1].reshape(N, N).detach().cpu().numpy()

    Xn = X.cpu().numpy()
    Yn = Y.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Electrostatic Potential — Parallel-Plate Capacitor  (Laplace)", fontsize=13)

    s = 7
    im0 = axes[0].contourf(Xn, Yn, phi_pred, levels=20, cmap="RdBu_r")
    axes[0].quiver(Xn[::s, ::s], Yn[::s, ::s], Ex[::s, ::s], Ey[::s, ::s],
                   scale=12, color="k", alpha=0.7)
    axes[0].set_title("PINN  φ  +  E-field vectors")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(Xn, Yn, phi_exact, levels=20, cmap="RdBu_r")
    axes[1].set_title("Exact  φ = y")
    axes[1].set_xlabel("x")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].contourf(Xn, Yn, err, levels=20, cmap="hot_r")
    axes[2].set_title(f"|PINN − Exact|  (max = {err.max():.2e})")
    axes[2].set_xlabel("x")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    out = RESULTS / "01_laplace_capacitor.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")

    fig2, ax = plt.subplots(figsize=(7, 3))
    ax.semilogy(losses)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training loss — Laplace capacitor")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = RESULTS / "01_laplace_capacitor_loss.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out2}")

    l2 = float(np.linalg.norm(err) / np.linalg.norm(phi_exact + 1e-12))
    print(f"  L2 relative error = {l2:.4e}")


if __name__ == "__main__":
    print("=== 01 Electrostatic Laplace Capacitor ===")
    net, losses = train()
    plot_results(net, losses)
    print("Done.\n")
