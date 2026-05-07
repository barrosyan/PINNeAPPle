"""
Magnetostatic vector potential of an infinite current-carrying wire.

2-D cross-section in the xy-plane.  The wire carries current in the z direction;
by symmetry the vector potential is purely A_z(x,y).

PDE : nabla^2 A_z = -mu0 * J_z(x,y)   on  [-1,1]^2
      J_z = J0   for  r < r0   (circular wire cross-section)
      J_z = 0    outside

BC  : A_z = 0  on all boundaries

Magnetic field: B_x = dA_z/dy,  B_y = -dA_z/dx
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
torch.manual_seed(3)

MU0 = 1.0   # normalised units
J0  = 1.0   # current density inside wire
R0  = 0.35  # wire radius


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


def J_z(xy: torch.Tensor) -> torch.Tensor:
    """Smooth approximation of the step function for the current region."""
    r = torch.sqrt(xy[:, 0:1] ** 2 + xy[:, 1:2] ** 2)
    return J0 * torch.sigmoid((R0 - r) * 25.0)


def pde_residual(net, xy):
    u = net(xy)
    du = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(du[:, 0].sum(), xy, create_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(du[:, 1].sum(), xy, create_graph=True)[0][:, 1:2]
    return uxx + uyy + MU0 * J_z(xy)


def boundary_pts(N, device):
    t = torch.linspace(-1, 1, N, device=device).unsqueeze(1)
    neg1 = -torch.ones_like(t)
    pos1 =  torch.ones_like(t)
    return torch.cat([
        torch.cat([t,    neg1], 1),
        torch.cat([t,    pos1], 1),
        torch.cat([neg1, t   ], 1),
        torch.cat([pos1, t   ], 1),
    ], dim=0)


def train():
    net = MLP().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6000, eta_min=1e-5)

    bnd = boundary_pts(256, DEVICE)

    losses = []
    for epoch in range(6000):
        xy = (torch.rand(2048, 2, device=DEVICE) * 2.0 - 1.0).requires_grad_(True)
        res = pde_residual(net, xy)
        loss = (res ** 2).mean() + 20.0 * (net(bnd) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        losses.append(float(loss.item()))
        if epoch % 1000 == 0:
            print(f"  epoch {epoch:5d}  loss={loss.item():.3e}")

    return net, losses


def plot_results(net, losses):
    N = 120
    xv = torch.linspace(-1, 1, N, device=DEVICE)
    yv = torch.linspace(-1, 1, N, device=DEVICE)
    X, Y = torch.meshgrid(xv, yv, indexing="ij")
    xy_g = torch.stack([X.flatten(), Y.flatten()], dim=1)

    with torch.no_grad():
        Az = net(xy_g).reshape(N, N).cpu().numpy()

    # B = curl A: B_x = dA_z/dy, B_y = -dA_z/dx
    xyr = xy_g.requires_grad_(True)
    Az_g = net(xyr)
    dA = torch.autograd.grad(Az_g, xyr, torch.ones_like(Az_g))[0]
    Bx = ( dA[:, 1]).reshape(N, N).detach().cpu().numpy()
    By = (-dA[:, 0]).reshape(N, N).detach().cpu().numpy()
    B_mag = np.sqrt(Bx ** 2 + By ** 2) + 1e-12

    Xn, Yn = X.cpu().numpy(), Y.cpu().numpy()

    # Wire boundary circle for illustration
    theta = np.linspace(0, 2 * np.pi, 200)
    xc, yc = R0 * np.cos(theta), R0 * np.sin(theta)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Magnetostatics — Infinite Current Wire  ∇²A_z = −μ₀J_z", fontsize=13)

    im0 = axes[0].contourf(Xn, Yn, Az, levels=25, cmap="viridis")
    axes[0].plot(xc, yc, "w--", lw=1.2, label=f"wire r={R0}")
    axes[0].set_title("Vector potential  A_z(x,y)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].legend(fontsize=8)
    plt.colorbar(im0, ax=axes[0])

    axes[1].streamplot(xv.cpu().numpy(), yv.cpu().numpy(),
                       Bx.T, By.T,
                       color=np.log1p(B_mag.T), cmap="plasma",
                       linewidth=0.9, density=1.5, arrowsize=0.9)
    axes[1].plot(xc, yc, "w--", lw=1.2)
    axes[1].set_title("Magnetic field  B(x,y)")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")

    # Radial profile of A_z along y=0
    with torch.no_grad():
        xline = torch.linspace(-1, 1, 300, device=DEVICE).unsqueeze(1)
        yline = torch.zeros_like(xline)
        Az_line = net(torch.cat([xline, yline], 1)).squeeze().cpu().numpy()
    x_line = xline.squeeze().cpu().numpy()
    axes[2].plot(x_line, Az_line, label="PINN A_z(x,0)")
    axes[2].axvline(-R0, color="gray", ls="--", lw=0.8, label="wire edge")
    axes[2].axvline( R0, color="gray", ls="--", lw=0.8)
    axes[2].set_xlabel("x"); axes[2].set_ylabel("A_z")
    axes[2].set_title("Radial profile  y=0")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = RESULTS / "04_magnetostatics_wire.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")

    fig2, ax = plt.subplots(figsize=(7, 3))
    ax.semilogy(losses)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training loss — Magnetostatics wire")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "04_magnetostatics_wire_loss.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("=== 04 Magnetostatics — Current-Carrying Wire ===")
    net, losses = train()
    plot_results(net, losses)
    print("Done.\n")
