"""
Electric potential of a dipole (two opposite Gaussian charge blobs).

PDE : nabla^2 phi = -(rho_plus - rho_minus)   on  [-1,1]^2
      rho_pm = +/-A * exp( -|r - r_pm|^2 / (2*sigma^2) )
      r_+ = (0.3, 0),  r_- = (-0.3, 0)   (centres shifted to [0,2] domain)

BC  : phi = 0 on all boundaries (grounded enclosure)

Analytic dipole field in free space:  phi ~ cos(theta)/r^2  (for reference)
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
torch.manual_seed(7)

A     = 4.0
SIGMA = 0.12
# domain [-1,1]^2; charges at (±0.4, 0)
X_POS = ( 0.4, 0.0)
X_NEG = (-0.4, 0.0)


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


def _gauss(xy, cx, cy):
    r2 = (xy[:, 0:1] - cx) ** 2 + (xy[:, 1:2] - cy) ** 2
    return A * torch.exp(-r2 / (2.0 * SIGMA ** 2))


def source(xy):
    """Dipole source: rho_+ - rho_-"""
    return _gauss(xy, *X_POS) - _gauss(xy, *X_NEG)


def pde_residual(net, xy):
    u = net(xy)
    du = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(du[:, 0].sum(), xy, create_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(du[:, 1].sum(), xy, create_graph=True)[0][:, 1:2]
    return uxx + uyy + source(xy)


def boundary_pts(N, device):
    t = torch.linspace(-1, 1, N, device=device).unsqueeze(1)
    neg1 = -torch.ones_like(t)
    pos1 =  torch.ones_like(t)
    return torch.cat([
        torch.cat([t,    neg1], 1),   # bottom  y=-1
        torch.cat([t,    pos1], 1),   # top     y=+1
        torch.cat([neg1, t   ], 1),   # left    x=-1
        torch.cat([pos1, t   ], 1),   # right   x=+1
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
        phi = net(xy_g).reshape(N, N).cpu().numpy()

    # E-field via autograd
    xyr = xy_g.requires_grad_(True)
    phi_g = net(xyr)
    dphi = torch.autograd.grad(phi_g, xyr, torch.ones_like(phi_g))[0]
    Ex = -dphi[:, 0].reshape(N, N).detach().cpu().numpy()
    Ey = -dphi[:, 1].reshape(N, N).detach().cpu().numpy()
    E_mag = np.sqrt(Ex ** 2 + Ey ** 2) + 1e-10

    Xn, Yn = X.cpu().numpy(), Y.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Electric Dipole — Poisson  ∇²φ = −(ρ₊ − ρ₋)", fontsize=13)

    # Left: equipotential contours + field lines
    levels = np.linspace(phi.min(), phi.max(), 30)
    cs = axes[0].contour(Xn, Yn, phi, levels=levels, cmap="RdBu_r", linewidths=0.8)
    axes[0].streamplot(xv.cpu().numpy(), yv.cpu().numpy(),
                       Ex.T, Ey.T,
                       color=np.log1p(E_mag.T), cmap="autumn", linewidth=0.9,
                       density=1.4, arrowsize=1.0)
    axes[0].plot(*X_POS, "r+", ms=12, mew=2, label="+q")
    axes[0].plot(*X_NEG, "b_", ms=12, mew=2, label="−q")
    axes[0].set_title("Equipotentials  +  E-field lines")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].legend(loc="upper right")
    axes[0].set_xlim(-1, 1); axes[0].set_ylim(-1, 1)

    # Right: filled potential contour
    im = axes[1].contourf(Xn, Yn, phi, levels=30, cmap="RdBu_r")
    axes[1].plot(*X_POS, "r+", ms=12, mew=2)
    axes[1].plot(*X_NEG, "b_", ms=12, mew=2)
    axes[1].set_title("Potential φ(x,y)")
    axes[1].set_xlabel("x")
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    out = RESULTS / "03_electric_dipole.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")

    fig2, ax = plt.subplots(figsize=(7, 3))
    ax.semilogy(losses)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training loss — Electric dipole")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "03_electric_dipole_loss.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("=== 03 Electric Dipole ===")
    net, losses = train()
    plot_results(net, losses)
    print("Done.\n")
