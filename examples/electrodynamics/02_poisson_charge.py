"""
Electric potential from a localised charge distribution.

PDE : nabla^2 phi = -rho(x,y)   on  [0,1]^2
      rho(x,y) = A * exp( -((x-0.5)^2 + (y-0.5)^2) / (2*sigma^2) )

BC  : phi = 0 on all four walls (grounded box)

No closed-form exact solution; a fine-grid FDM reference is computed
at evaluation time to measure accuracy.
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
torch.manual_seed(0)

A     = 5.0    # charge amplitude
SIGMA = 0.08   # Gaussian width


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


def rho(xy: torch.Tensor) -> torch.Tensor:
    r2 = (xy[:, 0:1] - 0.5) ** 2 + (xy[:, 1:2] - 0.5) ** 2
    return A * torch.exp(-r2 / (2.0 * SIGMA ** 2))


def pde_residual(net, xy):
    """nabla^2 phi + rho = 0  =>  residual = nabla^2 phi + rho"""
    u = net(xy)
    du = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(du[:, 0].sum(), xy, create_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(du[:, 1].sum(), xy, create_graph=True)[0][:, 1:2]
    return uxx + uyy + rho(xy)


def fdm_reference(N=200):
    """5-point Laplacian FDM on N×N grid, Dirichlet phi=0 on boundary."""
    h = 1.0 / (N + 1)
    x = np.linspace(h, 1 - h, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    src = A * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * SIGMA ** 2))

    # Assemble sparse system (interior only)
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve
    n2 = N * N
    M = lil_matrix((n2, n2))
    idx = lambda i, j: i * N + j
    for i in range(N):
        for j in range(N):
            k = idx(i, j)
            M[k, k] = -4.0
            if i > 0:     M[k, idx(i-1, j)] = 1.0
            if i < N - 1: M[k, idx(i+1, j)] = 1.0
            if j > 0:     M[k, idx(i, j-1)] = 1.0
            if j < N - 1: M[k, idx(i, j+1)] = 1.0
    rhs = -(h ** 2) * src.flatten()
    phi = spsolve(M.tocsr(), rhs).reshape(N, N)
    return x, phi


def train():
    net = MLP().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6000, eta_min=1e-5)

    N_COL = 3072
    N_BND = 256

    t = torch.rand(N_BND, 1, device=DEVICE)
    bnd_pts = torch.cat([
        torch.cat([t,               torch.zeros_like(t)], 1),   # bottom
        torch.cat([t,               torch.ones_like(t)],  1),   # top
        torch.cat([torch.zeros_like(t), t],               1),   # left
        torch.cat([torch.ones_like(t),  t],               1),   # right
    ], dim=0)

    losses = []
    for epoch in range(6000):
        xy = torch.rand(N_COL, 2, device=DEVICE).requires_grad_(True)
        res = pde_residual(net, xy)
        loss_pde = (res ** 2).mean()
        loss_bc  = (net(bnd_pts) ** 2).mean()
        loss = loss_pde + 20.0 * loss_bc

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
    xy_g = torch.stack([X.flatten(), Y.flatten()], dim=1)

    with torch.no_grad():
        phi_pred = net(xy_g).reshape(N, N).cpu().numpy()
        rho_val  = rho(xy_g).reshape(N, N).cpu().numpy()

    Xn, Yn = X.cpu().numpy(), Y.cpu().numpy()

    # FDM reference (optional — needs scipy)
    phi_ref = None
    try:
        xf, phi_ref = fdm_reference(N=150)
        Xf, Yf = np.meshgrid(xf, xf, indexing="ij")
    except Exception:
        pass

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Electrostatic Potential — Gaussian Charge Distribution  (Poisson)", fontsize=13)

    im0 = axes[0].contourf(Xn, Yn, phi_pred, levels=20, cmap="plasma")
    axes[0].set_title("PINN  φ (predicted)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(Xn, Yn, rho_val, levels=20, cmap="inferno")
    axes[1].set_title("Source  ρ(x,y)")
    axes[1].set_xlabel("x")
    plt.colorbar(im1, ax=axes[1])

    if phi_ref is not None:
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator((xf, xf), phi_ref, bounds_error=False, fill_value=0.0)
        pts = np.stack([Xn.flatten(), Yn.flatten()], axis=1)
        phi_ref_nn = interp(pts).reshape(N, N)
        err = np.abs(phi_pred - phi_ref_nn)
        im2 = axes[2].contourf(Xn, Yn, err, levels=20, cmap="hot_r")
        axes[2].set_title(f"|PINN − FDM|  (max={err.max():.2e})")
        plt.colorbar(im2, ax=axes[2])
    else:
        axes[2].text(0.5, 0.5, "scipy not available\nfor FDM reference",
                     ha="center", va="center", transform=axes[2].transAxes)

    for ax in axes:
        ax.set_xlabel("x")
    plt.tight_layout()
    out = RESULTS / "02_poisson_charge.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")

    # Loss curve
    fig2, ax = plt.subplots(figsize=(7, 3))
    ax.semilogy(losses)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training loss — Poisson charge")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "02_poisson_charge_loss.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("=== 02 Poisson — Gaussian Charge Distribution ===")
    net, losses = train()
    plot_results(net, losses)
    print("Done.\n")
