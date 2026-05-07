"""
TM₁₁ mode of a rectangular metallic waveguide / resonant cavity.

The transverse magnetic (TM) modes satisfy the 2-D Helmholtz eigenvalue problem:

    nabla^2 E_z + k^2 E_z = 0      on  [0,a] x [0,b]
    E_z = 0                          on all walls  (PEC boundary)

For a = b = 1 the lowest TM mode is TM₁₁:
    E_z = sin(pi*x) * sin(pi*y)
    k^2  = (pi/a)^2 + (pi/b)^2 = 2*pi^2

Training strategy: k² is prescribed as the known resonant value; the network
must find the mode shape.  A normalization point E_z(0.5, 0.5) = 1 prevents the
trivial zero solution.
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
torch.manual_seed(21)

K2 = 2.0 * (torch.pi ** 2)   # resonant wavenumber squared for TM₁₁


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


def helmholtz_residual(net, xy, k2):
    u = net(xy)
    du = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(du[:, 0].sum(), xy, create_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(du[:, 1].sum(), xy, create_graph=True)[0][:, 1:2]
    return uxx + uyy + k2 * u


def train():
    net = MLP().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=8000, eta_min=1e-5)

    N_COL = 2048
    N_BND = 256

    t = torch.rand(N_BND, 1, device=DEVICE)
    bnd = torch.cat([
        torch.cat([t,               torch.zeros_like(t)], 1),
        torch.cat([t,               torch.ones_like(t)],  1),
        torch.cat([torch.zeros_like(t), t],               1),
        torch.cat([torch.ones_like(t),  t],               1),
    ], dim=0)

    # Normalization point at mode peak
    xnorm = torch.tensor([[0.5, 0.5]], device=DEVICE)

    losses = []
    for epoch in range(8000):
        xy = torch.rand(N_COL, 2, device=DEVICE).requires_grad_(True)
        res = helmholtz_residual(net, xy, K2)
        loss_pde = (res ** 2).mean()
        loss_bc  = (net(bnd) ** 2).mean()
        loss_norm = (net(xnorm) - 1.0) ** 2

        loss = loss_pde + 20.0 * loss_bc + 50.0 * loss_norm

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
    xv = torch.linspace(0, 1, N, device=DEVICE)
    yv = torch.linspace(0, 1, N, device=DEVICE)
    X, Y = torch.meshgrid(xv, yv, indexing="ij")
    xy_g = torch.stack([X.flatten(), Y.flatten()], dim=1)

    with torch.no_grad():
        Ez_pred  = net(xy_g).reshape(N, N).cpu().numpy()

    Ez_exact = (torch.sin(torch.pi * X) * torch.sin(torch.pi * Y)).cpu().numpy()

    # Normalize exact to same sign/scale as prediction at (0.5, 0.5)
    center_pred = Ez_pred[N // 2, N // 2]
    center_exact = Ez_exact[N // 2, N // 2]
    if abs(center_exact) > 1e-8:
        Ez_exact_scaled = Ez_exact * (center_pred / center_exact)
    else:
        Ez_exact_scaled = Ez_exact

    err = np.abs(Ez_pred - Ez_exact_scaled)
    Xn, Yn = X.cpu().numpy(), Y.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "TM₁₁ Waveguide Mode  ∇²E_z + k²E_z = 0   (k² = 2π²,  a=b=1)",
        fontsize=13
    )

    im0 = axes[0].contourf(Xn, Yn, Ez_pred, levels=25, cmap="RdBu_r")
    axes[0].contour(Xn, Yn, Ez_pred, levels=10, colors="k", linewidths=0.4, alpha=0.5)
    axes[0].set_title("PINN  E_z(x,y)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(Xn, Yn, Ez_exact_scaled, levels=25, cmap="RdBu_r")
    axes[1].set_title("Exact  sin(πx)sin(πy)")
    axes[1].set_xlabel("x")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].contourf(Xn, Yn, err, levels=25, cmap="hot_r")
    l2 = float(np.linalg.norm(err) / (np.linalg.norm(Ez_exact_scaled) + 1e-12))
    axes[2].set_title(f"|PINN − Exact|   L2={l2:.3e}")
    axes[2].set_xlabel("x")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    out = RESULTS / "06_tm_waveguide.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")

    # Cross-sections at x=0.5 and y=0.5
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig2.suptitle("Cross-sections of TM₁₁ mode", fontsize=12)

    xn = xv.cpu().numpy()
    yn = yv.cpu().numpy()

    mid = N // 2
    ax1.plot(yn, Ez_pred[mid, :],       "r--", lw=2, label="PINN")
    ax1.plot(yn, Ez_exact_scaled[mid,:], "k-",  lw=2, label="Exact")
    ax1.set_title("E_z(0.5, y)"); ax1.set_xlabel("y")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(xn, Ez_pred[:, mid],       "r--", lw=2, label="PINN")
    ax2.plot(xn, Ez_exact_scaled[:, mid],"k-", lw=2, label="Exact")
    ax2.set_title("E_z(x, 0.5)"); ax2.set_xlabel("x")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS / "06_tm_waveguide_cross.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig3, ax = plt.subplots(figsize=(7, 3))
    ax.semilogy(losses)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training loss — TM waveguide")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "06_tm_waveguide_loss.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved cross-sections + loss -> {RESULTS}")


if __name__ == "__main__":
    print("=== 06 TM Waveguide — Helmholtz Eigenmode ===")
    net, losses = train()
    plot_results(net, losses)
    print("Done.\n")
