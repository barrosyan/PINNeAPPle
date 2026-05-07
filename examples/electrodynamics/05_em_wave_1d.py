"""
1-D transverse electromagnetic wave  (TE / TM plane wave).

PDE : E_tt = c^2 * E_xx      x in [0,1],  t in [0,1]
IC  : E(x,0)   = sin(pi*x)
      E_t(x,0) = 0
BC  : E(0,t) = E(1,t) = 0   (perfect electric conductor walls)

Exact: E(x,t) = sin(pi*x) * cos(pi*c*t)

The higher the wave speed c, the harder the problem for a PINN.
c = 1 gives k*T ~ 1 (mild), c = 2 is already challenging.
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
torch.manual_seed(13)

C = 1.0   # wave speed


class MLP(nn.Module):
    """SIREN-inspired init with Tanh for stable training."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def wave_residual(net, xt):
    """Returns (u, u_tt - c^2 u_xx)."""
    u = net(xt)
    du = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(du[:, 0].sum(), xt, create_graph=True)[0][:, 0:1]
    utt = torch.autograd.grad(du[:, 1].sum(), xt, create_graph=True)[0][:, 1:2]
    return u, utt - C ** 2 * uxx


def du_dt(net, xt):
    """dE/dt  for the zero-velocity IC."""
    u = net(xt)
    du = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0]
    return du[:, 1:2]


def train():
    net = MLP().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=8000, eta_min=5e-6)

    # Collocation in space-time
    N_COL = 3072
    N_IC  = 512
    N_BC  = 512

    # IC points: t=0, x uniform
    x_ic = torch.rand(N_IC, 1, device=DEVICE)
    t0   = torch.zeros(N_IC, 1, device=DEVICE)
    xt_ic = torch.cat([x_ic, t0], dim=1)
    E_ic_exact = torch.sin(torch.pi * x_ic)

    # BC points: x=0 and x=1
    t_bc = torch.rand(N_BC, 1, device=DEVICE)
    xt_left  = torch.cat([torch.zeros_like(t_bc), t_bc], 1)
    xt_right = torch.cat([torch.ones_like(t_bc),  t_bc], 1)

    losses = []
    for epoch in range(8000):
        xt = torch.rand(N_COL, 2, device=DEVICE).requires_grad_(True)
        _, res = wave_residual(net, xt)
        loss_pde = (res ** 2).mean()

        loss_ic_val = ((net(xt_ic) - E_ic_exact) ** 2).mean()

        xt_ic_g = xt_ic.clone().detach().requires_grad_(True)
        loss_ic_vel = (du_dt(net, xt_ic_g) ** 2).mean()

        loss_bc = (net(xt_left) ** 2).mean() + (net(xt_right) ** 2).mean()

        loss = loss_pde + 20.0 * (loss_ic_val + loss_ic_vel) + 20.0 * loss_bc

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        losses.append(float(loss.item()))
        if epoch % 1000 == 0:
            print(f"  epoch {epoch:5d}  loss={loss.item():.3e}")

    return net, losses


def plot_results(net, losses):
    Nx, Nt = 100, 100
    xv = torch.linspace(0, 1, Nx, device=DEVICE)
    tv = torch.linspace(0, 1, Nt, device=DEVICE)
    X, T = torch.meshgrid(xv, tv, indexing="ij")
    xt_g = torch.stack([X.flatten(), T.flatten()], dim=1)

    with torch.no_grad():
        E_pred = net(xt_g).reshape(Nx, Nt).cpu().numpy()

    E_exact = (torch.sin(torch.pi * X) * torch.cos(torch.pi * C * T)).cpu().numpy()
    err = np.abs(E_pred - E_exact)

    Xn, Tn = X.cpu().numpy(), T.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f"1-D EM Wave  E_tt = c²E_xx  (c={C},  Exact: sin(πx)cos(πct))", fontsize=13)

    im0 = axes[0].contourf(Tn, Xn, E_pred, levels=25, cmap="seismic")
    axes[0].set_title("PINN  E(x,t)")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("x")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(Tn, Xn, E_exact, levels=25, cmap="seismic")
    axes[1].set_title("Exact  E(x,t)")
    axes[1].set_xlabel("t")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].contourf(Tn, Xn, err, levels=25, cmap="hot_r")
    l2 = float(np.linalg.norm(err) / (np.linalg.norm(E_exact) + 1e-12))
    axes[2].set_title(f"|PINN − Exact|  L2={l2:.3e}")
    axes[2].set_xlabel("t")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    out = RESULTS / "05_em_wave_1d.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")

    # Snapshots at several times
    fig2, axes2 = plt.subplots(1, 4, figsize=(16, 3.5))
    fig2.suptitle("Snapshots  E(x, t*)", fontsize=12)
    for ax, ti in zip(axes2, [0.0, 0.25, 0.5, 0.75]):
        xn = xv.cpu().numpy()
        t_snap = torch.full((Nx, 1), ti, device=DEVICE)
        xt_snap = torch.cat([xv.unsqueeze(1), t_snap], dim=1)
        with torch.no_grad():
            e_p = net(xt_snap).squeeze().cpu().numpy()
        e_ex = np.sin(np.pi * xn) * np.cos(np.pi * C * ti)
        ax.plot(xn, e_ex, "k-",  lw=1.5, label="Exact")
        ax.plot(xn, e_p,  "r--", lw=1.5, label="PINN")
        ax.set_title(f"t = {ti}")
        ax.set_xlabel("x"); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(RESULTS / "05_em_wave_snapshots.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig3, ax = plt.subplots(figsize=(7, 3))
    ax.semilogy(losses)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training loss — EM wave 1D")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "05_em_wave_loss.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved snapshots + loss -> {RESULTS}")


if __name__ == "__main__":
    print("=== 05 Electromagnetic Wave 1D ===")
    net, losses = train()
    plot_results(net, losses)
    print("Done.\n")
