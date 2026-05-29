"""Visualization 06 — Structural Mechanics: Plate Deflection + Von Mises Stress.

Physics
-------
Thin elastic plate with uniform transverse load, modelled via biharmonic
decomposition:

    Δ²w = q / D    (Kirchhoff plate equation)

Decomposed as two coupled Poisson equations:
    −Δw = m        (moment)
    −Δm = q / D    (load)

    q/D = 1  (normalised load)

Domain : [0, 1]² (clamped on all edges: w = ∂w/∂n = 0)

For the clamped plate PINN we enforce:
  • w = 0   on all four boundaries
  • ∂w/∂n = 0  on all four boundaries
  • −Δw = m  in Ω
  • −Δm = 1  in Ω

Von Mises stress proxy (bending stresses proportional to second derivatives):
    σ_xx  ≈ −z · w_xx    (bending moment Mxx / plate constant)
    σ_yy  ≈ −z · w_yy
    σ_xy  ≈ −z · w_xy
    σ_VM  = √(σ_xx² − σ_xx·σ_yy + σ_yy² + 3·σ_xy²)
    (evaluated at the outer fibre z = 1)

What this shows
---------------
  • Deflection maximum at the centre, zero on clamped edges.
  • Von Mises stress maximum near the mid-edges where bending is largest.
  • Matches classical thin-plate theory.

Run
---
    python -m examples.visualizations.viz_06_structural
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

N_COL   = 3_000
N_BC    = 400
EPOCHS  = 10_000
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"


# ── Networks ────────────────────────────────────────────────────────────────────
def make_net(out: int = 1) -> nn.Module:
    return nn.Sequential(
        nn.Linear(2, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, out),
    ).to(DEVICE)


# ── Training ──────────────────────────────────────────────────────────────────
def train() -> tuple[nn.Module, nn.Module]:
    torch.manual_seed(0)
    net_w = make_net(1)   # deflection
    net_m = make_net(1)   # moment
    params = list(net_w.parameters()) + list(net_m.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    for ep in range(EPOCHS):
        opt.zero_grad()

        # Interior collocation
        xy = (torch.rand(N_COL, 2, device=DEVICE)).requires_grad_(True)
        w  = net_w(xy)
        m  = net_m(xy)

        # −Δm = 1  →  m_xx + m_yy = −1
        gm   = torch.autograd.grad(m.sum(), xy, create_graph=True)[0]
        m_xx = torch.autograd.grad(gm[:,0].sum(), xy, create_graph=True)[0][:,0:1]
        m_yy = torch.autograd.grad(gm[:,1].sum(), xy, create_graph=True)[0][:,1:2]
        l_pde_m = (m_xx + m_yy + 1.0).pow(2).mean()

        # −Δw = m  →  w_xx + w_yy = −m
        gw   = torch.autograd.grad(w.sum(), xy, create_graph=True)[0]
        w_xx = torch.autograd.grad(gw[:,0].sum(), xy, create_graph=True)[0][:,0:1]
        w_yy = torch.autograd.grad(gw[:,1].sum(), xy, create_graph=True)[0][:,1:2]
        l_pde_w = (w_xx + w_yy + m).pow(2).mean()

        # BCs: w = 0 on all four walls
        n_bc = N_BC
        t_    = torch.rand(n_bc, 1, device=DEVICE)
        bnd_x = torch.cat([torch.zeros(n_bc, 1, device=DEVICE),
                            torch.ones(n_bc, 1, device=DEVICE)])
        bnd_y = torch.cat([t_, t_])
        xy_bc_x = torch.cat([bnd_x, bnd_y], 1)
        l_bc_w  = net_w(xy_bc_x).pow(2).mean()

        bnd_y2 = torch.cat([torch.zeros(n_bc, 1, device=DEVICE),
                             torch.ones(n_bc, 1, device=DEVICE)])
        bnd_x2 = torch.cat([t_, t_])
        xy_bc_y = torch.cat([bnd_x2, bnd_y2], 1)
        l_bc_w += net_w(xy_bc_y).pow(2).mean()

        # Normal derivative ∂w/∂n = 0 on walls (via autograd)
        xy_bc_xg = xy_bc_x.clone().requires_grad_(True)
        w_bc   = net_w(xy_bc_xg)
        gw_bc  = torch.autograd.grad(w_bc.sum(), xy_bc_xg, create_graph=True)[0]
        l_dn_w = gw_bc[:, 0].pow(2).mean()

        xy_bc_yg = xy_bc_y.clone().requires_grad_(True)
        w_bc2  = net_w(xy_bc_yg)
        gw_bc2 = torch.autograd.grad(w_bc2.sum(), xy_bc_yg, create_graph=True)[0]
        l_dn_w += gw_bc2[:, 1].pow(2).mean()

        loss = (l_pde_m + l_pde_w +
                10.0 * l_bc_w + 5.0 * l_dn_w)
        loss.backward()
        opt.step(); sch.step()

        if ep % 2000 == 0:
            print(f"  ep {ep:5d}  loss={float(loss):.4e}")

    return net_w, net_m


# ── Visualize ──────────────────────────────────────────────────────────────────
def visualize(net_w: nn.Module, net_m: nn.Module) -> None:
    NX = 80
    xs = np.linspace(0, 1, NX, dtype=np.float32)
    xg, yg = np.meshgrid(xs, xs)
    xy_flat = torch.tensor(
        np.stack([xg.ravel(), yg.ravel()], axis=1), device=DEVICE,
        requires_grad=True)

    w_pred  = net_w(xy_flat)

    # Second derivatives for Von Mises
    gw   = torch.autograd.grad(w_pred.sum(), xy_flat, create_graph=True)[0]
    w_xx = torch.autograd.grad(gw[:,0].sum(), xy_flat,
                                create_graph=True)[0][:,0].detach().cpu().numpy()
    w_yy = torch.autograd.grad(gw[:,1].sum(), xy_flat,
                                create_graph=True)[0][:,1].detach().cpu().numpy()
    w_xy = torch.autograd.grad(gw[:,0].sum(), xy_flat,
                                create_graph=True)[0][:,1].detach().cpu().numpy()

    w_np   = w_pred.detach().cpu().numpy().reshape(NX, NX)
    w_xx_g = w_xx.reshape(NX, NX)
    w_yy_g = w_yy.reshape(NX, NX)
    w_xy_g = w_xy.reshape(NX, NX)

    # Von Mises at outer fibre z=h/2 (Kirchhoff plate bending stresses)
    s_xx = -w_xx_g
    s_yy = -w_yy_g
    s_xy = -w_xy_g
    vm = np.sqrt(s_xx**2 - s_xx * s_yy + s_yy**2 + 3 * s_xy**2)

    # Displ comparison to series (centre deflection ≈ 0.00126 for clamped plate)
    print(f"  Centre deflection w(0.5,0.5) ≈ {w_np[NX//2, NX//2]:.5f}")
    print(f"  (clamped plate theory: ≈ 0.00126)")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="#0d1117")
    fig.patch.set_facecolor("#0d1117")

    for ax in axes:
        ax.set_facecolor("#0d1117")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=7)

    # Panel 1: deflection
    ax = axes[0]
    im = ax.imshow(w_np, origin="lower", extent=[0, 1, 0, 1],
                   cmap="Blues", aspect="equal")
    ax.contour(xg, yg, w_np, levels=10,
               colors="white", linewidths=0.5, alpha=0.6)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors="#8b949e", labelsize=7)
    cb.set_label("w  (deflection)", color="#8b949e", fontsize=8)
    ax.set_title("Plate deflection  w(x,y)",
                 color="#e6edf3", fontsize=9)
    ax.set_xlabel("x", color="#8b949e"); ax.set_ylabel("y", color="#8b949e")

    # Panel 2: Von Mises stress
    ax = axes[1]
    im2 = ax.imshow(vm, origin="lower", extent=[0, 1, 0, 1],
                    cmap="jet", aspect="equal")
    ax.contour(xg, yg, vm, levels=10,
               colors="white", linewidths=0.4, alpha=0.4)
    cb2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(colors="#8b949e", labelsize=7)
    cb2.set_label("σ_VM  (Von Mises)", color="#8b949e", fontsize=8)
    ax.set_title("Von Mises stress  σ_VM(x,y)",
                 color="#e6edf3", fontsize=9)
    ax.set_xlabel("x", color="#8b949e"); ax.set_ylabel("y", color="#8b949e")

    # Panel 3: Bending curvature |w_xx + w_yy| (Laplacian of w ≈ moment)
    ax = axes[2]
    curvature = np.abs(w_xx_g + w_yy_g)
    im3 = ax.imshow(curvature, origin="lower", extent=[0, 1, 0, 1],
                    cmap="plasma", aspect="equal")
    cb3 = plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
    cb3.ax.tick_params(colors="#8b949e", labelsize=7)
    cb3.set_label("|Δw|  (moment)", color="#8b949e", fontsize=8)
    ax.set_title("Bending moment  |Δw(x,y)|",
                 color="#e6edf3", fontsize=9)
    ax.set_xlabel("x", color="#8b949e"); ax.set_ylabel("y", color="#8b949e")

    fig.suptitle(
        "Clamped Plate Under Uniform Load  Δ²w = 1  ·  PINNeAPPle",
        color="#e6edf3", fontsize=11, y=1.02)
    plt.tight_layout()
    out = "viz_06_structural.png"
    plt.savefig(out, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Saved {out}")


def main() -> None:
    print("─" * 55)
    print("  Viz 06 — Structural Mechanics (Plate + Von Mises)")
    print("─" * 55)
    print(f"  Training on {DEVICE}...")
    net_w, net_m = train()
    visualize(net_w, net_m)


if __name__ == "__main__":
    main()
