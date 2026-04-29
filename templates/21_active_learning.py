"""21_active_learning.py — Residual-based active learning for PINNs.

Demonstrates:
- ResidualActiveSampler: adaptively adds collocation points where PDE residual is largest
- VarianceActiveSampler: uses MC-Dropout uncertainty to guide sampling
- CombinedActiveSampler: hybrid residual + variance strategy
- Comparison of active vs. uniform sampling convergence
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_data.active_learning import (
    ResidualActiveSampler,
    VarianceActiveSampler,
    CombinedActiveSampler,
)


# ---------------------------------------------------------------------------
# Problem: 2D Poisson  Δu = f  on [0,1]²  (same as template 01)
# Exact u(x,y) = sin(πx)sin(πy)
# ---------------------------------------------------------------------------

import math

def f_source(xy: torch.Tensor) -> torch.Tensor:
    x, y = xy[:, 0:1], xy[:, 1:2]
    return -2.0 * math.pi ** 2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)


def poisson_residual(model: nn.Module, xy: torch.Tensor) -> torch.Tensor:
    xy = xy.requires_grad_(True)
    u  = model(xy)
    if hasattr(u, "y"):
        u = u.y
    g  = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
    u_xx = torch.autograd.grad(g[:, 0:1].sum(), xy, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(g[:, 1:2].sum(), xy, create_graph=True)[0][:, 1:2]
    return u_xx + u_yy - f_source(xy)


def build_pinn(dropout_p: float = 0.1) -> nn.Module:
    return nn.Sequential(
        nn.Linear(2, 64), nn.Tanh(),
        nn.Dropout(dropout_p),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Dropout(dropout_p),
        nn.Linear(64, 1),
    )


def train_step(model, optimizer, xy_col, xy_bc, u_bc, n_steps: int = 200):
    for _ in range(n_steps):
        optimizer.zero_grad()
        res  = poisson_residual(model, xy_col.clone())
        l_pd = res.pow(2).mean()
        out  = model(xy_bc)
        if hasattr(out, "y"):
            out = out.y
        l_bc = (out - u_bc).pow(2).mean()
        loss = l_pd + 10 * l_bc
        loss.backward()
        optimizer.step()
    return float(loss.item())


def l2_error(model, device) -> float:
    n = 50
    x_ = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(x_, x_)
    xy_t = torch.tensor(
        np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        u_pred = model(xy_t).cpu().numpy().ravel()
    u_ex = (np.sin(math.pi * xx) * np.sin(math.pi * yy)).ravel()
    return float(np.sqrt(((u_pred - u_ex)**2).mean()) / np.sqrt((u_ex**2).mean()))


def main():
    torch.manual_seed(99)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    N_INIT   = 256      # initial collocation points
    N_ADD    = 128      # points added per active iteration
    N_ROUNDS = 8        # active learning rounds
    STEPS    = 400      # training steps per round
    CAND     = 4096     # candidate pool size

    # Boundary conditions (Dirichlet u=0)
    from pinneaple_geom.csg import CSGRectangle
    rect = CSGRectangle(x_min=0, y_min=0, x_max=1, y_max=1)
    xy_bc_np = rect.sample_boundary(n=256, seed=0)
    xy_bc = torch.tensor(xy_bc_np, dtype=torch.float32, device=device)
    u_bc  = torch.zeros(len(xy_bc), 1, device=device)

    # Candidate pool
    xy_cand_np = np.random.rand(CAND, 2).astype(np.float32)
    xy_cand    = torch.tensor(xy_cand_np, device=device)

    results = {}

    # =========================================================================
    # Strategy A: uniform (no active learning — baseline)
    # =========================================================================
    print("\n[A] Uniform sampling baseline ...")
    xy_col_u = torch.rand(N_INIT, 2, device=device)
    model_u  = build_pinn().to(device)
    opt_u    = torch.optim.Adam(model_u.parameters(), lr=1e-3)
    err_u    = []
    for r in range(N_ROUNDS):
        train_step(model_u, opt_u, xy_col_u, xy_bc, u_bc, STEPS)
        # Add uniformly sampled points
        new_pts = torch.rand(N_ADD, 2, device=device)
        xy_col_u = torch.cat([xy_col_u, new_pts], dim=0)
        err_u.append(l2_error(model_u, device))
        print(f"  round {r+1}: n_col={xy_col_u.shape[0]}  L2={err_u[-1]:.4e}")
    results["uniform"] = err_u

    # =========================================================================
    # Strategy B: Residual active sampling
    # =========================================================================
    print("\n[B] Residual active sampling ...")
    xy_col_r = torch.rand(N_INIT, 2, device=device)
    model_r  = build_pinn().to(device)
    opt_r    = torch.optim.Adam(model_r.parameters(), lr=1e-3)

    res_sampler = ResidualActiveSampler(
        residual_fn=poisson_residual,
        n_add=N_ADD,
        temperature=1.0,
    )
    err_r = []
    for r in range(N_ROUNDS):
        train_step(model_r, opt_r, xy_col_r, xy_bc, u_bc, STEPS)
        new_pts = res_sampler.sample(model_r, xy_cand, device=device)
        xy_col_r = torch.cat([xy_col_r, new_pts], dim=0)
        err_r.append(l2_error(model_r, device))
        print(f"  round {r+1}: n_col={xy_col_r.shape[0]}  L2={err_r[-1]:.4e}")
    results["residual"] = err_r

    # =========================================================================
    # Strategy C: Combined residual + variance
    # =========================================================================
    print("\n[C] Combined (residual + variance) active sampling ...")
    xy_col_c = torch.rand(N_INIT, 2, device=device)
    model_c  = build_pinn(dropout_p=0.1).to(device)
    opt_c    = torch.optim.Adam(model_c.parameters(), lr=1e-3)

    var_sampler = VarianceActiveSampler(n_mc=30, n_add=N_ADD // 2)
    comb_sampler = CombinedActiveSampler(
        residual_sampler=ResidualActiveSampler(
            residual_fn=poisson_residual, n_add=N_ADD // 2
        ),
        variance_sampler=var_sampler,
    )
    err_c = []
    for r in range(N_ROUNDS):
        train_step(model_c, opt_c, xy_col_c, xy_bc, u_bc, STEPS)
        new_pts = comb_sampler.sample(model_c, xy_cand, device=device)
        xy_col_c = torch.cat([xy_col_c, new_pts], dim=0)
        err_c.append(l2_error(model_c, device))
        print(f"  round {r+1}: n_col={xy_col_c.shape[0]}  L2={err_c[-1]:.4e}")
    results["combined"] = err_c

    # =========================================================================
    # Visualisation
    # =========================================================================
    rounds = list(range(1, N_ROUNDS + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, err in results.items():
        axes[0].semilogy(rounds, err, marker="o", label=label)
    axes[0].set_xlabel("Active learning round")
    axes[0].set_ylabel("Relative L2 error")
    axes[0].set_title("Active learning convergence (2D Poisson)")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    # Visualise where residual sampler placed points in the last round
    xy_vis = xy_col_r.detach().cpu().numpy()
    axes[1].scatter(xy_vis[:N_INIT, 0], xy_vis[:N_INIT, 1],
                    s=3, c="blue", alpha=0.4, label="Initial uniform")
    axes[1].scatter(xy_vis[N_INIT:, 0], xy_vis[N_INIT:, 1],
                    s=3, c="red", alpha=0.6, label="Residual-added")
    axes[1].set_title("Collocation distribution (Residual strategy)")
    axes[1].legend(fontsize=8)
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("21_active_learning_result.png", dpi=120)
    print("Saved 21_active_learning_result.png")


if __name__ == "__main__":
    main()
