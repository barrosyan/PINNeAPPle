"""06_time_marching.py — Time-marching PINN for stiff temporal PDEs.

Demonstrates:
- TimeMarchingTrainer with sequential time windows
- 1D Allen-Cahn equation: u_t = 0.0001 u_xx + 5u(1-u²)
- Each window uses the previous window's solution as IC
"""

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_train.time_marching import TimeMarchingTrainer


# ---------------------------------------------------------------------------
# Problem: Allen-Cahn PDE
#   u_t - ε² u_xx - u(1 - u²) = 0
#   on x ∈ [−1, 1],  t ∈ [0, 1]
#   IC: u(x,0) = x² cos(πx)
#   BC: periodic (enforced via spatial coordinate embedding)
# ---------------------------------------------------------------------------

EPS2 = 0.0001   # diffusivity squared


def allen_cahn_residual(model: nn.Module, xt: torch.Tensor) -> torch.Tensor:
    """Allen-Cahn residual: u_t - ε²u_xx - u(1-u²) = 0."""
    xt = xt.requires_grad_(True)
    out = model(xt)
    if hasattr(out, "y"):
        out = out.y

    grad1 = torch.autograd.grad(out.sum(), xt, create_graph=True)[0]
    u_t = grad1[:, 1:2]
    u_x = grad1[:, 0:1]
    u_xx = torch.autograd.grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]
    return u_t - EPS2 * u_xx - out * (1.0 - out ** 2)


def ic_fn(xt: torch.Tensor) -> torch.Tensor:
    """Initial condition: u(x,0) = x² cos(πx)."""
    x = xt[:, 0:1]
    return x ** 2 * torch.cos(math.pi * x)


def make_model():
    return nn.Sequential(
        nn.Linear(2, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 1),
    )


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Spatial domain: x ∈ [−1, 1]
    x_domain = torch.linspace(-1, 1, 256, device=device).unsqueeze(1)

    # --- TimeMarchingTrainer setup ----------------------------------------
    trainer = TimeMarchingTrainer(
        model_factory=make_model,
        t_start=0.0,
        t_end=1.0,
        n_windows=5,
        epochs_per_window=2000,
        n_col=2000,
        ic_weight=10.0,
    )

    print(f"Marching through {trainer.n_windows} windows...")
    models = trainer.march(
        pde_residual_fn=allen_cahn_residual,
        ic_fn=ic_fn,
        x_domain=x_domain,
        device=device,
    )
    print(f"Trained {len(models)} window models.")

    # --- Stitched prediction ----------------------------------------------
    n_x, n_t = 128, 100
    x_vis = np.linspace(-1, 1, n_x)
    t_vis = np.linspace(0, 1, n_t)
    xx, tt = np.meshgrid(x_vis, t_vis)

    u_full = np.zeros_like(xx)
    window_edges = trainer.window_edges

    for w_idx, (t_lo, t_hi) in enumerate(window_edges):
        # Find time indices belonging to this window
        t_mask = (t_vis >= t_lo) & (t_vis <= t_hi)
        t_sub = t_vis[t_mask]
        if len(t_sub) == 0:
            continue
        xxw, ttw = np.meshgrid(x_vis, t_sub)
        xt_in = torch.tensor(
            np.stack([xxw.ravel(), ttw.ravel()], axis=1),
            dtype=torch.float32, device=device,
        )
        with torch.no_grad():
            u_out = models[w_idx](xt_in)
            if hasattr(u_out, "y"):
                u_out = u_out.y
            u_out = u_out.cpu().numpy().reshape(len(t_sub), n_x)
        u_full[t_mask, :] = u_out

    # --- Visualization ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im = axes[0].contourf(xx, tt, u_full, levels=50, cmap="RdBu")
    plt.colorbar(im, ax=axes[0])
    axes[0].set_title("Allen-Cahn u(x,t) — Time Marching")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")

    # Plot snapshots at a few times
    snap_times = [0.0, 0.2, 0.5, 0.8, 1.0]
    for t_snap in snap_times:
        t_idx = np.argmin(np.abs(t_vis - t_snap))
        axes[1].plot(x_vis, u_full[t_idx, :], label=f"t={t_snap:.1f}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("u")
    axes[1].set_title("Allen-Cahn snapshots")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("06_time_marching_result.png", dpi=120)
    print("Saved 06_time_marching_result.png")


if __name__ == "__main__":
    main()
