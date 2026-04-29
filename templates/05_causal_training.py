"""05_causal_training.py — Causal PINN for time-dependent PDEs.

Demonstrates:
- CausalPINNTrainer for temporal PDE training
- 1D heat equation:  u_t = α u_xx,  u(x,0) = sin(πx)
- Exact solution:    u(x,t) = sin(πx) exp(-α π² t)
- Causal weight visualization per time chunk
"""

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_train.causal import CausalPINNTrainer, CausalWeightScheduler


# ---------------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------------

ALPHA = 0.05   # thermal diffusivity
T_END = 1.0    # final time


def exact_solution(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.sin(math.pi * x) * np.exp(-ALPHA * math.pi**2 * t)


# ---------------------------------------------------------------------------
# PDE residual and IC functions
# ---------------------------------------------------------------------------

def heat_residual(model: nn.Module, xt: torch.Tensor) -> torch.Tensor:
    """Heat equation residual: u_t - α u_xx = 0."""
    xt = xt.requires_grad_(True)
    out = model(xt)
    if hasattr(out, "y"):
        out = out.y

    grad1 = torch.autograd.grad(out.sum(), xt, create_graph=True)[0]
    u_t = grad1[:, 1:2]                # dt
    u_x = grad1[:, 0:1]               # dx
    u_xx = torch.autograd.grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]
    return u_t - ALPHA * u_xx          # (N, 1)


def ic_loss_fn(model: nn.Module) -> torch.Tensor:
    """IC loss: u(x,0) = sin(πx)."""
    x = torch.linspace(0, 1, 256, device=next(model.parameters()).device)
    xt_ic = torch.stack([x, torch.zeros_like(x)], dim=1)
    out = model(xt_ic)
    if hasattr(out, "y"):
        out = out.y
    target = torch.sin(math.pi * x).unsqueeze(1)
    return (out - target).pow(2).mean()


# ---------------------------------------------------------------------------
# Simple MLP model
# ---------------------------------------------------------------------------

def make_model():
    return nn.Sequential(
        nn.Linear(2, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    )


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Model ------------------------------------------------------------
    model = make_model().to(device)

    # --- Causal trainer ---------------------------------------------------
    trainer = CausalPINNTrainer(
        model=model,
        epsilon=1.0,        # causal decay strength
        n_time_chunks=50,
        update_every=200,
    )

    print("Training with causal weighting...")
    history = trainer.train(
        pde_residual_fn=heat_residual,
        ic_loss_fn=ic_loss_fn,
        t_range=(0.0, T_END),
        n_epochs=8000,
        n_col=3000,
        ic_weight=10.0,
        lr=1e-3,
        print_every=1000,
    )

    causality = trainer.scheduler.causality_metric()
    print(f"\nCausality metric (std of log-weights): {causality:.4f}")
    print("Training complete.")

    # --- Evaluation -------------------------------------------------------
    n_x, n_t = 64, 64
    x_lin = np.linspace(0, 1, n_x)
    t_lin = np.linspace(0, T_END, n_t)
    xx, tt = np.meshgrid(x_lin, t_lin)
    xt_vis = torch.tensor(
        np.stack([xx.ravel(), tt.ravel()], axis=1), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        u_pred = model(xt_vis)
        if hasattr(u_pred, "y"):
            u_pred = u_pred.y
        u_pred = u_pred.cpu().numpy().reshape(n_t, n_x)

    u_exact = exact_solution(xx, tt)
    l2_err = (np.sqrt(((u_pred - u_exact) ** 2).mean())
              / np.sqrt((u_exact ** 2).mean() + 1e-14))
    print(f"Relative L2 error: {l2_err:.4e}")

    # --- Causal weight visualization -------------------------------------
    t_chunks = torch.linspace(0, T_END, trainer.scheduler.n_time_chunks + 1)
    t_mid = 0.5 * (t_chunks[:-1] + t_chunks[1:])
    chunk_w = trainer.scheduler._weights_from_cache(t_mid.unsqueeze(1)).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].contourf(xx, tt, u_pred, levels=30, cmap="hot")
    plt.colorbar(im0, ax=axes[0])
    axes[0].set_title("Predicted u(x,t)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")

    im1 = axes[1].contourf(xx, tt, u_exact, levels=30, cmap="hot")
    plt.colorbar(im1, ax=axes[1])
    axes[1].set_title("Exact u(x,t)")
    axes[1].set_xlabel("x")

    axes[2].bar(t_mid.numpy(), chunk_w, width=T_END / trainer.scheduler.n_time_chunks,
                color="steelblue", edgecolor="none")
    axes[2].set_xlabel("Time chunk midpoint")
    axes[2].set_ylabel("Causal weight")
    axes[2].set_title("Causal weights per time chunk")

    plt.tight_layout()
    plt.savefig("05_causal_training_result.png", dpi=120)
    print("Saved 05_causal_training_result.png")


if __name__ == "__main__":
    main()
