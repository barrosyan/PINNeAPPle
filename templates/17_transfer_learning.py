"""17_transfer_learning.py — Transfer learning for parametric PDEs.

Demonstrates:
- FineTuner: fine-tune a pre-trained PINN on a new physical regime
- ParametricFreezer: freeze backbone layers, train only the head
- LayerWiseLRScheduler: different learning rates per layer group
- Comparison of fine-tuned vs. from-scratch training speed
"""

import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_transfer.finetuner import FineTuner, FineTunerConfig
from pinneaple_transfer.freezer import ParametricFreezer
from pinneaple_transfer.lr_scheduler import LayerWiseLRScheduler


# ---------------------------------------------------------------------------
# Task: 1D heat equation  u_t = α u_xx  on [0,1] × [0,T]
# Steady-state surrogate: α_source = 0.1 → α_target = 0.5
# Exact steady state: u(x) = sin(πx) (same BCs, different transient speed)
# We train a PINN on α_source and fine-tune to α_target.
# ---------------------------------------------------------------------------

import math

ALPHA_SOURCE = 0.1
ALPHA_TARGET = 0.5


def heat_residual(model: nn.Module, xy: torch.Tensor, alpha: float) -> torch.Tensor:
    """Steady-state heat:  α u_xx = 0  →  residual = u_xx."""
    xy.requires_grad_(True)
    u = model(xy)
    if hasattr(u, "y"):
        u = u.y
    u_x = torch.autograd.grad(u.sum(), xy, create_graph=True)[0][:, 0:1]
    u_xx = torch.autograd.grad(u_x.sum(), xy, create_graph=True)[0][:, 0:1]
    return alpha * u_xx      # should be 0 at steady state


def bc_loss(model: nn.Module, device) -> torch.Tensor:
    x_bc = torch.tensor([[0.0], [1.0]], device=device)
    u_bc = torch.zeros(2, 1, device=device)
    out = model(x_bc)
    if hasattr(out, "y"):
        out = out.y
    return (out - u_bc).pow(2).mean()


def build_pinn() -> nn.Module:
    return nn.Sequential(
        nn.Linear(1, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    )


def train_from_scratch(model: nn.Module, alpha: float, device,
                       n_epochs: int = 3000) -> list[float]:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    x_col = torch.linspace(0, 1, 300, device=device).unsqueeze(1)
    for _ in range(n_epochs):
        opt.zero_grad()
        res = heat_residual(model, x_col.clone(), alpha)
        loss = res.pow(2).mean() + 10 * bc_loss(model, device)
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
    return history


def main():
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Step 1: pre-train on source domain α=0.1 ----------------------------
    print(f"Pre-training on α_source={ALPHA_SOURCE} ...")
    source_model = build_pinn().to(device)
    _ = train_from_scratch(source_model, ALPHA_SOURCE, device, n_epochs=4000)
    print("Pre-training complete.")

    # --- Step 2: Baseline — train from scratch on α_target -------------------
    print(f"\nTraining from scratch on α_target={ALPHA_TARGET} ...")
    scratch_model = build_pinn().to(device)
    scratch_hist = train_from_scratch(scratch_model, ALPHA_TARGET, device, n_epochs=2000)

    # --- Step 3: ParametricFreezer — freeze first 2 layers -------------------
    frozen_model = copy.deepcopy(source_model)
    freezer = ParametricFreezer(model=frozen_model)
    freezer.freeze_layers(layer_indices=[0, 1])      # freeze layers 0,1 (input + 1st hidden)
    n_frozen = freezer.n_frozen_params()
    print(f"\nFreezer: {n_frozen} parameters frozen.")

    # --- Step 4: FineTuner with LayerWiseLR ----------------------------------
    ft_config = FineTunerConfig(
        n_epochs=2000,
        base_lr=5e-4,
        weight_decay=1e-5,
        patience=200,
        device=str(device),
    )

    # Build LayerWiseLRScheduler: lower lr for frozen-adjacent layers
    layer_lrs = {
        "0": 1e-5,   # layer 0 (nearly frozen)
        "2": 1e-4,   # layer 2
        "4": 5e-4,   # layer 4 (output region)
        "6": 5e-4,   # output layer
    }
    lw_scheduler = LayerWiseLRScheduler(model=frozen_model, layer_lrs=layer_lrs)

    x_col = torch.linspace(0, 1, 300, device=device).unsqueeze(1)

    def ft_loss_fn(model, epoch):
        res = heat_residual(model, x_col.clone(), ALPHA_TARGET)
        return res.pow(2).mean() + 10 * bc_loss(model, device)

    finetuner = FineTuner(
        model=frozen_model,
        config=ft_config,
        loss_fn=ft_loss_fn,
        lr_scheduler=lw_scheduler,
    )
    print(f"Fine-tuning to α_target={ALPHA_TARGET} ...")
    ft_hist = finetuner.train()
    print("Fine-tuning complete.")

    # --- Evaluation -----------------------------------------------------------
    x_vis = torch.linspace(0, 1, 200, device=device).unsqueeze(1)
    u_exact = torch.sin(math.pi * x_vis).cpu().numpy().ravel()

    with torch.no_grad():
        u_ft    = frozen_model(x_vis).cpu().numpy().ravel()
        u_scr   = scratch_model(x_vis).cpu().numpy().ravel()

    x_plot = x_vis.cpu().numpy().ravel()

    # --- Plot ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].semilogy(scratch_hist, label=f"From scratch (α={ALPHA_TARGET})", alpha=0.8)
    ft_loss_series = ft_hist.get("total", ft_hist.get("loss", []))
    axes[0].semilogy(ft_loss_series, label="Fine-tuned (frozen backbone)", alpha=0.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Transfer vs. from-scratch convergence")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(x_plot, u_exact, "k--", label="Exact")
    axes[1].plot(x_plot, u_ft,    "b-",  label="Fine-tuned")
    axes[1].plot(x_plot, u_scr,   "r-",  label="From scratch")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("u(x)")
    axes[1].set_title(f"Solution at α_target={ALPHA_TARGET}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("17_transfer_learning_result.png", dpi=120)
    print("Saved 17_transfer_learning_result.png")


if __name__ == "__main__":
    main()
