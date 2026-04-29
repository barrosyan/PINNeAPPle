"""12_meta_learning.py — Meta-learning for parametric PDEs.

Demonstrates:
- PDETaskSampler for a family of 1D Burgers equations (varying viscosity ν)
- ReptileTrainer with ReptileConfig
- Fast adaptation of meta-model to a new (unseen) ν value
- Comparison: meta-init vs random-init adaptation speed
"""

import copy
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_meta.task_sampler import PDETaskSampler
from pinneaple_meta.reptile import ReptileTrainer
from pinneaple_meta.config import ReptileConfig


# ---------------------------------------------------------------------------
# Task family: 1D Burgers  u_t + u u_x - ν u_xx = 0
# Steady state (for simplicity): u u_x - ν u_xx = 0 on [0,1]
# BC: u(0) = 1, u(1) = 0  → analytic: u ≈ (1 - x)  for small ν
# ---------------------------------------------------------------------------

def burgers_steady_physics_factory(params: dict):
    """Return a physics loss function for steady Burgers with given ν."""
    nu = float(params["nu"])

    def physics_fn(model: nn.Module, batch: dict) -> tuple[torch.Tensor, dict]:
        x_col = batch["x_col"]
        x_col = x_col.requires_grad_(True)
        out = model(x_col)
        if hasattr(out, "y"):
            out = out.y

        u_x = torch.autograd.grad(out.sum(), x_col, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x_col, create_graph=True)[0]

        # PDE residual
        residual = out * u_x - nu * u_xx
        loss_pde = residual.pow(2).mean()

        # BCs: u(0)=1, u(1)=0
        x_bc = torch.tensor([[0.0], [1.0]], dtype=x_col.dtype, device=x_col.device)
        u_bc = torch.tensor([[1.0], [0.0]], dtype=x_col.dtype, device=x_col.device)
        out_bc = model(x_bc)
        if hasattr(out_bc, "y"):
            out_bc = out_bc.y
        loss_bc = (out_bc - u_bc).pow(2).mean()

        total = loss_pde + 10.0 * loss_bc
        return total, {"pde": float(loss_pde.item()), "bc": float(loss_bc.item())}

    return physics_fn


def burgers_data_factory(params: dict) -> dict:
    """Provide collocation points for each task."""
    x_col = torch.linspace(0, 1, 200).unsqueeze(1)   # (200, 1)
    return {"x_col": x_col}


def make_model():
    return nn.Sequential(
        nn.Linear(1, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    )


def adapt(model: nn.Module, physics_fn, n_steps: int = 50, lr: float = 0.01) -> list:
    """Run n_steps gradient steps on a single task; return loss history."""
    m = copy.deepcopy(model)
    optimizer = torch.optim.SGD(m.parameters(), lr=lr)
    batch = {"x_col": torch.linspace(0, 1, 200).unsqueeze(1)}
    history = []
    for _ in range(n_steps):
        optimizer.zero_grad()
        total, _ = physics_fn(m, batch)
        total.backward()
        optimizer.step()
        history.append(float(total.item()))
    return history


def main():
    torch.manual_seed(0)
    device = "cpu"

    # --- PDETaskSampler ---------------------------------------------------
    sampler = PDETaskSampler(
        param_ranges={"nu": (0.005, 0.1)},
        physics_fn_factory=burgers_steady_physics_factory,
        data_factory=burgers_data_factory,
        input_dim=1,
        n_col=200,
    )

    # --- ReptileConfig ----------------------------------------------------
    config = ReptileConfig(
        n_inner_steps=10,
        inner_lr=0.01,
        outer_lr=0.1,
        n_tasks_per_batch=1,
        n_meta_epochs=200,          # small for demo
        device=device,
        epsilon=1.0,
        checkpoint_every=0,         # disable checkpointing for demo
    )

    # --- ReptileTrainer ---------------------------------------------------
    meta_model = make_model().to(device)
    trainer = ReptileTrainer(
        model=meta_model,
        config=config,
        task_sampler=sampler,
    )

    print(f"Meta-training Reptile for {config.n_meta_epochs} epochs...")
    trainer.train()
    print("Meta-training complete.")

    # --- Fast adaptation to a new ν value --------------------------------
    nu_test = 0.008   # unseen viscosity (low → steep gradient)
    print(f"\nFast adaptation to nu_test = {nu_test}")

    test_physics = burgers_steady_physics_factory({"nu": nu_test})

    # Meta-initialised model
    meta_hist = adapt(meta_model, test_physics, n_steps=100, lr=0.01)

    # Random-initialised model (baseline)
    rand_model = make_model().to(device)
    rand_hist  = adapt(rand_model, test_physics, n_steps=100, lr=0.01)

    # --- Prediction comparison --------------------------------------------
    x_vis = torch.linspace(0, 1, 100).unsqueeze(1).to(device)

    # Adapt both models for 100 steps
    meta_adapted = copy.deepcopy(meta_model)
    rand_adapted = copy.deepcopy(rand_model)
    for m, hist_fn in [(meta_adapted, meta_hist), (rand_adapted, rand_hist)]:
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        batch = {"x_col": torch.linspace(0, 1, 200).unsqueeze(1)}
        for _ in range(100):
            opt.zero_grad()
            loss, _ = test_physics(m, batch)
            loss.backward()
            opt.step()

    with torch.no_grad():
        u_meta = meta_adapted(x_vis).squeeze().numpy()
        u_rand = rand_adapted(x_vis).squeeze().numpy()

    # Approximate analytic: linear (valid for large ν limit)
    u_exact = 1.0 - x_vis.squeeze().numpy()

    # --- Plot -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].semilogy(meta_hist[:100], label="Meta-init (Reptile)")
    axes[0].semilogy(rand_hist[:100], label="Random-init")
    axes[0].set_xlabel("Adaptation step")
    axes[0].set_ylabel("Task loss")
    axes[0].set_title(f"Adaptation to nu={nu_test} (100 steps)")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(x_vis.numpy(), u_exact, "k--", label="Exact (lin. approx)")
    axes[1].plot(x_vis.numpy(), u_meta,  "b-",  label="Meta-adapted")
    axes[1].plot(x_vis.numpy(), u_rand,  "r-",  label="Random-adapted")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("u(x)")
    axes[1].set_title(f"Solution after 100 adaptation steps  (nu={nu_test})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("12_meta_learning_result.png", dpi=120)
    print("Saved 12_meta_learning_result.png")


if __name__ == "__main__":
    main()
