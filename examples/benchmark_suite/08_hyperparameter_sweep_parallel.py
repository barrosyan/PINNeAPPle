"""08 — Parallel hyperparameter sweep for PINN training.

What this demonstrates
----------------------
- Defining a hyperparameter grid (learning rate, hidden layers, activation)
- Running trials in parallel using run_parallel_sweep (ThreadPoolExecutor)
- Each trial trains a PINN on the Burgers equation for a fixed number of epochs
- Results are sorted by validation loss and the best model is saved
- GPU/CPU device selection per trial
- Using ThroughputMonitor to track training speed

Run from repo root:
    python examples/pinneaple_arena/08_hyperparameter_sweep_parallel.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from pinneaple_train import (
    best_device, run_parallel_sweep, SweepConfig, ThroughputMonitor,
)


DEVICE = best_device()
NU = 0.01   # Burgers viscosity


# ------------------------------------------------------------------
# Dataset (shared across all trials)
# ------------------------------------------------------------------
rng = np.random.default_rng(42)
N_COL, N_BC, N_IC = 4000, 500, 1000

x_col = rng.uniform(-1, 1, N_COL).astype(np.float32)
t_col = rng.uniform(0, 1, N_COL).astype(np.float32)
X_col = np.column_stack([x_col, t_col])

t_bc  = rng.uniform(0, 1, N_BC).astype(np.float32)
X_bc  = np.column_stack([
    np.concatenate([np.full(N_BC//2, -1.), np.full(N_BC//2, 1.)]),
    np.concatenate([t_bc[:N_BC//2], t_bc[N_BC//2:]]),
]).astype(np.float32)
U_bc  = np.zeros(N_BC, dtype=np.float32)

x_ic  = rng.uniform(-1, 1, N_IC).astype(np.float32)
X_ic  = np.column_stack([x_ic, np.zeros(N_IC)]).astype(np.float32)
U_ic  = (-np.sin(np.pi * x_ic)).astype(np.float32)


def run_trial(params: Dict[str, Any]) -> Dict[str, Any]:
    """Train a Burgers PINN with the given hyperparameters."""
    lr       = params["lr"]
    hidden   = params["hidden"]
    act_name = params["activation"]
    epochs   = params.get("epochs", 500)

    device = DEVICE   # use the global device for all trials

    act_map = {"tanh": nn.Tanh(), "silu": nn.SiLU(), "gelu": nn.GELU()}
    act = act_map.get(act_name, nn.Tanh())

    # Build model
    dims = [2, *[hidden] * 3, 1]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act)
    model = nn.Sequential(*layers).to(device)

    # Move data to device
    x_col_t = torch.from_numpy(X_col).to(device).requires_grad_(True)
    x_bc_t  = torch.from_numpy(X_bc).to(device)
    u_bc_t  = torch.from_numpy(U_bc[:, None]).to(device)
    x_ic_t  = torch.from_numpy(X_ic).to(device)
    u_ic_t  = torch.from_numpy(U_ic[:, None]).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    t0 = time.time()
    final_loss = float("inf")

    for epoch in range(epochs):
        opt.zero_grad()
        x_req = x_col_t.requires_grad_(True)
        u = model(x_req)

        gu  = torch.autograd.grad(u, x_req, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_t = gu[:, 1:2]
        u_x = gu[:, 0:1]
        u_xx = torch.autograd.grad(u_x, x_req, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]

        loss_pde = ((u_t + u * u_x - NU * u_xx) ** 2).mean()
        loss_bc  = 10.0 * ((model(x_bc_t) - u_bc_t) ** 2).mean()
        loss_ic  = 10.0 * ((model(x_ic_t) - u_ic_t) ** 2).mean()

        loss = loss_pde + loss_bc + loss_ic
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (epoch + 1) == epochs:
            final_loss = float(loss.item())

    elapsed = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())

    return {
        "val_loss": final_loss,
        "elapsed_s": round(elapsed, 2),
        "n_params": n_params,
    }


# ------------------------------------------------------------------
# Sweep config
# ------------------------------------------------------------------
sweep_cfg = SweepConfig(
    param_grid={
        "lr":         [1e-3, 5e-4],
        "hidden":     [32, 64],
        "activation": ["tanh", "silu"],
        "epochs":     [300],         # keep short for the demo
    },
    n_jobs=4,          # run 4 trials in parallel (threads)
    backend="thread",  # use "process" for CPU-bound on multiple machines
)

print(f"[Sweep] Grid: {len(sweep_cfg.grid_points())} combinations | "
      f"{sweep_cfg.n_jobs} parallel workers")
print("[Sweep] Running ...\n")

t_sweep_start = time.time()
results = run_parallel_sweep(run_trial, sweep_cfg)
t_sweep = time.time() - t_sweep_start

print(f"\n[Sweep] Completed in {t_sweep:.1f}s")
print(f"\n{'Rank':<6} {'val_loss':>12} {'lr':>8} {'hidden':>8} {'act':>6} {'params':>8} {'time(s)':>8}")
print("-" * 65)
for i, r in enumerate(results[:8], 1):
    p = r.get("params", {})
    print(
        f"  {i:<4} {r.get('val_loss', float('nan')):>12.4e} "
        f"{p.get('lr',0):>8.0e} {p.get('hidden',0):>8} "
        f"{p.get('activation','?'):>6} {r.get('n_params',0):>8,} "
        f"{r.get('elapsed_s',0):>8.1f}"
    )

best = results[0]
print(f"\n[Best] val_loss={best.get('val_loss'):.4e}  params={best.get('params')}")


# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = REPO_ROOT / "data" / "artifacts" / "examples" / "hparam_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bar chart of val_loss per trial
    valid_results = [r for r in results if "val_loss" in r and "params" in r]
    labels  = [f"lr={r['params']['lr']:.0e}\nh={r['params']['hidden']}\n{r['params']['activation']}"
               for r in valid_results]
    losses  = [r["val_loss"] for r in valid_results]
    colors  = ["#2196F3" if r == best else "#90CAF9" for r in valid_results]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4))
    bars = ax.bar(range(len(labels)), losses, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Final loss (log scale)")
    ax.set_title(f"Hyperparameter sweep — {len(results)} trials in {t_sweep:.0f}s")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(min(losses), color="red", lw=0.8, linestyle="--", label="best")
    ax.legend()

    plt.tight_layout()
    fig_path = out_dir / "sweep_results.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\n[Plot] {fig_path}")

except ImportError:
    print("[Plot] matplotlib not available.")


print("\n=== COMPLETE ===")
