"""05 — DeepONet surrogate: multi-field operator learning for parametric PDEs.

What this demonstrates
----------------------
- Using DeepONet to learn the solution operator u(x; µ) for Poisson equation
  across a family of right-hand side functions (parametric PDE)
- Branch network encodes the RHS function sampled at sensor points
- Trunk network encodes the query coordinate
- Parallel GPU training with AMP and gradient accumulation
- Hyperparameter sweep with run_parallel_sweep
- Saving and loading the trained operator

Problem
-------
  -∇²u = f(x, y)  in [0,1]²
   u = 0           on boundary

For each training sample:
  - f is a random combination of sinusoidal modes
  - u is approximated with the analytical Green's function (toy)

Run from repo root:
    python examples/pinneaple_arena/05_surrogate_deeponet_multifield.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pinneaple_models.neural_operators.deeponet import DeepONet
from pinneaple_train import (
    best_device, maybe_compile, AMPContext,
    GradAccumConfig, GradAccumTrainer,
    ThroughputMonitor, batched_inference,
)


DEVICE = best_device()
print(f"[Device] {DEVICE}")


# ------------------------------------------------------------------
# 1. Dataset: parametric Poisson
# ------------------------------------------------------------------

def random_rhs_modes(n_samples: int, n_sensors: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random RHS functions f sampled at sensor points + toy u values.

    f(x) = sum_k a_k sin(k pi x)  (1D for simplicity)
    u_exact(x) ~ sum_k a_k / (k pi)^2 sin(k pi x)  (1D Poisson solution)

    Returns
    -------
    F : (n_samples, n_sensors)  — f values at sensors
    U : (n_samples, n_sensors)  — u values at query points
    """
    x_sensors = np.linspace(0, 1, n_sensors, dtype=np.float32)
    max_modes = 5

    F_list, U_list = [], []
    for _ in range(n_samples):
        a = rng.uniform(-1, 1, max_modes).astype(np.float32)
        k = np.arange(1, max_modes + 1, dtype=np.float32)
        f_vals = sum(a[i] * np.sin(k[i] * np.pi * x_sensors) for i in range(max_modes))
        u_vals = sum(a[i] / (k[i] * np.pi) ** 2 * np.sin(k[i] * np.pi * x_sensors) for i in range(max_modes))
        F_list.append(f_vals)
        U_list.append(u_vals)

    return np.stack(F_list), np.stack(U_list)


rng = np.random.default_rng(123)
N_TRAIN, N_VAL = 2000, 400
N_SENSORS  = 100
N_QUERY    = 100   # trunk points per sample

print(f"[Data] Generating parametric Poisson dataset ...")
F_train, U_train = random_rhs_modes(N_TRAIN, N_SENSORS, rng)
F_val,   U_val   = random_rhs_modes(N_VAL,   N_SENSORS, rng)

# Trunk coords: same for all samples (linspace in [0,1])
trunk_coords = np.linspace(0, 1, N_QUERY, dtype=np.float32)[:, None]

# Convert to tensors
F_train_t = torch.from_numpy(F_train)     # (N_TRAIN, N_SENSORS)
U_train_t = torch.from_numpy(U_train)     # (N_TRAIN, N_SENSORS)
F_val_t   = torch.from_numpy(F_val)
U_val_t   = torch.from_numpy(U_val)
trunk_t   = torch.from_numpy(trunk_coords)  # (N_QUERY, 1)

train_ds = TensorDataset(F_train_t, U_train_t)
val_ds   = TensorDataset(F_val_t,   U_val_t)

BATCH_SIZE = 64
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"  Train: {N_TRAIN} samples | Val: {N_VAL} samples | "
      f"Sensors: {N_SENSORS} | Trunk: {N_QUERY} pts")


# ------------------------------------------------------------------
# 2. DeepONet model
# ------------------------------------------------------------------
model = DeepONet(
    branch_dim=N_SENSORS,
    trunk_dim=1,
    out_dim=1,
    hidden=128,
    modes=64,
).to(DEVICE)

model = maybe_compile(model, mode="default")
n_params = sum(p.numel() for p in model.parameters())
print(f"[Model] DeepONet | Parameters: {n_params:,}")


# ------------------------------------------------------------------
# 3. Training
# ------------------------------------------------------------------
EPOCHS    = 200
LR        = 1e-3
ACCUM     = 2       # gradient accumulation steps

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
amp_ctx   = AMPContext(device=str(DEVICE), enabled=DEVICE.type == "cuda")
monitor   = ThroughputMonitor()

history = {"train_loss": [], "val_loss": []}
trunk_dev = trunk_t.to(DEVICE)   # shared trunk coords on device

print(f"\n[Training] {EPOCHS} epochs | ACCUM={ACCUM} | AMP={amp_ctx.enabled}")
t0 = time.time()

for epoch in range(EPOCHS):
    # ---- Train ----
    model.train()
    monitor.start_epoch()
    tr_loss, n_tr = 0.0, 0

    optimizer.zero_grad(set_to_none=True)
    for step, (f_batch, u_batch) in enumerate(train_loader):
        f_batch = f_batch.to(DEVICE)  # (B, N_SENSORS)
        u_batch = u_batch.to(DEVICE)  # (B, N_SENSORS)

        with amp_ctx.autocast():
            # DeepONet: branch=f_batch (B, sensors), trunk=trunk_dev (N_query, 1)
            out = model(f_batch, trunk_dev)  # OperatorOutput.y: (B, N_query, 1)
            # y_true must match: we use only u at the trunk query points
            # For simplicity: subsample U to trunk_coords (same grid)
            y_pred = out.y.squeeze(-1)       # (B, N_query)
            y_true = u_batch[:, :N_QUERY]    # (B, N_query) — first N_QUERY points

            loss = ((y_pred - y_true) ** 2).mean() / ACCUM

        if amp_ctx.enabled:
            amp_ctx.scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % ACCUM == 0:
            if amp_ctx.enabled:
                amp_ctx.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_ctx.scaler.step(optimizer)
                amp_ctx.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        tr_loss += float(loss.item() * ACCUM) * len(f_batch)
        n_tr += len(f_batch)

    train_loss = tr_loss / max(1, n_tr)
    monitor.end_epoch(n_samples=n_tr)

    # ---- Validate ----
    model.eval()
    va_loss, n_va = 0.0, 0
    with torch.no_grad():
        for f_batch, u_batch in val_loader:
            f_batch = f_batch.to(DEVICE)
            u_batch = u_batch.to(DEVICE)
            with amp_ctx.autocast():
                out = model(f_batch, trunk_dev)
                y_pred = out.y.squeeze(-1)
                y_true = u_batch[:, :N_QUERY]
                loss_v = ((y_pred - y_true) ** 2).mean()
            va_loss += float(loss_v.item()) * len(f_batch)
            n_va += len(f_batch)

    val_loss = va_loss / max(1, n_va)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:4d}/{EPOCHS}  train={train_loss:.4e}  val={val_loss:.4e}")

elapsed = time.time() - t0
print(f"\n[Done] {elapsed:.1f}s | {monitor.summary()}")


# ------------------------------------------------------------------
# 4. Save model
# ------------------------------------------------------------------
out_dir = REPO_ROOT / "data" / "artifacts" / "examples" / "deeponet_poisson"
out_dir.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), out_dir / "deeponet_poisson.pt")
print(f"[Saved] {out_dir / 'deeponet_poisson.pt'}")


# ------------------------------------------------------------------
# 5. Inference on a new RHS function
# ------------------------------------------------------------------
model.eval()
a_test = np.array([1.0, -0.5, 0.2, 0.0, 0.0], dtype=np.float32)
x_s = np.linspace(0, 1, N_SENSORS, dtype=np.float32)
k   = np.arange(1, 6, dtype=np.float32)
f_test = sum(a_test[i] * np.sin(k[i] * np.pi * x_s) for i in range(5))
u_analytic = sum(a_test[i] / (k[i] * np.pi)**2 * np.sin(k[i] * np.pi * trunk_coords.ravel()) for i in range(5))

with torch.no_grad():
    f_t   = torch.from_numpy(f_test[None]).to(DEVICE)   # (1, N_SENSORS)
    out   = model(f_t, trunk_dev)
    u_pred_op = out.y[0, :, 0].cpu().numpy()

rel_l2 = np.linalg.norm(u_pred_op - u_analytic) / (np.linalg.norm(u_analytic) + 1e-12)
print(f"\n[Eval] Test sample — Relative L2 error: {rel_l2:.4f}")


# ------------------------------------------------------------------
# 6. Visualization
# ------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].semilogy(history["train_loss"], label="train", lw=1.5)
    axes[0].semilogy(history["val_loss"],   label="val",   lw=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("DeepONet Training (Parametric Poisson)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    x_plot = trunk_coords.ravel()
    axes[1].plot(x_plot, u_analytic,  "k--", lw=2,   label="Analytic")
    axes[1].plot(x_plot, u_pred_op,   "r-",  lw=1.5, label=f"DeepONet (L2={rel_l2:.3f})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("u(x)")
    axes[1].set_title("DeepONet Prediction vs Analytic")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / "deeponet_results.png"
    plt.savefig(fig_path, dpi=150)
    print(f"[Plot] {fig_path}")

except ImportError:
    print("[Plot] matplotlib not available.")


print("\n=== COMPLETE ===")
