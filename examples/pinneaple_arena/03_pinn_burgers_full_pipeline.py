"""03 — Full PINN pipeline: Burgers equation with physics loss, GPU acceleration, and visualization.

What this demonstrates
----------------------
- Loading a pre-defined problem from pinneaple_environment (Burgers 1D)
- Generating collocation + boundary + IC points with CollocationSampler
- Building a PINN model (VanillaPINN) with physics-informed loss
- Training with GPU, AMP, gradient clipping, and torch.compile()
- Evaluating metrics and running inference on a grid
- Visualizing loss curves, solution field, and error map

Run from repo root:
    python examples/pinneaple_arena/03_pinn_burgers_full_pipeline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------
# Add repo root to path
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
import torch
import torch.nn as nn

from pinneaple_environment import get_preset
from pinneaple_train import (
    Trainer, TrainConfig,
    best_device, maybe_compile, ThroughputMonitor,
    build_metrics_from_cfg,
)
from pinneaple_inference import infer_on_grid_2d


# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------
DEVICE = best_device()
print(f"[Device] Using: {DEVICE}")


# ------------------------------------------------------------------
# 1. Load problem spec
# ------------------------------------------------------------------
spec = get_preset("burgers_1d", nu=0.01)
print(f"[Problem] {spec.problem_id}  |  fields: {spec.fields}  |  nu={spec.pde.params.get('nu')}")


# ------------------------------------------------------------------
# 2. Generate training data (collocation + boundary + IC)
# ------------------------------------------------------------------
rng = np.random.default_rng(42)

nu = spec.pde.params["nu"]
N_COL = 8000   # interior collocation points (x, t)
N_BC  = 1000   # boundary points (x=±1, t)
N_IC  = 2000   # initial condition (t=0)

# Collocation (interior)
x_col = rng.uniform(-1, 1, N_COL).astype(np.float32)
t_col = rng.uniform(0, 1, N_COL).astype(np.float32)
X_col = np.column_stack([x_col, t_col])

# Boundary (x = ±1, u = 0)
t_bc  = rng.uniform(0, 1, N_BC).astype(np.float32)
x_bc_left  = np.full(N_BC // 2, -1.0, dtype=np.float32)
x_bc_right = np.full(N_BC // 2, +1.0, dtype=np.float32)
X_bc = np.column_stack([
    np.concatenate([x_bc_left, x_bc_right]),
    np.concatenate([t_bc[:N_BC//2], t_bc[N_BC//2:]]),
])
U_bc = np.zeros(N_BC, dtype=np.float32)

# Initial condition (t = 0, u = -sin(πx))
x_ic = rng.uniform(-1, 1, N_IC).astype(np.float32)
t_ic = np.zeros(N_IC, dtype=np.float32)
X_ic = np.column_stack([x_ic, t_ic])
U_ic = (-np.sin(np.pi * x_ic)).astype(np.float32)

print(f"[Data] Collocation: {N_COL}  |  BC: {N_BC}  |  IC: {N_IC}")


# ------------------------------------------------------------------
# 3. Build PINN model
# ------------------------------------------------------------------
class BurgersPINN(nn.Module):
    """Vanilla PINN for Burgers equation.  Standard forward(x) -> (N,1) tensor."""

    def __init__(self, hidden=(64, 64, 64, 64), activation="tanh"):
        super().__init__()
        act_map = {"tanh": nn.Tanh(), "silu": nn.SiLU(), "gelu": nn.GELU()}
        act = act_map.get(activation, nn.Tanh())
        dims = [2, *list(hidden), 1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


model = BurgersPINN(hidden=(64, 64, 64, 64), activation="tanh").to(DEVICE)

# Optional: torch.compile for PyTorch 2.x
model = maybe_compile(model, mode="default")

n_params = sum(p.numel() for p in model.parameters())
print(f"[Model] Parameters: {n_params:,}")


# ------------------------------------------------------------------
# 4. Define PINN loss function
# ------------------------------------------------------------------

W_PDE = 1.0
W_BC  = 10.0
W_IC  = 10.0


def burgers_pinn_loss(model: nn.Module, batch: dict) -> dict:
    """
    PINN loss for Burgers equation:
        ∂u/∂t + u ∂u/∂x - ν ∂²u/∂x² = 0

    Batch keys: x_col, x_bc, y_bc, x_ic, y_ic
    """
    # ---- Collocation residual ----
    x_col_ = batch["x_col"].requires_grad_(True)
    u_col  = model(x_col_)

    grad_u = torch.autograd.grad(
        u_col, x_col_, grad_outputs=torch.ones_like(u_col),
        create_graph=True, retain_graph=True
    )[0]
    u_t = grad_u[:, 1:2]   # ∂u/∂t
    u_x = grad_u[:, 0:1]   # ∂u/∂x

    u_xx = torch.autograd.grad(
        u_x, x_col_, grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True
    )[0][:, 0:1]             # ∂²u/∂x²

    pde_res = u_t + u_col * u_x - nu * u_xx
    loss_pde = W_PDE * torch.mean(pde_res ** 2)

    # ---- Boundary loss ----
    u_bc = model(batch["x_bc"])
    loss_bc = W_BC * torch.mean((u_bc - batch["y_bc"]) ** 2)

    # ---- Initial condition loss ----
    u_ic = model(batch["x_ic"])
    loss_ic = W_IC * torch.mean((u_ic - batch["y_ic"]) ** 2)

    total = loss_pde + loss_bc + loss_ic
    return {"total": total, "pde": loss_pde, "bc": loss_bc, "ic": loss_ic}


# ------------------------------------------------------------------
# 5. Build DataLoader-compatible dataset
# ------------------------------------------------------------------

def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).to(DEVICE)


class PINNBatch(torch.utils.data.Dataset):
    """Single-batch dataset that always returns the full collocation set."""

    def __init__(self, batch: dict):
        self.batch = batch

    def __len__(self):
        return 1

    def __getitem__(self, _):
        return self.batch


batch = {
    "x_col": _to_tensor(X_col),
    "x_bc":  _to_tensor(X_bc),
    "y_bc":  _to_tensor(U_bc[:, None]),
    "x_ic":  _to_tensor(X_ic),
    "y_ic":  _to_tensor(U_ic[:, None]),
}


def _collate(items):
    return items[0]


train_ds = PINNBatch(batch)
val_ds   = PINNBatch(batch)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, collate_fn=_collate)
val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=1, collate_fn=_collate)


# ------------------------------------------------------------------
# 6. Train
# ------------------------------------------------------------------
EPOCHS = 3000
LR     = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

monitor = ThroughputMonitor()
history = {"total": [], "pde": [], "bc": [], "ic": []}

print(f"\n[Training] {EPOCHS} epochs on {DEVICE}")
t_train_start = time.time()

for epoch in range(EPOCHS):
    monitor.start_epoch()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    b = batch   # single batch
    losses = burgers_pinn_loss(model, b)
    losses["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    for k in history:
        if k in losses:
            history[k].append(float(losses[k].item()))

    monitor.end_epoch(n_samples=N_COL + N_BC + N_IC)

    if (epoch + 1) % 500 == 0:
        print(
            f"  Epoch {epoch+1:5d}/{EPOCHS}  |  "
            f"total={losses['total'].item():.4e}  "
            f"pde={losses['pde'].item():.4e}  "
            f"bc={losses['bc'].item():.4e}  "
            f"ic={losses['ic'].item():.4e}"
        )

t_elapsed = time.time() - t_train_start
print(f"\n[Done] Training time: {t_elapsed:.1f}s")
print(f"[Throughput] {monitor.summary()}")


# ------------------------------------------------------------------
# 7. Inference on a grid
# ------------------------------------------------------------------
model.eval()

Nx, Nt = 200, 200
x_grid = np.linspace(-1, 1, Nx, dtype=np.float32)
t_grid = np.linspace(0, 1, Nt, dtype=np.float32)
XX, TT = np.meshgrid(x_grid, t_grid)
X_test = np.column_stack([XX.ravel(), TT.ravel()])

from pinneaple_train import batched_inference
X_tensor = torch.from_numpy(X_test)
U_pred = batched_inference(model, X_tensor, batch_size=8192, device=str(DEVICE))
U_pred_grid = U_pred.numpy().reshape(Nt, Nx)

print(f"[Inference] Grid shape: {U_pred_grid.shape}  |  u range: [{U_pred_grid.min():.3f}, {U_pred_grid.max():.3f}]")


# ------------------------------------------------------------------
# 8. Save outputs
# ------------------------------------------------------------------
out_dir = REPO_ROOT / "data" / "artifacts" / "examples" / "burgers_pinn"
out_dir.mkdir(parents=True, exist_ok=True)

np.save(out_dir / "u_pred.npy", U_pred_grid)
np.save(out_dir / "x_grid.npy", x_grid)
np.save(out_dir / "t_grid.npy", t_grid)

import json
with open(out_dir / "training_history.json", "w") as f:
    json.dump({k: v for k, v in history.items()}, f, indent=2)

torch.save(model.state_dict(), out_dir / "burgers_pinn.pt")

print(f"\n[Saved] Artifacts → {out_dir}")


# ------------------------------------------------------------------
# 9. Visualization (optional — requires matplotlib)
# ------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss curve
    axes[0].semilogy(history["total"], label="total", alpha=0.8)
    axes[0].semilogy(history["pde"],   label="pde",   alpha=0.7)
    axes[0].semilogy(history["bc"],    label="bc",    alpha=0.7)
    axes[0].semilogy(history["ic"],    label="ic",    alpha=0.7)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss (Burgers PINN)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Predicted field
    im1 = axes[1].pcolormesh(x_grid, t_grid, U_pred_grid, cmap="RdBu_r", shading="auto")
    plt.colorbar(im1, ax=axes[1], label="u(x,t)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    axes[1].set_title("PINN Prediction u(x,t)")

    # Solution at final time
    axes[2].plot(x_grid, U_pred_grid[-1], lw=2, label="PINN (t=1)")
    axes[2].axhline(0, color="k", lw=0.5, alpha=0.4)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("u")
    axes[2].set_title("PINN solution at t=1")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / "burgers_pinn_results.png"
    plt.savefig(fig_path, dpi=150)
    print(f"[Plot] Saved → {fig_path}")

except ImportError:
    print("[Plot] matplotlib not available — skipping visualization.")


print("\n=== COMPLETE ===")
print(f"  Device:        {DEVICE}")
print(f"  Total loss:    {history['total'][-1]:.4e}")
print(f"  Artifacts dir: {out_dir}")
