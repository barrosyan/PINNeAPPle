"""pinneaple_train example 03: Physics-aware training (PINN-style).

We learn y(x) ~ sin(x) on [-pi, pi] while enforcing the ODE:

    y''(x) + y(x) = 0

This demonstrates:
- CombinedLoss(supervised + physics)
- PhysicsLossHook with autograd residuals
- Training logs + "best" checkpoint

Run
---
python examples/pinneaple_train/03_physics_aware_pinn_ode.py
"""

from __future__ import annotations

import math
import os
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pinneaple_train import (
    Trainer,
    TrainConfig,
    CombinedLoss,
    SupervisedLoss,
    PhysicsLossHook,
    default_metrics,
)


# -----------------------------
# 1) Dataset: per item we provide one supervised point + one collocation point
# -----------------------------
class ODESampleDataset(Dataset):
    def __init__(self, n: int, *, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        # supervised points (x, y=sin(x))
        self.x = (2 * math.pi) * (torch.rand(n, 1, generator=g) - 0.5)  # [-pi, pi]
        self.y = torch.sin(self.x)

        # collocation points for physics residual
        self.x_col = (2 * math.pi) * (torch.rand(n, 1, generator=g) - 0.5)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "x": self.x[idx],
            "y": self.y[idx],
            "x_col": self.x_col[idx],
        }


# -----------------------------
# 2) Model
# -----------------------------
class MLP(nn.Module):
    def __init__(self, width: int = 64, depth: int = 3):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept x as (B,1) or (B,1,1)
        if x.ndim == 3:
            x = x.squeeze(-1)
        return self.net(x)


# -----------------------------
# 3) Physics loss: y'' + y = 0 on collocation points
# -----------------------------
def ode_residual_loss(model: nn.Module, batch: Dict[str, Any]):
    with torch.enable_grad():
        x_col = batch["x_col"]
        if x_col.ndim == 3:
            x_col = x_col.squeeze(-1)

        x_col = x_col.clone().detach().requires_grad_(True)
        y_col = model(x_col)

        (dy_dx,) = torch.autograd.grad(
            y_col,
            x_col,
            grad_outputs=torch.ones_like(y_col),
            create_graph=True,
            retain_graph=True,
        )

        (d2y_dx2,) = torch.autograd.grad(
            dy_dx,
            x_col,
            grad_outputs=torch.ones_like(dy_dx),
            create_graph=True,
            retain_graph=True,
        )

        r = d2y_dx2 + y_col
        loss = torch.mean(r**2)

    comps = {"residual_rmse": float(torch.sqrt(torch.mean(r.detach() ** 2)).item())}
    return loss, comps

def main() -> None:
    os.makedirs("examples/_runs", exist_ok=True)

    train_ds = ODESampleDataset(4096, seed=1)
    val_ds = ODESampleDataset(1024, seed=2)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = MLP(width=64, depth=4)

    combined = CombinedLoss(
        supervised=SupervisedLoss("mse"),
        physics=PhysicsLossHook(ode_residual_loss),
        w_supervised=1.0,
        w_physics=0.25,  # trade-off knob
    )

    def loss_fn(m, y_hat, batch):
        return combined(m, y_hat, batch)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        metrics=default_metrics(),
    )

    cfg = TrainConfig(
        epochs=15,
        lr=1e-3,
        device="cpu",
        log_dir="examples/_runs",
        run_name="pinn_ode",
        seed=123,
        deterministic=False,
        amp=False,
        save_best=True,
    )

    out = trainer.fit(train_loader, val_loader, cfg)
    print("best_val:", out.get("best_val"))
    print("best_path:", out.get("best_path"))

    # Quick sanity check: evaluate a few points
    with torch.no_grad():
        xs = torch.linspace(-math.pi, math.pi, 5).unsqueeze(1)
        yh = model(xs)
        print("x:", xs.squeeze(1).tolist())
        print("pred:", yh.squeeze(1).tolist())
        print("true:", torch.sin(xs).squeeze(1).tolist())


if __name__ == "__main__":
    main()
