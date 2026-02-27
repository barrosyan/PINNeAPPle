"""pinneaple_train example 02: DataModule + robust preprocessing.

What this shows
--------------
- Building loaders with pinneaple_train.DataModule (train/val/test split).
- Leakage-safe *time* split (no shuffle across the time axis).
- A preprocessing pipeline that:
  - fills missing values (NaNs)
  - normalizes inputs using statistics fitted on *train batches only*
- Training with Trainer + audit logging.

Run
---
python examples/pinneaple_train/02_datamodule_splits_preprocess.py

Artifacts
---------
- examples/_runs/train_dm.jsonl (epoch logs)
- examples/_runs/train_dm.best.pt (best checkpoint)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Any, List

import torch
from torch import nn

from pinneaple_train import (
    DataModule,
    SplitSpec,
    Trainer,
    TrainConfig,
    CombinedLoss,
    SupervisedLoss,
    default_metrics,
)
from pinneaple_train.preprocess import PreprocessPipeline, MissingValueStep, NormalizeStep


# -----------------------------
# 1) Synthetic time-series dataset with NaNs
# -----------------------------
@dataclass
class WindowItem:
    x: torch.Tensor  # (T, D)
    y: torch.Tensor  # (1,) regression target
    meta: Dict[str, Any]


def make_windows(*, n_series: int = 12, T: int = 64, D: int = 3, window: int = 24) -> List[Dict[str, Any]]:
    """Create windowed samples from multiple synthetic series.

    Each item is a dict compatible with pinneaple_train.ItemAdapter.
    """
    g = torch.Generator().manual_seed(123)
    items: List[Dict[str, Any]] = []

    for sid in range(n_series):
        # Base signal: mixture of sines + drift + noise
        t = torch.linspace(0, 10.0, T)
        base = (
            0.6 * torch.sin(2 * math.pi * (0.15 + 0.02 * sid) * t)
            + 0.25 * torch.sin(2 * math.pi * (0.05 + 0.01 * sid) * t + 0.3)
            + 0.02 * t
        )
        noise = 0.05 * torch.randn(T, generator=g)
        s = base + noise

        # Build multivariate x(t) with correlated features
        x_full = torch.stack(
            [
                s,
                torch.cos(s) + 0.02 * torch.randn(T, generator=g),
                torch.tanh(s) + 0.02 * torch.randn(T, generator=g),
            ],
            dim=-1,
        )  # (T, 3)

        # Inject missing values (NaNs)
        miss = torch.rand_like(x_full, generator=g) < 0.03
        x_full = x_full.masked_fill(miss, float("nan"))

        # Windowed forecasting target: next-step of feature 0
        for i in range(0, T - window - 1):
            x = x_full[i : i + window]  # (window, D)
            y = x_full[i + window, 0].unsqueeze(0)  # (1,)
            items.append(
                {
                    "x": x,
                    "y": y,
                    "meta": {"series_id": str(sid), "t0": int(i)},
                }
            )

    return items


# -----------------------------
# 2) Simple model: flatten window -> MLP -> scalar
# -----------------------------
class WindowMLP(nn.Module):
    def __init__(self, window: int, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window * d_in, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B,T,D), got {tuple(x.shape)}")
        return self.net(x.flatten(start_dim=1))


def main() -> None:
    os.makedirs("examples/_runs", exist_ok=True)

    window = 24
    items = make_windows(n_series=14, T=72, D=3, window=window)

    # Leakage-safe split: time-based (keeps order). Works well for forecasting demos.
    dm = DataModule(
        dataset=items,
        split=SplitSpec(method="time", train=0.75, val=0.15, test=0.10, seed=7),
        batch_size=64,
        num_workers=0,
        pin_memory=False,
    )

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # Preprocess: fill NaNs then normalize x using train statistics
    preprocess = PreprocessPipeline(
        steps=[
            MissingValueStep(key="x", strategy="ffill"),
            NormalizeStep(key="x", dim=(0, 1), store_key="x_scaler"),
        ]
    )

    # Fit preprocessing ONLY using train batches
    train_batches = []
    for i, b in enumerate(train_loader):
        train_batches.append(b)
        if i >= 8:  # a handful of batches is enough for the demo
            break
    preprocess.fit(train_batches)

    model = WindowMLP(window=window, d_in=3)

    # Loss wrapper: returns a dict with mandatory "total".
    combined = CombinedLoss(supervised=SupervisedLoss("mse"), physics=None, w_supervised=1.0, w_physics=0.0)

    def loss_fn(m, y_hat, batch):
        return combined(m, y_hat, batch)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        metrics=default_metrics(),
        preprocess=preprocess,
    )

    cfg = TrainConfig(
        epochs=8,
        lr=2e-3,
        device="cpu",
        log_dir="examples/_runs",
        run_name="train_dm",
        seed=42,
        deterministic=False,
        amp=False,
        save_best=True,
    )

    out = trainer.fit(train_loader, val_loader, cfg)
    print("best_val:", out.get("best_val"))
    print("best_path:", out.get("best_path"))


if __name__ == "__main__":
    main()