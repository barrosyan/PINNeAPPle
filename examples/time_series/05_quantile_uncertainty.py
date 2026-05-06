"""Probabilistic forecasting via QuantileHead (uncertainty module).

This shows how to:
  - define a *point* forecaster (simple MLP) for one target
  - wrap it with QuantileHead to predict multiple quantiles per horizon step
  - train with pinball loss for each quantile

Notes:
  - This is a minimal demo. In real projects you would use the Trainer API
    (pinneaple_train) + BacktestRunner for fold-safe evaluation.

Run:
  python examples/pinneaple_timeseries/05_quantile_uncertainty.py
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn

from pinneaple_timeseries import TimeSeriesSpec, TSDataModule
from pinneaple_timeseries.uncertainty.quantile import QuantileHead, QuantileConfig, pinball_loss_torch


def make_series(T: int = 5000, seed: int = 3) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=float)
    y = 0.001 * t + 0.8 * np.sin(2 * math.pi * t / 24.0) + 0.35 * rng.standard_normal(T)
    return torch.tensor(y.reshape(T, 1), dtype=torch.float32)


class PointMLP(nn.Module):
    """Predicts horizon steps directly from history (single target)."""

    def __init__(self, input_len: int, horizon: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),  # (B, L, 1) -> (B, L)
            nn.Linear(input_len, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, H)


def main() -> None:
    torch.manual_seed(0)

    series = make_series()
    spec = TimeSeriesSpec(input_len=96, horizon=24, stride=1, target_offset=0)
    train_loader, val_loader = TSDataModule(series=series, spec=spec, batch_size=256, val_ratio=0.2).make_loaders()

    base = PointMLP(input_len=spec.input_len, horizon=spec.horizon)
    qcfg = QuantileConfig(quantiles=(0.1, 0.5, 0.9))
    model = QuantileHead(base=base, cfg=qcfg, hidden_dim=64)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    def step(batch):
        x, y = batch  # x: (B, L, 1), y: (B, H, 1)
        y = y[..., 0]  # (B, H)
        q = model(x)  # (B, H, Q)
        loss = 0.0
        for qi, qv in enumerate(qcfg.quantiles):
            loss = loss + pinball_loss_torch(q[..., qi], y, q=float(qv))
        return loss / len(qcfg.quantiles)

    # Quick training loop
    for epoch in range(5):
        model.train()
        tr = []
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = step(batch)
            loss.backward()
            opt.step()
            tr.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            va = [float(step(b).cpu()) for b in val_loader]
        print(f"Epoch {epoch+1:02d}  train_pinball={np.mean(tr):.4f}  val_pinball={np.mean(va):.4f}")

    # Show one example
    model.eval()
    x0, y0 = next(iter(val_loader))
    with torch.no_grad():
        q0 = model(x0[:1])  # (1, H, Q)
    print("\nOne sample (first 5 horizon steps):")
    print("y_true:", y0[0, :5, 0].cpu().numpy())
    print("q10  :", q0[0, :5, 0].cpu().numpy())
    print("q50  :", q0[0, :5, 1].cpu().numpy())
    print("q90  :", q0[0, :5, 2].cpu().numpy())


if __name__ == "__main__":
    main()