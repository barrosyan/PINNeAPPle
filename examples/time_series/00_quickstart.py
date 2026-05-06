"""Quickstart: TimeSeriesSpec + TSDataModule + naive baselines.

This example is meant to be the fastest way to validate that `pinneaple_timeseries`
is installed and working end-to-end.

It demonstrates:
  - defining a forecasting window spec (history length / horizon)
  - building deterministic train/val loaders (time-ordered; no leakage)
  - running strong zero-ML baselines (naive / seasonal naive / drift)
  - computing a couple of simple metrics

Run:
  python examples/pinneaple_timeseries/00_quickstart.py
"""

from __future__ import annotations

import math
import numpy as np
import torch

from pinneaple_timeseries import (
    TimeSeriesSpec,
    TSDataModule,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    DriftForecaster,
)
from pinneaple_timeseries.metrics_ext.point import mae, rmse, smape


def make_synthetic(T: int = 3000, F: int = 1, seed: int = 7) -> torch.Tensor:
    """Seasonal + trend + noise (multivariate supported via F)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=float)
    trend = 0.0008 * t
    seasonal = 0.8 * np.sin(2 * math.pi * t / 24.0) + 0.3 * np.sin(2 * math.pi * t / 168.0)
    y = trend + seasonal + 0.25 * rng.standard_normal(T)
    y = y.reshape(T, 1)
    if F > 1:
        # Create F correlated features by mixing.
        W = rng.normal(size=(1, F))
        x = y @ W + 0.05 * rng.standard_normal((T, F))
        return torch.tensor(x, dtype=torch.float32)
    return torch.tensor(y, dtype=torch.float32)


def main() -> None:
    series = make_synthetic(T=4000, F=1)  # (T, F)
    spec = TimeSeriesSpec(input_len=96, horizon=24, stride=1, target_offset=0)

    dm = TSDataModule(series=series, spec=spec, batch_size=128, val_ratio=0.2)
    train_loader, val_loader = dm.make_loaders()

    # Grab the whole validation set (small enough for this quickstart).
    Xv, Yv = [], []
    for xb, yb in val_loader:
        Xv.append(xb)
        Yv.append(yb)
    Xv = torch.cat(Xv, dim=0)  # (N, L, F)
    Yv = torch.cat(Yv, dim=0)  # (N, H, C)

    # Baselines operate on numpy arrays.
    x_hist = Xv[..., 0].numpy()   # (N, L) -> using first feature as target history
    y_true = Yv[..., 0].numpy()   # (N, H)

    preds = {}
    preds["naive"] = NaiveForecaster().predict(x_hist, horizon=spec.horizon)
    preds["seasonal_naive_24"] = SeasonalNaiveForecaster(season_length=24).predict(x_hist, horizon=spec.horizon)
    preds["drift"] = DriftForecaster().predict(x_hist, horizon=spec.horizon)

    print("Validation metrics (lower is better):\n")
    for name, yhat in preds.items():
        m_mae = mae(y_true, yhat)
        m_rmse = rmse(y_true, yhat)
        m_smape = smape(y_true, yhat)
        print(f"- {name:16s}  MAE={m_mae:8.4f}  RMSE={m_rmse:8.4f}  sMAPE={m_smape:8.3f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()