"""Walk-forward backtesting for baselines (fast, no training).

Why this matters:
  - You should *always* beat strong baselines before trusting a deep model.
  - Backtesting with expanding/rolling windows mimics real deployment.

Shows:
  - ExpandingWindowSplitter + BacktestRunner
  - comparing naive / seasonal naive / drift across folds
  - reporting fold metrics and aggregate

Run:
  python examples/pinneaple_timeseries/04_backtest_baselines.py
"""

from __future__ import annotations

import math
import numpy as np

from pinneaple_timeseries import (
    ExpandingWindowSplitter,
    BacktestRunner,
    BacktestConfig,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    DriftForecaster,
)
from pinneaple_timeseries.metrics_ext.point import mae, rmse, smape


def make_series(T: int = 2500, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=float)
    y = 0.002 * t + 0.7 * np.sin(2 * math.pi * t / 24.0) + 0.25 * rng.standard_normal(T)
    return y.astype(float)


def eval_point(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "smape": smape(y_true, y_pred)}


def main() -> None:
    y = make_series()

    input_len = 96
    horizon = 24

    splitter = ExpandingWindowSplitter(
        initial_train_size=1200,
        step=48,
        horizon=horizon,
        input_len=input_len,
        max_folds=6,
    )

    runner = BacktestRunner(cfg=BacktestConfig())
    baselines = {
        "naive": NaiveForecaster(),
        "seasonal_naive_24": SeasonalNaiveForecaster(season_length=24),
        "drift": DriftForecaster(),
    }

    print("Backtest folds:")
    splits = splitter.split(y)
    print(f"- total folds: {len(splits)}")

    # Evaluate each baseline per fold
    results = {name: [] for name in baselines}
    for i, sp in enumerate(splits):
        y_train = y[sp.train_slice]
        y_test = y[sp.test_slice]  # length == horizon

        x_hist = y_train[-input_len:]  # most recent history at fold boundary

        for name, model in baselines.items():
            yhat = model.predict(x_hist.reshape(1, -1), horizon=horizon)[0]
            results[name].append(eval_point(y_test, yhat))

        print(f"\nFold {i+1}: train={sp.train_slice}  test={sp.test_slice}")
        for name in baselines:
            m = results[name][-1]
            print(f"  - {name:16s} MAE={m['mae']:7.4f} RMSE={m['rmse']:7.4f} sMAPE={m['smape']:7.2f}%")

    # Aggregate
    print("\n=== Aggregate (mean across folds) ===")
    for name, ms in results.items():
        agg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0].keys()}
        print(f"- {name:16s} MAE={agg['mae']:7.4f} RMSE={agg['rmse']:7.4f} sMAPE={agg['smape']:7.2f}%")


if __name__ == "__main__":
    main()