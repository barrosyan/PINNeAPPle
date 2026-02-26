"""
End-to-end time series forecasting pipeline demo for PINNeAPPle.

What this script shows:
  1) Formal problem definition with ForecastProblemSpec
  2) Time-series audit (stationarity, heteroscedasticity, autocorrelation, regime breaks)
  3) Walk-forward validation splitters (expanding and rolling windows)
  4) Strong baselines (naive / seasonal naive / drift) evaluated per fold
  5) Deep learning model (simple LSTM forecaster) evaluated with BacktestRunner
  6) Fold-safe preprocessing (fit only on the training fold inside Trainer.fit)

Optional dependencies:
  - statsmodels (for ADF/KPSS/ARCH/ACF/PACF)
  - ruptures (for changepoint detection)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from pinneaple_timeseries.problem import ForecastProblemSpec
from pinneaple_timeseries.datamodule import TSDataModule
from pinneaple_timeseries.audit.tests import TSAuditor
from pinneaple_timeseries.validation.splitters import ExpandingWindowSplitter, RollingWindowSplitter, Split
from pinneaple_timeseries.validation.backtest import BacktestRunner, BacktestConfig

from pinneaple_timeseries.metrics_ext.point import mae as np_mae, rmse as np_rmse, smape as np_smape

from pinneaple_train.trainer import Trainer, TrainConfig
from pinneaple_train.losses import CombinedLoss, SupervisedLoss
from pinneaple_train.metrics import regression_metrics_bundle
from pinneaple_train.preprocess import (
    PreprocessPipeline,
    MissingValueStep,
    WinsorizeStep,
    RobustScaleStep,
    NormalizeStep,
)


# ---------------------------
# 0) Utilities
# ---------------------------
def set_global_seed(seed: int = 123) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_synthetic_series(T: int = 600) -> np.ndarray:
    """
    Synthetic series with:
      - trend
      - yearly-ish seasonality
      - heteroscedastic noise increasing over time
      - occasional outliers
    """
    t = np.arange(T, dtype=float)

    trend = 0.003 * t
    season = 1.2 * np.sin(2 * np.pi * t / 50.0) + 0.4 * np.cos(2 * np.pi * t / 12.0)

    noise_scale = 0.1 + 0.002 * t
    noise = noise_scale * np.random.randn(T)

    y = 2.0 + trend + season + noise

    # add outliers
    idx = np.random.choice(np.arange(50, T - 50), size=8, replace=False)
    y[idx] += np.random.choice([-1, 1], size=len(idx)) * (2.0 + 2.0 * np.random.rand(len(idx)))

    # add some missing values
    miss = np.random.choice(np.arange(50, T - 50), size=6, replace=False)
    y[miss] = np.nan

    return y

# ---------------------------
# 1) Baselines (fold-safe)
# ---------------------------
def baseline_predict_from_window(x: np.ndarray, horizon: int, kind: str, season_length: int = 12) -> np.ndarray:
    """
    Baselines computed from the *input window only*, so they never see the future.
    x: (L, F) numpy array (we'll use x[:, 0] as target history)
    returns: (H,) predicted target
    """
    hist = x[:, 0].astype(float)
    hist = hist[~np.isnan(hist)]
    if len(hist) == 0:
        return np.full((horizon,), 0.0, dtype=float)

    if kind == "naive":
        return np.full((horizon,), float(hist[-1]), dtype=float)

    if kind == "seasonal_naive":
        s = int(season_length)
        if len(hist) >= s:
            val = float(hist[-s])
        else:
            val = float(hist[-1])
        return np.full((horizon,), val, dtype=float)

    if kind == "drift":
        if len(hist) < 2:
            return np.full((horizon,), float(hist[-1]), dtype=float)
        slope = (float(hist[-1]) - float(hist[0])) / max(1, (len(hist) - 1))
        return np.asarray([float(hist[-1]) + (i + 1) * slope for i in range(horizon)], dtype=float)

    raise ValueError(f"Unknown baseline kind: {kind}")

def evaluate_baselines_on_split(
    datamodule: TSDataModule,
    split: Split,
    *,
    season_length: int = 12,
) -> Dict[str, float]:
    """
    Evaluate baselines on the validation indices of one split.
    Returns MAE/RMSE/sMAPE for each baseline kind.
    """
    ds = datamodule.dataset()
    H = datamodule.spec.horizon

    # collect val forecasts and truths
    truths = []
    preds = {"naive": [], "seasonal_naive": [], "drift": []}

    for idx in split.val_idx:
        x_t, y_t = ds[idx]  # torch tensors: x (L,F), y (H,F)
        x_np = x_t.detach().cpu().numpy()
        y_np = y_t.detach().cpu().numpy()[:, 0]  # target channel

        truths.append(y_np)

        preds["naive"].append(baseline_predict_from_window(x_np, H, "naive", season_length))
        preds["seasonal_naive"].append(baseline_predict_from_window(x_np, H, "seasonal_naive", season_length))
        preds["drift"].append(baseline_predict_from_window(x_np, H, "drift", season_length))

    y_true = np.asarray(truths, dtype=float)      # (N_val, H)
    out: Dict[str, float] = {}

    for k, arr in preds.items():
        y_pred = np.asarray(arr, dtype=float)     # (N_val, H)
        # aggregate across horizons by flattening
        out[f"{k}_mae"] = np_mae(y_pred.reshape(-1), y_true.reshape(-1))
        out[f"{k}_rmse"] = np_rmse(y_pred.reshape(-1), y_true.reshape(-1))
        out[f"{k}_smape"] = np_smape(y_pred.reshape(-1), y_true.reshape(-1))

    return out

# ---------------------------
# 2) A simple DL model (LSTM forecaster)
# ---------------------------
class LSTMForecaster(nn.Module):
    """
    Simple sequence-to-multihorizon forecaster:
      input:  (B, L, F)
      output: (B, H, F)
    """
    def __init__(self, in_dim: int, hidden_dim: int, horizon: int, out_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.horizon = int(horizon)
        self.out_dim = int(out_dim)

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, self.horizon * self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]  # (B, hidden)
        y = self.head(h_last).view(x.shape[0], self.horizon, self.out_dim)
        return y


# ---------------------------
# 3) Main
# ---------------------------
set_global_seed(123)

# ---------------------------------------------------------
# 1) Declare the problem formally
# ---------------------------------------------------------
problem = ForecastProblemSpec(
    freq="D",
    time_col=None,
    target_cols=("y",),
    feature_cols=tuple(),
    exog_past_cols=tuple(),
    exog_future_cols=tuple(),
    input_len=96,
    horizon=24,
    horizon_type="direct",
    target_offset=0,
    stride=1,
    objective="point",
    metrics=("mae", "rmse", "smape"),
    notes="Demo: point forecasting with walk-forward backtesting.",
)

spec = problem.to_timeseries_spec()

# ---------------------------------------------------------
# 2) Build a series tensor (T, F) -> here F=1
# ---------------------------------------------------------
y = make_synthetic_series(T=700)  # numpy (T,)
series = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # (T,1)

# DataModule wraps the windowed dataset
dm = TSDataModule(series=series, spec=spec, batch_size=64, num_workers=0)

# ---------------------------------------------------------
# 3) Audit the series (diagnostics)
# ---------------------------------------------------------
auditor = TSAuditor(nlags=40, arch_lags=12)
audit_report = auditor.run(y, meta={"name": "synthetic_demo", "T": len(y), "freq": problem.freq})
print("\n=== AUDIT REPORT (keys only) ===")
for sec_name, sec in audit_report.sections.items():
    print(f"- {sec_name}: {list(sec.results.keys())}")
# If you want the full dict:
print(audit_report.to_dict())

# ---------------------------------------------------------
# 4) Create Splits (walk-forward)
# ---------------------------------------------------------
ds = dm.dataset()
n_samples = len(ds)

# Expanding window: start with some training windows, validate on next block
expanding = ExpandingWindowSplitter(
    initial_train_size=250,
    val_size=60,
    step_size=30,  # how much the train_end expands each fold
    gap=0,
    max_folds=4,
)
expanding_splits = list(expanding.split(n_samples))

# Rolling window: fixed-size training window that slides forward
rolling = RollingWindowSplitter(
    train_size=250,
    val_size=60,
    step_size=30,
    gap=0,
    max_folds=4,
)
rolling_splits = list(rolling.split(n_samples))

print("\n=== SPLITS ===")
print(f"Dataset windows: n_samples={n_samples}")
print(f"Expanding folds: {len(expanding_splits)} | Rolling folds: {len(rolling_splits)}")
print(f"Example expanding split[0]: train={len(expanding_splits[0].train_idx)} val={len(expanding_splits[0].val_idx)}")

# ---------------------------------------------------------
# 5) Compare baselines vs model (per fold, expanding)
# ---------------------------------------------------------
print("\n=== BASELINES (Expanding) ===")
for sp in expanding_splits:
    b = evaluate_baselines_on_split(dm, sp, season_length=50)  # choose a season_length matching synthetic season
    print(f"Fold {sp.fold} | "
            f"naive MAE={b['naive_mae']:.4f} | "
            f"seasonal MAE={b['seasonal_naive_mae']:.4f} | "
            f"drift MAE={b['drift_mae']:.4f}")

# ---------------------------------------------------------
# 6) BacktestRunner with fold-safe preprocessing + metrics
# ---------------------------------------------------------
def preprocess_factory() -> PreprocessPipeline:
    # IMPORTANT: these steps will be fit on the TRAIN FOLD ONLY inside Trainer.fit()
    return PreprocessPipeline(
        steps=[
            MissingValueStep(key="x", strategy="ffill", enabled=True),
            WinsorizeStep(key="x", q_low=0.01, q_high=0.99, enabled=True),
            RobustScaleStep(key="x", enabled=True),
            NormalizeStep(key="x", dim=(0, 1), enabled=True),  # standardize after robust scaling
        ]
    )

def model_factory() -> nn.Module:
    # input features F=1, output features F=1
    return LSTMForecaster(in_dim=1, hidden_dim=64, horizon=spec.horizon, out_dim=1, num_layers=1, dropout=0.0)

def trainer_factory(*, model: nn.Module, preprocess: PreprocessPipeline | None) -> Trainer:
    loss_fn = CombinedLoss(supervised=SupervisedLoss("mse"))
    metrics = regression_metrics_bundle()  # {mse, mae, rmse, r2}
    return Trainer(model=model, loss_fn=loss_fn, metrics=metrics, preprocess=preprocess)

train_cfg = TrainConfig(
    epochs=10,
    lr=1e-3,
    weight_decay=0.0,
    grad_clip=1.0,
    amp=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=123,
    deterministic=False,
    log_dir="runs",
    run_name="timeseries_demo_lstm",
    save_best=True,
)

bt = BacktestRunner(
    trainer_factory=trainer_factory,
    model_factory=model_factory,
    preprocess_factory=preprocess_factory,
    datamodule=dm,
    splits=expanding_splits,              # choose expanding or rolling here
    train_cfg=train_cfg,
    backtest_cfg=BacktestConfig(
        reset_model_each_fold=True,
        shuffle_train=True,              # window order in training batches can be shuffled safely
        val_batch_size=256,
    ),
)

print("\n=== MODEL BACKTEST (Expanding) ===")
result = bt.run()
print("Aggregate metrics:", result.agg)
print("Per-fold summary:")
for r in result.folds:
    # last logged metrics keys depend on trainer/metrics bundle
    print(f"  Fold {int(r['fold'])}: best_val={r.get('best_val'):.6f} | "
            f"val_mae={r.get('val_mae', float('nan')):.6f} | "
            f"val_rmse={r.get('val_rmse', float('nan')):.6f}")

# Optional: do the same for rolling splits
# bt_rolling = BacktestRunner(..., splits=rolling_splits, ...)
# result_rolling = bt_rolling.run()

print("\nDone.")
