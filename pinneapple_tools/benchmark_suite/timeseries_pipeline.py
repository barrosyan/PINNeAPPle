"""TimeSeriesBenchmarkSpec — declarative time-series benchmark pipeline.

Uses exclusively existing PINNeAPPle modules — nothing built from scratch:

  pinneapple_timeseries.datasets.windowed  WindowedTimeSeriesDataset (sliding windows)
  pinneapple_timeseries.spec               TimeSeriesSpec
  pinneapple_train.trainer                 Trainer, TrainConfig  (training loop)
  pinneapple_train.losses                  CombinedLoss, SupervisedLoss
  pinneapple_train.normalizers             StandardScaler
  pinneapple_train.metrics                 MetricBundle + individual metric classes
  pinneapple_data.datasets                 load_dataset  (data loading)
  pinneapple_models / TSModelCatalog       model registry fallback

Usage
-----
    from pinneapple_tools.benchmark_suite import TimeSeriesBenchmarkSpec

    spec = TimeSeriesBenchmarkSpec(
        source   = "lorenz63",
        models   = ["lstm", "nbeats", "tcn"],
        metrics  = ["mse", "mae", "rmse"],
        horizon  = 50,
        lookback = 100,
        plots    = True,
    )
    report = spec.run()
    report.save("outputs/ts_benchmark.json")
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .report import BenchmarkReport, ModelRunResult


# ---------------------------------------------------------------------------
# Data loading  (pinneapple_data.datasets + file-path fallbacks)
# ---------------------------------------------------------------------------

def _load_ts_data(source: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load time series.  Returns (t, X) where X is (n_steps, n_features)."""
    try:
        from pinneapple_data.datasets import load_dataset
        data = load_dataset(source)
        X = data.get("X")
        t = data.get("t", np.arange(len(X) if X is not None else 0))
        if X is not None:
            return t, np.atleast_2d(X).T if X.ndim == 1 else X
    except (KeyError, ImportError):
        pass

    if source.startswith("synthetic:"):
        name = source.split(":", 1)[1].lower()
        try:
            from pinneapple_data.datasets import load_dataset
            data = load_dataset(name)
            X = data.get("X")
            t = data.get("t", np.arange(len(X)))
            if X is not None:
                return t, np.atleast_2d(X).T if X.ndim == 1 else X
        except Exception:
            pass
        t = np.arange(2000) * 0.05
        X = np.sin(2 * math.pi * 0.5 * t).reshape(-1, 1)
        return t, X

    p = Path(source)
    if p.exists():
        if p.suffix in (".csv", ".txt"):
            try:
                import pandas as pd
                df = pd.read_csv(p)
                return np.arange(len(df)), df.values.astype(np.float32)
            except ImportError:
                arr = np.loadtxt(p, delimiter=",")
                return np.arange(len(arr)), arr
        elif p.suffix == ".npz":
            data = dict(np.load(p))
            X = data.get("X", data.get("data", list(data.values())[0]))
            t = data.get("t", np.arange(len(X)))
            return t, X
        elif p.suffix == ".npy":
            X = np.load(p)
            return np.arange(len(X)), X

    raise ValueError(
        f"Cannot load time series from source='{source}'. "
        "Use a pinneapple dataset ID, 'synthetic:<name>', or a file path."
    )


# ---------------------------------------------------------------------------
# Metrics  (MetricBundle from pinneapple_train.metrics)
# ---------------------------------------------------------------------------

def _metric_bundle(requested: List[str]) -> Tuple[Any, bool]:
    """Build a MetricBundle from requested names; second return is mape_flag."""
    from pinneapple_neural.trainer.metrics import (
        MSE, MAE, RMSE, R2, RelL2, MaxError, MetricBundle,
    )
    _map = {
        "mse":         MSE(),
        "rmse":        RMSE(),
        "mae":         MAE(),
        "r2":          R2(),
        "l2_rel":      RelL2(name="l2_rel"),
        "relative_l2": RelL2(name="relative_l2"),
        "max_err":     MaxError(name="max_err"),
        "linf":        MaxError(name="linf"),
        "max":         MaxError(name="max"),
    }
    mape_flag = False
    metrics = []
    for m in requested:
        ml = m.lower()
        if ml in ("mape", "mean_absolute_percentage_error"):
            mape_flag = True
        elif ml in _map:
            metrics.append(_map[ml])
    return MetricBundle(metrics=metrics), mape_flag


# ---------------------------------------------------------------------------
# Loss  (CombinedLoss from pinneapple_train.losses, with feature alignment)
# ---------------------------------------------------------------------------

class _TSTrainer:
    """Trainer subclass that handles TSOutput.y_hat unwrapping."""

    def __init__(self, model, loss_fn):
        from pinneapple_neural.trainer.trainer import Trainer
        self._trainer = Trainer(model, loss_fn)

    def fit(self, train_loader, val_loader, cfg):
        # Patch _unwrap_pred to also handle TSOutput.y_hat
        original_unwrap = self._trainer._unwrap_pred

        def _unwrap_ts(y_hat, batch=None):
            if hasattr(y_hat, "y_hat") and isinstance(y_hat.y_hat, torch.Tensor):
                return y_hat.y_hat
            return original_unwrap(y_hat, batch)

        self._trainer._unwrap_pred = _unwrap_ts
        return self._trainer.fit(train_loader, val_loader, cfg)


def _make_ts_loss():
    """Alignment-aware supervised loss — delegates to CombinedLoss."""
    from pinneapple_neural.trainer.losses import CombinedLoss, SupervisedLoss
    _base = CombinedLoss(supervised=SupervisedLoss("mse"))

    def loss_fn(model, y_hat, batch):
        yb = batch.get("y")
        if yb is not None and y_hat.dim() >= 1 and yb.dim() >= 1:
            if y_hat.shape[-1] < yb.shape[-1]:
                batch = {**batch, "y": yb[..., :y_hat.shape[-1]]}
            elif y_hat.shape[-1] > yb.shape[-1]:
                y_hat = y_hat[..., :yb.shape[-1]]
        return _base(model, y_hat, batch)

    return loss_fn


# ---------------------------------------------------------------------------
# Model factory  (pinneapple_timeseries.models + TSModelCatalog + ModelRegistry)
# ---------------------------------------------------------------------------

def _build_ts_model(name: str, n_features: int, horizon: int,
                    lookback: int) -> Any:
    """Build any time-series model via pinneapple_timeseries or ModelRegistry."""
    from pinneapple_systems.time_series.models import (
        LSTMForecaster, GRUForecaster, TCNForecaster, NBeats, TFTForecaster,
        MLPForecaster, RecurrentConfig, TCNConfig, NBeatsConfig, TFTConfig,
        XGBoostForecaster, LightGBMForecaster, RandomForestForecaster,
    )

    nl = name.lower().replace("-", "_").replace(" ", "_")

    if nl in ("lstm", "lstm_forecaster"):
        return LSTMForecaster(RecurrentConfig(
            input_len=lookback, horizon=horizon,
            n_features=n_features, n_targets=n_features,
            hidden_size=64, num_layers=2, dropout=0.1,
        ))

    if nl in ("gru", "gru_forecaster"):
        return GRUForecaster(RecurrentConfig(
            input_len=lookback, horizon=horizon,
            n_features=n_features, n_targets=n_features,
            hidden_size=64, num_layers=2, dropout=0.1,
        ))

    if nl in ("tcn", "tcn_forecaster"):
        return TCNForecaster(TCNConfig(
            input_len=lookback, horizon=horizon,
            n_features=n_features, n_targets=n_features,
            n_channels=64, n_layers=4, kernel_size=3, dropout=0.1,
        ))

    if nl in ("nbeats", "n_beats", "n-beats"):
        # NBeats forward uses only the first feature; always build univariate
        return NBeats(NBeatsConfig(
            input_len=lookback, horizon=horizon, n_features=1,
            n_blocks=2, layer_width=64,
        ))

    if nl in ("transformer", "tft", "temporal_fusion"):
        return TFTForecaster(TFTConfig(
            input_len=lookback, horizon=horizon,
            n_features=n_features, n_targets=n_features,
            hidden_size=64, num_heads=4, num_lstm_layers=2, dropout=0.1,
        ))

    if nl in ("mlp", "feedforward", "fc"):
        return MLPForecaster(hidden_layer_sizes=(256, 128), max_iter=500)

    if nl in ("xgboost", "xgb"):
        return XGBoostForecaster(n_estimators=200, max_depth=6)

    if nl in ("lgbm", "lightgbm"):
        return LightGBMForecaster(n_estimators=200)

    if nl in ("randomforest", "rf"):
        return RandomForestForecaster(n_estimators=100)

    # Fallback: TSModelCatalog — wraps ModelRegistry for all TS families
    try:
        from pinneapple_systems.time_series.registry import TSModelCatalog
        return TSModelCatalog().build(nl, input_len=lookback, horizon=horizon,
                                     n_features=n_features, n_targets=n_features)
    except Exception:
        pass

    # Ultimate fallback: ModelRegistry — all 75+ registered models
    try:
        from pinneapple_neural.architectures import ModelRegistry
        return ModelRegistry.build(nl, input_len=lookback, horizon=horizon,
                                   n_features=n_features, n_targets=n_features)
    except Exception:
        pass

    raise ValueError(
        f"Unknown model '{name}'. "
        "Built-in shortcuts: lstm, gru, tcn, nbeats, transformer, mlp, xgboost, lgbm, randomforest. "
        "Extended: any model in TSModelCatalog or ModelRegistry."
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _predict_neural(model: nn.Module, loader) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on a DataLoader; return (y_pred, y_true) numpy arrays."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            pred = out.y_hat if hasattr(out, "y_hat") else out
            preds.append(pred.numpy())
            trues.append(y.numpy())
    return np.concatenate(preds, 0), np.concatenate(trues, 0)


def _train_sklearn(model: Any,
                   X_train: np.ndarray, y_train: np.ndarray
                   ) -> Tuple[List[Dict], Dict[str, float]]:
    N_tr, L, F = X_train.shape
    N_tr2, H, _ = y_train.shape
    model.fit(X_train.reshape(N_tr, L * F), y_train.reshape(N_tr2, H * F))
    return [], {}


def _predict_sklearn(model: Any, X_test: np.ndarray,
                     horizon: int, n_features: int) -> np.ndarray:
    N, L, F = X_test.shape
    return model.predict(X_test.reshape(N, L * F)).reshape(N, horizon, n_features)


# ---------------------------------------------------------------------------
# TimeSeriesBenchmarkSpec
# ---------------------------------------------------------------------------

class TimeSeriesBenchmarkSpec:
    """Declarative time-series benchmark pipeline.

    Parameters
    ----------
    source : str
        Dataset ID (e.g. "lorenz63"), file path, or "synthetic:<name>".
    models : list of str
        Any model name: built-in shortcuts (lstm, gru, tcn, nbeats, transformer,
        mlp, xgboost, lgbm, randomforest) or any name in TSModelCatalog /
        ModelRegistry (ts_fno, informer, autoformer, …).
    metrics : list of str
        "mse", "rmse", "mae", "mape", "r2", "l2_rel", "max_err".
    target_cols : list of int or None
        Column indices to forecast.  None = all columns.
    horizon : int
        Forecast horizon (steps).
    lookback : int
        Input window length (steps).
    test_size : float
        Fraction of windows reserved for testing.
    plots : bool
        Save forecast and loss-curve plots to output_dir.
    epochs : int
        Training epochs (neural models).
    lr : float
        Initial learning rate.
    batch_size : int
        Mini-batch size.
    seed : int
        Random seed.
    output_dir : str
        Directory for plots and JSON reports.
    """

    def __init__(
        self,
        source: str,
        models: Sequence[str] = ("lstm",),
        metrics: Sequence[str] = ("mse", "mae", "rmse"),
        target_cols: Optional[Sequence[int]] = None,
        horizon: int = 50,
        lookback: int = 100,
        test_size: float = 0.2,
        plots: bool = True,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 256,
        seed: int = 42,
        output_dir: str = "outputs",
    ):
        self.source = source
        self.models = list(models)
        self.metrics = list(metrics)
        self.target_cols = target_cols
        self.horizon = horizon
        self.lookback = lookback
        self.test_size = test_size
        self.plots = plots
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.output_dir = Path(output_dir)

    def _plot(self, y_pred: np.ndarray, y_test: np.ndarray,
              model_id: str, history: List[Dict]) -> List[str]:
        paths: List[str] = []
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return paths

        self.output_dir.mkdir(parents=True, exist_ok=True)
        prefix = self.output_dir / f"ts_{self.source.replace('/', '_')}_{model_id}"

        if history:
            fig, ax = plt.subplots(figsize=(7, 4))
            ep = [r["epoch"] for r in history]
            lo = [r["loss"] for r in history]
            ax.semilogy(ep, lo, lw=2)
            ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
            ax.set_title(f"{self.source} | {model_id} — Training Loss")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            p = str(prefix) + "_loss.png"
            fig.savefig(p, dpi=100, bbox_inches="tight"); plt.close(fig)
            paths.append(p)

        try:
            n_show = min(5, len(y_pred))
            idx_show = len(y_pred) - n_show
            fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 3), sharey=True)
            if n_show == 1:
                axes = [axes]
            for i, ax in enumerate(axes):
                ax.plot(y_test[idx_show + i, :, 0], label="true", lw=2)
                ax.plot(y_pred[idx_show + i, :, 0], label="pred", ls="--", lw=2)
                ax.set_title(f"sample {idx_show + i}")
                if i == 0:
                    ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            fig.suptitle(f"{self.source} | {model_id} — Forecast (feature 0)", fontsize=10)
            fig.tight_layout()
            p2 = str(prefix) + "_forecast.png"
            fig.savefig(p2, dpi=100, bbox_inches="tight"); plt.close(fig)
            paths.append(p2)
        except Exception:
            pass

        return paths

    def run(self) -> BenchmarkReport:
        """Execute the full time-series benchmark pipeline."""
        from pinneapple_neural.trainer.normalizers import StandardScaler
        from pinneapple_systems.time_series.datasets.windowed import WindowedTimeSeriesDataset
        from pinneapple_systems.time_series.spec import TimeSeriesSpec
        from torch.utils.data import DataLoader, Subset

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        report = BenchmarkReport(
            benchmark_type="timeseries",
            created_at=BenchmarkReport.now_timestamp(),
        )

        # 1. Load
        _, X = _load_ts_data(self.source)
        if self.target_cols is not None:
            X = X[:, self.target_cols]
        n_features = X.shape[1]

        # 2. Normalize — StandardScaler from pinneapple_train.normalizers
        scaler = StandardScaler()
        X_t = torch.tensor(X, dtype=torch.float32)
        scaler.fit(X_t)
        X_norm_t = scaler.transform(X_t)           # (T, F) tensor

        # 3. Windowed dataset + time-ordered train/val split
        spec_ts = TimeSeriesSpec(input_len=self.lookback, horizon=self.horizon)
        try:
            full_ds = WindowedTimeSeriesDataset(X_norm_t, spec_ts)
        except ValueError as exc:
            raise ValueError(
                f"Dataset '{self.source}' has {len(X_norm_t)} samples but "
                f"lookback={self.lookback}+horizon={self.horizon}="
                f"{self.lookback + self.horizon} needed per window. "
                "Reduce lookback/horizon or use a larger dataset."
            ) from exc

        split = max(1, int(len(full_ds) * (1 - self.test_size)))
        train_ds = Subset(full_ds, range(split))
        val_ds   = Subset(full_ds, range(split, len(full_ds)))
        if len(val_ds) == 0:
            val_ds = Subset(full_ds, range(len(full_ds) - 1, len(full_ds)))

        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True, drop_last=False)
        val_loader   = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        print(f"\n{'-' * 60}")
        print(f"  TimeSeriesBenchmarkSpec  ->  {self.source}")
        print(f"  n_features={n_features}  lookback={self.lookback}  "
              f"horizon={self.horizon}")
        print(f"  train windows={len(train_ds)}  test windows={len(val_ds)}")
        print(f"  Models: {self.models}")
        print(f"{'-' * 60}")

        report.problem_info = {
            "source": self.source,
            "n_samples": int(len(X_norm_t)),
            "n_features": n_features,
            "n_train_windows": int(len(train_ds)),
            "n_test_windows": int(len(val_ds)),
            "horizon": self.horizon,
            "lookback": self.lookback,
        }
        report.config = {
            "models": self.models, "metrics": self.metrics,
            "test_size": self.test_size, "epochs": self.epochs,
            "lr": self.lr, "batch_size": self.batch_size, "seed": self.seed,
        }

        all_plots: List[str] = []

        # 4. Train & evaluate each model
        for model_name in self.models:
            print(f"\n  > Model: {model_name}")
            t_start = time.time()
            try:
                model = _build_ts_model(model_name, n_features,
                                        self.horizon, self.lookback)
                is_neural = isinstance(model, nn.Module)

                if is_neural:
                    from pinneapple_neural.trainer.trainer import TrainConfig

                    n_params = sum(p.numel() for p in model.parameters())
                    print(f"    params = {n_params:,}")

                    trainer = _TSTrainer(model, _make_ts_loss())
                    cfg = TrainConfig(
                        epochs=self.epochs, lr=self.lr,
                        grad_clip=1.0, seed=self.seed,
                        save_best=False,
                        log_dir=str(self.output_dir / "runs"),
                        run_name=f"{self.source}_{model_name}",
                    )
                    trainer.fit(train_loader, val_loader, cfg)
                    y_pred, y_test = _predict_neural(model, val_loader)
                    history: List[Dict] = []   # Trainer logs to file; no per-epoch dict returned

                else:
                    n_params = 0
                    print("    sklearn model")
                    X_train = np.concatenate([x.numpy() for x, y in train_loader], 0)
                    y_train = np.concatenate([y.numpy() for x, y in train_loader], 0)
                    X_test  = np.concatenate([x.numpy() for x, y in val_loader], 0)
                    y_test  = np.concatenate([y.numpy() for x, y in val_loader], 0)
                    _train_sklearn(model, X_train, y_train)
                    y_pred  = _predict_sklearn(model, X_test, self.horizon, n_features)
                    history = []

                elapsed = time.time() - t_start

                # Align feature dimension (e.g. NBeats outputs 1 feature)
                n_out = y_pred.shape[-1]
                y_test_m = y_test[..., :n_out]

                # Metrics — MetricBundle from pinneapple_train.metrics
                bundle, mape_flag = _metric_bundle(self.metrics)
                pred_t = torch.tensor(y_pred.flatten(), dtype=torch.float32)
                true_t = torch.tensor(y_test_m.flatten(), dtype=torch.float32)
                metrics_out = bundle.compute(pred_t, true_t)
                if mape_flag:
                    denom = np.abs(y_test_m.flatten()) + 1e-8
                    metrics_out["mape"] = float(
                        np.mean(np.abs(y_pred.flatten() - y_test_m.flatten()) / denom) * 100
                    )

                print(f"    metrics: {metrics_out}")
                print(f"    time: {elapsed:.1f}s")

                report.model_results[model_name] = ModelRunResult(
                    model_id=model_name, n_params=n_params,
                    training_time_s=elapsed, metrics=metrics_out, history=history,
                )

                if self.plots:
                    if y_pred.shape[-1] < n_features:
                        pad = np.zeros((*y_pred.shape[:-1], n_features - y_pred.shape[-1]),
                                       dtype=y_pred.dtype)
                        y_pred_plot = np.concatenate([y_pred, pad], axis=-1)
                    else:
                        y_pred_plot = y_pred
                    all_plots.extend(self._plot(y_pred_plot, y_test, model_name, history))

            except Exception as exc:
                elapsed = time.time() - t_start
                print(f"    ERROR: {exc}")
                import traceback; traceback.print_exc()
                report.model_results[model_name] = ModelRunResult(
                    model_id=model_name, n_params=0,
                    training_time_s=elapsed, metrics={},
                    history=[], error_message=str(exc),
                )

        # 5. Leaderboard
        primary = self.metrics[0] if self.metrics else "mse"
        scored = [
            (mid, r.metrics.get(primary, float("inf")))
            for mid, r in report.model_results.items()
            if not r.error_message
        ]
        scored.sort(key=lambda x: x[1])
        report.leaderboard = [
            {"rank": i + 1, "model": mid, primary: score}
            for i, (mid, score) in enumerate(scored)
        ]
        for i, (mid, _) in enumerate(scored):
            report.model_results[mid].rank = i + 1
        report.best_model = scored[0][0] if scored else None
        report.plots_saved = all_plots

        report.print_summary()
        return report
