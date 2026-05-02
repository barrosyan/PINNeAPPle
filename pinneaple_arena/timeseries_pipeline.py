"""TimeSeriesBenchmarkSpec — declarative time-series benchmark pipeline.

Usage
-----
    from pinneaple_arena import TimeSeriesBenchmarkSpec

    spec = TimeSeriesBenchmarkSpec(
        source  = "lorenz63",             # dataset ID, file path, or "synthetic:sine"
        models  = ["lstm", "nbeats", "tcn"],
        metrics = ["mse", "mae", "rmse"],
        horizon = 50,
        lookback = 100,
        plots   = True,
    )
    report = spec.run()
    report.print_summary()
    report.save("outputs/ts_benchmark.json")
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .report import BenchmarkReport, ModelRunResult


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def _ts_metrics(pred: np.ndarray, true: np.ndarray,
                requested: List[str]) -> Dict[str, float]:
    diff = pred - true
    out: Dict[str, float] = {}
    for m in requested:
        ml = m.lower()
        if ml == "mse":
            out[m] = float(np.mean(diff**2))
        elif ml == "rmse":
            out[m] = float(np.sqrt(np.mean(diff**2)))
        elif ml == "mae":
            out[m] = float(np.mean(np.abs(diff)))
        elif ml in ("mape", "mean_absolute_percentage_error"):
            denom = np.abs(true) + 1e-8
            out[m] = float(np.mean(np.abs(diff) / denom) * 100)
        elif ml in ("r2", "r_squared"):
            ss_res = np.sum(diff**2)
            ss_tot = np.sum((true - true.mean())**2) + 1e-12
            out[m] = float(1.0 - ss_res / ss_tot)
        elif ml in ("l2_rel", "relative_l2"):
            out[m] = float(np.linalg.norm(diff) / (np.linalg.norm(true) + 1e-12))
        elif ml in ("max_err", "linf"):
            out[m] = float(np.max(np.abs(diff)))
        else:
            out[m] = float("nan")
    return out


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def _load_ts_data(source: str, target_col: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Load time series data.  Returns (t, X) where X is (n_steps, n_features)."""

    # 1. Pinneaple dataset registry
    try:
        from pinneaple_data.datasets import load_dataset
        data = load_dataset(source)
        X = data.get("X")
        t = data.get("t", np.arange(len(X) if X is not None else 0))
        if X is not None:
            return t, np.atleast_2d(X).T if X.ndim == 1 else X
    except (KeyError, ImportError):
        pass

    # 2. Synthetic generators  (source = "synthetic:<name>")
    if source.startswith("synthetic:"):
        name = source.split(":", 1)[1].lower()
        try:
            from pinneaple_data.datasets import load_dataset
            data = load_dataset(name)
            X = data.get("X")
            t = data.get("t", np.arange(len(X)))
            if X is not None:
                return t, np.atleast_2d(X).T if X.ndim == 1 else X
        except Exception:
            pass
        # Fallback: generate simple sine
        t = np.arange(2000) * 0.05
        X = np.sin(2 * math.pi * 0.5 * t).reshape(-1, 1)
        return t, X

    # 3. File path (CSV / NPZ / NPY)
    p = Path(source)
    if p.exists():
        if p.suffix in (".csv", ".txt"):
            try:
                import pandas as pd
                df = pd.read_csv(p)
                t = np.arange(len(df))
                return t, df.values.astype(np.float32)
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

    # 4. Try real-world dataset registry
    try:
        from pinneaple_data.datasets.real_world import load_real_dataset
        data = load_real_dataset(source)
        X = data.get("X", data.get("values"))
        t = data.get("t", np.arange(len(X)))
        if X is not None:
            return t, np.atleast_2d(X).T if X.ndim == 1 else X
    except Exception:
        pass

    raise ValueError(
        f"Cannot load time series from source='{source}'. "
        "Use a pinneaple dataset ID, 'synthetic:<name>', or a file path."
    )


def _make_windows(X: np.ndarray, lookback: int, horizon: int,
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding window -> (n_samples, lookback, n_features), (n_samples, horizon, n_features)."""
    n, d = X.shape
    xs, ys = [], []
    for i in range(n - lookback - horizon + 1):
        xs.append(X[i: i+lookback])
        ys.append(X[i+lookback: i+lookback+horizon])
    if not xs:
        return (np.zeros((0, lookback, d), dtype=np.float32),
                np.zeros((0, horizon, d), dtype=np.float32))
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------

class _Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.std + self.mean


# -----------------------------------------------------------------------------
# Built-in PyTorch time series models
# -----------------------------------------------------------------------------

class _LSTMModel(nn.Module):
    def __init__(self, n_features: int, horizon: int, lookback: int,
                 hidden: int = 64, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, horizon * n_features)
        self.horizon = horizon
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        pred = self.head(out[:, -1, :])
        return pred.view(-1, self.horizon, self.n_features)


class _GRUModel(nn.Module):
    def __init__(self, n_features: int, horizon: int, lookback: int,
                 hidden: int = 64, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, horizon * n_features)
        self.horizon = horizon
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        pred = self.head(out[:, -1, :])
        return pred.view(-1, self.horizon, self.n_features)


class _TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()
        self.causal_trim = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.conv1(x)[..., :-self.causal_trim or None])
        h = self.relu(self.conv2(h)[..., :-self.causal_trim or None])
        return self.relu(h + self.skip(x))


class _TCNModel(nn.Module):
    def __init__(self, n_features: int, horizon: int, lookback: int,
                 hidden: int = 64, n_blocks: int = 4, kernel: int = 3):
        super().__init__()
        layers = []
        in_ch = n_features
        for i in range(n_blocks):
            layers.append(_TCNBlock(in_ch, hidden, kernel, dilation=2**i))
            in_ch = hidden
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, horizon * n_features)
        self.horizon = horizon
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F) -> (B, F, L)
        h = self.net(x.transpose(1, 2))
        pred = self.head(h[..., -1])
        return pred.view(-1, self.horizon, self.n_features)


class _NBeatsBlock(nn.Module):
    def __init__(self, lookback: int, horizon: int, hidden: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(lookback, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.theta_b = nn.Linear(hidden, lookback)
        self.theta_f = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(x)
        return self.theta_b(h), self.theta_f(h)


class _NBeatsModel(nn.Module):
    def __init__(self, n_features: int, horizon: int, lookback: int,
                 hidden: int = 64, n_stacks: int = 2, n_blocks: int = 2):
        super().__init__()
        self.horizon = horizon
        self.n_features = n_features
        self.lookback = lookback
        self.stacks = nn.ModuleList([
            nn.ModuleList([_NBeatsBlock(lookback, horizon, hidden)
                           for _ in range(n_blocks)])
            for _ in range(n_stacks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F) — handle each feature independently for simplicity
        B, L, F = x.shape
        preds = []
        for f in range(F):
            x_f = x[:, :, f]   # (B, L)
            residual = x_f
            forecast_f = torch.zeros(B, self.horizon, device=x.device)
            for stack in self.stacks:
                for block in stack:
                    backcast, forecast = block(residual)
                    residual = residual - backcast
                    forecast_f = forecast_f + forecast
            preds.append(forecast_f.unsqueeze(-1))
        return torch.cat(preds, dim=-1)   # (B, H, F)


class _TransformerModel(nn.Module):
    def __init__(self, n_features: int, horizon: int, lookback: int,
                 d_model: int = 64, nhead: int = 4, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.head = nn.Linear(d_model, horizon * n_features)
        self.horizon = horizon
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.encoder(h)
        pred = self.head(h[:, -1, :])
        return pred.view(-1, self.horizon, self.n_features)


class _MLPModel(nn.Module):
    def __init__(self, n_features: int, horizon: int, lookback: int,
                 hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lookback * n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, horizon * n_features),
        )
        self.horizon = horizon
        self.n_features = n_features
        self.lookback = lookback

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        pred = self.net(x.reshape(B, -1))
        return pred.view(B, self.horizon, self.n_features)


# -----------------------------------------------------------------------------
# Model factory
# -----------------------------------------------------------------------------

def _build_ts_model(name: str, n_features: int, horizon: int,
                    lookback: int) -> nn.Module:
    name_l = name.lower().replace("-", "_").replace(" ", "_")

    # Try pinneaple_timeseries first
    try:
        if name_l in ("lstm", "lstm_forecaster"):
            from pinneaple_timeseries.models.recurrent import LSTMForecaster, RecurrentConfig
            cfg = RecurrentConfig(input_size=n_features, hidden_size=128,
                                  num_layers=2, horizon=horizon,
                                  lookback=lookback, dropout=0.1)
            return LSTMForecaster(cfg)
    except Exception:
        pass

    try:
        if name_l in ("nbeats", "n_beats"):
            from pinneaple_timeseries.models.nbeats import NBeats, NBeatsConfig
            cfg = NBeatsConfig(input_size=lookback * n_features,
                               horizon=horizon, n_stacks=2, n_blocks=3)
            return NBeats(cfg)
    except Exception:
        pass

    try:
        if name_l in ("tcn", "tcn_forecaster"):
            from pinneaple_timeseries.models.tcn import TCNForecaster, TCNConfig
            cfg = TCNConfig(input_size=n_features, horizon=horizon,
                            lookback=lookback, hidden_size=64)
            return TCNForecaster(cfg)
    except Exception:
        pass

    try:
        if name_l in ("tft", "temporal_fusion_transformer"):
            from pinneaple_timeseries.models.tft import TFTForecaster, TFTConfig
            cfg = TFTConfig(input_size=n_features, horizon=horizon,
                            lookback=lookback, hidden_size=64)
            return TFTForecaster(cfg)
    except Exception:
        pass

    # Fallback: built-in pure PyTorch implementations
    if name_l in ("lstm", "lstm_forecaster"):
        return _LSTMModel(n_features, horizon, lookback)
    if name_l in ("gru", "gru_forecaster"):
        return _GRUModel(n_features, horizon, lookback)
    if name_l in ("tcn", "temporal_convolutional_network"):
        return _TCNModel(n_features, horizon, lookback)
    if name_l in ("nbeats", "n_beats", "n-beats"):
        return _NBeatsModel(n_features, horizon, lookback)
    if name_l in ("transformer", "tft", "temporal_fusion"):
        return _TransformerModel(n_features, horizon, lookback)
    if name_l in ("mlp", "feedforward", "fc"):
        return _MLPModel(n_features, horizon, lookback)

    # Try ModelRegistry
    try:
        from pinneaple_models.registry import ModelRegistry
        return ModelRegistry.build(name_l, in_dim=lookback*n_features,
                                   out_dim=horizon*n_features)
    except Exception:
        pass

    raise ValueError(
        f"Unknown time series model '{name}'. "
        "Supported: lstm, gru, tcn, nbeats, transformer, mlp."
    )


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def _train_ts(model: nn.Module,
              X_train: np.ndarray, y_train: np.ndarray,
              epochs: int, lr: float, batch_size: int = 256,
              log_every: int = 50) -> Tuple[List[Dict], Dict[str, float]]:
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    n = len(X_t)
    history: List[Dict] = []
    model.train()

    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i: i+batch_size]
            xb = X_t[idx]; yb = y_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if n_batches > 0:
            scheduler.step()

        if (epoch + 1) % log_every == 0 or epoch == 0:
            history.append({"epoch": epoch+1, "loss": epoch_loss / max(n_batches, 1)})

    model.eval()
    return history, {"final_loss": history[-1]["loss"] if history else float("nan")}


def _eval_ts(model: nn.Module, X_test: np.ndarray,
             y_test: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        pred = model(torch.tensor(X_test, dtype=torch.float32))
    return pred.numpy()


# -----------------------------------------------------------------------------
# TimeSeriesBenchmarkSpec
# -----------------------------------------------------------------------------

class TimeSeriesBenchmarkSpec:
    """Declarative time-series benchmark pipeline.

    Parameters
    ----------
    source : str
        Dataset ID (e.g. "lorenz63"), file path, or "synthetic:<name>".
    models : list of str
        Model names: "lstm", "gru", "tcn", "nbeats", "transformer", "mlp".
    metrics : list of str
        Metrics: "mse", "rmse", "mae", "mape", "r2", "l2_rel".
    target_cols : list of int or None
        Column indices to forecast. If None, all columns.
    horizon : int
        Forecast horizon (steps).
    lookback : int
        Input window length (steps).
    test_size : float
        Fraction of data reserved for testing.
    plots : bool
        Save forecast plots to output_dir.
    epochs : int
        Training epochs per model.
    lr : float
        Initial learning rate.
    batch_size : int
        Mini-batch size.
    seed : int
        Random seed.
    output_dir : str
        Directory for plots and report JSON.
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

    def _plot(self, model: nn.Module, X_test: np.ndarray, y_test: np.ndarray,
              y_pred: np.ndarray, norm: _Normalizer,
              model_id: str, history: List[Dict]) -> List[str]:
        paths: List[str] = []
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return paths

        self.output_dir.mkdir(parents=True, exist_ok=True)
        prefix = self.output_dir / f"ts_{self.source.replace('/', '_')}_{model_id}"

        # Loss curve
        fig, ax = plt.subplots(figsize=(7, 4))
        ep = [r["epoch"] for r in history]
        lo = [r["loss"] for r in history]
        ax.semilogy(ep, lo, lw=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
        ax.set_title(f"{self.source} | {model_id} ? Training Loss")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = str(prefix) + "_loss.png"
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

        # Forecast comparison for first feature
        try:
            # y_pred / y_test: (n_test, horizon, n_features)
            # Take last sample for visualization
            n_show = min(5, len(y_pred))
            idx_show = len(y_pred) - n_show

            fig, axes = plt.subplots(1, n_show, figsize=(4*n_show, 3), sharey=True)
            if n_show == 1:
                axes = [axes]
            for i, ax in enumerate(axes):
                true_seq = y_test[idx_show + i, :, 0]
                pred_seq = y_pred[idx_show + i, :, 0]
                ax.plot(true_seq, label="true", lw=2)
                ax.plot(pred_seq, label="pred", ls="--", lw=2)
                ax.set_title(f"sample {idx_show+i}")
                if i == 0:
                    ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            fig.suptitle(f"{self.source} | {model_id} ? Forecast (feature 0)", fontsize=10)
            fig.tight_layout()
            p2 = str(prefix) + "_forecast.png"
            fig.savefig(p2, dpi=100, bbox_inches="tight")
            plt.close(fig)
            paths.append(p2)
        except Exception:
            pass

        return paths

    def run(self) -> BenchmarkReport:
        """Execute the full time-series benchmark pipeline."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        report = BenchmarkReport(
            benchmark_type="timeseries",
            created_at=BenchmarkReport.now_timestamp(),
        )

        # 1. Load data
        print(f"\n{'-'*60}")
        print(f"  TimeSeriesBenchmarkSpec  ->  {self.source}")
        t, X = _load_ts_data(self.source)

        # Select target columns
        if self.target_cols is not None:
            X = X[:, self.target_cols]
        n_features = X.shape[1]

        # 2. Normalize
        norm = _Normalizer().fit(X)
        X_norm = norm.transform(X)

        # 3. Create all sliding windows first, then split (handles small datasets)
        X_all, y_all = _make_windows(X_norm, self.lookback, self.horizon)
        if len(X_all) == 0:
            raise ValueError(
                f"Dataset '{self.source}' has {len(X_norm)} samples, but "
                f"lookback={self.lookback} + horizon={self.horizon} = "
                f"{self.lookback + self.horizon} samples are needed per window. "
                "Reduce lookback/horizon or use a larger dataset."
            )
        split = max(1, int(len(X_all) * (1 - self.test_size)))
        X_train, X_test = X_all[:split], X_all[split:]
        y_train, y_test = y_all[:split], y_all[split:]
        # Ensure at least 1 test window
        if len(X_test) == 0:
            X_test, y_test = X_all[-1:], y_all[-1:]

        print(f"  n_features={n_features}  lookback={self.lookback}  "
              f"horizon={self.horizon}")
        print(f"  train windows={len(X_train)}  test windows={len(X_test)}")
        print(f"  Models: {self.models}")
        print(f"{'-'*60}")

        report.problem_info = {
            "source": self.source,
            "n_samples": int(len(X_norm)),
            "n_features": n_features,
            "n_train_windows": int(len(X_train)),
            "n_test_windows": int(len(X_test)),
            "horizon": self.horizon,
            "lookback": self.lookback,
        }
        report.config = {
            "models": self.models,
            "metrics": self.metrics,
            "test_size": self.test_size,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }

        all_plots: List[str] = []

        # 4. Train & evaluate each model
        for model_name in self.models:
            print(f"\n  > Model: {model_name}")
            t_start = time.time()
            try:
                model = _build_ts_model(model_name, n_features,
                                        self.horizon, self.lookback)
                n_params = sum(p.numel() for p in model.parameters())
                print(f"    params = {n_params:,}")

                history, _ = _train_ts(
                    model, X_train, y_train,
                    epochs=self.epochs, lr=self.lr,
                    batch_size=self.batch_size,
                    log_every=max(1, self.epochs // 8),
                )
                elapsed = time.time() - t_start

                y_pred = _eval_ts(model, X_test, y_test)

                # Flatten for metrics
                metrics_out = _ts_metrics(
                    y_pred.flatten(), y_test.flatten(), self.metrics
                )
                print(f"    metrics: {metrics_out}")
                print(f"    time: {elapsed:.1f}s")

                result = ModelRunResult(
                    model_id=model_name,
                    n_params=n_params,
                    training_time_s=elapsed,
                    metrics=metrics_out,
                    history=history,
                )
                report.model_results[model_name] = result

                if self.plots:
                    plots = self._plot(model, X_test, y_test, y_pred,
                                       norm, model_name, history)
                    all_plots.extend(plots)

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
            {"rank": i+1, "model": mid, primary: score}
            for i, (mid, score) in enumerate(scored)
        ]
        for i, (mid, _) in enumerate(scored):
            report.model_results[mid].rank = i + 1
        report.best_model = scored[0][0] if scored else None
        report.plots_saved = all_plots

        report.print_summary()
        return report
