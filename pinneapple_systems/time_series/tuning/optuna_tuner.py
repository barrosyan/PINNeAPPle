"""Optuna-based HPO for all timeseries forecasters (neural + classical)."""
from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Tuple, Type

import numpy as np

_OPTUNA_AVAILABLE = False
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Temporal split helper (no data leakage)
# ---------------------------------------------------------------------------

def temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronological 3-way split → (X_tr, y_tr, X_val, y_val, X_te, y_te)."""
    n = len(X)
    i_val  = int(n * (1 - val_frac - test_frac))
    i_test = int(n * (1 - test_frac))
    return (
        X[:i_val],   y[:i_val],
        X[i_val:i_test], y[i_val:i_test],
        X[i_test:], y[i_test:],
    )


# ---------------------------------------------------------------------------
# Classical ML tuner
# ---------------------------------------------------------------------------

_CLASSICAL_SEARCH_SPACES: Dict[str, Callable] = {
    "xgboost": lambda t: {
        "n_estimators":   t.suggest_int("n_estimators", 100, 600),
        "max_depth":      t.suggest_int("max_depth", 3, 10),
        "learning_rate":  t.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample":      t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
    },
    "lightgbm": lambda t: {
        "n_estimators":  t.suggest_int("n_estimators", 100, 600),
        "num_leaves":    t.suggest_int("num_leaves", 20, 150),
        "learning_rate": t.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample":     t.suggest_float("subsample", 0.5, 1.0),
    },
    "random_forest": lambda t: {
        "n_estimators":     t.suggest_int("n_estimators", 50, 400),
        "max_depth":        t.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 20),
    },
    "catboost": lambda t: {
        "iterations":     t.suggest_int("iterations", 100, 600),
        "depth":          t.suggest_int("depth", 3, 10),
        "learning_rate":  t.suggest_float("learning_rate", 1e-3, 0.3, log=True),
    },
}

_CLASSICAL_FACTORY: Dict[str, Any] = {}


def _get_classical_factory():
    from ..models.classical import (
        XGBoostForecaster, LightGBMForecaster,
        RandomForestForecaster, CatBoostForecaster,
    )
    return {
        "xgboost":      XGBoostForecaster,
        "lightgbm":     LightGBMForecaster,
        "random_forest": RandomForestForecaster,
        "catboost":     CatBoostForecaster,
    }


class ClassicalTuner:
    """
    Optuna HPO wrapper for classical ML forecasters.

    Usage::
        tuner = ClassicalTuner("xgboost", n_trials=50)
        best_model, best_params = tuner.fit(X_train, y_train, X_val, y_val)
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        n_trials: int = 30,
        direction: str = "minimize",
        metric: str = "mae",
        timeout: Optional[int] = None,
    ):
        if not _OPTUNA_AVAILABLE:
            raise ImportError("pip install optuna")
        self.model_type = model_type.lower()
        self.n_trials   = n_trials
        self.direction  = direction
        self.metric     = metric
        self.timeout    = timeout
        self.best_params_: Optional[Dict] = None

    def fit(
        self,
        X_tr: np.ndarray, y_tr: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ):
        factories = _get_classical_factory()
        space_fn  = _CLASSICAL_SEARCH_SPACES[self.model_type]
        cls       = factories[self.model_type]

        def objective(trial):
            params = space_fn(trial)
            model  = cls(**params)
            model.fit(X_tr, y_tr)
            y_hat  = model.predict(X_val)
            return float(np.mean(np.abs(y_val - y_hat)))

        study = optuna.create_study(direction=self.direction)
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=False)
        self.best_params_ = study.best_params
        best_model = cls(**self.best_params_)
        best_model.fit(np.vstack([X_tr, X_val]), np.vstack([y_tr, y_val]) if y_tr.ndim > 1 else np.concatenate([y_tr, y_val]))
        return best_model, self.best_params_


# ---------------------------------------------------------------------------
# Neural tuner (PyTorch)
# ---------------------------------------------------------------------------

_NEURAL_SEARCH_SPACES: Dict[str, Callable] = {
    "lstm": lambda t, cfg: {
        "hidden_size": t.suggest_categorical("hidden_size", [64, 128, 256]),
        "num_layers":  t.suggest_int("num_layers", 1, 4),
        "dropout":     t.suggest_float("dropout", 0.0, 0.4),
        "bidirectional": t.suggest_categorical("bidirectional", [False, True]),
    },
    "gru": lambda t, cfg: {
        "hidden_size": t.suggest_categorical("hidden_size", [64, 128, 256]),
        "num_layers":  t.suggest_int("num_layers", 1, 4),
        "dropout":     t.suggest_float("dropout", 0.0, 0.4),
    },
    "tcn": lambda t, cfg: {
        "n_channels":  t.suggest_categorical("n_channels", [32, 64, 128]),
        "n_layers":    t.suggest_int("n_layers", 3, 8),
        "kernel_size": t.suggest_categorical("kernel_size", [3, 5, 7]),
        "dropout":     t.suggest_float("dropout", 0.0, 0.4),
    },
    "nbeats": lambda t, cfg: {
        "n_blocks":    t.suggest_int("n_blocks", 1, 5),
        "n_layers":    t.suggest_int("n_layers", 2, 6),
        "layer_width": t.suggest_categorical("layer_width", [128, 256, 512]),
    },
    "tft": lambda t, cfg: {
        "hidden_size":      t.suggest_categorical("hidden_size", [32, 64, 128]),
        "num_heads":        t.suggest_categorical("num_heads", [2, 4, 8]),
        "num_lstm_layers":  t.suggest_int("num_lstm_layers", 1, 3),
        "dropout":          t.suggest_float("dropout", 0.0, 0.3),
    },
}


def _build_neural(model_type: str, params: Dict, base_cfg: Dict):
    from ..models import LSTMForecaster, GRUForecaster, TCNForecaster, NBeats, TFTForecaster
    from ..models import RecurrentConfig, TCNConfig, NBeatsConfig, TFTConfig
    merged = {**base_cfg, **params}
    if model_type == "lstm":
        return LSTMForecaster(RecurrentConfig(**merged))
    if model_type == "gru":
        return GRUForecaster(RecurrentConfig(**merged))
    if model_type == "tcn":
        return TCNForecaster(TCNConfig(**merged))
    if model_type == "nbeats":
        return NBeats(NBeatsConfig(**merged))
    if model_type == "tft":
        return TFTForecaster(TFTConfig(**merged))
    raise ValueError(f"Unknown neural model type: {model_type}")


class NeuralTuner:
    """
    Optuna HPO for PyTorch forecasters.

    Usage::
        tuner = NeuralTuner("tft", n_trials=20, input_len=64, horizon=16, n_features=5)
        best_model, best_params = tuner.fit(X_tr_t, y_tr_t, X_val_t, y_val_t)
    """

    def __init__(
        self,
        model_type: str = "lstm",
        n_trials: int = 20,
        epochs_per_trial: int = 20,
        lr: float = 1e-3,
        batch_size: int = 64,
        input_len: int = 64,
        horizon: int = 16,
        n_features: int = 1,
        n_targets: int = 1,
        device: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        if not _OPTUNA_AVAILABLE:
            raise ImportError("pip install optuna")
        import torch
        self.model_type      = model_type.lower()
        self.n_trials        = n_trials
        self.epochs          = epochs_per_trial
        self.lr              = lr
        self.batch_size      = batch_size
        self.device          = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.timeout         = timeout
        self.base_cfg        = dict(input_len=input_len, horizon=horizon,
                                    n_features=n_features, n_targets=n_targets)
        self.best_params_: Optional[Dict] = None

    def _train_eval(self, model, X_tr, y_tr, X_val, y_val) -> float:
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        dev   = self.device
        model = model.to(dev)
        opt   = torch.optim.Adam(model.parameters(), lr=self.lr)
        ds    = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                              torch.tensor(y_tr, dtype=torch.float32))
        dl    = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        model.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                loss   = torch.nn.functional.mse_loss(model(xb).y_hat, yb)
                opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            Xv = torch.tensor(X_val, dtype=torch.float32).to(dev)
            yv = torch.tensor(y_val, dtype=torch.float32).to(dev)
            val_loss = float(torch.mean(torch.abs(model(Xv).y_hat - yv)).item())
        return val_loss

    def fit(self, X_tr, y_tr, X_val, y_val):
        space_fn = _NEURAL_SEARCH_SPACES[self.model_type]

        def objective(trial):
            params = space_fn(trial, self.base_cfg)
            model  = _build_neural(self.model_type, params, self.base_cfg)
            return self._train_eval(model, X_tr, y_tr, X_val, y_val)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=False)
        self.best_params_ = study.best_params
        best_model = _build_neural(self.model_type, self.best_params_, self.base_cfg)
        import numpy as np
        X_all = np.concatenate([X_tr, X_val], axis=0)
        y_all = np.concatenate([y_tr, y_val], axis=0)
        self._train_eval(best_model, X_all, y_all, X_val, y_val)
        return best_model, self.best_params_
