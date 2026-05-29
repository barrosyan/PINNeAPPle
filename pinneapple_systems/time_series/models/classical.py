"""
Classical ML forecasting models: XGBoost, LightGBM, RandomForest, CatBoost, GPR.

All follow a common ForecastModel interface:
  fit(X, y) → self
  predict(X) → np.ndarray (n_samples, horizon)
  predict_with_uncertainty(X, n_samples) → (mean, std)

Window/lag features are expected to be pre-computed by the caller
(see features/engineering.py and tuning/optuna_tuner.py).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class ForecastModel(ABC):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ForecastModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def predict_with_uncertainty(
        self, X: np.ndarray, n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Default: returns (predict, zeros) — override for probabilistic models."""
        mu = self.predict(X)
        return mu, np.zeros_like(mu)

    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        y_hat = self.predict(X)
        return {
            "mae":  float(mean_absolute_error(y, y_hat)),
            "rmse": float(np.sqrt(mean_squared_error(y, y_hat))),
            "r2":   float(r2_score(y, y_hat)),
        }


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

class XGBoostForecaster(ForecastModel):
    """XGBoost multi-output regressor (wraps MultiOutputRegressor for horizon>1)."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        **kwargs: Any,
    ):
        self._params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            **kwargs,
        )
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostForecaster":
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("pip install xgboost")
        from sklearn.multioutput import MultiOutputRegressor
        base = xgb.XGBRegressor(**self._params, verbosity=0)
        if y.ndim == 1 or y.shape[1] == 1:
            self._model = base
            self._model.fit(X, y.ravel())
        else:
            self._model = MultiOutputRegressor(base, n_jobs=-1)
            self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

class LightGBMForecaster(ForecastModel):
    def __init__(
        self,
        n_estimators: int = 300,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        **kwargs: Any,
    ):
        self._params = dict(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            subsample=subsample,
            verbose=-1,
            **kwargs,
        )
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMForecaster":
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("pip install lightgbm")
        from sklearn.multioutput import MultiOutputRegressor
        base = lgb.LGBMRegressor(**self._params)
        if y.ndim == 1 or y.shape[1] == 1:
            self._model = base
            self._model.fit(X, y.ravel())
        else:
            self._model = MultiOutputRegressor(base, n_jobs=-1)
            self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

class RandomForestForecaster(ForecastModel):
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        **kwargs: Any,
    ):
        self._params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            **kwargs,
        )
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestForecaster":
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor
        base = RandomForestRegressor(**self._params)
        if y.ndim == 1 or y.shape[1] == 1:
            self._model = base
            self._model.fit(X, y.ravel())
        else:
            self._model = MultiOutputRegressor(base)
            self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_with_uncertainty(self, X: np.ndarray, n_samples: int = 100):
        mu = self.predict(X)
        # RF uncertainty: std across individual tree predictions
        if hasattr(self._model, "estimators_"):
            tree_preds = np.stack([t.predict(X) for t in self._model.estimators_], axis=0)
        else:  # MultiOutputRegressor
            tree_preds = np.stack(
                [np.stack([t.predict(X[:, 0:1]) for t in e.estimators_], 0).mean(0)
                 for e in self._model.estimators_], axis=-1
            )
        std = tree_preds.std(axis=0)
        return mu, std


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------

class CatBoostForecaster(ForecastModel):
    def __init__(
        self,
        iterations: int = 300,
        depth: int = 6,
        learning_rate: float = 0.05,
        **kwargs: Any,
    ):
        self._params = dict(
            iterations=iterations, depth=depth,
            learning_rate=learning_rate,
            verbose=0,
            **kwargs,
        )
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CatBoostForecaster":
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            raise ImportError("pip install catboost")
        from sklearn.multioutput import MultiOutputRegressor
        base = CatBoostRegressor(**self._params)
        if y.ndim == 1 or y.shape[1] == 1:
            self._model = base
            self._model.fit(X, y.ravel())
        else:
            self._model = MultiOutputRegressor(base)
            self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


# ---------------------------------------------------------------------------
# GPR (Gaussian Process Regression)
# ---------------------------------------------------------------------------

class GPRForecaster(ForecastModel):
    """
    Gaussian Process Regressor — natively probabilistic.
    Best for small datasets (< ~5000 samples).
    Multi-output: one GP per output dimension.
    """

    def __init__(
        self,
        kernel=None,
        alpha: float = 1e-4,
        normalize_y: bool = True,
        **kwargs: Any,
    ):
        self._kernel = kernel
        self._alpha  = alpha
        self._norm_y = normalize_y
        self._models: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPRForecaster":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        if self._kernel is None:
            k = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-3)
        else:
            k = self._kernel
        y2d = y.reshape(len(y), -1)
        self._models = []
        for i in range(y2d.shape[1]):
            gpr = GaussianProcessRegressor(
                kernel=k, alpha=self._alpha, normalize_y=self._norm_y, n_restarts_optimizer=2
            )
            gpr.fit(X, y2d[:, i])
            self._models.append(gpr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.stack([m.predict(X) for m in self._models], axis=1).squeeze()

    def predict_with_uncertainty(self, X: np.ndarray, n_samples: int = 0):
        means, stds = [], []
        for m in self._models:
            mu, sigma = m.predict(X, return_std=True)
            means.append(mu); stds.append(sigma)
        return np.stack(means, axis=1).squeeze(), np.stack(stds, axis=1).squeeze()


# ---------------------------------------------------------------------------
# MLP (scikit-learn, for non-GPU use)
# ---------------------------------------------------------------------------

class MLPForecaster(ForecastModel):
    def __init__(
        self,
        hidden_layer_sizes: tuple = (128, 64),
        activation: str = "relu",
        max_iter: int = 1000,
        learning_rate_init: float = 1e-3,
        **kwargs: Any,
    ):
        self._params = dict(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            max_iter=max_iter,
            learning_rate_init=learning_rate_init,
            **kwargs,
        )
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPForecaster":
        from sklearn.neural_network import MLPRegressor
        from sklearn.multioutput import MultiOutputRegressor
        base = MLPRegressor(**self._params)
        if y.ndim == 1 or y.shape[1] == 1:
            self._model = base
            self._model.fit(X, y.ravel())
        else:
            self._model = MultiOutputRegressor(base)
            self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
