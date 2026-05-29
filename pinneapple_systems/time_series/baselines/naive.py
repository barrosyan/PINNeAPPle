from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class NaiveForecaster:
    """Predicts the last observed value for all future steps."""
    last_value_: Optional[float] = None

    def fit(self, y: np.ndarray) -> "NaiveForecaster":
        y = np.asarray(y, dtype=float).reshape(-1)
        self.last_value_ = float(y[~np.isnan(y)][-1])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self.last_value_ is None:
            raise RuntimeError("Call fit() before predict().")
        return np.full((int(horizon),), self.last_value_, dtype=float)


@dataclass
class SeasonalNaiveForecaster:
    """Repeats the value from the same season in the last cycle."""
    season_length: int = 12
    history_: Optional[np.ndarray] = None

    def fit(self, y: np.ndarray) -> "SeasonalNaiveForecaster":
        y = np.asarray(y, dtype=float).reshape(-1)
        self.history_ = y.copy()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self.history_ is None:
            raise RuntimeError("Call fit() before predict().")
        h = int(horizon)
        s = int(self.season_length)
        y = self.history_
        out = np.zeros((h,), dtype=float)
        for i in range(h):
            idx = len(y) - s + (i % s)
            out[i] = float(y[idx]) if 0 <= idx < len(y) else float(y[~np.isnan(y)][-1])
        return out


@dataclass
class DriftForecaster:
    """Linearly extrapolates from the first to the last training point."""
    first_: Optional[float] = None
    last_: Optional[float] = None
    n_: Optional[int] = None

    def fit(self, y: np.ndarray) -> "DriftForecaster":
        y = np.asarray(y, dtype=float).reshape(-1)
        yv = y[~np.isnan(y)]
        if len(yv) < 2:
            raise ValueError("DriftForecaster requires at least 2 non-NaN points.")
        self.first_ = float(yv[0])
        self.last_ = float(yv[-1])
        self.n_ = int(len(yv))
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self.first_ is None or self.last_ is None or self.n_ is None:
            raise RuntimeError("Call fit() before predict().")
        h = int(horizon)
        slope = (self.last_ - self.first_) / max(1, (self.n_ - 1))
        return np.asarray([self.last_ + (i + 1) * slope for i in range(h)], dtype=float)