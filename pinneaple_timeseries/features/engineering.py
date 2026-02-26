from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np


def make_lags(y: np.ndarray, lags: Sequence[int]) -> Dict[str, np.ndarray]:
    """Create lag features for a 1D target series."""
    y = np.asarray(y, dtype=float).reshape(-1)
    out: Dict[str, np.ndarray] = {}
    for lag in lags:
        lag = int(lag)
        if lag <= 0:
            continue
        feat = np.full_like(y, np.nan, dtype=float)
        feat[lag:] = y[:-lag]
        out[f"lag_{lag}"] = feat
    return out


def rolling_stats(
    y: np.ndarray,
    windows: Sequence[int],
    stats: Sequence[str] = ("mean", "std", "min", "max"),
) -> Dict[str, np.ndarray]:
    """Compute rolling statistics over the target series."""
    y = np.asarray(y, dtype=float).reshape(-1)
    out: Dict[str, np.ndarray] = {}
    for w in windows:
        w = int(w)
        if w <= 1:
            continue
        for stat in stats:
            arr = np.full_like(y, np.nan, dtype=float)
            for i in range(w - 1, len(y)):
                seg = y[i - w + 1 : i + 1]
                if stat == "mean":
                    arr[i] = float(np.nanmean(seg))
                elif stat == "std":
                    arr[i] = float(np.nanstd(seg))
                elif stat == "min":
                    arr[i] = float(np.nanmin(seg))
                elif stat == "max":
                    arr[i] = float(np.nanmax(seg))
                else:
                    raise ValueError(f"Unknown rolling stat: {stat}")
            out[f"roll_{stat}_{w}"] = arr
    return out


def fourier_features(t: np.ndarray, periods: Sequence[float], K: int = 1) -> Dict[str, np.ndarray]:
    """Fourier seasonality features: sin/cos harmonics for each period."""
    t = np.asarray(t, dtype=float).reshape(-1)
    out: Dict[str, np.ndarray] = {}
    for period in periods:
        period = float(period)
        for k in range(1, int(K) + 1):
            angle = 2.0 * np.pi * k * t / period
            out[f"sin_P{period:g}_k{k}"] = np.sin(angle)
            out[f"cos_P{period:g}_k{k}"] = np.cos(angle)
    return out


@dataclass
class TSFeatureEngineer:
    """Convenience wrapper to generate a feature dictionary."""
    lags: Sequence[int] = (1, 2, 3, 12, 24)
    rolling_windows: Sequence[int] = (3, 7, 14, 30)
    rolling_stats_set: Sequence[str] = ("mean", "std")
    fourier_periods: Sequence[float] = (12.0,)
    fourier_K: int = 2

    def transform(self, *, y: np.ndarray, t: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        features: Dict[str, np.ndarray] = {}
        features.update(make_lags(y, self.lags))
        features.update(rolling_stats(y, self.rolling_windows, stats=self.rolling_stats_set))
        if t is not None and self.fourier_periods:
            features.update(fourier_features(t, self.fourier_periods, K=self.fourier_K))
        return features