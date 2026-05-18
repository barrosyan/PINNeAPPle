from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


def mae(y_pred, y_true) -> float:
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    return float(np.nanmean(np.abs(y_pred - y_true)))


def rmse(y_pred, y_true) -> float:
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.nanmean((y_pred - y_true) ** 2)))


def mape(y_pred, y_true, eps: float = 1e-8) -> float:
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.nanmean(np.abs((y_pred - y_true) / denom)) * 100.0)


def smape(y_pred, y_true, eps: float = 1e-8) -> float:
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.nanmean(np.abs(y_pred - y_true) / denom) * 100.0)


def mase(y_pred, y_true, y_train, season_length: int = 1, eps: float = 1e-8) -> float:
    """Mean Absolute Scaled Error (MASE)."""
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)

    s = int(season_length)
    if len(y_train) <= s:
        scale = np.nanmean(np.abs(np.diff(y_train)))
    else:
        scale = np.nanmean(np.abs(y_train[s:] - y_train[:-s]))

    scale = float(np.maximum(scale, eps))
    return float(np.nanmean(np.abs(y_pred - y_true)) / scale)


@dataclass
class PointMetrics:
    season_length: int = 1

    def compute(
        self,
        *,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        y_train: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        out = {"mae": mae(y_pred, y_true), "rmse": rmse(y_pred, y_true), "mape": mape(y_pred, y_true), "smape": smape(y_pred, y_true)}
        if y_train is not None:
            out["mase"] = mase(y_pred, y_true, y_train, season_length=self.season_length)
        return out


def multi_horizon_mae(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Overall MAE and MAE per horizon step for (N,H) or (N,H,C)."""
    yp = np.asarray(y_pred, dtype=float)
    yt = np.asarray(y_true, dtype=float)

    if yp.ndim == 3:
        yp = np.nanmean(yp, axis=2)
        yt = np.nanmean(yt, axis=2)

    out: Dict[str, float] = {"mae": float(np.nanmean(np.abs(yp - yt)))}
    H = yp.shape[1]
    for h in range(H):
        out[f"mae_h{h+1}"] = float(np.nanmean(np.abs(yp[:, h] - yt[:, h])))
    return out