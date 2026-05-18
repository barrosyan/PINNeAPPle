from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


def pinball_loss(y_pred_q: np.ndarray, y_true: np.ndarray, q: float) -> float:
    y_pred_q = np.asarray(y_pred_q, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    q = float(q)
    e = y_true - y_pred_q
    return float(np.nanmean(np.maximum(q * e, (q - 1.0) * e)))


def coverage(y_lo: np.ndarray, y_hi: np.ndarray, y_true: np.ndarray) -> float:
    y_lo = np.asarray(y_lo, dtype=float)
    y_hi = np.asarray(y_hi, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    return float(np.nanmean((y_true >= y_lo) & (y_true <= y_hi)))


def mean_interval_width(y_lo: np.ndarray, y_hi: np.ndarray) -> float:
    y_lo = np.asarray(y_lo, dtype=float)
    y_hi = np.asarray(y_hi, dtype=float)
    return float(np.nanmean(y_hi - y_lo))


@dataclass
class ProbabilisticMetrics:
    quantiles: Sequence[float] = (0.1, 0.5, 0.9)

    def compute(self, *, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Compute pinball loss per quantile and basic interval metrics."""
        yp = np.asarray(y_pred, dtype=float)
        yt = np.asarray(y_true, dtype=float)

        if yp.ndim == 2:
            yp = yp[:, None, :]
        if yt.ndim == 1:
            yt = yt[:, None]
        if yp.ndim != 3:
            raise ValueError(f"Expected y_pred with 3 dims after normalization, got {yp.shape}")

        out: Dict[str, float] = {}
        qs = list(self.quantiles)

        for qi, q in enumerate(qs):
            out[f"pinball_q{q:g}"] = pinball_loss(yp[:, :, qi], yt, q)

        if len(qs) >= 2:
            lo_i = int(np.argmin(qs))
            hi_i = int(np.argmax(qs))
            out["coverage"] = coverage(yp[:, :, lo_i], yp[:, :, hi_i], yt)
            out["mean_interval_width"] = mean_interval_width(yp[:, :, lo_i], yp[:, :, hi_i])

        return out