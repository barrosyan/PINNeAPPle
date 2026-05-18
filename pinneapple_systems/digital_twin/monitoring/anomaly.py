"""Anomaly detection for digital twins.

Compares model predictions against real-time observations and flags
deviations beyond configurable thresholds.

Algorithms available
--------------------
- ThresholdDetector   : simple upper/lower band check
- ZScoreDetector      : rolling z-score against a baseline window
- MahalanobisDetector : multivariate Mahalanobis distance (covariance-aware)
- ResidualDetector    : checks innovation residuals from the Kalman filter
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnomalyEvent:
    """Records a detected anomaly."""

    timestamp: float
    sensor_id: str
    field_name: str
    observed: float
    predicted: float
    score: float                          # anomaly score (higher = more anomalous)
    detector: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ThresholdDetector:
    """
    Simple threshold-based anomaly detector.

    Flags observations where |obs - pred| > threshold (absolute) or
    |obs - pred| / (|pred| + eps) > rel_threshold (relative).
    """

    def __init__(
        self,
        thresholds: Dict[str, float],
        rel_thresholds: Optional[Dict[str, float]] = None,
        eps: float = 1e-8,
    ) -> None:
        self.thresholds = thresholds
        self.rel_thresholds = rel_thresholds or {}
        self.eps = float(eps)
        self.events: List[AnomalyEvent] = []

    def check(
        self,
        timestamp: float,
        sensor_id: str,
        observed: Dict[str, float],
        predicted: Dict[str, float],
    ) -> List[AnomalyEvent]:
        new_events: List[AnomalyEvent] = []
        for fname, obs_val in observed.items():
            pred_val = predicted.get(fname, float("nan"))
            if np.isnan(pred_val):
                continue
            diff = abs(obs_val - pred_val)
            thresh = self.thresholds.get(fname, float("inf"))
            rel_thresh = self.rel_thresholds.get(fname, float("inf"))
            rel_diff = diff / (abs(pred_val) + self.eps)
            if diff > thresh or rel_diff > rel_thresh:
                ev = AnomalyEvent(
                    timestamp=timestamp,
                    sensor_id=sensor_id,
                    field_name=fname,
                    observed=obs_val,
                    predicted=pred_val,
                    score=max(diff / max(thresh, 1e-12), rel_diff / max(rel_thresh, 1e-12)),
                    detector="threshold",
                )
                new_events.append(ev)
                logger.warning(
                    f"[ANOMALY] {sensor_id}/{fname}: obs={obs_val:.4g} "
                    f"pred={pred_val:.4g} diff={diff:.4g} (score={ev.score:.2f})"
                )
        self.events.extend(new_events)
        return new_events


class ZScoreDetector:
    """
    Rolling z-score anomaly detector.

    Maintains a window of residuals (obs - pred) and flags when
    the current residual exceeds z_threshold standard deviations
    from the window mean.
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        window_size: int = 100,
        min_samples: int = 10,
    ) -> None:
        self.z_threshold = float(z_threshold)
        self.window_size = int(window_size)
        self.min_samples = int(min_samples)
        self._windows: Dict[str, Deque[float]] = {}
        self.events: List[AnomalyEvent] = []

    def _get_window(self, key: str) -> Deque[float]:
        if key not in self._windows:
            self._windows[key] = deque(maxlen=self.window_size)
        return self._windows[key]

    def check(
        self,
        timestamp: float,
        sensor_id: str,
        observed: Dict[str, float],
        predicted: Dict[str, float],
    ) -> List[AnomalyEvent]:
        new_events: List[AnomalyEvent] = []
        for fname, obs_val in observed.items():
            pred_val = predicted.get(fname, float("nan"))
            if np.isnan(pred_val):
                continue
            residual = obs_val - pred_val
            key = f"{sensor_id}/{fname}"
            win = self._get_window(key)
            win.append(residual)

            if len(win) < self.min_samples:
                continue

            arr = np.array(win)
            mu, sigma = arr.mean(), arr.std()
            if sigma < 1e-12:
                continue
            z = abs((residual - mu) / sigma)
            if z > self.z_threshold:
                ev = AnomalyEvent(
                    timestamp=timestamp,
                    sensor_id=sensor_id,
                    field_name=fname,
                    observed=obs_val,
                    predicted=pred_val,
                    score=float(z),
                    detector="zscore",
                    metadata={"z": float(z), "mu": float(mu), "sigma": float(sigma)},
                )
                new_events.append(ev)
                logger.warning(
                    f"[ANOMALY/ZScore] {key}: z={z:.2f} > threshold={self.z_threshold}"
                )
        self.events.extend(new_events)
        return new_events


class MahalanobisDetector:
    """
    Multivariate Mahalanobis distance anomaly detector.

    Uses the state covariance P from the Kalman filter to compute
    the Mahalanobis distance of the innovation vector:
        d = sqrt( delta^T P^{-1} delta )

    Thresholds can be set using chi-squared quantiles.
    """

    def __init__(self, threshold: float = 3.0) -> None:
        self.threshold = float(threshold)
        self.events: List[AnomalyEvent] = []

    def check_vector(
        self,
        timestamp: float,
        delta: np.ndarray,
        covariance: np.ndarray,
        *,
        sensor_id: str = "system",
        field_name: str = "state",
    ) -> Optional[AnomalyEvent]:
        """
        Check a residual vector delta against covariance.

        delta : (n,) innovation vector
        covariance : (n,n) covariance matrix
        """
        try:
            cov_inv = np.linalg.inv(covariance + np.eye(len(delta)) * 1e-12)
            d = float(np.sqrt(delta @ cov_inv @ delta))
        except np.linalg.LinAlgError:
            return None

        if d > self.threshold:
            ev = AnomalyEvent(
                timestamp=timestamp,
                sensor_id=sensor_id,
                field_name=field_name,
                observed=float("nan"),
                predicted=float("nan"),
                score=d,
                detector="mahalanobis",
                metadata={"mahalanobis_distance": d, "threshold": self.threshold},
            )
            self.events.append(ev)
            logger.warning(
                f"[ANOMALY/Mahalanobis] {sensor_id}: d={d:.2f} > threshold={self.threshold}"
            )
            return ev
        return None


class AnomalyMonitor:
    """
    Composite anomaly monitor that runs multiple detectors in sequence.

    Usage
    -----
    >>> monitor = AnomalyMonitor()
    >>> monitor.add_detector(ThresholdDetector({"p": 500.0}))
    >>> monitor.add_detector(ZScoreDetector(z_threshold=3.0))
    >>> events = monitor.check(ts, "sensor_01", obs, pred)
    """

    def __init__(self) -> None:
        self._detectors: List[Any] = []
        self.all_events: List[AnomalyEvent] = []

    def add_detector(self, detector: Any) -> None:
        self._detectors.append(detector)

    def check(
        self,
        timestamp: float,
        sensor_id: str,
        observed: Dict[str, float],
        predicted: Dict[str, float],
    ) -> List[AnomalyEvent]:
        events: List[AnomalyEvent] = []
        for det in self._detectors:
            if hasattr(det, "check"):
                evts = det.check(timestamp, sensor_id, observed, predicted)
                events.extend(evts)
        self.all_events.extend(events)
        return events

    def recent_events(self, n: int = 20) -> List[AnomalyEvent]:
        return self.all_events[-n:]

    def clear(self) -> None:
        self.all_events.clear()
        for det in self._detectors:
            if hasattr(det, "events"):
                det.events.clear()
