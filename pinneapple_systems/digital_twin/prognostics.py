"""Predictive maintenance: trend-based Remaining Useful Life (RUL) estimation.

This module adds a prognostics capability on top of the digital twin's
existing *diagnostic* signals — the anomaly severity scores produced by
``pinneapple_systems.digital_twin.monitoring.anomaly`` (``ThresholdDetector``,
``ZScoreDetector``, ``MahalanobisDetector``, ``ResidualDetector``) and the
rolling state/observation history kept by ``DigitalTwin``/``SystemState``.
Every one of those detectors already emits a scalar ``AnomalyEvent.score``
per timestep where *higher = more anomalous* (see ``monitoring/anomaly.py``);
this module treats that scalar score (or any other user-supplied scalar
"health indicator") as a **degradation indicator time series** and fits a
trend to it in order to extrapolate forward to a failure threshold.

Technique: trend-based / regression-based RUL estimation
----------------------------------------------------------
Given a scalar health/degradation indicator ``h(t)`` sampled at times
``t``, we fit a parametric degradation *path* model and extrapolate it
forward to the time ``t*`` at which it crosses a user-specified
``failure_threshold``. Remaining useful life is then::

    RUL = t* - t_now

Two standard degradation-path models are supported:

- ``"linear"``:      ``h(t) = a + b*t``
- ``"exponential"``: ``h(t) = h0 * exp(k*t)``, fit by ordinary least
  squares on ``ln(h(t)) = ln(h0) + k*t``. The exponential form is the
  textbook model for many wear / fatigue-crack-growth / corrosion
  degradation processes, where the *rate* of degradation grows
  proportionally to the current damage level.

Both cases reduce, after an optional log-transform, to fitting a straight
line ``y = a + b*t`` by OLS and solving for the crossing time
``t* = (y_threshold - a) / b``. This is the same "general path model"
approach to trend-based RUL used throughout the prognostics-and-health-
management (PHM) literature — see e.g.:

    Coble, J. & Hines, J.W., "Applying the General Path Model to
    Estimation of Remaining Useful Life," International Journal of
    Prognostics and Health Management, 2011.

    Meeker, W.Q. & Escobar, L.A., "Statistical Methods for Reliability
    Data," Wiley, 1998 (Ch. 13, degradation/general-path models,
    including the log-linear treatment of exponential degradation).

Uncertainty quantification
---------------------------
The OLS fit yields a parameter covariance matrix for ``(a, b)``. We
propagate that covariance to the crossing time ``t*`` via the **delta
method**: treating ``t* = g(a, b) = (y_threshold - a) / b`` as a smooth
function of the (asymptotically normal) regression coefficients,

    Var(t*) ~= grad(g)^T * Cov(a, b) * grad(g)

with ``grad(g) = (-1/b, -t*/b)``. A two-sided prediction interval for
``t*`` (and hence for RUL) is then ``t* +/- c * sqrt(Var(t*))`` where
``c`` is the Student-t critical value at ``dof = n - 2`` (via SciPy if
available) or, as a documented fallback when SciPy is not importable, the
corresponding standard-normal critical value.

**Limitations, stated explicitly**: the delta method is a first-order
(linearized) approximation — for the exponential model applied to a ratio
of two correlated Gaussian parameters this is not the exact fiducial
interval one would get from e.g. Fieller's theorem, and for very small
samples or near-zero slopes the linearization can understate the true
tail width. This module does not silently pretend otherwise: fit quality
(R^2 in the fitted, i.e. possibly log-transformed, space) and sample size
are both surfaced in ``RULEstimate``, and ``is_extrapolation_reliable``
is set to ``False`` whenever the fit is poor, the sample is small, the
trend has no discernible slope, or the fitted trend does not actually
point toward the failure threshold in the future — precisely so callers
do not over-trust a bad extrapolation.

Usage
-----
Standalone, from plain arrays::

    >>> from pinneapple_systems.digital_twin.prognostics import estimate_rul
    >>> rul = estimate_rul(health_scores, timestamps, failure_threshold=10.0,
    ...                     model="linear", confidence_level=0.90)
    >>> rul.estimated_rul, rul.rul_confidence_interval, rul.is_extrapolation_reliable

Driven by a live ``DigitalTwin``'s accumulated anomaly-detector history::

    >>> from pinneapple_systems.digital_twin.prognostics import estimate_rul_from_twin
    >>> rul = estimate_rul_from_twin(twin, failure_threshold=10.0, detector="zscore")

This module only *consumes* ``AnomalyEvent``/``SystemState`` objects; it
does not modify anomaly detection or state-tracking logic.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy import stats as _scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - SciPy is a repo-wide dependency,
    _SCIPY_AVAILABLE = False  # but this module stays usable without it.


DEGRADATION_MODELS: Tuple[str, ...] = ("linear", "exponential")


# ---------------------------------------------------------------------------
# Normal-quantile fallback (used only when SciPy is unavailable)
# ---------------------------------------------------------------------------

def _norm_ppf(p: float) -> float:
    """Inverse CDF of the standard normal distribution.

    Rational approximation (P.J. Acklam's algorithm), accurate to about
    1.15e-9 over the full (0, 1) domain. Used only as a fallback critical
    value source when SciPy's Student-t quantile function is unavailable;
    it ignores small-sample (finite degrees-of-freedom) corrections, which
    is a documented limitation of that fallback path.
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must lie strictly between 0 and 1")

    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
             ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def _two_sided_critical_value(confidence_level: float, dof: int) -> float:
    """Two-sided critical value for a `confidence_level` prediction interval.

    Uses the Student-t distribution with `dof` degrees of freedom via SciPy
    when available (correct small-sample behaviour); falls back to the
    standard-normal quantile otherwise (see `_norm_ppf` docstring).
    """
    alpha = 1.0 - confidence_level
    q = 1.0 - alpha / 2.0
    if _SCIPY_AVAILABLE and dof > 0:
        return float(_scipy_stats.t.ppf(q, dof))
    return _norm_ppf(q)


# ---------------------------------------------------------------------------
# Health indicator history
# ---------------------------------------------------------------------------

@dataclass
class HealthIndicatorHistory:
    """A scalar health/degradation-indicator time series.

    Light wrapper so `estimate_rul` can be driven either from plain
    arrays (`HealthIndicatorHistory.from_arrays`) or from a live
    `DigitalTwin`'s accumulated anomaly-detector event history
    (`HealthIndicatorHistory.from_twin`), without requiring a separate
    data pipeline.

    Attributes
    ----------
    timestamps : np.ndarray, shape (N,)
        Monotonically increasing sample times (any consistent time unit).
    values : np.ndarray, shape (N,)
        The scalar health/degradation indicator at each timestamp. For
        anomaly-detector-sourced series this is `AnomalyEvent.score`
        (higher = more anomalous / less healthy, per `monitoring/anomaly.py`).
    source : str
        Free-form provenance label (e.g. "custom", "digital_twin.anomaly_monitor").
    """

    timestamps: np.ndarray
    values: np.ndarray
    source: str = "custom"

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=np.float64)
        self.values = np.asarray(self.values, dtype=np.float64)
        if self.timestamps.shape != self.values.shape:
            raise ValueError(
                f"timestamps shape {self.timestamps.shape} != values shape "
                f"{self.values.shape}"
            )
        order = np.argsort(self.timestamps)
        self.timestamps = self.timestamps[order]
        self.values = self.values[order]

    def __len__(self) -> int:
        return int(self.values.shape[0])

    @classmethod
    def from_arrays(
        cls,
        timestamps: Union[Sequence[float], np.ndarray],
        values: Union[Sequence[float], np.ndarray],
        source: str = "custom",
    ) -> "HealthIndicatorHistory":
        return cls(timestamps=np.asarray(timestamps, dtype=np.float64),
                    values=np.asarray(values, dtype=np.float64),
                    source=source)

    @classmethod
    def from_twin(
        cls,
        twin: Any,
        *,
        detector: Optional[str] = None,
        sensor_id: Optional[str] = None,
        field_name: Optional[str] = None,
        n: Optional[int] = None,
    ) -> "HealthIndicatorHistory":
        """Build a health-indicator series from a live `DigitalTwin`.

        Pulls `AnomalyEvent`s from `twin.anomaly_monitor.all_events` (the
        accumulated event log every `AnomalyMonitor` already keeps — see
        `monitoring/anomaly.py`) and uses each event's `.timestamp` /
        `.score`. Optionally filter by `detector` (e.g. "zscore",
        "threshold", "mahalanobis", "residual"), `sensor_id`, and/or
        `field_name`; `n` keeps only the most recent `n` matching events.

        Raises
        ------
        ValueError
            If no matching anomaly events are found (e.g. detector name
            typo, or the twin has not observed any anomalies yet).
        """
        monitor = getattr(twin, "anomaly_monitor", None)
        events = list(getattr(monitor, "all_events", []) or [])

        if detector is not None:
            events = [e for e in events if e.detector == detector]
        if sensor_id is not None:
            events = [e for e in events if e.sensor_id == sensor_id]
        if field_name is not None:
            events = [e for e in events if e.field_name == field_name]

        events = sorted(events, key=lambda e: e.timestamp)
        if n is not None:
            events = events[-n:]

        if not events:
            raise ValueError(
                "No matching AnomalyEvent history found on this DigitalTwin "
                f"(detector={detector!r}, sensor_id={sensor_id!r}, "
                f"field_name={field_name!r}). The twin's anomaly_monitor has "
                f"{len(getattr(monitor, 'all_events', []) or [])} total events."
            )

        ts = np.array([e.timestamp for e in events], dtype=np.float64)
        vals = np.array([e.score for e in events], dtype=np.float64)
        return cls(timestamps=ts, values=vals, source="digital_twin.anomaly_monitor")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class RULEstimate:
    """Result of a trend-based Remaining Useful Life estimation.

    Attributes
    ----------
    estimated_rul : float
        Point estimate of remaining useful life (`failure_time_estimate -
        current_time`), in the same time unit as the input timestamps.
        Can be negative if the fitted trend indicates the health indicator
        has *already* crossed `failure_threshold` as of `current_time`, or
        `float('inf')` if the fitted trend has no slope toward the
        threshold (see `reliability_reasons`).
    rul_confidence_interval : (float, float)
        Two-sided `confidence_level` interval on `estimated_rul`, from
        delta-method propagation of the trend fit's parameter covariance
        (see module docstring for the method and its limitations).
    failure_time_estimate : float
        Absolute time at which the fitted trend crosses `failure_threshold`.
    degradation_model_used : str
        `"linear"` or `"exponential"`.
    fit_quality : float
        R^2 of the OLS fit, in the space it was actually fit (i.e. for the
        exponential model, R^2 of the log-linear fit `ln(h)` vs `t`).
    is_extrapolation_reliable : bool
        `False` whenever the extrapolation should not be trusted at face
        value: too few points, poor fit quality, a degenerate/no-slope
        trend, a non-finite confidence interval, or a trend that does not
        point toward the failure threshold in the future. See
        `reliability_reasons` for exactly which check(s) failed.
    current_time, current_health : float
        The reference time/health level the RUL is measured from.
    failure_threshold, confidence_level : float
        Echoed inputs, for convenience.
    n_points : int
        Number of (timestamp, health) samples the trend was fit on.
    trend_params : dict
        Fitted parameters: `{"a":, "b":}` for linear (`h = a + b*t`), or
        `{"h0":, "k":, "ln_h0":}` for exponential (`h = h0*exp(k*t)`).
    reliability_reasons : list[str]
        Human-readable reasons behind `is_extrapolation_reliable` (empty
        when reliable).
    metadata : dict
        Extra diagnostic info (dof, standard error, critical value, etc).
    """

    estimated_rul: float
    rul_confidence_interval: Tuple[float, float]
    failure_time_estimate: float
    degradation_model_used: str
    fit_quality: float
    is_extrapolation_reliable: bool
    current_time: float
    current_health: float
    failure_threshold: float
    confidence_level: float
    n_points: int
    trend_params: Dict[str, float] = field(default_factory=dict)
    reliability_reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimated_rul": self.estimated_rul,
            "rul_confidence_interval": self.rul_confidence_interval,
            "failure_time_estimate": self.failure_time_estimate,
            "degradation_model_used": self.degradation_model_used,
            "fit_quality": self.fit_quality,
            "is_extrapolation_reliable": self.is_extrapolation_reliable,
            "current_time": self.current_time,
            "current_health": self.current_health,
            "failure_threshold": self.failure_threshold,
            "confidence_level": self.confidence_level,
            "n_points": self.n_points,
            "trend_params": dict(self.trend_params),
            "reliability_reasons": list(self.reliability_reasons),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Core estimation
# ---------------------------------------------------------------------------

def _ols_line(t: np.ndarray, y: np.ndarray) -> Tuple[float, float, np.ndarray, float, float]:
    """Fit y = a + b*t by OLS.

    Returns (a, b, cov_ab (2x2), r2, sse) where cov_ab is the covariance
    matrix of (a, b) estimated from the residual variance (i.e. the usual
    unscaled-design-matrix covariance times s^2 = SSE / (n - 2); zero when
    n <= 2, in which case the fit is a perfect (degenerate) interpolant).
    """
    n = t.shape[0]
    X = np.column_stack([np.ones(n), t])
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Could not fit a trend: timestamps are degenerate (e.g. all "
            "identical), so the design matrix is singular."
        ) from exc

    beta = XtX_inv @ X.T @ y
    a, b = float(beta[0]), float(beta[1])

    resid = y - X @ beta
    sse = float(resid @ resid)
    y_mean = float(y.mean())
    sst = float(((y - y_mean) ** 2).sum())
    if sst > 1e-300:
        r2 = 1.0 - sse / sst
    else:
        # Degenerate: indicator is (numerically) constant.
        r2 = 1.0 if sse < 1e-12 else 0.0

    dof = n - 2
    s2 = sse / dof if dof > 0 else 0.0
    cov_ab = s2 * XtX_inv

    return a, b, cov_ab, r2, sse


def estimate_rul(
    health_indicator_series: Union[Sequence[float], np.ndarray, HealthIndicatorHistory],
    timestamps: Optional[Union[Sequence[float], np.ndarray]] = None,
    failure_threshold: Optional[float] = None,
    model: str = "linear",
    confidence_level: float = 0.90,
    *,
    current_time: Optional[float] = None,
    min_points: int = 4,
    r2_reliable_threshold: float = 0.7,
) -> RULEstimate:
    """Trend-based Remaining Useful Life (RUL) estimation.

    Fits a degradation trend to a scalar health/degradation-indicator time
    series and extrapolates to the time it crosses `failure_threshold`.
    See the module docstring for the underlying method (general-path-model
    trend fitting) and its uncertainty propagation (delta method).

    Parameters
    ----------
    health_indicator_series : array-like or HealthIndicatorHistory
        The scalar health indicator at each sample time (e.g. anomaly
        severity scores — higher = less healthy — from
        `monitoring/anomaly.py`, or any other user-supplied scalar signal
        that trends toward a known failure threshold). May be a
        `HealthIndicatorHistory`, in which case `timestamps` is ignored.
    timestamps : array-like, optional
        Sample times matching `health_indicator_series`. Required unless
        `health_indicator_series` is a `HealthIndicatorHistory`.
    failure_threshold : float
        The health-indicator value that defines failure.
    model : {"linear", "exponential"}
        Degradation path model. `"linear"`: h(t) = a + b*t. `"exponential"`:
        h(t) = h0*exp(k*t) (requires all health values > 0), the standard
        model for wear/fatigue/corrosion-type degradation.
    confidence_level : float
        Two-sided confidence level for `rul_confidence_interval`, e.g. 0.90.
    current_time : float, optional
        Reference "now" to measure RUL from. Defaults to the last
        (largest) timestamp in the series.
    min_points : int
        Minimum number of samples required to consider the extrapolation
        reliable (the fit itself only requires >= 2 points, but very short
        series fail the `is_extrapolation_reliable` check).
    r2_reliable_threshold : float
        Minimum fit R^2 (in the fitted space) to consider the
        extrapolation reliable.

    Returns
    -------
    RULEstimate
    """
    if isinstance(health_indicator_series, HealthIndicatorHistory):
        hist = health_indicator_series
    else:
        if timestamps is None:
            raise ValueError(
                "timestamps is required when health_indicator_series is not "
                "a HealthIndicatorHistory"
            )
        hist = HealthIndicatorHistory.from_arrays(timestamps, health_indicator_series)

    if failure_threshold is None:
        raise ValueError("failure_threshold is required")
    if model not in DEGRADATION_MODELS:
        raise ValueError(f"model must be one of {DEGRADATION_MODELS}, got {model!r}")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be in (0, 1)")

    t = hist.timestamps
    y = hist.values
    n = t.shape[0]
    if n < 2:
        raise ValueError(f"Need at least 2 data points to fit a trend, got {n}")

    if current_time is None:
        current_time = float(t[-1])

    reasons: List[str] = []

    # --- fit in (possibly log-transformed) space -------------------------
    if model == "exponential":
        if np.any(y <= 0):
            raise ValueError(
                "Exponential degradation model requires all health-indicator "
                "values to be strictly positive (fit is done on ln(h)); got "
                f"a non-positive value (min={float(np.min(y)):.4g}). Use "
                "model='linear', or shift/rescale the indicator so it is "
                "positive, e.g. anomaly severity scores which are already >= 0 "
                "may need a small positive floor."
            )
        if failure_threshold <= 0:
            raise ValueError(
                "Exponential degradation model requires a strictly positive "
                "failure_threshold (fit is done on ln(h))."
            )
        yt = np.log(y)
        thresh_t = math.log(float(failure_threshold))
    else:
        yt = y
        thresh_t = float(failure_threshold)

    a, b, cov_ab, r2, sse = _ols_line(t, yt)
    dof = n - 2

    current_level_t = a + b * current_time
    current_health = float(math.exp(current_level_t)) if model == "exponential" else float(current_level_t)

    # --- crossing time + delta-method CI ----------------------------------
    b_is_degenerate = abs(b) < 1e-15 * (1.0 + abs(a))

    if b_is_degenerate:
        reasons.append("fitted trend has ~zero slope; it never crosses the failure threshold")
        t_star = math.inf
        rul = math.inf
        ci = (math.inf, math.inf)
        se_t = math.inf
    else:
        t_star = (thresh_t - a) / b
        rul = t_star - current_time

        grad = np.array([-1.0 / b, -t_star / b])
        var_t = float(grad @ cov_ab @ grad)
        var_t = max(var_t, 0.0)  # guard tiny negative values from fp error
        se_t = math.sqrt(var_t)

        crit = _two_sided_critical_value(confidence_level, dof)
        lo, hi = t_star - crit * se_t, t_star + crit * se_t
        ci = (lo - current_time, hi - current_time)

    # --- reliability checks --------------------------------------------
    if n < min_points:
        reasons.append(f"only {n} data points available (need >= {min_points})")
    if r2 < r2_reliable_threshold:
        reasons.append(f"poor trend fit: R^2={r2:.3f} < {r2_reliable_threshold}")
    if not b_is_degenerate and rul <= 0:
        reasons.append(
            "fitted crossing time is not in the future: health indicator is "
            "already at/beyond the failure threshold as of current_time"
        )
    if not np.isfinite(ci[0]) or not np.isfinite(ci[1]):
        if not b_is_degenerate:
            reasons.append("confidence interval on RUL is non-finite")

    is_reliable = len(reasons) == 0

    if model == "exponential":
        trend_params = {"h0": float(math.exp(a)), "k": float(b), "ln_h0": float(a)}
    else:
        trend_params = {"a": float(a), "b": float(b)}

    metadata = {
        "dof": dof,
        "sse": sse,
        "scipy_used_for_critical_value": bool(_SCIPY_AVAILABLE and dof > 0),
        "fit_space": "log(health)" if model == "exponential" else "health",
    }
    if not b_is_degenerate:
        metadata["standard_error_of_crossing_time"] = se_t

    return RULEstimate(
        estimated_rul=float(rul),
        rul_confidence_interval=(float(ci[0]), float(ci[1])),
        failure_time_estimate=float(t_star),
        degradation_model_used=model,
        fit_quality=float(r2),
        is_extrapolation_reliable=is_reliable,
        current_time=float(current_time),
        current_health=current_health,
        failure_threshold=float(failure_threshold),
        confidence_level=float(confidence_level),
        n_points=n,
        trend_params=trend_params,
        reliability_reasons=reasons,
        metadata=metadata,
    )


def estimate_rul_from_twin(
    twin: Any,
    failure_threshold: float,
    *,
    detector: Optional[str] = None,
    sensor_id: Optional[str] = None,
    field_name: Optional[str] = None,
    n: Optional[int] = None,
    model: str = "linear",
    confidence_level: float = 0.90,
    **kwargs: Any,
) -> RULEstimate:
    """Convenience wrapper: estimate RUL directly from a live `DigitalTwin`.

    Pulls the health-indicator series from the twin's accumulated
    `AnomalyEvent` history (`twin.anomaly_monitor.all_events`, i.e. exactly
    the anomaly severity scores the twin has already been computing — see
    `HealthIndicatorHistory.from_twin`) and forwards to `estimate_rul`.

    This is optional polish on top of the standalone, array-based
    `estimate_rul`, which remains the mandatory core API and works without
    any live `DigitalTwin` at all.

    Parameters
    ----------
    twin : DigitalTwin
        A live (or previously-run) digital twin whose `anomaly_monitor`
        has accumulated `AnomalyEvent`s.
    failure_threshold : float
        Health-indicator (anomaly score) value that defines failure.
    detector, sensor_id, field_name : str, optional
        Filters passed through to `HealthIndicatorHistory.from_twin`.
    n : int, optional
        Keep only the most recent `n` matching anomaly events.
    model, confidence_level, **kwargs
        Forwarded to `estimate_rul`.
    """
    history = HealthIndicatorHistory.from_twin(
        twin, detector=detector, sensor_id=sensor_id, field_name=field_name, n=n
    )
    return estimate_rul(
        history,
        failure_threshold=failure_threshold,
        model=model,
        confidence_level=confidence_level,
        **kwargs,
    )
