"""Tests for ``pinneapple_systems.digital_twin.prognostics`` (trend-based RUL).

Validates the core, standalone ``estimate_rul`` API against synthetic
degradation series with an analytically-known crossing time: a known
linear trend plus noise, and a known exponential trend plus noise. Also
checks the ``is_extrapolation_reliable`` flag behaves honestly (False for
short/noisy/poor-fit series, True for clean well-fit ones), and exercises
the optional ``estimate_rul_from_twin`` convenience wrapper against a
minimal object that mimics the ``DigitalTwin.anomaly_monitor.all_events``
shape (a live ``DigitalTwin`` needs a trained model/streams which is out
of scope here; the wrapper only reads ``twin.anomaly_monitor.all_events``,
so a small stand-in is enough to exercise it honestly).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pinneapple_systems.digital_twin.monitoring.anomaly import AnomalyEvent, AnomalyMonitor
from pinneapple_systems.digital_twin.prognostics import (
    HealthIndicatorHistory,
    RULEstimate,
    estimate_rul,
    estimate_rul_from_twin,
)


# ---------------------------------------------------------------------------
# Linear degradation
# ---------------------------------------------------------------------------

def test_linear_rul_recovers_known_crossing_time_low_noise():
    """h(t) = a + b*t with small noise: RUL should be recovered accurately
    and the CI should bracket the true crossing time."""
    rng = np.random.default_rng(0)
    a_true, b_true = 1.0, 0.5
    t = np.linspace(0.0, 20.0, 60)
    noise_std = 0.05
    h = a_true + b_true * t + rng.normal(scale=noise_std, size=t.shape)

    threshold = 15.0  # above the series' final value (h(20) = 11), so the
    # crossing is genuinely in the future relative to current_time = t[-1].
    true_t_star = (threshold - a_true) / b_true  # = 28.0
    current_time = float(t[-1])
    true_rul = true_t_star - current_time  # = 8.0

    result = estimate_rul(h, t, failure_threshold=threshold, model="linear",
                           confidence_level=0.90)

    assert isinstance(result, RULEstimate)
    assert result.degradation_model_used == "linear"
    assert result.n_points == len(t)
    assert result.fit_quality > 0.95

    assert abs(result.estimated_rul - true_rul) < 1.0, (
        f"estimated RUL {result.estimated_rul} vs true {true_rul}"
    )

    lo, hi = result.rul_confidence_interval
    assert lo <= result.estimated_rul <= hi
    # True RUL should fall inside (or very near) the 90% interval for a
    # well-behaved, low-noise linear fit.
    assert lo - 0.5 <= true_rul <= hi + 0.5

    assert result.is_extrapolation_reliable is True
    assert result.reliability_reasons == []


def test_linear_rul_failure_time_estimate_matches_analytic_solution():
    """Sanity check on failure_time_estimate directly against the closed-form
    crossing time of a noiseless linear series (perfect fit, R^2 == 1)."""
    a_true, b_true = 2.0, 1.5
    t = np.linspace(0.0, 10.0, 20)
    h = a_true + b_true * t  # no noise

    threshold = 25.0
    true_t_star = (threshold - a_true) / b_true

    result = estimate_rul(h, t, failure_threshold=threshold, model="linear")

    assert result.fit_quality == pytest.approx(1.0, abs=1e-6)
    assert result.failure_time_estimate == pytest.approx(true_t_star, abs=1e-6)
    assert result.trend_params["a"] == pytest.approx(a_true, abs=1e-6)
    assert result.trend_params["b"] == pytest.approx(b_true, abs=1e-6)


# ---------------------------------------------------------------------------
# Exponential degradation
# ---------------------------------------------------------------------------

def test_exponential_rul_recovers_known_crossing_time_low_noise():
    """h(t) = h0*exp(k*t) with small multiplicative noise: RUL should be
    recovered accurately via the log-linear fit."""
    rng = np.random.default_rng(1)
    h0_true, k_true = 1.0, 0.15
    t = np.linspace(0.0, 20.0, 60)
    clean = h0_true * np.exp(k_true * t)
    # Small multiplicative (log-normal) noise keeps values positive, as
    # required by the exponential model's log-transform.
    h = clean * np.exp(rng.normal(scale=0.03, size=t.shape))

    threshold = 30.0  # above the series' final value (h(20) ~= 20.1), so the
    # crossing is genuinely in the future relative to current_time = t[-1].
    true_t_star = math.log(threshold / h0_true) / k_true
    current_time = float(t[-1])
    true_rul = true_t_star - current_time

    result = estimate_rul(h, t, failure_threshold=threshold, model="exponential",
                           confidence_level=0.90)

    assert result.degradation_model_used == "exponential"
    assert result.fit_quality > 0.95
    assert abs(result.estimated_rul - true_rul) < 1.5, (
        f"estimated RUL {result.estimated_rul} vs true {true_rul}"
    )

    lo, hi = result.rul_confidence_interval
    assert lo <= result.estimated_rul <= hi
    assert lo - 0.75 <= true_rul <= hi + 0.75

    assert result.is_extrapolation_reliable is True
    assert result.trend_params["k"] == pytest.approx(k_true, rel=0.2)
    assert result.trend_params["h0"] == pytest.approx(h0_true, rel=0.3)


def test_exponential_model_requires_positive_values():
    t = np.linspace(0.0, 10.0, 10)
    h = np.linspace(-1.0, 5.0, 10)  # contains non-positive values
    with pytest.raises(ValueError, match="strictly positive"):
        estimate_rul(h, t, failure_threshold=10.0, model="exponential")


# ---------------------------------------------------------------------------
# Reliability flag behaviour
# ---------------------------------------------------------------------------

def test_reliability_flag_false_for_noisy_poor_fit_series():
    """A health indicator dominated by noise (near-zero true trend, large
    noise) should produce a poor fit and be flagged unreliable."""
    rng = np.random.default_rng(2)
    t = np.linspace(0.0, 20.0, 40)
    h = 5.0 + rng.normal(scale=5.0, size=t.shape)  # no real trend, huge noise

    result = estimate_rul(h, t, failure_threshold=100.0, model="linear")

    assert result.is_extrapolation_reliable is False
    assert len(result.reliability_reasons) > 0


def test_reliability_flag_false_for_too_short_series():
    """Even a perfect 2-3 point fit should be flagged unreliable: too few
    points to trust an extrapolation (default min_points=4)."""
    t = np.array([0.0, 1.0, 2.0])
    h = 1.0 + 2.0 * t  # perfect linear trend, R^2 == 1

    result = estimate_rul(h, t, failure_threshold=50.0, model="linear")

    assert result.fit_quality == pytest.approx(1.0, abs=1e-6)
    assert result.n_points == 3
    assert result.is_extrapolation_reliable is False
    assert any("data points" in r for r in result.reliability_reasons)


def test_reliability_flag_true_for_clean_well_fit_series():
    rng = np.random.default_rng(3)
    t = np.linspace(0.0, 30.0, 50)
    h = 0.5 + 0.3 * t + rng.normal(scale=0.02, size=t.shape)

    result = estimate_rul(h, t, failure_threshold=20.0, model="linear")

    assert result.fit_quality > 0.99
    assert result.n_points >= 4
    assert result.is_extrapolation_reliable is True
    assert result.reliability_reasons == []


def test_reliability_flag_false_when_trend_moves_away_from_threshold():
    """A trend that is decreasing while the failure threshold is *above*
    the current level (or vice versa) never reaches the threshold in the
    future; RUL should be flagged unreliable rather than silently wrong."""
    t = np.linspace(0.0, 10.0, 20)
    h = 10.0 - 0.5 * t  # decreasing trend

    # Threshold is above the current (and increasing-in-the-past) level,
    # so the fitted line's crossing point lies in the past, not the future.
    result = estimate_rul(h, t, failure_threshold=50.0, model="linear")

    assert result.is_extrapolation_reliable is False
    assert result.estimated_rul <= 0
    assert any("not in the future" in r for r in result.reliability_reasons)


def test_zero_slope_series_flagged_unreliable_with_infinite_rul():
    t = np.linspace(0.0, 10.0, 10)
    h = np.full_like(t, 5.0)  # perfectly flat, no trend

    result = estimate_rul(h, t, failure_threshold=50.0, model="linear")

    assert math.isinf(result.estimated_rul)
    assert result.is_extrapolation_reliable is False


# ---------------------------------------------------------------------------
# HealthIndicatorHistory
# ---------------------------------------------------------------------------

def test_health_indicator_history_sorts_by_timestamp():
    ts = [3.0, 1.0, 2.0]
    vals = [30.0, 10.0, 20.0]
    hist = HealthIndicatorHistory.from_arrays(ts, vals)

    assert list(hist.timestamps) == [1.0, 2.0, 3.0]
    assert list(hist.values) == [10.0, 20.0, 30.0]
    assert len(hist) == 3


def test_estimate_rul_accepts_health_indicator_history_directly():
    t = np.linspace(0.0, 10.0, 20)
    h = 1.0 + 2.0 * t
    hist = HealthIndicatorHistory.from_arrays(t, h)

    result = estimate_rul(hist, failure_threshold=15.0, model="linear")
    assert result.failure_time_estimate == pytest.approx(7.0, abs=1e-6)


def test_estimate_rul_requires_timestamps_for_plain_arrays():
    with pytest.raises(ValueError):
        estimate_rul([1.0, 2.0, 3.0], failure_threshold=10.0)


# ---------------------------------------------------------------------------
# estimate_rul_from_twin (optional convenience wrapper)
# ---------------------------------------------------------------------------

class _FakeTwin:
    """Minimal stand-in exposing only what estimate_rul_from_twin reads:
    ``twin.anomaly_monitor.all_events`` (a real AnomalyMonitor instance,
    populated exactly as DigitalTwin populates it in twin.py)."""

    def __init__(self) -> None:
        self.anomaly_monitor = AnomalyMonitor()


def test_estimate_rul_from_twin_uses_anomaly_event_history():
    twin = _FakeTwin()
    rng = np.random.default_rng(4)
    t0 = 1_000.0
    # Simulate a growing anomaly severity score over time, as ZScoreDetector
    # would append to AnomalyMonitor.all_events on each check().
    for i in range(30):
        ts = t0 + i * 1.0
        score = 1.0 + 0.4 * i + rng.normal(scale=0.05)
        ev = AnomalyEvent(
            timestamp=ts, sensor_id="pump_1", field_name="p",
            observed=0.0, predicted=0.0, score=float(score), detector="zscore",
        )
        twin.anomaly_monitor.all_events.append(ev)

    result = estimate_rul_from_twin(twin, failure_threshold=20.0, detector="zscore")

    assert isinstance(result, RULEstimate)
    assert result.n_points == 30
    assert result.degradation_model_used == "linear"
    assert result.estimated_rul > 0
    assert result.is_extrapolation_reliable is True


def test_estimate_rul_from_twin_raises_when_no_matching_events():
    twin = _FakeTwin()
    with pytest.raises(ValueError, match="No matching AnomalyEvent"):
        estimate_rul_from_twin(twin, failure_threshold=10.0, detector="threshold")


def test_estimate_rul_from_twin_filters_by_detector_name():
    twin = _FakeTwin()
    for i in range(10):
        twin.anomaly_monitor.all_events.append(
            AnomalyEvent(timestamp=float(i), sensor_id="s", field_name="f",
                         observed=0.0, predicted=0.0, score=float(i),
                         detector="zscore")
        )
    for i in range(10):
        twin.anomaly_monitor.all_events.append(
            AnomalyEvent(timestamp=float(i), sensor_id="s", field_name="f",
                         observed=0.0, predicted=0.0, score=float(100 + i),
                         detector="threshold")
        )

    hist = HealthIndicatorHistory.from_twin(twin, detector="zscore")
    assert len(hist) == 10
    assert hist.values.max() < 20.0
