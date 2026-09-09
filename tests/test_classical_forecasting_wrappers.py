"""Tests for pinneapple_systems.time_series.classical_wrappers.

Covers the real statsmodels (SARIMAX) and pmdarima (auto_arima) forecaster
wrappers: fitting/forecasting a synthetic AR(1)-like series and checking
that output is real, finite, and sane (not NaN/inf, right shape, roughly
tracks the series level).
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_systems.time_series.classical_wrappers import (
    StatsmodelsARIMAForecaster,
    AutoARIMAForecaster,
)


def _make_ar1_series(n: int = 200, phi: float = 0.7, c: float = 5.0, seed: int = 0) -> np.ndarray:
    """Synthetic AR(1): x_t = c*(1-phi) + phi*x_{t-1} + eps_t, mean-reverting to c."""
    rng = np.random.default_rng(seed)
    x = np.empty(n, dtype=float)
    x[0] = c
    for t in range(1, n):
        x[t] = c * (1 - phi) + phi * x[t - 1] + rng.normal(0, 0.5)
    return x


class TestStatsmodelsARIMAForecaster:
    def setup_method(self):
        pytest.importorskip("statsmodels", reason="StatsmodelsARIMAForecaster requires statsmodels")

    def test_fit_predict_shapes_and_finiteness(self):
        y = _make_ar1_series(n=150, phi=0.7, c=5.0, seed=1)
        f = StatsmodelsARIMAForecaster(order=(1, 0, 0))
        f.fit(y)
        horizon = 12
        y_hat = f.predict(horizon)

        assert isinstance(y_hat, np.ndarray)
        assert y_hat.shape == (horizon,)
        assert np.all(np.isfinite(y_hat))
        # AR(1) mean-reverts to c=5; forecast should land in a sane range,
        # not blow up or collapse to zero.
        assert np.all(np.abs(y_hat) < 100)
        assert np.abs(y_hat.mean() - 5.0) < 5.0

    def test_predict_with_uncertainty(self):
        y = _make_ar1_series(n=150, phi=0.6, c=2.0, seed=2)
        f = StatsmodelsARIMAForecaster(order=(1, 0, 0))
        f.fit(y)
        mean, std = f.predict_with_uncertainty(horizon=10)

        assert mean.shape == (10,)
        assert std.shape == (10,)
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(std))
        assert np.all(std >= 0)
        # Uncertainty should grow (or at least not shrink much) with horizon
        # for a stationary AR(1) approaching its unconditional variance.
        assert std[-1] >= std[0] - 1e-6

    def test_differenced_order_runs(self):
        # d=1 exercises the integration path.
        y = np.cumsum(_make_ar1_series(n=120, phi=0.5, c=0.0, seed=3)) + 10.0
        f = StatsmodelsARIMAForecaster(order=(1, 1, 1))
        f.fit(y)
        y_hat = f.predict(horizon=5)
        assert y_hat.shape == (5,)
        assert np.all(np.isfinite(y_hat))

    def test_predict_before_fit_raises(self):
        f = StatsmodelsARIMAForecaster()
        with pytest.raises(RuntimeError):
            f.predict(5)

    def test_nan_input_raises(self):
        y = _make_ar1_series(n=50)
        y[10] = np.nan
        f = StatsmodelsARIMAForecaster()
        with pytest.raises(ValueError):
            f.fit(y)


class TestAutoARIMAForecaster:
    def setup_method(self):
        pytest.importorskip("pmdarima", reason="AutoARIMAForecaster requires pmdarima")

    def test_fit_predict_selects_order_and_forecasts(self):
        y = _make_ar1_series(n=150, phi=0.75, c=3.0, seed=4)
        f = AutoARIMAForecaster(seasonal=False, max_p=3, max_q=3, max_d=1)
        f.fit(y)

        assert f.order_ is not None
        assert isinstance(f.order_, tuple) and len(f.order_) == 3

        horizon = 15
        y_hat = f.predict(horizon)
        assert y_hat.shape == (horizon,)
        assert np.all(np.isfinite(y_hat))
        assert np.all(np.abs(y_hat) < 100)
        assert np.abs(y_hat.mean() - 3.0) < 5.0

    def test_predict_with_uncertainty(self):
        y = _make_ar1_series(n=150, phi=0.65, c=1.0, seed=5)
        f = AutoARIMAForecaster(seasonal=False, max_p=3, max_q=3, max_d=1)
        f.fit(y)
        mean, std = f.predict_with_uncertainty(horizon=8)

        assert mean.shape == (8,)
        assert std.shape == (8,)
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(std))
        assert np.all(std >= 0)

    def test_predict_before_fit_raises(self):
        f = AutoARIMAForecaster()
        with pytest.raises(RuntimeError):
            f.predict(5)

    def test_nan_input_raises(self):
        y = _make_ar1_series(n=50)
        y[5] = np.nan
        f = AutoARIMAForecaster()
        with pytest.raises(ValueError):
            f.fit(y)


def test_package_level_import():
    """Both forecasters must be exported from the top-level time_series package."""
    from pinneapple_systems.time_series import StatsmodelsARIMAForecaster as PkgSM
    from pinneapple_systems.time_series import AutoARIMAForecaster as PkgAA
    assert PkgSM is StatsmodelsARIMAForecaster
    assert PkgAA is AutoARIMAForecaster
