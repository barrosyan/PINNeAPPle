"""Real wrappers around the classical statistical forecasting libraries.

`statsmodels` and `pmdarima` are listed in this package's `timeseries` extra
(`pyproject.toml`) but, prior to this module, were only genuinely wired into
statistical *tests* (ADF/KPSS/ACF/PACF in `audit/tests.py`,
`eda/plots.py`) and STL decomposition (`pinneapple_simulation`) — never into
an actual forecasting model. `pinneapple_neural.architectures.classical_ts.ARIMA`
looks like an ARIMA model but is a hand-rolled ridge-regression AR(p) on
Δ^d(x); it does not call `statsmodels` and has no MA component. `pmdarima`
had zero references anywhere in the repo.

This module closes that gap with two forecasters that follow the same
`.fit(y) -> self` / `.predict(horizon) -> np.ndarray` convention used by
`pinneapple_systems.time_series.baselines.naive.NaiveForecaster` and friends:

- `StatsmodelsARIMAForecaster` — real wrapper around
  `statsmodels.tsa.statespace.sarimax.SARIMAX` (a strict superset of
  `statsmodels.tsa.arima.model.ARIMA` that additionally supports seasonal
  orders), giving genuine ARIMA/SARIMA fitting (including the MA terms the
  hand-rolled `classical_ts.ARIMA` omits).
- `AutoARIMAForecaster` — real wrapper around `pmdarima.auto_arima`, which
  automatically selects (p,d,q)(P,D,Q,m) via stepwise search — a genuine
  capability neither the hand-rolled ARIMA nor the plain statsmodels wrapper
  above provides.

Both treat their backing library as optional (try/except ImportError with an
actionable `pip install ...` message), following the pattern already used by
`models/classical.py` (`XGBoostForecaster`, `CatBoostForecaster`, ...).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# statsmodels SARIMAX (covers plain ARIMA via seasonal_order=(0,0,0,0))
# ---------------------------------------------------------------------------

@dataclass
class StatsmodelsARIMAForecaster:
    """Real ARIMA/SARIMA forecaster wrapping `statsmodels`.

    Uses `statsmodels.tsa.statespace.sarimax.SARIMAX`, which fits the same
    ARIMA(p,d,q) family as `statsmodels.tsa.arima.model.ARIMA` but also
    supports an optional seasonal order (P,D,Q,m) — set
    `seasonal_order=(0, 0, 0, 0)` (the default) to fall back to plain
    non-seasonal ARIMA.

    Example:
        >>> f = StatsmodelsARIMAForecaster(order=(2, 1, 1))
        >>> f.fit(y)
        >>> y_hat = f.predict(horizon=10)
    """

    order: Tuple[int, int, int] = (1, 0, 0)
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0)
    trend: Optional[str] = None
    enforce_stationarity: bool = True
    enforce_invertibility: bool = True
    fit_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._result = None
        self._n_obs: Optional[int] = None

    @staticmethod
    def _import_statsmodels():
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError as exc:
            raise ImportError(
                "StatsmodelsARIMAForecaster requires statsmodels. "
                "Install it with: pip install statsmodels "
                "(or: pip install 'pinneapple[timeseries]')"
            ) from exc
        return SARIMAX

    def fit(self, y: np.ndarray) -> "StatsmodelsARIMAForecaster":
        SARIMAX = self._import_statsmodels()
        y = np.asarray(y, dtype=float).reshape(-1)
        if np.isnan(y).any():
            raise ValueError(
                "StatsmodelsARIMAForecaster.fit() received NaNs; impute the "
                "series first (see preparation.imputer.TimeSeriesImputer)."
            )
        if len(y) < max(sum(self.order), 2) + 1:
            raise ValueError(
                f"Series too short (n={len(y)}) for order={self.order}."
            )
        model = SARIMAX(
            y,
            order=tuple(self.order),
            seasonal_order=tuple(self.seasonal_order),
            trend=self.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )
        self._result = model.fit(disp=False, **self.fit_kwargs)
        self._n_obs = len(y)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("Call fit() before predict().")
        h = int(horizon)
        forecast = self._result.get_forecast(steps=h)
        return np.asarray(forecast.predicted_mean, dtype=float).reshape(h)

    def predict_with_uncertainty(
        self, horizon: int, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (mean, std) using the model's analytic forecast variance."""
        if self._result is None:
            raise RuntimeError("Call fit() before predict_with_uncertainty().")
        h = int(horizon)
        forecast = self._result.get_forecast(steps=h)
        mean = np.asarray(forecast.predicted_mean, dtype=float).reshape(h)
        se = np.asarray(forecast.se_mean, dtype=float).reshape(h)
        return mean, se

    def summary(self) -> str:
        if self._result is None:
            raise RuntimeError("Call fit() before summary().")
        return str(self._result.summary())


# ---------------------------------------------------------------------------
# pmdarima auto_arima — automatic order selection
# ---------------------------------------------------------------------------

@dataclass
class AutoARIMAForecaster:
    """Real forecaster wrapping `pmdarima.auto_arima`.

    Automatically selects the (p,d,q) — and, if `seasonal=True`, seasonal
    (P,D,Q,m) — order via stepwise AIC-minimizing search, then fits the
    resulting SARIMAX model. This is a genuine differentiator from both the
    hand-rolled `classical_ts.ARIMA` (fixed order, no search) and
    `StatsmodelsARIMAForecaster` above (also a fixed, user-specified order).

    Example:
        >>> f = AutoARIMAForecaster(seasonal=False)
        >>> f.fit(y)
        >>> y_hat = f.predict(horizon=10)
        >>> f.order_   # the (p, d, q) auto_arima selected
    """

    seasonal: bool = False
    m: int = 1  # seasonal period; only used when seasonal=True
    max_p: int = 5
    max_q: int = 5
    max_d: int = 2
    max_P: int = 2
    max_Q: int = 2
    max_D: int = 1
    stepwise: bool = True
    suppress_warnings: bool = True
    auto_arima_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._model = None
        self.order_: Optional[Tuple[int, int, int]] = None
        self.seasonal_order_: Optional[Tuple[int, int, int, int]] = None

    @staticmethod
    def _import_pmdarima():
        try:
            import pmdarima as pm
        except ImportError as exc:
            raise ImportError(
                "AutoARIMAForecaster requires pmdarima. "
                "Install it with: pip install pmdarima "
                "(or: pip install 'pinneapple[timeseries]')"
            ) from exc
        return pm

    def fit(self, y: np.ndarray) -> "AutoARIMAForecaster":
        pm = self._import_pmdarima()
        y = np.asarray(y, dtype=float).reshape(-1)
        if np.isnan(y).any():
            raise ValueError(
                "AutoARIMAForecaster.fit() received NaNs; impute the series "
                "first (see preparation.imputer.TimeSeriesImputer)."
            )
        min_len = 10 if not self.seasonal else max(10, 2 * self.m)
        if len(y) < min_len:
            raise ValueError(
                f"Series too short (n={len(y)}) for auto_arima "
                f"(seasonal={self.seasonal}, m={self.m})."
            )
        self._model = pm.auto_arima(
            y,
            seasonal=self.seasonal,
            m=self.m if self.seasonal else 1,
            max_p=self.max_p,
            max_q=self.max_q,
            max_d=self.max_d,
            max_P=self.max_P,
            max_Q=self.max_Q,
            max_D=self.max_D,
            stepwise=self.stepwise,
            suppress_warnings=self.suppress_warnings,
            error_action="ignore",
            **self.auto_arima_kwargs,
        )
        self.order_ = tuple(self._model.order)
        self.seasonal_order_ = tuple(self._model.seasonal_order)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        h = int(horizon)
        y_hat = self._model.predict(n_periods=h)
        return np.asarray(y_hat, dtype=float).reshape(h)

    def predict_with_uncertainty(
        self, horizon: int, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (mean, std) derived from pmdarima's confidence intervals."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict_with_uncertainty().")
        h = int(horizon)
        y_hat, conf_int = self._model.predict(n_periods=h, return_conf_int=True, alpha=alpha)
        y_hat = np.asarray(y_hat, dtype=float).reshape(h)
        conf_int = np.asarray(conf_int, dtype=float)
        # Approximate std from a (1 - alpha) symmetric CI half-width.
        from scipy.stats import norm
        z = norm.ppf(1 - alpha / 2.0)
        half_width = (conf_int[:, 1] - conf_int[:, 0]) / 2.0
        std = half_width / z
        return y_hat, std

    def summary(self) -> str:
        if self._model is None:
            raise RuntimeError("Call fit() before summary().")
        return str(self._model.summary())


__all__ = ["StatsmodelsARIMAForecaster", "AutoARIMAForecaster"]
