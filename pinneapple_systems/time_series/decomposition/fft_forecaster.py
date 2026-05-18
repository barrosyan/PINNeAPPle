"""Pure FFT forecaster: fits dominant sinusoids and extrapolates."""
from __future__ import annotations
from typing import Optional

import numpy as np


class FFTForecaster:
    """
    Fits a signal using its top-K dominant frequency components (via FFT)
    and extrapolates them forward.

    Usage::
        model = FFTForecaster(n_harmonics=10)
        model.fit(y_train)
        y_pred = model.predict(horizon=24)
    """

    def __init__(self, n_harmonics: int = 10, detrend: bool = True):
        self.n_harmonics = n_harmonics
        self.detrend     = detrend
        self._freqs:   Optional[np.ndarray] = None
        self._amps:    Optional[np.ndarray] = None
        self._phases:  Optional[np.ndarray] = None
        self._trend_slope:     float = 0.0
        self._trend_intercept: float = 0.0
        self._n:       int = 0
        self._mean:    float = 0.0

    # ------------------------------------------------------------------
    def fit(self, y: np.ndarray) -> "FFTForecaster":
        y  = np.asarray(y, dtype=float).ravel()
        self._n    = len(y)
        self._mean = float(y.mean())

        if self.detrend:
            t = np.arange(self._n, dtype=float)
            slope, intercept = np.polyfit(t, y, 1)
            self._trend_slope     = slope
            self._trend_intercept = intercept
            y = y - (slope * t + intercept)

        fft_vals = np.fft.rfft(y)
        freqs    = np.fft.rfftfreq(self._n)
        amps     = np.abs(fft_vals)

        # Select top-K by amplitude (skip DC)
        idx = np.argsort(amps[1:])[::-1][: self.n_harmonics] + 1
        self._freqs  = freqs[idx]
        self._amps   = amps[idx] * 2.0 / self._n   # two-sided amplitude
        self._phases = np.angle(fft_vals[idx])
        return self

    # ------------------------------------------------------------------
    def predict(self, horizon: int, start: Optional[int] = None) -> np.ndarray:
        if self._freqs is None:
            raise RuntimeError("Call fit() before predict().")
        if start is None:
            start = self._n
        t = np.arange(start, start + horizon, dtype=float)
        signal = np.zeros(horizon)
        for freq, amp, phase in zip(self._freqs, self._amps, self._phases):
            signal += amp * np.cos(2 * np.pi * freq * t + phase)
        if self.detrend:
            signal += self._trend_slope * t + self._trend_intercept
        return signal

    # ------------------------------------------------------------------
    def fit_predict(self, y: np.ndarray, horizon: int) -> np.ndarray:
        self.fit(y)
        return self.predict(horizon)

    # ------------------------------------------------------------------
    def reconstruct(self) -> np.ndarray:
        """Reconstruct the training signal from fitted harmonics."""
        return self.predict(self._n, start=0)
