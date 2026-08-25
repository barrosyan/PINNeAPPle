"""Hilbert-Huang Transform + Neural Network on residual decomposition forecaster."""
from __future__ import annotations
from typing import Optional, Any

import numpy as np


class HHTNNForecaster:
    """
    Two-stage forecaster:
      1. EMD (Empirical Mode Decomposition) decomposes the signal into IMFs.
         Dominant IMFs are extrapolated via Hilbert instantaneous frequency.
      2. A neural network models the residual (sum of discarded IMFs + trend).

    Dependencies: PyEMD (pip install EMD-signal), scipy.

    Usage::
        model = HHTNNForecaster(n_imfs=5, input_len=32, horizon=16)
        model.fit(y_train)
        y_pred = model.predict()
    """

    def __init__(
        self,
        n_imfs: int = 5,
        input_len: int = 32,
        horizon: int = 16,
        nn_model: Optional[Any] = None,
        nn_epochs: int = 50,
        nn_lr: float = 1e-3,
    ):
        self.n_imfs    = n_imfs
        self.input_len = input_len
        self.horizon   = horizon
        self.nn_epochs = nn_epochs
        self.nn_lr     = nn_lr
        self._nn_model = nn_model
        self._imfs: Optional[np.ndarray] = None
        self._inst_freqs: Optional[np.ndarray] = None
        self._inst_amps:  Optional[np.ndarray] = None
        self._residual_nn = None
        self._y_train: Optional[np.ndarray] = None
        self._n_fit: int = 0
        self._trend: Optional[np.ndarray] = None
        self._trend_slope: float = 0.0
        self._trend_intercept: float = 0.0

    # ------------------------------------------------------------------
    def _default_nn(self):
        from ..models.classical import MLPForecaster
        return MLPForecaster(
            hidden_layer_sizes=(128, 64),
            max_iter=self.nn_epochs,
            learning_rate_init=self.nn_lr,
        )

    def _emd_decompose(self, y: np.ndarray):
        try:
            from PyEMD import EMD
        except ImportError:
            raise ImportError("pip install EMD-signal")
        emd  = EMD()
        imfs = emd(y)  # shape (n_found, N); last row is the non-oscillatory residual
        return imfs

    @staticmethod
    def _is_oscillatory(imf: np.ndarray) -> bool:
        """True if `imf` actually crosses zero enough to be a real IMF.

        A monotonic (or nearly monotonic) trend has essentially no zero
        crossings around its own mean; the Hilbert instantaneous
        frequency of such a component has no physical meaning per the
        IMF definition of Huang et al. (1998), so it must be routed to
        a plain trend extrapolation instead.
        """
        centered = imf - np.mean(imf)
        signs = np.sign(centered)
        zero_crossings = int(np.sum(np.diff(signs) != 0))
        return zero_crossings >= 2

    def _hilbert_extrapolate(self, imf: np.ndarray, horizon: int) -> np.ndarray:
        """Extrapolate a single IMF using last instantaneous frequency + amplitude."""
        from scipy.signal import hilbert
        analytic = hilbert(imf)
        inst_amp   = np.abs(analytic)
        inst_phase = np.unwrap(np.angle(analytic))
        inst_freq  = np.diff(inst_phase) / (2 * np.pi)

        # Use mean of last 10% of signal as steady-state estimate
        tail = max(1, len(imf) // 10)
        amp_est  = float(np.mean(inst_amp[-tail:]))
        freq_est = float(np.mean(inst_freq[-tail:]))
        phase_0  = inst_phase[-1]

        t = np.arange(1, horizon + 1, dtype=float)
        return amp_est * np.cos(phase_0 + 2 * np.pi * freq_est * t)

    # ------------------------------------------------------------------
    def fit(self, y: np.ndarray) -> "HHTNNForecaster":
        y = np.asarray(y, dtype=float).ravel()
        self._y_train = y

        self._n_fit = len(y)
        imfs = self._emd_decompose(y)

        # PyEMD always appends the non-oscillatory residual/trend as the
        # final row so that sum(imfs) reconstructs the signal exactly; it
        # is not a true IMF and must not be Hilbert-extrapolated.
        if len(imfs) > 1:
            candidates, trend = imfs[:-1], np.array(imfs[-1], dtype=float, copy=True)
        else:
            candidates, trend = imfs, np.zeros_like(y)

        oscillatory = []
        for imf in candidates:
            if self._is_oscillatory(imf):
                oscillatory.append(imf)
            else:
                # Degenerate "IMF" that never actually oscillates: fold it
                # into the trend so it gets a linear extrapolation instead
                # of a spurious Hilbert-derived cosine.
                trend = trend + imf

        k = min(self.n_imfs, len(oscillatory))
        self._imfs = np.array(oscillatory[:k]) if oscillatory else np.zeros((0, len(y)))
        self._trend = trend

        t = np.arange(self._n_fit, dtype=float)
        if self._n_fit > 1:
            p = np.polyfit(t, trend, 1)
            self._trend_slope, self._trend_intercept = float(p[0]), float(p[1])
        else:
            self._trend_slope = 0.0
            self._trend_intercept = float(trend[0]) if self._n_fit else 0.0

        # Build HHT reconstruction of dominant IMFs + trend
        hht_recon = self._imfs.sum(axis=0) + trend
        residual  = y - hht_recon

        # NN on residual
        from ..features.engineering import window_features
        X, Y = window_features(residual, self.input_len, self.horizon)
        nn = self._nn_model if self._nn_model is not None else self._default_nn()
        nn.fit(X, Y)
        self._residual_nn = nn
        return self

    # ------------------------------------------------------------------
    def predict(self) -> np.ndarray:
        if self._imfs is None:
            raise RuntimeError("Call fit() before predict().")

        hht_pred = np.zeros(self.horizon)
        for imf in self._imfs:
            hht_pred += self._hilbert_extrapolate(imf, self.horizon)

        t_future = np.arange(self._n_fit, self._n_fit + self.horizon, dtype=float)
        hht_pred += self._trend_slope * t_future + self._trend_intercept

        # NN residual prediction from last training window
        hht_recon = self._imfs.sum(axis=0) + self._trend
        residual  = self._y_train - hht_recon
        last_win  = residual[-self.input_len:].reshape(1, -1)
        nn_pred   = self._residual_nn.predict(last_win).ravel()

        return hht_pred + nn_pred[:self.horizon]

    # ------------------------------------------------------------------
    def fit_predict(self, y: np.ndarray) -> np.ndarray:
        self.fit(y)
        return self.predict()
