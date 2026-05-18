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
        imfs = emd(y)  # shape (n_found, N)
        return imfs

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

        imfs = self._emd_decompose(y)
        k    = min(self.n_imfs, len(imfs))
        self._imfs = imfs[:k]

        # Build HHT reconstruction of dominant IMFs
        hht_recon = self._imfs.sum(axis=0)
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

        # NN residual prediction from last training window
        hht_recon = self._imfs.sum(axis=0)
        residual  = self._y_train - hht_recon
        last_win  = residual[-self.input_len:].reshape(1, -1)
        nn_pred   = self._residual_nn.predict(last_win).ravel()

        return hht_pred + nn_pred[:self.horizon]

    # ------------------------------------------------------------------
    def fit_predict(self, y: np.ndarray) -> np.ndarray:
        self.fit(y)
        return self.predict()
