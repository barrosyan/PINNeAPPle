"""FFT + Neural Network on residual decomposition forecaster."""
from __future__ import annotations
from typing import Optional, Dict, Any

import numpy as np


class FFTNNForecaster:
    """
    Two-stage forecaster:
      1. FFT component captures global periodicity / trend.
      2. A neural network (default: MLP) models the residual.

    The final forecast is FFT extrapolation + NN residual prediction.

    Usage::
        from pinneaple_timeseries.decomposition import FFTNNForecaster
        from pinneaple_timeseries.models import LSTMForecaster, RecurrentConfig

        model = FFTNNForecaster(n_harmonics=8, input_len=32, horizon=16)
        model.fit(y_train)
        y_pred = model.predict()
    """

    def __init__(
        self,
        n_harmonics: int = 10,
        detrend: bool = True,
        input_len: int = 32,
        horizon: int = 16,
        nn_model: Optional[Any] = None,
        nn_epochs: int = 50,
        nn_lr: float = 1e-3,
        nn_batch: int = 32,
        device: Optional[str] = None,
    ):
        self.n_harmonics = n_harmonics
        self.detrend     = detrend
        self.input_len   = input_len
        self.horizon     = horizon
        self.nn_epochs   = nn_epochs
        self.nn_lr       = nn_lr
        self.nn_batch    = nn_batch
        self._device     = device
        self._nn_model   = nn_model  # external model; if None, default MLP is built
        self._fft_model  = None
        self._y_train: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def _default_nn(self):
        from ..models.classical import MLPForecaster
        return MLPForecaster(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=self.nn_epochs,
            learning_rate_init=self.nn_lr,
        )

    def _build_windows(self, residual: np.ndarray):
        from ..features.engineering import window_features
        return window_features(residual, self.input_len, self.horizon)

    # ------------------------------------------------------------------
    def fit(self, y: np.ndarray) -> "FFTNNForecaster":
        y = np.asarray(y, dtype=float).ravel()
        self._y_train = y

        # Stage 1: FFT
        from .fft_forecaster import FFTForecaster
        self._fft_model = FFTForecaster(n_harmonics=self.n_harmonics, detrend=self.detrend)
        self._fft_model.fit(y)
        fft_recon = self._fft_model.reconstruct()
        residual  = y - fft_recon

        # Stage 2: NN on residual
        X, Y = self._build_windows(residual)
        nn = self._nn_model if self._nn_model is not None else self._default_nn()
        nn.fit(X, Y)
        self._nn = nn
        return self

    # ------------------------------------------------------------------
    def predict(self) -> np.ndarray:
        if self._fft_model is None:
            raise RuntimeError("Call fit() before predict().")
        fft_pred = self._fft_model.predict(self.horizon)

        # Residual: last window of training residual
        fft_recon = self._fft_model.reconstruct()
        residual  = self._y_train - fft_recon
        last_win  = residual[-self.input_len:].reshape(1, -1)
        nn_pred   = self._nn.predict(last_win).ravel()

        return fft_pred + nn_pred[:self.horizon]

    # ------------------------------------------------------------------
    def fit_predict(self, y: np.ndarray) -> np.ndarray:
        self.fit(y)
        return self.predict()
