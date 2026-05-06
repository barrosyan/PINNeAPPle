"""Hybrid FFT decomposition + LSTM residual forecaster (multivariate-ready).

Architecture:
  1. FFTDecomposer  — per-channel FFT that extracts trend + dominant harmonics
                      and exposes deterministic reconstruction / extrapolation.
  2. ResidualLSTMForecaster — PyTorch LSTM that models only the *residual* after
                              FFT reconstruction. Plugs directly into the standard
                              Trainer + TSDataModule pipeline.
  3. FFTLSTMPipeline — high-level orchestrator that wires the two stages and
                       exposes fit_fft / residuals_tensor / fft_extrapolation helpers.

Multivariate support:
  Each channel receives its own FFTForecaster instance; the LSTM processes the
  joint residual tensor (B, L, F) and predicts (B, H, n_targets).

Anti-overfitting in ResidualLSTMForecaster:
  - Input LayerNorm (normalises across feature dimension, batch-size agnostic)
  - LSTM inter-layer dropout
  - LayerNorm on the final LSTM hidden state
  - Dropout before the linear head
  - Weight-decay (L2) is handled at the optimizer level via TrainConfig.weight_decay
  - Gradient clipping via TrainConfig.grad_clip
  - EarlyStopping via the standard callback from pinneaple_train.callbacks
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..base import TSModelBase, TSOutput
from .fft_forecaster import FFTForecaster


# ---------------------------------------------------------------------------
# 1. Multivariate FFT decomposer (numpy / sklearn-style API)
# ---------------------------------------------------------------------------

class FFTDecomposer:
    """Fits one FFTForecaster per channel; computes residuals and extrapolations.

    Args:
        n_harmonics:     fixed number of dominant harmonics per channel.
        detrend:         remove linear trend before FFT.
        auto_harmonics:  auto-select n_harmonics per channel using amplitude
                         threshold instead of a fixed count.
        auto_threshold:  fraction of max amplitude for automatic selection.
    """

    def __init__(
        self,
        n_harmonics: int = 10,
        detrend: bool = True,
        auto_harmonics: bool = True,
        auto_threshold: float = 0.05,
    ) -> None:
        self.n_harmonics = n_harmonics
        self.detrend = detrend
        self.auto_harmonics = auto_harmonics
        self.auto_threshold = auto_threshold

        self._models: List[FFTForecaster] = []
        self._n_channels: int = 0
        self._n_fit: int = 0

    # ------------------------------------------------------------------
    def fit(self, y: np.ndarray) -> "FFTDecomposer":
        """Fit one FFT model per channel.  y: (T,) or (T, F)."""
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        T, F = y.shape
        self._n_channels = F
        self._n_fit = T
        self._models = []
        for f in range(F):
            n_h = self._auto_select(y[:, f]) if self.auto_harmonics else self.n_harmonics
            m = FFTForecaster(n_harmonics=n_h, detrend=self.detrend)
            m.fit(y[:, f])
            self._models.append(m)
        return self

    def _auto_select(self, y: np.ndarray) -> int:
        """Choose n_harmonics as the count of peaks above auto_threshold × max."""
        yc = y - y.mean()
        if self.detrend:
            t = np.arange(len(y), dtype=float)
            slope, intercept = np.polyfit(t, yc, 1)
            yc = yc - (slope * t + intercept)
        amps = np.abs(np.fft.rfft(yc)[1:])
        if amps.max() == 0:
            return 1
        threshold = self.auto_threshold * amps.max()
        count = int((amps >= threshold).sum())
        return max(1, min(count, min(self.n_harmonics, len(amps))))

    # ------------------------------------------------------------------
    def reconstruct(self) -> np.ndarray:
        """Reconstruct the training-length deterministic signal.  Returns (T_fit, F)."""
        self._check_fitted()
        return np.stack([m.reconstruct() for m in self._models], axis=1)

    def extrapolate(self, start: int, horizon: int) -> np.ndarray:
        """Extrapolate deterministic component beyond the training window.

        Returns (horizon, F).
        """
        self._check_fitted()
        return np.stack([m.predict(horizon, start=start) for m in self._models], axis=1)

    def residual(self, y: np.ndarray) -> np.ndarray:
        """Compute y minus the deterministic FFT reconstruction.

        Handles any T (including T > T_fit via extrapolation).
        Returns (T, F).
        """
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        T = y.shape[0]

        if T <= self._n_fit:
            recon = self.reconstruct()[:T]
        else:
            recon_train = self.reconstruct()                              # (T_fit, F)
            recon_future = self.extrapolate(                              # (T-T_fit, F)
                start=self._n_fit, horizon=T - self._n_fit
            )
            recon = np.concatenate([recon_train, recon_future], axis=0)  # (T, F)

        return y - recon

    def dominant_periods(self) -> List[List[float]]:
        """Return dominant periods (1/freq) per channel, sorted descending."""
        self._check_fitted()
        out = []
        for m in self._models:
            with np.errstate(divide="ignore"):
                p = np.where(m._freqs > 0, 1.0 / m._freqs, np.inf)
            out.append(sorted(p.tolist(), reverse=True))
        return out

    def power_spectrum(self, y: np.ndarray) -> List[dict]:
        """Return FFT power spectrum per channel for exploratory analysis.

        Returns a list (one per channel) of dicts with keys:
          freqs, periods, power (all numpy arrays, positive freqs only).
        """
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        out = []
        for f in range(y.shape[1]):
            yf = y[:, f] - y[:, f].mean()
            fft_vals = np.fft.rfft(yf)
            freqs = np.fft.rfftfreq(len(yf))
            power = np.abs(fft_vals) ** 2
            mask = freqs > 0
            with np.errstate(divide="ignore"):
                periods = np.where(freqs[mask] > 0, 1.0 / freqs[mask], np.inf)
            out.append({
                "freqs":   freqs[mask],
                "periods": periods,
                "power":   power[mask],
            })
        return out

    def _check_fitted(self) -> None:
        if not self._models:
            raise RuntimeError("Call fit() before using FFTDecomposer.")


# ---------------------------------------------------------------------------
# 2. Residual LSTM (PyTorch, plugs into Trainer + TSDataModule)
# ---------------------------------------------------------------------------

@dataclass
class ResidualLSTMConfig:
    """Hyperparameters for ResidualLSTMForecaster."""
    input_len:   int   = 64
    horizon:     int   = 16
    n_features:  int   = 1     # number of input channels (all channels as features)
    n_targets:   int   = 1     # number of output channels (channels to forecast)
    hidden_size: int   = 128
    num_layers:  int   = 2
    dropout:     float = 0.20  # applied between LSTM layers and before head
    layer_norm:  bool  = True  # LayerNorm on LSTM output


class ResidualLSTMForecaster(TSModelBase):
    """LSTM forecaster trained on FFT residuals.

    Input:  (B, L, n_features)  — windowed residual series
    Output: TSOutput(y_hat=(B, H, n_targets), extras={})

    Anti-overfitting checklist (use together with optimizer weight_decay and
    EarlyStopping from pinneaple_train.callbacks):
      ✓ Input LayerNorm  (safe for any batch size, unlike BatchNorm)
      ✓ LSTM inter-layer dropout
      ✓ LayerNorm on LSTM last-step hidden state
      ✓ Dropout before linear head
      (set weight_decay in TrainConfig, grad_clip ≥ 0.5 recommended)
    """

    def __init__(self, cfg: Optional[ResidualLSTMConfig] = None, **kw):
        super().__init__()
        self.cfg = cfg or ResidualLSTMConfig(**kw)
        c = self.cfg

        # Input normalisation: per-feature LayerNorm over the time axis
        self.input_norm = nn.LayerNorm(c.n_features)

        self.lstm = nn.LSTM(
            input_size=c.n_features,
            hidden_size=c.hidden_size,
            num_layers=c.num_layers,
            dropout=c.dropout if c.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.post_norm = nn.LayerNorm(c.hidden_size) if c.layer_norm else nn.Identity()
        self.head_drop = nn.Dropout(p=c.dropout)
        self.head = nn.Linear(c.hidden_size, c.horizon * c.n_targets)

    def forward(self, x: torch.Tensor) -> TSOutput:
        # x: (B, L, F)
        B = x.shape[0]
        xn = self.input_norm(x)                       # (B, L, F)  — normalise features
        out, _ = self.lstm(xn)                        # (B, L, hidden)
        last = self.post_norm(out[:, -1, :])          # (B, hidden)
        last = self.head_drop(last)
        pred = self.head(last)                        # (B, H * n_targets)
        pred = pred.view(B, self.cfg.horizon, self.cfg.n_targets)
        return TSOutput(y_hat=pred, extras={})


# ---------------------------------------------------------------------------
# 3. High-level FFT-LSTM orchestrator
# ---------------------------------------------------------------------------

class FFTLSTMPipeline:
    """Orchestrates the two-stage hybrid FFT + LSTM pipeline.

    Stage 1 (FFT): fits a multivariate FFTDecomposer on the *training* portion
      of the series, capturing global trend and dominant periodic components.

    Stage 2 (LSTM): trains a ResidualLSTMForecaster on the stochastic residual
      (original − FFT reconstruction) using the standard Trainer infrastructure.

    Final forecast = FFT extrapolation  +  LSTM residual prediction

    Typical usage::

        pipeline = FFTLSTMPipeline(
            n_harmonics=12, input_len=96, horizon=24, n_features=3, n_targets=3
        )

        # Step 1 — fit FFT on training data (numpy)
        pipeline.fit_fft(train_np)                   # (T_train, F)

        # Step 2 — compute residual series (for ALL splits)
        res_tensor = pipeline.residuals_tensor(full_series_np)  # (T, F)

        # Step 3 — build TSDataModule on residuals, train with Trainer
        dm  = TSDataModule(series=res_tensor, spec=spec, ...)
        mdl = pipeline.build_lstm()
        trainer.fit(train_loader, val_loader, cfg)   # standard Trainer call

        # Step 4 — inference: reconstruct final forecast
        fft_part  = pipeline.fft_extrapolation(start=T_train, horizon=24)
        lstm_part = mdl(x_residual_window).y_hat.numpy()
        forecast  = fft_part + lstm_part
    """

    def __init__(
        self,
        *,
        n_harmonics: int = 10,
        detrend: bool = True,
        auto_harmonics: bool = True,
        auto_threshold: float = 0.05,
        # LSTM
        input_len:   int   = 64,
        horizon:     int   = 16,
        n_features:  int   = 1,
        n_targets:   int   = 1,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.20,
    ) -> None:
        self.decomposer = FFTDecomposer(
            n_harmonics=n_harmonics,
            detrend=detrend,
            auto_harmonics=auto_harmonics,
            auto_threshold=auto_threshold,
        )
        self.lstm_cfg = ResidualLSTMConfig(
            input_len=input_len,
            horizon=horizon,
            n_features=n_features,
            n_targets=n_targets,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    def fit_fft(self, y: np.ndarray) -> "FFTLSTMPipeline":
        """Fit the FFT decomposer on training data.  y: (T_train,) or (T_train, F)."""
        self.decomposer.fit(y)
        return self

    def residuals_tensor(self, y: np.ndarray) -> torch.Tensor:
        """Compute residuals for any split and return as float32 tensor (T, F)."""
        res = self.decomposer.residual(y)             # (T, F) numpy
        return torch.tensor(res, dtype=torch.float32)

    def fft_extrapolation(self, start: int, horizon: int) -> np.ndarray:
        """Deterministic FFT forecast beyond training.  Returns (horizon, F)."""
        return self.decomposer.extrapolate(start=start, horizon=horizon)

    def build_lstm(self) -> ResidualLSTMForecaster:
        """Instantiate a fresh ResidualLSTMForecaster with the configured LSTM params."""
        return ResidualLSTMForecaster(cfg=self.lstm_cfg)

    def dominant_periods(self) -> List[List[float]]:
        """Dominant periods per channel from the fitted FFT decomposer."""
        return self.decomposer.dominant_periods()

    def power_spectrum(self, y: np.ndarray) -> List[dict]:
        """Power spectrum per channel (delegates to FFTDecomposer)."""
        return self.decomposer.power_spectrum(y)
