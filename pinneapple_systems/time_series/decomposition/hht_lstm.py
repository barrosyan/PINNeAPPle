"""Hilbert-Huang Transform (EMD-based) + LSTM residual forecaster (multivariate-ready).

Architecture:
  1. HHTDecomposer  — per-channel Empirical Mode Decomposition that extracts dominant
                      IMFs (Intrinsic Mode Functions) and the linear trend; extrapolates
                      them forward via Hilbert instantaneous frequency/amplitude.
  2. HHTLSTMPipeline — high-level orchestrator mirroring FFTLSTMPipeline; exposes
                       fit_hht / residuals_tensor / hht_extrapolation helpers.
  3. The residual LSTM (ResidualLSTMForecaster from fft_lstm.py) is reused directly.

Key difference from FFT approach:
  • FFT assumes stationarity with a fixed frequency basis.
  • EMD is fully data-adaptive — IMFs are extracted from the signal itself, capturing
    non-stationary amplitude and frequency modulation.
  • Better for signals with evolving seasonality; less reliable for very long extrapolation.

Dependencies: PyEMD (pip install EMD-signal), scipy.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Internal: single-channel EMD + Hilbert state
# ---------------------------------------------------------------------------

class _ChannelHHT:
    """EMD decomposition and Hilbert-based extrapolation for one channel."""

    def __init__(
        self,
        n_imfs: int = 6,
        dominant_energy_ratio: float = 0.80,
        detrend: bool = True,
    ) -> None:
        self.n_imfs = n_imfs
        self.dominant_energy_ratio = dominant_energy_ratio
        self.detrend = detrend

        self._n_fit: int = 0
        self._trend_slope: float = 0.0
        self._trend_intercept: float = 0.0
        self._dominant_imfs: List[np.ndarray] = []
        self._amp_est: List[float] = []
        self._freq_est: List[float] = []
        self._phase_0: List[float] = []
        self._recon_train: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(self, y: np.ndarray) -> "_ChannelHHT":
        try:
            from PyEMD import EMD
        except ImportError:
            raise ImportError(
                "PyEMD is required for HHT decomposition. "
                "Install with: pip install EMD-signal"
            )
        from scipy.signal import hilbert

        y = np.asarray(y, dtype=float).ravel()
        T = len(y)
        self._n_fit = T
        t = np.arange(T, dtype=float)

        # Linear detrend
        if self.detrend and T > 1:
            p = np.polyfit(t, y, 1)
            self._trend_slope = float(p[0])
            self._trend_intercept = float(p[1])
        else:
            self._trend_slope = 0.0
            self._trend_intercept = float(np.mean(y))
        y_detrended = y - (self._trend_slope * t + self._trend_intercept)

        # EMD decomposition
        try:
            emd = EMD()
            imfs = emd(y_detrended)  # (n_found, T)
        except Exception:
            imfs = y_detrended[None, :]

        # Select dominant IMFs by cumulative energy ratio
        energies = np.array([float(np.sum(imf ** 2)) for imf in imfs])
        total_energy = float(energies.sum())
        sorted_idx = np.argsort(energies)[::-1]

        self._dominant_imfs = []
        self._amp_est = []
        self._freq_est = []
        self._phase_0 = []

        cumulative = 0.0
        for idx in sorted_idx:
            if len(self._dominant_imfs) >= self.n_imfs:
                break
            imf = imfs[int(idx)]
            analytic = hilbert(imf)
            inst_amp = np.abs(analytic)
            inst_phase = np.unwrap(np.angle(analytic))
            inst_freq = np.diff(inst_phase) / (2.0 * np.pi)

            tail = max(10, T // 10)
            amp_est = float(np.mean(inst_amp[-tail:]))
            freq_est = float(np.clip(np.mean(np.abs(inst_freq[-tail:])), 1e-6, 0.499))

            self._dominant_imfs.append(imf)
            self._amp_est.append(amp_est)
            self._freq_est.append(freq_est)
            self._phase_0.append(float(inst_phase[-1]))

            cumulative += energies[int(idx)]
            if total_energy > 0 and cumulative / total_energy >= self.dominant_energy_ratio:
                break

        # Build training reconstruction
        recon = np.zeros(T, dtype=float)
        for imf in self._dominant_imfs:
            recon += imf
        recon += self._trend_slope * t + self._trend_intercept
        self._recon_train = recon
        return self

    def reconstruct(self) -> np.ndarray:
        if self._recon_train is None:
            raise RuntimeError("Call fit() first.")
        return self._recon_train.copy()

    def extrapolate(self, start: int, horizon: int) -> np.ndarray:
        """Extrapolate from absolute time index 'start' for 'horizon' steps."""
        result = np.zeros(horizon, dtype=float)
        n_fit = self._n_fit
        for h in range(horizon):
            t_abs = start + h
            val = self._trend_slope * float(t_abs) + self._trend_intercept
            for i, imf in enumerate(self._dominant_imfs):
                if t_abs < n_fit:
                    val += imf[t_abs]
                else:
                    delta_t = float(t_abs - n_fit + 1)
                    val += self._amp_est[i] * np.cos(
                        self._phase_0[i] + 2.0 * np.pi * self._freq_est[i] * delta_t
                    )
            result[h] = val
        return result

    def imf_info(self) -> Dict:
        periods = [1.0 / max(f, 1e-9) for f in self._freq_est]
        return {
            "n_dominant": len(self._dominant_imfs),
            "periods": periods,
            "amplitudes": list(self._amp_est),
            "frequencies": list(self._freq_est),
        }


# ---------------------------------------------------------------------------
# Multivariate HHT decomposer (sklearn-style API)
# ---------------------------------------------------------------------------

class HHTDecomposer:
    """Fits one EMD decomposer per channel; computes residuals and extrapolations.

    Args:
        n_imfs:                max number of dominant IMFs per channel.
        dominant_energy_ratio: fraction of signal energy captured by dominant IMFs.
        detrend:               remove linear trend before EMD.
    """

    def __init__(
        self,
        n_imfs: int = 6,
        dominant_energy_ratio: float = 0.80,
        detrend: bool = True,
    ) -> None:
        self.n_imfs = n_imfs
        self.dominant_energy_ratio = dominant_energy_ratio
        self.detrend = detrend
        self._models: List[_ChannelHHT] = []
        self._n_channels: int = 0
        self._n_fit: int = 0

    def fit(self, y: np.ndarray) -> "HHTDecomposer":
        """Fit one HHT model per channel.  y: (T,) or (T, F)."""
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        T, F = y.shape
        self._n_channels = F
        self._n_fit = T
        self._models = []
        for f in range(F):
            m = _ChannelHHT(
                n_imfs=self.n_imfs,
                dominant_energy_ratio=self.dominant_energy_ratio,
                detrend=self.detrend,
            )
            m.fit(y[:, f])
            self._models.append(m)
        return self

    def reconstruct(self) -> np.ndarray:
        """Return HHT reconstruction over the training period. (T_fit, F)"""
        return np.stack([m.reconstruct() for m in self._models], axis=1)

    def extrapolate(self, start: int, horizon: int) -> np.ndarray:
        """Extrapolate from absolute position 'start' for 'horizon' steps. (H, F)"""
        return np.stack(
            [m.extrapolate(start, horizon) for m in self._models], axis=1
        )

    def residual(self, y: np.ndarray) -> np.ndarray:
        """Compute residuals for y (T, F). For T > T_fit, extrapolation is used."""
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        T = len(y)
        recon = np.zeros((T, self._n_channels), dtype=float)
        if T <= self._n_fit:
            recon[:T] = self.reconstruct()[:T]
        else:
            recon[:self._n_fit] = self.reconstruct()
            extra = T - self._n_fit
            recon[self._n_fit:] = self.extrapolate(self._n_fit, extra)
        return y - recon

    def dominant_imf_counts(self) -> List[int]:
        """Number of dominant IMFs selected per channel."""
        return [len(m._dominant_imfs) for m in self._models]

    def imf_info(self) -> List[Dict]:
        """Per-channel IMF diagnostic information."""
        return [m.imf_info() for m in self._models]


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

class HHTLSTMPipeline:
    """Orchestrates HHTDecomposer + LSTM residual forecasting.

    Mirrors FFTLSTMPipeline API so both pipelines can be used interchangeably.

    Usage::

        pipeline = HHTLSTMPipeline(n_imfs=6, input_len=96, horizon=24, n_features=3)
        pipeline.fit_hht(train_np)                          # fit on train only
        res_tensor = pipeline.residuals_tensor(full_np)     # (T, F) residuals
        hht_part   = pipeline.hht_extrapolation(850, 24)    # (24, 3) numpy
    """

    def __init__(
        self,
        n_imfs: int = 6,
        dominant_energy_ratio: float = 0.80,
        detrend: bool = True,
        input_len: int = 64,
        horizon: int = 16,
        n_features: int = 1,
        n_targets: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.20,
        layer_norm: bool = True,
    ) -> None:
        self._decomposer: Optional[HHTDecomposer] = None
        self.n_imfs = n_imfs
        self.dominant_energy_ratio = dominant_energy_ratio
        self.detrend = detrend
        self.input_len = input_len
        self.horizon = horizon
        self.n_features = n_features
        self.n_targets = n_targets
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.layer_norm = layer_norm

    def fit_hht(self, y: np.ndarray) -> "HHTLSTMPipeline":
        """Fit HHT decomposer on training data (no leakage from val/test)."""
        self._decomposer = HHTDecomposer(
            n_imfs=self.n_imfs,
            dominant_energy_ratio=self.dominant_energy_ratio,
            detrend=self.detrend,
        )
        self._decomposer.fit(y)
        return self

    def residuals_tensor(self, y: np.ndarray) -> torch.Tensor:
        """Compute residuals for the full series as a (T, F) float32 tensor."""
        if self._decomposer is None:
            raise RuntimeError("Call fit_hht() first.")
        res = self._decomposer.residual(y)
        return torch.tensor(res, dtype=torch.float32)

    def hht_extrapolation(self, start: int, horizon: int) -> np.ndarray:
        """Extrapolate HHT reconstruction from 'start' for 'horizon' steps. (H, F)"""
        if self._decomposer is None:
            raise RuntimeError("Call fit_hht() first.")
        return self._decomposer.extrapolate(start, horizon)

    def dominant_imf_counts(self) -> List[int]:
        if self._decomposer is None:
            raise RuntimeError("Call fit_hht() first.")
        return self._decomposer.dominant_imf_counts()

    def imf_info(self) -> List[Dict]:
        if self._decomposer is None:
            raise RuntimeError("Call fit_hht() first.")
        return self._decomposer.imf_info()
