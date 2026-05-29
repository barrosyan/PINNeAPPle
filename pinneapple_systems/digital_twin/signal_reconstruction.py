"""Signal reconstruction from incoming data streams.

Provides methods to reconstruct continuous, clean signals from the sparse,
noisy observations arriving via digital twin streams.  All methods operate
on the ``Observation`` objects accumulated by ``DigitalTwin._last_observations``.

Methods
-------
spectral     : FFT-based — fit dominant harmonics and extrapolate/interpolate
ssa          : Singular Spectrum Analysis — decompose + keep dominant components
wavelet      : Wavelet soft-thresholding denoising + reconstruction
interpolate  : Scipy-based gap-filling (linear, cubic, PCHIP)
model_based  : Use the twin's physics model to fill gaps in space or time

All methods return ``ReconstructionResult`` containing the reconstructed
signal, timestamps, and quality metrics.

Usage
-----
>>> from pinneapple_systems.digital_twin import DigitalTwin
>>> from pinneapple_systems.digital_twin.signal_reconstruction import SignalReconstructor

>>> twin = DigitalTwin(model, ["u", "v"])
>>> rec = SignalReconstructor(twin)

>>> # Reconstruct sensor 'inlet' field 'u' using SSA
>>> result = rec.reconstruct("inlet", "u", method="ssa")
>>> result.signal          # np.ndarray — reconstructed signal
>>> result.timestamps      # np.ndarray — corresponding times
>>> result.quality         # dict with SNR, RMSE vs raw, etc.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReconstructionResult:
    """Output of a signal reconstruction pass."""
    sensor_id: str
    field_name: str
    method: str
    timestamps: np.ndarray          # (N,) UTC seconds
    signal: np.ndarray              # (N,) reconstructed values
    raw_timestamps: np.ndarray      # original sparse observations
    raw_values: np.ndarray
    quality: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return len(self.signal)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "field_name": self.field_name,
            "method": self.method,
            "n_points": self.n_points,
            "quality": self.quality,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_series(
    observations: list,
    sensor_id: str,
    field_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pull (timestamps, values) for a given sensor+field from obs list."""
    ts, vals = [], []
    for obs in observations:
        if obs.sensor_id == sensor_id and field_name in obs.values:
            ts.append(obs.timestamp)
            vals.append(float(obs.values[field_name]))
    if not ts:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float32)
    order = np.argsort(ts)
    return np.array(ts, dtype=np.float64)[order], np.array(vals, dtype=np.float32)[order]


def _quality_metrics(
    raw: np.ndarray,
    reconstructed_at_raw: np.ndarray,
) -> Dict[str, float]:
    """Compute quality metrics comparing reconstructed to raw values."""
    if len(raw) == 0 or len(reconstructed_at_raw) == 0:
        return {}
    diff = raw - reconstructed_at_raw
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    signal_power = float(np.mean(raw ** 2)) + 1e-12
    snr = float(10.0 * np.log10(signal_power / (np.mean(diff ** 2) + 1e-12)))
    return {"rmse": rmse, "snr_db": snr, "max_abs_error": float(np.max(np.abs(diff)))}


# ---------------------------------------------------------------------------
# Individual reconstruction methods (standalone functions)
# ---------------------------------------------------------------------------

def reconstruct_spectral(
    timestamps: np.ndarray,
    values: np.ndarray,
    *,
    n_harmonics: int = 10,
    out_times: Optional[np.ndarray] = None,
    min_points: int = 8,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    FFT-based reconstruction: fit the dominant harmonics and evaluate on a
    dense grid (or ``out_times`` if provided).

    Returns (out_timestamps, reconstructed_signal, metadata).
    """
    if len(timestamps) < min_points:
        raise ValueError(f"Need at least {min_points} points for spectral reconstruction.")

    t = timestamps - timestamps[0]
    T_total = t[-1] - t[0] if t[-1] > t[0] else 1.0

    # Interpolate to uniform grid first
    t_uniform = np.linspace(0.0, T_total, num=max(256, len(timestamps) * 4))
    v_uniform = np.interp(t_uniform, t, values)

    # FFT
    N = len(t_uniform)
    dt = t_uniform[1] - t_uniform[0]
    F = np.fft.rfft(v_uniform)
    freqs = np.fft.rfftfreq(N, d=dt)

    # Keep only dominant harmonics (by magnitude)
    magnitudes = np.abs(F)
    dc = F[0]
    F_filtered = np.zeros_like(F)
    F_filtered[0] = dc
    top_k = np.argsort(magnitudes[1:])[::-1][:n_harmonics] + 1
    F_filtered[top_k] = F[top_k]

    # Reconstruct on uniform grid
    v_recon_uniform = np.fft.irfft(F_filtered, n=N).astype(np.float32)

    # Evaluate at requested output times
    if out_times is not None:
        t_out = out_times - timestamps[0]
        v_out = np.interp(t_out, t_uniform, v_recon_uniform).astype(np.float32)
        ts_out = out_times
    else:
        v_out = v_recon_uniform
        ts_out = t_uniform + timestamps[0]

    energy_kept = float(np.sum(np.abs(F_filtered) ** 2) / (np.sum(magnitudes ** 2) + 1e-12))
    meta = {"n_harmonics": n_harmonics, "energy_retained": energy_kept, "N_uniform": N}
    return ts_out, v_out, meta


def reconstruct_ssa(
    timestamps: np.ndarray,
    values: np.ndarray,
    *,
    window: int = 0,          # 0 = auto (T // 4)
    n_components: int = 5,
    out_times: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    SSA-based reconstruction: decompose the signal and reconstruct from the
    dominant components (noise suppression).

    Uses pinneapple_simulation.numerical_solvers.ssa.ssa_decompose internally.
    """
    from pinneapple_simulation.numerical_solvers.ssa import ssa_decompose

    t = timestamps - timestamps[0]
    T_total = t[-1] - t[0] if t[-1] > t[0] else 1.0
    t_uniform = np.linspace(0.0, T_total, num=max(256, len(timestamps) * 4))
    v_uniform = np.interp(t_uniform, t, values)

    L = window if window > 0 else max(2, len(t_uniform) // 4)
    comps, sv, _ = ssa_decompose(v_uniform, L=L, r=n_components)
    v_recon_uniform = comps.sum(axis=0).astype(np.float32)

    if out_times is not None:
        t_out = out_times - timestamps[0]
        v_out = np.interp(t_out, t_uniform, v_recon_uniform).astype(np.float32)
        ts_out = out_times
    else:
        v_out = v_recon_uniform
        ts_out = t_uniform + timestamps[0]

    explained = float(np.sum(sv ** 2) / (np.sum(sv ** 2) + 1e-12))
    meta = {"window": L, "n_components": n_components, "explained_variance": explained}
    return ts_out, v_out, meta


def reconstruct_wavelet(
    timestamps: np.ndarray,
    values: np.ndarray,
    *,
    wavelet: str = "db4",
    level: int = 4,
    threshold_mode: str = "soft",   # "soft" | "hard"
    threshold_sigma_mult: float = 3.0,
    out_times: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Wavelet soft-thresholding denoising + reconstruction.

    Uses pywt via pinneapple_simulation wavelet helper.
    """
    try:
        import pywt
    except ImportError:
        raise ImportError("pywavelets required: pip install PyWavelets")

    t = timestamps - timestamps[0]
    T_total = t[-1] - t[0] if t[-1] > t[0] else 1.0
    N = max(256, len(timestamps) * 4)
    # Round up to next power of 2 for clean wavelet decomposition
    N_pow2 = int(2 ** np.ceil(np.log2(N)))
    t_uniform = np.linspace(0.0, T_total, num=N_pow2)
    v_uniform = np.interp(t_uniform, t, values.astype(float))

    coeffs = pywt.wavedec(v_uniform, wavelet=wavelet, level=level, mode="periodization")

    # Estimate noise from finest detail level
    detail = coeffs[-1]
    sigma = float(np.median(np.abs(detail)) / 0.6745 + 1e-10)
    thresh = threshold_sigma_mult * sigma * np.sqrt(2.0 * np.log(N_pow2))

    # Threshold all detail levels
    denoised = [coeffs[0]] + [
        pywt.threshold(c, thresh, mode=threshold_mode) for c in coeffs[1:]
    ]
    v_recon_uniform = pywt.waverec(denoised, wavelet=wavelet, mode="periodization")
    v_recon_uniform = v_recon_uniform[:N_pow2].astype(np.float32)

    if out_times is not None:
        t_out = out_times - timestamps[0]
        v_out = np.interp(t_out, t_uniform, v_recon_uniform).astype(np.float32)
        ts_out = out_times
    else:
        v_out = v_recon_uniform
        ts_out = t_uniform + timestamps[0]

    meta = {"wavelet": wavelet, "level": level, "sigma_noise": sigma,
            "threshold": float(thresh), "threshold_mode": threshold_mode}
    return ts_out, v_out, meta


def reconstruct_interpolate(
    timestamps: np.ndarray,
    values: np.ndarray,
    *,
    kind: str = "cubic",     # "linear" | "cubic" | "pchip" | "akima"
    out_times: Optional[np.ndarray] = None,
    n_out: int = 512,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Scipy interpolation-based reconstruction (gap filling / upsampling).

    ``kind``: passed to scipy.interpolate.interp1d, or "pchip"/"akima" for
    shape-preserving methods.
    """
    try:
        from scipy.interpolate import interp1d, PchipInterpolator, Akima1DInterpolator
    except ImportError:
        raise ImportError("scipy required: pip install scipy")

    t = timestamps - timestamps[0]

    if out_times is not None:
        t_out = out_times - timestamps[0]
        ts_out = out_times
    else:
        t_out = np.linspace(0.0, t[-1], num=n_out)
        ts_out = t_out + timestamps[0]

    if kind == "pchip":
        itp = PchipInterpolator(t, values)
        v_out = itp(t_out).astype(np.float32)
    elif kind == "akima":
        itp = Akima1DInterpolator(t, values)
        v_out = itp(t_out).astype(np.float32)
    else:
        itp = interp1d(t, values, kind=kind, bounds_error=False,
                       fill_value=(values[0], values[-1]))
        v_out = itp(t_out).astype(np.float32)

    meta = {"kind": kind, "n_raw": len(timestamps), "n_out": len(t_out)}
    return ts_out, v_out, meta


def reconstruct_model_based(
    timestamps: np.ndarray,
    values: np.ndarray,
    *,
    model: Any,
    sensor_coord: Dict[str, float],
    coord_names: List[str],
    field_name: str,
    field_index: int = 0,
    out_times: Optional[np.ndarray] = None,
    n_out: int = 512,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Use a trained physics model to reconstruct the time series at a given
    spatial location by querying model(x, y, t).

    The model is queried at the sensor coordinate across the output time grid.
    Useful to fill gaps or extrapolate beyond the observation window.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch required.")

    if out_times is not None:
        t_out = out_times
    else:
        t_out = np.linspace(timestamps[0], timestamps[-1], num=n_out)

    n = len(t_out)
    coord_cols = [
        np.full(n, sensor_coord.get(c, 0.0), dtype=np.float32)
        for c in coord_names if c != "t"
    ]
    # Add time column if coord_names includes "t" or by convention
    t_col = (t_out - t_out[0]).astype(np.float32)
    if "t" in coord_names:
        idx_t = coord_names.index("t")
        coord_cols.insert(idx_t, t_col)
    else:
        coord_cols.append(t_col)

    X = np.column_stack(coord_cols)
    dev = torch.device(device)
    X_t = torch.from_numpy(X).to(dev)

    with torch.no_grad():
        out = model(X_t)
    if hasattr(out, "y"):
        out = out.y
    if isinstance(out, torch.Tensor):
        out_np = out.cpu().float().numpy()
    else:
        out_np = np.asarray(out, dtype=np.float32)

    if out_np.ndim == 1:
        v_out = out_np
    else:
        v_out = out_np[:, field_index] if field_index < out_np.shape[1] else out_np[:, 0]

    meta = {"coord": sensor_coord, "n_out": n, "field_index": field_index}
    return t_out, v_out.astype(np.float32), meta


# ---------------------------------------------------------------------------
# SignalReconstructor — high-level interface bound to a DigitalTwin
# ---------------------------------------------------------------------------

class SignalReconstructor:
    """
    Convenience wrapper that reconstructs signals from the observation history
    held by a live ``DigitalTwin`` (or from any list of ``Observation`` objects).

    Parameters
    ----------
    twin_or_observations :
        A ``DigitalTwin`` instance (uses its ``_last_observations`` buffer) or
        a plain list of ``Observation`` objects.
    default_method : str
        Default reconstruction method: "spectral" | "ssa" | "wavelet" |
        "interpolate" | "model_based"
    default_n_out : int
        Number of output points when no ``out_times`` is given.
    """

    METHODS = ("spectral", "ssa", "wavelet", "interpolate", "model_based")

    def __init__(
        self,
        twin_or_observations: Any,
        *,
        default_method: str = "ssa",
        default_n_out: int = 512,
    ) -> None:
        self._source = twin_or_observations
        self.default_method = default_method
        self.default_n_out = default_n_out

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def _get_observations(self) -> list:
        from .twin import DigitalTwin
        if isinstance(self._source, DigitalTwin):
            return list(self._source._last_observations)
        return list(self._source)

    def available_sensors(self) -> Dict[str, List[str]]:
        """Return {sensor_id: [field_name, ...]} from buffered observations."""
        obs = self._get_observations()
        sensors: Dict[str, set] = {}
        for o in obs:
            if o.sensor_id not in sensors:
                sensors[o.sensor_id] = set()
            sensors[o.sensor_id].update(o.values.keys())
        return {sid: sorted(fields) for sid, fields in sensors.items()}

    def raw_series(
        self,
        sensor_id: str,
        field_name: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (timestamps, values) for a sensor/field without reconstruction."""
        obs = self._get_observations()
        return _extract_series(obs, sensor_id, field_name)

    # ------------------------------------------------------------------
    # Main reconstruction entry point
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        sensor_id: str,
        field_name: str,
        *,
        method: Optional[str] = None,
        out_times: Optional[np.ndarray] = None,
        n_out: Optional[int] = None,
        # Method-specific options forwarded via kwargs
        **kwargs: Any,
    ) -> ReconstructionResult:
        """
        Reconstruct the signal for ``sensor_id / field_name``.

        Parameters
        ----------
        sensor_id   : sensor identifier
        field_name  : field to reconstruct
        method      : override default_method
        out_times   : optional array of output timestamps (UTC seconds)
        n_out       : number of output points (used if out_times is None)
        **kwargs    : passed to the specific reconstruction function
        """
        method = method or self.default_method
        if method not in self.METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from {self.METHODS}")

        obs = self._get_observations()
        raw_ts, raw_vals = _extract_series(obs, sensor_id, field_name)

        if len(raw_ts) < 2:
            raise ValueError(
                f"Not enough observations for sensor '{sensor_id}' field '{field_name}'. "
                f"Got {len(raw_ts)}, need ≥ 2."
            )

        n = n_out or self.default_n_out
        fn_kwargs = dict(out_times=out_times)
        if out_times is None:
            fn_kwargs["n_out"] = n

        if method == "spectral":
            fn_kwargs.update({k: v for k, v in kwargs.items()
                              if k in ("n_harmonics", "min_points")})
            ts_out, v_out, meta = reconstruct_spectral(raw_ts, raw_vals, **fn_kwargs)

        elif method == "ssa":
            fn_kwargs.update({k: v for k, v in kwargs.items()
                              if k in ("window", "n_components")})
            ts_out, v_out, meta = reconstruct_ssa(raw_ts, raw_vals, **fn_kwargs)

        elif method == "wavelet":
            fn_kwargs.update({k: v for k, v in kwargs.items()
                              if k in ("wavelet", "level", "threshold_mode", "threshold_sigma_mult")})
            ts_out, v_out, meta = reconstruct_wavelet(raw_ts, raw_vals, **fn_kwargs)

        elif method == "interpolate":
            fn_kwargs.update({k: v for k, v in kwargs.items()
                              if k in ("kind",)})
            ts_out, v_out, meta = reconstruct_interpolate(raw_ts, raw_vals, **fn_kwargs)

        elif method == "model_based":
            required = ("model", "sensor_coord", "coord_names")
            for req in required:
                if req not in kwargs:
                    raise ValueError(f"model_based requires '{req}' in kwargs.")
            fn_kwargs.update({k: v for k, v in kwargs.items()
                              if k in ("model", "sensor_coord", "coord_names",
                                       "field_index", "device")})
            fn_kwargs["field_name"] = field_name
            ts_out, v_out, meta = reconstruct_model_based(raw_ts, raw_vals, **fn_kwargs)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Quality metrics: compare reconstruction at raw timestamps
        raw_ts_rel = np.clip(raw_ts, ts_out[0], ts_out[-1])
        recon_at_raw = np.interp(raw_ts_rel, ts_out, v_out)
        quality = _quality_metrics(raw_vals, recon_at_raw)

        return ReconstructionResult(
            sensor_id=sensor_id,
            field_name=field_name,
            method=method,
            timestamps=ts_out,
            signal=v_out,
            raw_timestamps=raw_ts,
            raw_values=raw_vals,
            quality=quality,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Convenience: reconstruct all sensors with one call
    # ------------------------------------------------------------------

    def reconstruct_all(
        self,
        method: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[Tuple[str, str], ReconstructionResult]:
        """Reconstruct all available sensor/field combinations."""
        results = {}
        for sid, fields in self.available_sensors().items():
            for fname in fields:
                try:
                    r = self.reconstruct(sid, fname, method=method, **kwargs)
                    results[(sid, fname)] = r
                except Exception as exc:
                    logger.warning(f"Reconstruction failed for {sid}/{fname}: {exc}")
        return results

    # ------------------------------------------------------------------
    # Integration: attach to DigitalTwin and return reconstructed state
    # ------------------------------------------------------------------

    def reconstruct_state(
        self,
        method: Optional[str] = None,
        at_time: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, float]]:
        """
        Reconstruct the instantaneous sensor values at ``at_time`` (default: now).

        Returns {sensor_id: {field_name: reconstructed_value}}.
        Useful for state estimation when sensor data arrives irregularly.
        """
        import time as _time
        t_query = np.array([at_time if at_time is not None else _time.time()])
        state: Dict[str, Dict[str, float]] = {}
        for sid, fields in self.available_sensors().items():
            state[sid] = {}
            for fname in fields:
                try:
                    r = self.reconstruct(
                        sid, fname, method=method, out_times=t_query, **kwargs
                    )
                    state[sid][fname] = float(r.signal[0])
                except Exception:
                    pass
        return state
