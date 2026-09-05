"""Extract dominant vibration/acoustic frequencies from an audio
recording (e.g. a struck beam, plate, or shell's impulse response) via
FFT peak-picking with sub-bin parabolic frequency refinement. Depends
only on numpy/scipy (already core PINNeAPPle dependencies).

The extracted natural frequencies are directly usable to calibrate or
validate a structural-dynamics preset (e.g. comparing against a beam's
predicted modal frequencies, or as a `DataConstraint` target for an
inverse-problem fit of an unknown material/geometry parameter that the
preset's frequency depends on).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks


def _parabolic_subbin_refine(mag: np.ndarray, idx: int, freqs: np.ndarray) -> Tuple[float, float]:
    """3-point parabolic interpolation around an FFT magnitude-spectrum
    peak bin -- standard sub-bin frequency refinement (same technique as
    `video_piv`'s sub-pixel peak fit, applied to a 1D spectrum instead of
    a 2D correlation surface)."""
    if idx <= 0 or idx >= len(mag) - 1:
        return float(freqs[idx]), float(mag[idx])
    y0, y1, y2 = mag[idx - 1], mag[idx], mag[idx + 1]
    denom = (y0 - 2 * y1 + y2)
    if abs(denom) < 1e-15:
        return float(freqs[idx]), float(mag[idx])
    delta = 0.5 * (y0 - y2) / denom
    df = freqs[1] - freqs[0]
    refined_freq = freqs[idx] + delta * df
    refined_mag = y1 - 0.25 * (y0 - y2) * delta
    return float(refined_freq), float(refined_mag)


def extract_dominant_frequencies(
    waveform: np.ndarray,
    sample_rate: float,
    n_peaks: int = 5,
    min_freq: float = 0.0,
    max_freq: Optional[float] = None,
    window: str = "hann",
) -> List[Tuple[float, float]]:
    """Return the `n_peaks` most prominent frequency components of
    `waveform`, as a list of (frequency_Hz, amplitude) tuples sorted by
    descending amplitude.

    Parameters
    ----------
    waveform : 1D array, a single-channel audio/vibration time series.
    sample_rate : samples per second (Hz).
    n_peaks : how many dominant frequencies to return.
    min_freq, max_freq : restrict the search to this frequency band (e.g.
        to exclude DC/very-low-frequency drift, or to focus on a known
        expected mode range). `max_freq` defaults to the Nyquist frequency.
    window : window function applied before the FFT to reduce spectral
        leakage ("hann" or "none").
    """
    waveform = np.asarray(waveform, dtype=np.float64).ravel()
    n = waveform.shape[0]
    if n < 4:
        raise ValueError("waveform is too short to extract frequency content from.")

    if window == "hann":
        w = np.hanning(n)
        windowed = waveform * w
    elif window == "none":
        windowed = waveform
    else:
        raise ValueError(f"Unknown window '{window}'. Use 'hann' or 'none'.")

    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    mag = np.abs(spectrum)

    if max_freq is None:
        max_freq = sample_rate / 2.0
    band = (freqs >= min_freq) & (freqs <= max_freq)
    band_idx = np.nonzero(band)[0]
    if band_idx.size == 0:
        return []

    peak_idx_local, _ = find_peaks(mag[band_idx])
    if peak_idx_local.size == 0:
        # fall back to the single global maximum in-band if the spectrum
        # has no interior local maximum (e.g. a monotonic band edge)
        peak_idx_local = np.array([np.argmax(mag[band_idx])])
    peak_idx = band_idx[peak_idx_local]

    refined = [_parabolic_subbin_refine(mag, int(i), freqs) for i in peak_idx]
    refined.sort(key=lambda t: t[1], reverse=True)
    return refined[:n_peaks]
