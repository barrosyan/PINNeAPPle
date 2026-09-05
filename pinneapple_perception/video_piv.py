"""Extract a velocity field from a video (image pair/sequence) using
cross-correlation Particle Image Velocimetry (PIV) -- the actual
standard technique experimental fluid dynamicists use to turn a video of
a seeded/textured flow into a velocity field, not a generic computer-
vision optical-flow method repurposed for this. Depends only on
numpy/scipy (both already core PINNeAPPle dependencies) -- no new
third-party dependency needed.

Algorithm (per pair of consecutive frames):
  1. Divide the first frame into square interrogation windows on a grid
     (with optional overlap).
  2. For each window, cross-correlate it (FFT-based, normalized) against
     the corresponding region of the second frame.
  3. The correlation peak location gives the most likely pixel
     displacement for that window; a 3-point parabolic fit around the
     peak in each axis gives sub-pixel accuracy (the standard PIV
     sub-pixel refinement, e.g. Raffel et al., "Particle Image
     Velocimetry: A Practical Guide").
  4. Displacement -> velocity via the frame interval `dt` and an optional
     physical calibration (`units_per_pixel`).

The extracted (x, y, u, v) field is a plain numpy array bundle -- pass it
directly as `x_data`/`y_data` to `solve_pde`/`compile_problem` (e.g. as a
`DataConstraint` on a `navier_stokes_incompressible`-family preset) to
train a PINN surrogate constrained by real video-derived velocity
observations, the same way any other sparse tracking/data-constraint
this library already supports is used.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def _parabolic_subpixel_peak(corr: np.ndarray, peak: Tuple[int, int]) -> Tuple[float, float]:
    """3-point parabolic sub-pixel refinement around an integer peak
    location in a 2D correlation surface (standard PIV sub-pixel
    estimator). Falls back to the integer peak if it sits on the
    correlation surface's edge (no neighbor on one side)."""
    py, px = peak
    h, w = corr.shape
    dy = dx = 0.0
    if 0 < px < w - 1:
        c_l, c_c, c_r = corr[py, px - 1], corr[py, px], corr[py, px + 1]
        denom = (c_l - 2 * c_c + c_r)
        if abs(denom) > 1e-12:
            dx = 0.5 * (c_l - c_r) / denom
    if 0 < py < h - 1:
        c_t, c_c, c_b = corr[py - 1, px], corr[py, px], corr[py + 1, px]
        denom = (c_t - 2 * c_c + c_b)
        if abs(denom) > 1e-12:
            dy = 0.5 * (c_t - c_b) / denom
    return float(py + dy), float(px + dx)


def _normalized_cross_correlate(window: np.ndarray, search: np.ndarray) -> np.ndarray:
    """FFT-based normalized cross-correlation of `window` against every
    same-size sub-region of the (larger) `search` region. Returns a
    correlation surface whose shape is
    (search.shape[0]-window.shape[0]+1, search.shape[1]-window.shape[1]+1).
    """
    wh, ww = window.shape
    sh, sw = search.shape
    out_h, out_w = sh - wh + 1, sw - ww + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("search region must be larger than the interrogation window")

    w_norm = window - window.mean()
    w_energy = np.sqrt(np.sum(w_norm ** 2)) + 1e-12

    # Correlation via FFT: correlate(search, window) == convolve(search, flip(window))
    fsize = (sh + wh - 1, sw + ww - 1)
    F_search = np.fft.rfft2(search, s=fsize)
    F_window = np.fft.rfft2(w_norm[::-1, ::-1], s=fsize)
    full = np.fft.irfft2(F_search * F_window, s=fsize)
    corr_full = full[wh - 1:wh - 1 + out_h, ww - 1:ww - 1 + out_w]

    # Local energy normalization (per-window mean-subtracted search patch norm)
    search_sq_cumsum = np.cumsum(np.cumsum(search ** 2, axis=0), axis=1)
    search_cumsum = np.cumsum(np.cumsum(search, axis=0), axis=1)

    def _window_sum(cs, i, j, h, w):
        total = cs[i + h - 1, j + w - 1]
        if i > 0:
            total -= cs[i - 1, j + w - 1]
        if j > 0:
            total -= cs[i + h - 1, j - 1]
        if i > 0 and j > 0:
            total += cs[i - 1, j - 1]
        return total

    denom = np.empty((out_h, out_w), dtype=np.float64)
    for i in range(out_h):
        for j in range(out_w):
            s_sum = _window_sum(search_cumsum, i, j, wh, ww)
            s_sq_sum = _window_sum(search_sq_cumsum, i, j, wh, ww)
            mean = s_sum / (wh * ww)
            var = max(s_sq_sum - (s_sum ** 2) / (wh * ww), 0.0)
            denom[i, j] = np.sqrt(var) * w_energy + 1e-12

    return corr_full / denom


def piv_velocity_field(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    window_size: int = 32,
    search_margin: int = 16,
    step: Optional[int] = None,
    dt: float = 1.0,
    units_per_pixel: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Cross-correlation PIV velocity field between two grayscale frames.

    Parameters
    ----------
    frame_a, frame_b : (H, W) arrays, same shape (grayscale/single-channel;
        pass e.g. `frame[..., :3].mean(axis=-1)` for an RGB frame).
    window_size : interrogation window side length, in pixels.
    search_margin : how many pixels beyond the window the search region in
        frame_b extends on each side (bounds the maximum detectable
        displacement to +/- search_margin pixels).
    step : grid spacing between window centers, in pixels (defaults to
        `window_size`, i.e. no overlap).
    dt : time interval between the two frames (s) -- divides displacement
        to give velocity.
    units_per_pixel : physical length per pixel (e.g. meters/pixel from a
        camera calibration) -- multiplies displacement to convert from
        pixels to physical length before dividing by dt.

    Returns
    -------
    dict with keys "x", "y" (window-center pixel coordinates, in physical
    units) and "u", "v" (velocity components, physical units per second).
    All four are flattened 1D arrays of the same length (one entry per
    interrogation window), directly usable as a `DataConstraint`'s
    x/y data for `solve_pde`.
    """
    frame_a = np.asarray(frame_a, dtype=np.float64)
    frame_b = np.asarray(frame_b, dtype=np.float64)
    if frame_a.shape != frame_b.shape:
        raise ValueError(f"frame_a and frame_b must have the same shape, got {frame_a.shape} vs {frame_b.shape}")
    h, w = frame_a.shape
    step = step or window_size

    xs, ys, us, vs = [], [], [], []
    half_w = window_size // 2
    margin = half_w + search_margin
    # Windows whose full search region (not just the interrogation window)
    # would be clipped by the image border are skipped entirely, rather
    # than searched over a truncated (and therefore asymmetric) region --
    # a truncated search region can't represent the full range of
    # displacements it was meant to cover, so a genuine large motion near
    # the edge has no valid match candidate inside it and the correlation
    # peak becomes spurious noise instead of just "less precise". This
    # matches standard PIV practice (see Raffel et al.): near-border
    # vectors are excluded rather than silently returned as unreliable
    # numbers indistinguishable from good ones.
    for cy in range(margin, h - margin, step):
        for cx in range(margin, w - margin, step):
            window = frame_a[cy - half_w:cy + half_w, cx - half_w:cx + half_w]

            sy0 = cy - margin
            sy1 = cy + margin
            sx0 = cx - margin
            sx1 = cx + margin
            search = frame_b[sy0:sy1, sx0:sx1]

            if search.shape[0] < window.shape[0] or search.shape[1] < window.shape[1]:
                continue

            corr = _normalized_cross_correlate(window, search)
            peak = np.unravel_index(np.argmax(corr), corr.shape)
            py, px = _parabolic_subpixel_peak(corr, peak)

            # Displacement of the window's top-left corner in the search
            # region, minus its top-left corner in frame_a's coordinates
            # (both measured from the search region's own origin) gives
            # the net pixel displacement.
            dy = (sy0 + py) - (cy - half_w)
            dx = (sx0 + px) - (cx - half_w)

            xs.append(cx * units_per_pixel)
            ys.append(cy * units_per_pixel)
            us.append(dx * units_per_pixel / dt)
            vs.append(dy * units_per_pixel / dt)

    return {
        "x": np.asarray(xs, dtype=np.float32),
        "y": np.asarray(ys, dtype=np.float32),
        "u": np.asarray(us, dtype=np.float32),
        "v": np.asarray(vs, dtype=np.float32),
    }


def piv_velocity_sequence(
    frames: np.ndarray,
    dt: float = 1.0,
    **piv_kwargs,
) -> Dict[str, np.ndarray]:
    """Run `piv_velocity_field` on every consecutive pair in a video
    (`frames`: (T, H, W) array) and stack the results with a `t` column,
    ready for a 3D (x, y, t) `DataConstraint`."""
    frames = np.asarray(frames)
    if frames.ndim != 3:
        raise ValueError(f"frames must be a (T, H, W) array, got shape {frames.shape}")
    xs, ys, ts, us, vs = [], [], [], [], []
    for i in range(frames.shape[0] - 1):
        field = piv_velocity_field(frames[i], frames[i + 1], dt=dt, **piv_kwargs)
        n = field["x"].shape[0]
        xs.append(field["x"])
        ys.append(field["y"])
        ts.append(np.full(n, i * dt, dtype=np.float32))
        us.append(field["u"])
        vs.append(field["v"])
    return {
        "x": np.concatenate(xs), "y": np.concatenate(ys), "t": np.concatenate(ts),
        "u": np.concatenate(us), "v": np.concatenate(vs),
    }
