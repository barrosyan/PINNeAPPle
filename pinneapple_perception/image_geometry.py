"""Extract boundary/geometry point clouds from an image -- e.g. a photo
or scan of a physical part's cross-section -- for use as a domain
boundary in `pinneapple_design.geometry` or as boundary-condition
selector data. Depends only on numpy/scipy (already core PINNeAPPle
dependencies).

Method: binarize the image (threshold), then extract the boundary as the
set of foreground pixels adjacent to a background pixel (a standard
morphological boundary: `foreground AND NOT erode(foreground)`), and
order them into a single closed contour by nearest-neighbor walking
(sufficient for simple, single-blob shapes; a genuinely disconnected or
highly concave/branching shape may need per-component ordering -- see
`extract_boundary_points`'s `ordered` parameter).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def _binary_erode(mask: np.ndarray) -> np.ndarray:
    """4-connected binary erosion without a scipy.ndimage.morphology
    dependency version mismatch -- implemented directly for portability."""
    eroded = mask.copy()
    eroded[1:, :] &= mask[:-1, :]
    eroded[:-1, :] &= mask[1:, :]
    eroded[:, 1:] &= mask[:, :-1]
    eroded[:, :-1] &= mask[:, 1:]
    return eroded


def extract_boundary_points(
    image: np.ndarray,
    threshold: Optional[float] = None,
    units_per_pixel: float = 1.0,
    ordered: bool = True,
) -> np.ndarray:
    """Extract the boundary of the foreground region(s) in `image` as a
    point cloud.

    Parameters
    ----------
    image : (H, W) grayscale array (float or int). RGB images should be
        converted to grayscale first (e.g. `image.mean(axis=-1)`).
    threshold : foreground/background cutoff. Defaults to the image's own
        mean value (Otsu-lite; good enough for a roughly bimodal image --
        pass an explicit value for anything else).
    units_per_pixel : physical length per pixel, applied to the returned
        coordinates.
    ordered : if True (default), order the boundary points into a single
        connected walk (nearest-neighbor greedy ordering) so the result
        can be used directly as a polygon/contour. If False, returns the
        unordered set of boundary pixel coordinates (faster, and correct
        for multi-component shapes where "ordered" doesn't make sense).

    Returns
    -------
    (N, 2) float32 array of (x, y) boundary point coordinates, in
    physical units (image row/column, i.e. y increases downward, matching
    standard image-array convention -- flip the y column if a
    y-increases-upward convention is needed downstream).
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"image must be a 2D grayscale array, got shape {image.shape}")
    if threshold is None:
        threshold = float(image.mean())

    mask = image >= threshold
    if not mask.any():
        return np.zeros((0, 2), dtype=np.float32)

    boundary_mask = mask & ~_binary_erode(mask)
    ys, xs = np.nonzero(boundary_mask)
    points = np.stack([xs, ys], axis=1).astype(np.float64)

    if ordered and points.shape[0] > 2:
        points = _order_by_nearest_neighbor(points)

    return (points * units_per_pixel).astype(np.float32)


def _order_by_nearest_neighbor(points: np.ndarray) -> np.ndarray:
    """Greedy nearest-neighbor walk through a point set -- turns an
    unordered boundary-pixel set into a traversable contour. O(N^2); fine
    for the few-hundred-to-few-thousand-point boundaries this function
    targets (a full image's boundary pixel count), not for arbitrarily
    large point clouds."""
    remaining = points.copy()
    ordered = [remaining[0]]
    remaining = np.delete(remaining, 0, axis=0)
    while remaining.shape[0] > 0:
        last = ordered[-1]
        dists = np.sum((remaining - last) ** 2, axis=1)
        idx = int(np.argmin(dists))
        ordered.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)
    return np.stack(ordered, axis=0)


def estimate_bounding_circle(points: np.ndarray) -> Tuple[float, float, float]:
    """Least-squares circle fit (algebraic Kasa method) to a boundary
    point cloud -- a quick, standard way to validate/characterize a
    roughly-circular extracted boundary (e.g. a pipe or shaft
    cross-section). Returns (center_x, center_y, radius)."""
    x, y = points[:, 0].astype(np.float64), points[:, 1].astype(np.float64)
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r = float(np.sqrt(sol[2] + cx ** 2 + cy ** 2))
    return float(cx), float(cy), r
