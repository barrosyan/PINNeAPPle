"""Rich SDF (Signed Distance Function) library for 2D and 3D shapes.

All functions accept numpy arrays of shape (..., D) or torch tensors of the
same shape, and return signed distances of shape (...,).

Convention: negative inside the shape, positive outside, zero on the boundary.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _norm(x, axis=-1):
    return np.sqrt(np.sum(x ** 2, axis=axis))


def _clamp(x, lo, hi):
    return np.clip(x, lo, hi)


def _dot(a, b, axis=-1):
    return np.sum(a * b, axis=axis)


# ---------------------------------------------------------------------------
# 2D SDF primitives
# ---------------------------------------------------------------------------

def sdf2d_circle(p: np.ndarray, center: Tuple[float, float], radius: float) -> np.ndarray:
    """SDF for a 2D circle. Negative inside."""
    c = np.asarray(center, dtype=np.float64)
    return _norm(p - c) - radius


def sdf2d_rectangle(
    p: np.ndarray,
    center: Tuple[float, float],
    half_extents: Tuple[float, float],
) -> np.ndarray:
    """SDF for an axis-aligned rectangle. Negative inside."""
    c = np.asarray(center, dtype=np.float64)
    h = np.asarray(half_extents, dtype=np.float64)
    q = np.abs(p - c) - h
    return _norm(np.maximum(q, 0.0)) + np.minimum(np.max(q, axis=-1), 0.0)


def sdf2d_rounded_rectangle(
    p: np.ndarray,
    center: Tuple[float, float],
    half_extents: Tuple[float, float],
    radius: float,
) -> np.ndarray:
    """SDF for a rectangle with rounded corners."""
    return sdf2d_rectangle(p, center, (half_extents[0] - radius, half_extents[1] - radius)) - radius


def sdf2d_ellipse(
    p: np.ndarray,
    center: Tuple[float, float],
    semi_axes: Tuple[float, float],
) -> np.ndarray:
    """Approximate SDF for a 2D ellipse (Inigo Quilez formula)."""
    c = np.asarray(center, dtype=np.float64)
    ab = np.asarray(semi_axes, dtype=np.float64)
    q = p - c
    # normalize to unit-circle space
    q_n = q / ab
    l = _norm(q_n)
    # approximate distance
    d = l - 1.0
    # scale back to world space (approximate)
    scale = np.min(ab)
    return d * scale


def sdf2d_capsule(
    p: np.ndarray,
    a: Tuple[float, float],
    b: Tuple[float, float],
    radius: float,
) -> np.ndarray:
    """SDF for a 2D capsule (stadium) between points a and b."""
    pa = p - np.asarray(a, dtype=np.float64)
    ba = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    h = _clamp(_dot(pa, ba) / _dot(ba, ba), 0.0, 1.0)[..., None]
    return _norm(pa - h * ba) - radius


def sdf2d_line_segment(
    p: np.ndarray,
    a: Tuple[float, float],
    b: Tuple[float, float],
    thickness: float = 0.0,
) -> np.ndarray:
    """SDF to a line segment (with optional thickness)."""
    return sdf2d_capsule(p, a, b, thickness)


def sdf2d_triangle(
    p: np.ndarray,
    v0: Tuple[float, float],
    v1: Tuple[float, float],
    v2: Tuple[float, float],
) -> np.ndarray:
    """SDF for a 2D triangle (signed: negative inside)."""
    p0 = np.asarray(v0, dtype=np.float64)
    p1 = np.asarray(v1, dtype=np.float64)
    p2 = np.asarray(v2, dtype=np.float64)

    def edge_dist(pa, pb):
        e = pb - pa
        w = p - pa
        b_proj = w - e * _clamp(_dot(w, e) / _dot(e, e), 0.0, 1.0)[..., None]
        return _dot(b_proj, b_proj)

    d = np.minimum(np.minimum(edge_dist(p0, p1), edge_dist(p1, p2)), edge_dist(p2, p0))

    def sign_part(pa, pb, pc):
        return (
            np.sign((p[:, 0] - pa[0]) * (pb[1] - pa[1]) - (p[:, 1] - pa[1]) * (pb[0] - pa[0]))
        )

    s = (
        sign_part(p0, p1, p2)
        + sign_part(p1, p2, p0)
        + sign_part(p2, p0, p1)
    ) < 2.0

    return np.where(s, 1.0, -1.0) * np.sqrt(d)


def sdf2d_annulus(
    p: np.ndarray,
    center: Tuple[float, float],
    r_inner: float,
    r_outer: float,
) -> np.ndarray:
    """SDF for a 2D annular region (ring). Negative between the two radii."""
    c = np.asarray(center, dtype=np.float64)
    r = _norm(p - c)
    d_outer = r - r_outer
    d_inner = r_inner - r
    return np.maximum(d_outer, d_inner)


def sdf2d_sector(
    p: np.ndarray,
    center: Tuple[float, float],
    radius: float,
    angle_start: float,
    angle_end: float,
) -> np.ndarray:
    """SDF for a circular sector (pie slice)."""
    c = np.asarray(center, dtype=np.float64)
    q = p - c
    angle = np.arctan2(q[:, 1], q[:, 0])

    a_start = float(angle_start) % (2 * math.pi)
    a_end = float(angle_end) % (2 * math.pi)

    # angle distance from sector
    if a_end >= a_start:
        inside_angle = (angle >= a_start) & (angle <= a_end)
    else:
        inside_angle = (angle >= a_start) | (angle <= a_end)

    d_circle = _norm(q) - radius
    # points inside sector angle: dist = d_circle
    # points outside: project to nearest sector edge
    d_edge1 = sdf2d_capsule(p, center, (center[0] + radius * math.cos(a_start), center[1] + radius * math.sin(a_start)), 0.0)
    d_edge2 = sdf2d_capsule(p, center, (center[0] + radius * math.cos(a_end), center[1] + radius * math.sin(a_end)), 0.0)

    return np.where(inside_angle, d_circle, np.minimum(d_edge1, d_edge2))


def sdf2d_convex_polygon(
    p: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    """SDF for a convex polygon given as ordered vertices (Nx2 array)."""
    verts = np.asarray(vertices, dtype=np.float64)
    n = len(verts)
    d = np.full(len(p), np.inf)
    sign = np.ones(len(p))

    for i in range(n):
        va = verts[i]
        vb = verts[(i + 1) % n]
        e = vb - va
        w = p - va
        proj = _clamp(_dot(w, e) / _dot(e, e), 0.0, 1.0)
        dist2 = np.sum((w - proj[..., None] * e) ** 2, axis=-1)
        d = np.minimum(d, dist2)
        # winding
        c1 = (p[:, 1] >= va[1])
        c2 = (p[:, 1] < vb[1])
        c3 = ((vb[0] - va[0]) * (p[:, 1] - va[1]) > (p[:, 0] - va[0]) * (vb[1] - va[1]))
        winding_sign = np.where(c1 & c2 & c3, -1.0, 1.0)
        winding_sign2 = np.where((~c1) & (~c2) & (~c3), -1.0, 1.0)
        sign *= winding_sign * winding_sign2

    return sign * np.sqrt(d)


# ---------------------------------------------------------------------------
# 3D SDF primitives
# ---------------------------------------------------------------------------

def sdf3d_sphere(p: np.ndarray, center: Tuple[float, float, float], radius: float) -> np.ndarray:
    """SDF for a 3D sphere."""
    c = np.asarray(center, dtype=np.float64)
    return _norm(p - c) - radius


def sdf3d_box(
    p: np.ndarray,
    center: Tuple[float, float, float],
    half_extents: Tuple[float, float, float],
) -> np.ndarray:
    """SDF for an axis-aligned 3D box."""
    c = np.asarray(center, dtype=np.float64)
    h = np.asarray(half_extents, dtype=np.float64)
    q = np.abs(p - c) - h
    return _norm(np.maximum(q, 0.0)) + np.minimum(np.max(q, axis=-1), 0.0)


def sdf3d_cylinder(
    p: np.ndarray,
    center: Tuple[float, float, float],
    radius: float,
    height: float,
    axis: int = 2,
) -> np.ndarray:
    """SDF for a finite cylinder aligned with `axis` (0=x, 1=y, 2=z)."""
    c = np.asarray(center, dtype=np.float64)
    axes = [0, 1, 2]
    radial_axes = [a for a in axes if a != axis]
    q_radial = p[:, radial_axes] - c[radial_axes]
    q_axial = p[:, axis] - c[axis]
    d_r = _norm(q_radial) - radius
    d_a = np.abs(q_axial) - height * 0.5
    d2 = np.minimum(np.maximum(d_r, d_a), 0.0) + _norm(
        np.stack([np.maximum(d_r, 0.0), np.maximum(d_a, 0.0)], axis=-1)
    )
    return d2


def sdf3d_torus(
    p: np.ndarray,
    center: Tuple[float, float, float],
    R: float,
    r: float,
    axis: int = 2,
) -> np.ndarray:
    """SDF for a torus with major radius R and tube radius r."""
    c = np.asarray(center, dtype=np.float64)
    axes = [0, 1, 2]
    radial_axes = [a for a in axes if a != axis]
    q_xy = p[:, radial_axes] - c[radial_axes]
    q_z = p[:, axis] - c[axis]
    q = np.stack([_norm(q_xy) - R, q_z], axis=-1)
    return _norm(q) - r


def sdf3d_capsule(
    p: np.ndarray,
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    radius: float,
) -> np.ndarray:
    """SDF for a 3D capsule between points a and b."""
    pa = p - np.asarray(a, dtype=np.float64)
    ba = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    h = _clamp(_dot(pa, ba) / _dot(ba, ba), 0.0, 1.0)[..., None]
    return _norm(pa - h * ba) - radius


def sdf3d_rounded_box(
    p: np.ndarray,
    center: Tuple[float, float, float],
    half_extents: Tuple[float, float, float],
    radius: float,
) -> np.ndarray:
    """SDF for a 3D box with rounded edges."""
    h = np.asarray(half_extents, dtype=np.float64) - radius
    return sdf3d_box(p, center, tuple(h)) - radius


def sdf3d_cone(
    p: np.ndarray,
    apex: Tuple[float, float, float],
    height: float,
    radius: float,
) -> np.ndarray:
    """SDF for an upward-pointing cone with apex at `apex`."""
    a = np.asarray(apex, dtype=np.float64)
    q = p - a
    d_axial = q[:, 2]
    d_radial = _norm(q[:, :2])
    slope = radius / height
    q2 = np.stack([d_radial, -d_axial], axis=-1)
    n = np.array([slope, 1.0]) / math.sqrt(1 + slope**2)
    d = _dot(q2, n) - 0.0
    inside = (d_axial >= 0) & (d_axial <= height) & (d_radial <= slope * d_axial)
    dist_edge = np.abs(d_radial - slope * d_axial) / math.sqrt(1 + slope**2)
    return np.where(inside, -dist_edge, dist_edge)


# ---------------------------------------------------------------------------
# Boolean operations
# ---------------------------------------------------------------------------

def sdf_union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Boolean union: min(a, b)."""
    return np.minimum(a, b)


def sdf_intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Boolean intersection: max(a, b)."""
    return np.maximum(a, b)


def sdf_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Boolean difference: a minus b = max(a, -b)."""
    return np.maximum(a, -b)


def sdf_smooth_union(a: np.ndarray, b: np.ndarray, k: float = 0.1) -> np.ndarray:
    """Smooth minimum (R-function). k controls blend radius."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return a * (1.0 - h) + b * h - k * h * (1.0 - h)


def sdf_smooth_intersection(a: np.ndarray, b: np.ndarray, k: float = 0.1) -> np.ndarray:
    """Smooth maximum."""
    h = np.clip(0.5 - 0.5 * (b - a) / k, 0.0, 1.0)
    return a * (1.0 - h) + b * h + k * h * (1.0 - h)


def sdf_smooth_difference(a: np.ndarray, b: np.ndarray, k: float = 0.1) -> np.ndarray:
    """Smooth difference."""
    return sdf_smooth_intersection(a, -b, k)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def sdf_translate(
    sdf_fn: Callable[[np.ndarray], np.ndarray],
    offset: Sequence[float],
) -> Callable[[np.ndarray], np.ndarray]:
    """Translate an SDF by applying an inverse offset to the query point."""
    off = np.asarray(offset, dtype=np.float64)
    def wrapped(p: np.ndarray) -> np.ndarray:
        return sdf_fn(p - off)
    return wrapped


def sdf_scale(
    sdf_fn: Callable[[np.ndarray], np.ndarray],
    factor: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Uniformly scale an SDF."""
    def wrapped(p: np.ndarray) -> np.ndarray:
        return sdf_fn(p / factor) * factor
    return wrapped


def sdf_rotate_2d(
    sdf_fn: Callable[[np.ndarray], np.ndarray],
    angle_rad: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Rotate a 2D SDF by angle_rad (counter-clockwise)."""
    c, s = math.cos(-angle_rad), math.sin(-angle_rad)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    def wrapped(p: np.ndarray) -> np.ndarray:
        p_rot = p @ R.T
        return sdf_fn(p_rot)
    return wrapped


def sdf_mirror_x(sdf_fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    """Mirror an SDF across the y-axis (flip x)."""
    def wrapped(p: np.ndarray) -> np.ndarray:
        q = p.copy()
        q[:, 0] = np.abs(q[:, 0])
        return sdf_fn(q)
    return wrapped


def sdf_mirror_y(sdf_fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    """Mirror an SDF across the x-axis (flip y)."""
    def wrapped(p: np.ndarray) -> np.ndarray:
        q = p.copy()
        q[:, 1] = np.abs(q[:, 1])
        return sdf_fn(q)
    return wrapped


# ---------------------------------------------------------------------------
# Modifiers
# ---------------------------------------------------------------------------

def sdf_onion(sdf_fn: Callable[[np.ndarray], np.ndarray], thickness: float) -> Callable[[np.ndarray], np.ndarray]:
    """Create a shell (onion skin) of given thickness."""
    def wrapped(p: np.ndarray) -> np.ndarray:
        return np.abs(sdf_fn(p)) - thickness
    return wrapped


def sdf_elongate_2d(
    sdf_fn: Callable[[np.ndarray], np.ndarray],
    dx: float = 0.0,
    dy: float = 0.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Elongate a 2D SDF shape along x and/or y axes."""
    h = np.array([dx, dy], dtype=np.float64) * 0.5
    def wrapped(p: np.ndarray) -> np.ndarray:
        q = np.abs(p) - h
        q_clamped = np.maximum(q, 0.0)
        q_shifted = np.minimum(q, 0.0)
        return sdf_fn(q_shifted + q_clamped)
    return wrapped


def sdf_repeat_2d(
    sdf_fn: Callable[[np.ndarray], np.ndarray],
    period: Tuple[float, float],
    repetitions: Optional[Tuple[int, int]] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Tile a 2D SDF with given (px, py) period."""
    px, py = float(period[0]), float(period[1])
    def wrapped(p: np.ndarray) -> np.ndarray:
        q = p.copy()
        if px > 0:
            q[:, 0] = p[:, 0] - px * np.round(p[:, 0] / px)
        if py > 0:
            q[:, 1] = p[:, 1] - py * np.round(p[:, 1] / py)
        if repetitions is not None:
            rx, ry = repetitions
            if px > 0 and rx > 0:
                q[:, 0] = np.clip(q[:, 0], -rx * px * 0.5, rx * px * 0.5)
            if py > 0 and ry > 0:
                q[:, 1] = np.clip(q[:, 1], -ry * py * 0.5, ry * py * 0.5)
        return sdf_fn(q)
    return wrapped


# ---------------------------------------------------------------------------
# SDF class: composable callable wrapper
# ---------------------------------------------------------------------------

class SDF:
    """Object-oriented wrapper around an SDF callable.

    Supports chaining operations:
        s = SDF(sdf2d_circle, (0, 0), 0.5)
        s2 = SDF(sdf2d_rectangle, (0.3, 0), (0.2, 0.2))
        combined = s | s2      # union
        diff = s - s2          # difference
        inter = s & s2         # intersection
    """

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], *args, **kwargs):
        if args or kwargs:
            self._fn = lambda p: fn(p, *args, **kwargs)
        else:
            self._fn = fn

    def __call__(self, p: np.ndarray) -> np.ndarray:
        return self._fn(p)

    def __or__(self, other: "SDF") -> "SDF":
        return SDF(lambda p: sdf_union(self(p), other(p)))

    def __and__(self, other: "SDF") -> "SDF":
        return SDF(lambda p: sdf_intersection(self(p), other(p)))

    def __sub__(self, other: "SDF") -> "SDF":
        return SDF(lambda p: sdf_difference(self(p), other(p)))

    def translate(self, offset: Sequence[float]) -> "SDF":
        return SDF(sdf_translate(self._fn, offset))

    def scale(self, factor: float) -> "SDF":
        return SDF(sdf_scale(self._fn, factor))

    def rotate(self, angle_rad: float) -> "SDF":
        return SDF(sdf_rotate_2d(self._fn, angle_rad))

    def onion(self, thickness: float) -> "SDF":
        return SDF(sdf_onion(self._fn, thickness))

    def smooth_union(self, other: "SDF", k: float = 0.1) -> "SDF":
        return SDF(lambda p: sdf_smooth_union(self(p), other(p), k))

    def sample_boundary(
        self,
        n: int,
        bounds_min: Tuple[float, ...],
        bounds_max: Tuple[float, ...],
        tol: float = 0.02,
        seed: int = 0,
    ) -> np.ndarray:
        """Sample n points near the zero-level-set (boundary) by rejection."""
        rng = np.random.default_rng(seed)
        bmin = np.asarray(bounds_min)
        bmax = np.asarray(bounds_max)
        dim = len(bmin)
        collected = []
        while sum(len(c) for c in collected) < n:
            cands = bmin + rng.random((n * 20, dim)) * (bmax - bmin)
            cands = cands.astype(np.float64)
            d = np.abs(self(cands))
            pts = cands[d < tol]
            if len(pts) > 0:
                collected.append(pts)
        return np.concatenate(collected, axis=0)[:n].astype(np.float32)

    def sample_interior(
        self,
        n: int,
        bounds_min: Tuple[float, ...],
        bounds_max: Tuple[float, ...],
        seed: int = 0,
    ) -> np.ndarray:
        """Sample n points inside the domain (sdf < 0) by rejection."""
        rng = np.random.default_rng(seed)
        bmin = np.asarray(bounds_min)
        bmax = np.asarray(bounds_max)
        dim = len(bmin)
        collected = []
        while sum(len(c) for c in collected) < n:
            cands = bmin + rng.random((n * 5, dim)) * (bmax - bmin)
            cands = cands.astype(np.float64)
            d = self(cands)
            pts = cands[d < 0]
            if len(pts) > 0:
                collected.append(pts)
        return np.concatenate(collected, axis=0)[:n].astype(np.float32)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def circle(center=(0.0, 0.0), radius=0.5) -> SDF:
    return SDF(sdf2d_circle, center, radius)


def rectangle(center=(0.0, 0.0), half_extents=(0.5, 0.5)) -> SDF:
    return SDF(sdf2d_rectangle, center, half_extents)


def ellipse(center=(0.0, 0.0), semi_axes=(0.5, 0.3)) -> SDF:
    return SDF(sdf2d_ellipse, center, semi_axes)


def annulus(center=(0.0, 0.0), r_inner=0.2, r_outer=0.5) -> SDF:
    return SDF(sdf2d_annulus, center, r_inner, r_outer)


def capsule2d(a=(0.0, -0.5), b=(0.0, 0.5), radius=0.2) -> SDF:
    return SDF(sdf2d_capsule, a, b, radius)


def sphere3d(center=(0.0, 0.0, 0.0), radius=0.5) -> SDF:
    return SDF(sdf3d_sphere, center, radius)


def box3d(center=(0.0, 0.0, 0.0), half_extents=(0.5, 0.5, 0.5)) -> SDF:
    return SDF(sdf3d_box, center, half_extents)


def cylinder3d(center=(0.0, 0.0, 0.0), radius=0.5, height=1.0, axis=2) -> SDF:
    return SDF(sdf3d_cylinder, center, radius, height, axis)


def torus3d(center=(0.0, 0.0, 0.0), R=0.5, r=0.15, axis=2) -> SDF:
    return SDF(sdf3d_torus, center, R, r, axis)
