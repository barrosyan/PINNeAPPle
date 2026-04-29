"""Constructive Solid Geometry (CSG) for 2D/3D PINN collocation domains.

Supports union, intersection, and difference of primitive shapes, enabling:
- L-shapes, T-junctions, annuli with holes
- Arbitrary 2D geometries for PINN collocation

All shapes implement a signed distance function (SDF) and interior / boundary
sampling that integrates with PINNeAPPle's PhysicsDomain2D pattern.

Convention for SDF:
    sdf(x) < 0  -->  inside the shape
    sdf(x) = 0  -->  on the boundary
    sdf(x) > 0  -->  outside the shape

Quick examples::

    from pinneaple_geom.csg import lshape, annulus, channel_with_hole

    domain  = lshape(2.0, 2.0, 1.0, 1.0)
    pts_int = domain.sample_interior(4096)   # (N,2) numpy float32
    pts_bnd = domain.sample_boundary(512)

    ring    = annulus(cx=0, cy=0, r_inner=0.2, r_outer=1.0)
    chan    = channel_with_hole(4.0, 1.0, 1.0, 0.5, 0.15)
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class SDFShape:
    """Base class for SDF-based 2D shapes.

    Subclasses must implement:
        sdf(x)           : (N,2) array -> (N,) signed distance
        sample_interior  : int -> (N,2) float32
        sample_boundary  : int -> (N,2) float32
        bounds_min / bounds_max : axis-aligned bounding box
    """

    bounds_min: Tuple[float, float] = (-1.0, -1.0)
    bounds_max: Tuple[float, float] = (1.0,  1.0)

    def sdf(self, x: np.ndarray) -> np.ndarray:
        """Signed distance function.  Negative inside, 0 on boundary, positive outside."""
        raise NotImplementedError

    def contains(self, x: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        """Boolean mask: True if point is strictly inside (or within tol of boundary)."""
        return self.sdf(np.asarray(x, dtype=np.float64)) <= tol

    def on_boundary(self, x: np.ndarray, tol: float = 1e-3) -> np.ndarray:
        """Boolean mask: True if point is near the boundary."""
        return np.abs(self.sdf(np.asarray(x, dtype=np.float64))) <= tol

    def sample_interior(self, n: int, *, seed: int = 0) -> np.ndarray:
        """Sample n interior collocation points by rejection sampling from bbox."""
        raise NotImplementedError

    def sample_boundary(self, n: int, *, seed: int = 0) -> np.ndarray:
        """Sample n points approximately on the boundary."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # CSG operators  (return new SDFShape objects)
    # ------------------------------------------------------------------

    def __add__(self, other: "SDFShape") -> "CSGUnion":
        """Union: a + b."""
        return CSGUnion(self, other)

    def __mul__(self, other: "SDFShape") -> "CSGIntersection":
        """Intersection: a * b."""
        return CSGIntersection(self, other)

    def __sub__(self, other: "SDFShape") -> "CSGDifference":
        """Difference: a - b."""
        return CSGDifference(self, other)

    # ------------------------------------------------------------------
    # Shared rejection sampler (works for any CSG tree)
    # ------------------------------------------------------------------

    def _rejection_sample_interior(
        self,
        n: int,
        seed: int = 0,
        oversample: int = 8,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        bmin = np.asarray(self.bounds_min, dtype=np.float64)
        bmax = np.asarray(self.bounds_max, dtype=np.float64)
        collected: List[np.ndarray] = []
        total = 0
        while total < n:
            cands = bmin + rng.random((n * oversample, 2)) * (bmax - bmin)
            mask = self.contains(cands)
            pts  = cands[mask]
            if len(pts) > 0:
                collected.append(pts)
                total += len(pts)
        return np.concatenate(collected, axis=0)[:n].astype(np.float32)

    def _rejection_sample_boundary(
        self,
        n: int,
        seed: int = 0,
        tol: float = 1e-2,
        oversample: int = 50,
    ) -> np.ndarray:
        """Sample boundary by rejection from bbox (slow but universal).

        For shapes with analytic boundary parametrisations the subclass
        should override this for efficiency.
        """
        rng = np.random.default_rng(seed)
        bmin = np.asarray(self.bounds_min, dtype=np.float64)
        bmax = np.asarray(self.bounds_max, dtype=np.float64)
        scale = np.linalg.norm(bmax - bmin)
        collected: List[np.ndarray] = []
        total = 0
        while total < n:
            cands = bmin + rng.random((n * oversample, 2)) * (bmax - bmin)
            d = np.abs(self.sdf(cands))
            mask = d <= tol * scale
            pts  = cands[mask]
            if len(pts) > 0:
                collected.append(pts)
                total += len(pts)
        return np.concatenate(collected, axis=0)[:n].astype(np.float32)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class CSGRectangle(SDFShape):
    """Axis-aligned rectangle.

    Parameters
    ----------
    x_min, y_min : lower-left corner
    x_max, y_max : upper-right corner
    """

    def __init__(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
    ) -> None:
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.x_max = float(x_max)
        self.y_max = float(y_max)
        self.bounds_min = (x_min, y_min)
        self.bounds_max = (x_max, y_max)
        # centre and half-extents for SDF
        self._cx = (x_min + x_max) * 0.5
        self._cy = (y_min + y_max) * 0.5
        self._hx = (x_max - x_min) * 0.5
        self._hy = (y_max - y_min) * 0.5

    def sdf(self, x: np.ndarray) -> np.ndarray:
        p = np.asarray(x, dtype=np.float64)
        qx = np.abs(p[:, 0] - self._cx) - self._hx
        qy = np.abs(p[:, 1] - self._cy) - self._hy
        return (
            np.sqrt(np.maximum(qx, 0.0)**2 + np.maximum(qy, 0.0)**2)
            + np.minimum(np.maximum(qx, qy), 0.0)
        )

    def sample_interior(self, n: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        pts = np.empty((n, 2), dtype=np.float32)
        pts[:, 0] = rng.uniform(self.x_min, self.x_max, n)
        pts[:, 1] = rng.uniform(self.y_min, self.y_max, n)
        return pts

    def sample_boundary(self, n: int, *, seed: int = 0) -> np.ndarray:
        """Sample uniformly on the 4 edges."""
        rng = np.random.default_rng(seed)
        W = self.x_max - self.x_min
        H = self.y_max - self.y_min
        perimeter = 2.0 * (W + H)
        # allocate proportionally
        n_bot = max(1, round(n * W / perimeter))
        n_top = max(1, round(n * W / perimeter))
        n_lft = max(1, round(n * H / perimeter))
        n_rgt = n - n_bot - n_top - n_lft
        if n_rgt < 1:
            n_rgt = 1

        def _seg(x0, y0, x1, y1, k):
            t = rng.uniform(0.0, 1.0, k)
            return np.stack(
                [x0 + t * (x1 - x0), y0 + t * (y1 - y0)], axis=1
            )

        parts = [
            _seg(self.x_min, self.y_min, self.x_max, self.y_min, n_bot),
            _seg(self.x_min, self.y_max, self.x_max, self.y_max, n_top),
            _seg(self.x_min, self.y_min, self.x_min, self.y_max, n_lft),
            _seg(self.x_max, self.y_min, self.x_max, self.y_max, n_rgt),
        ]
        return np.concatenate(parts, axis=0).astype(np.float32)


class CSGCircle(SDFShape):
    """2D circle.

    Parameters
    ----------
    center_x, center_y : centre coordinates
    radius             : radius
    """

    def __init__(self, center_x: float, center_y: float, radius: float) -> None:
        self.cx = float(center_x)
        self.cy = float(center_y)
        self.r  = float(radius)
        r = self.r
        self.bounds_min = (self.cx - r, self.cy - r)
        self.bounds_max = (self.cx + r, self.cy + r)

    def sdf(self, x: np.ndarray) -> np.ndarray:
        p = np.asarray(x, dtype=np.float64)
        dx = p[:, 0] - self.cx
        dy = p[:, 1] - self.cy
        return np.sqrt(dx**2 + dy**2) - self.r

    def sample_interior(self, n: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        # polar method: r = R * sqrt(u), theta uniform
        u   = rng.uniform(0.0, 1.0, n)
        th  = rng.uniform(0.0, 2.0 * math.pi, n)
        r   = self.r * np.sqrt(u)
        pts = np.stack([self.cx + r * np.cos(th), self.cy + r * np.sin(th)], axis=1)
        return pts.astype(np.float32)

    def sample_boundary(self, n: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        th  = rng.uniform(0.0, 2.0 * math.pi, n)
        pts = np.stack(
            [self.cx + self.r * np.cos(th), self.cy + self.r * np.sin(th)],
            axis=1,
        )
        return pts.astype(np.float32)


class CSGEllipse(SDFShape):
    """2D axis-aligned ellipse.

    Uses the Inigo Quilez approximate SDF.

    Parameters
    ----------
    center_x, center_y : centre coordinates
    a, b               : semi-axes (x and y directions)
    """

    def __init__(
        self,
        center_x: float,
        center_y: float,
        a: float,
        b: float,
    ) -> None:
        self.cx = float(center_x)
        self.cy = float(center_y)
        self.a  = float(a)
        self.b  = float(b)
        self.bounds_min = (self.cx - a, self.cy - b)
        self.bounds_max = (self.cx + a, self.cy + b)

    def sdf(self, x: np.ndarray) -> np.ndarray:
        """Approximate SDF (Inigo Quilez 2020)."""
        p  = np.asarray(x, dtype=np.float64)
        px = np.abs(p[:, 0] - self.cx)
        py = np.abs(p[:, 1] - self.cy)
        ab = np.array([self.a, self.b])
        # implicit value: (px/a)^2 + (py/b)^2 - 1
        q  = np.stack([px, py], axis=1) / ab  # normalized
        l  = np.sqrt(np.sum(q**2, axis=1))
        # approximate world-space distance
        scale = min(self.a, self.b)
        return (l - 1.0) * scale

    def sample_interior(self, n: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        # rejection from bounding box
        return self._rejection_sample_interior(n, seed=seed)

    def sample_boundary(self, n: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        th  = rng.uniform(0.0, 2.0 * math.pi, n)
        pts = np.stack(
            [self.cx + self.a * np.cos(th), self.cy + self.b * np.sin(th)],
            axis=1,
        )
        return pts.astype(np.float32)


class CSGPolygon(SDFShape):
    """Arbitrary polygon defined by an ordered list of 2D vertices.

    Uses the standard point-in-polygon winding-number test and a
    piecewise-linear SDF (exact for convex, approximate for concave).

    Parameters
    ----------
    vertices : (M, 2) array of vertices (must be closed ring or auto-closed)
    """

    def __init__(self, vertices: np.ndarray) -> None:
        verts = np.asarray(vertices, dtype=np.float64)
        if verts.shape[1] != 2:
            raise ValueError("vertices must be (M, 2)")
        # auto-close
        if not np.allclose(verts[0], verts[-1]):
            verts = np.vstack([verts, verts[:1]])
        self.verts = verts
        self.bounds_min = (float(verts[:, 0].min()), float(verts[:, 1].min()))
        self.bounds_max = (float(verts[:, 0].max()), float(verts[:, 1].max()))

    def sdf(self, x: np.ndarray) -> np.ndarray:
        """Point-to-polygon SDF.  Negative inside."""
        p  = np.asarray(x, dtype=np.float64)   # (N, 2)
        verts = self.verts
        N = len(p)
        M = len(verts) - 1  # number of edges
        d_min = np.full(N, np.inf)
        sign  = np.ones(N)

        for i in range(M):
            a = verts[i]       # (2,)
            b = verts[i + 1]   # (2,)
            # edge vector
            e  = b - a                          # (2,)
            w  = p - a                          # (N,2)
            t  = np.clip(
                np.sum(w * e, axis=1) / (np.dot(e, e) + 1e-12), 0.0, 1.0
            )                                   # (N,)
            proj = a + np.outer(t, e)           # (N,2)
            dist = np.sqrt(np.sum((p - proj)**2, axis=1))
            d_min = np.minimum(d_min, dist)

            # winding number contribution (y-crossing)
            cond = (
                (p[:, 1] >= a[1]) != (p[:, 1] >= b[1])
            ) & (
                p[:, 0] < a[0] + (p[:, 1] - a[1]) / (b[1] - a[1] + 1e-12) * (b[0] - a[0])
            )
            sign = np.where(cond, -sign, sign)

        return d_min * sign

    def sample_interior(self, n: int, *, seed: int = 0) -> np.ndarray:
        return self._rejection_sample_interior(n, seed=seed)

    def sample_boundary(self, n: int, *, seed: int = 0) -> np.ndarray:
        """Sample uniformly along all edges (proportional to edge length)."""
        rng = np.random.default_rng(seed)
        verts = self.verts
        M = len(verts) - 1
        lengths = np.array(
            [np.linalg.norm(verts[i + 1] - verts[i]) for i in range(M)]
        )
        probs = lengths / lengths.sum()
        counts = rng.multinomial(n, probs)
        pts_list = []
        for i, cnt in enumerate(counts):
            if cnt == 0:
                continue
            t   = rng.uniform(0.0, 1.0, cnt)
            seg = verts[i] + np.outer(t, verts[i + 1] - verts[i])
            pts_list.append(seg)
        return np.concatenate(pts_list, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# CSG Boolean operations
# ---------------------------------------------------------------------------

class _CSGBinary(SDFShape):
    """Internal base for binary CSG nodes."""

    def __init__(self, a: SDFShape, b: SDFShape) -> None:
        self.a = a
        self.b = b
        # combined bounding box (conservative)
        bmin_a = np.asarray(a.bounds_min)
        bmax_a = np.asarray(a.bounds_max)
        bmin_b = np.asarray(b.bounds_min)
        bmax_b = np.asarray(b.bounds_max)
        bmin = np.minimum(bmin_a, bmin_b)
        bmax = np.maximum(bmax_a, bmax_b)
        self.bounds_min = (float(bmin[0]), float(bmin[1]))
        self.bounds_max = (float(bmax[0]), float(bmax[1]))

    def sample_interior(self, n: int, *, seed: int = 0) -> np.ndarray:
        return self._rejection_sample_interior(n, seed=seed)

    def sample_boundary(self, n: int, *, seed: int = 0) -> np.ndarray:
        return self._rejection_sample_boundary(n, seed=seed)


class CSGUnion(_CSGBinary):
    """Boolean union of two SDFShapes: min(sdf_a, sdf_b).

    Use ``a + b`` as a shorthand.
    """

    def sdf(self, x: np.ndarray) -> np.ndarray:
        return np.minimum(self.a.sdf(x), self.b.sdf(x))


class CSGIntersection(_CSGBinary):
    """Boolean intersection of two SDFShapes: max(sdf_a, sdf_b).

    Use ``a * b`` as a shorthand.
    """

    def sdf(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(self.a.sdf(x), self.b.sdf(x))


class CSGDifference(_CSGBinary):
    """Boolean difference A minus B: max(sdf_a, -sdf_b).

    Use ``a - b`` as a shorthand.
    """

    def sdf(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(self.a.sdf(x), -self.b.sdf(x))


# ---------------------------------------------------------------------------
# Convenience domain factories
# ---------------------------------------------------------------------------

def lshape(
    width1: float,
    height1: float,
    width2: float,
    height2: float,
) -> CSGDifference:
    """L-shaped domain.

    The big rectangle is [0, width1] x [0, height1].
    The upper-right rectangular notch is [width2, width1] x [height2, height1].

    Parameters
    ----------
    width1, height1 : outer bounding box
    width2, height2 : corner of the cut (defines the notch)

    Returns a CSGDifference that can be used as a CSG domain.

    Example::

        dom = lshape(2.0, 2.0, 1.0, 1.0)  # symmetric L-shape
        pts = dom.sample_interior(4096)
    """
    big = CSGRectangle(0.0, 0.0, width1, height1)
    cut = CSGRectangle(width2, height2, width1, height1)
    return big - cut


def annulus(
    cx: float,
    cy: float,
    r_inner: float,
    r_outer: float,
) -> CSGDifference:
    """Annular (ring) domain: outer circle minus inner circle.

    Parameters
    ----------
    cx, cy           : centre
    r_inner, r_outer : inner and outer radii

    Example::

        dom = annulus(0, 0, 0.25, 1.0)
        pts = dom.sample_interior(2048)
    """
    if r_inner >= r_outer:
        raise ValueError(f"r_inner ({r_inner}) must be < r_outer ({r_outer})")
    outer = CSGCircle(cx, cy, r_outer)
    inner = CSGCircle(cx, cy, r_inner)
    return outer - inner


def channel_with_hole(
    length: float,
    height: float,
    hole_cx: float,
    hole_cy: float,
    hole_r: float,
) -> CSGDifference:
    """Rectangular channel with a circular obstacle (hole).

    Typical usage for Navier-Stokes around a cylinder::

        dom = channel_with_hole(4.0, 1.0, 1.0, 0.5, 0.1)
        pts_col = dom.sample_interior(8192)   # collocation
        pts_bnd = dom.sample_boundary(1024)   # boundary

    Parameters
    ----------
    length, height : channel dimensions; lower-left at (0, 0)
    hole_cx, hole_cy, hole_r : obstacle centre and radius
    """
    channel = CSGRectangle(0.0, 0.0, length, height)
    hole    = CSGCircle(hole_cx, hole_cy, hole_r)
    return channel - hole


def t_junction(
    h_length: float,
    h_height: float,
    v_x_start: float,
    v_width: float,
    v_height: float,
) -> CSGUnion:
    """T-junction domain: horizontal channel plus a vertical branch.

    Parameters
    ----------
    h_length, h_height : horizontal channel  [0, h_length] x [0, h_height]
    v_x_start          : x-start of the vertical branch
    v_width, v_height  : width and height of the branch (extends upward from h_height)

    Example::

        dom = t_junction(4.0, 1.0, 1.5, 1.0, 2.0)
    """
    horiz = CSGRectangle(0.0, 0.0, h_length, h_height)
    vert  = CSGRectangle(v_x_start, h_height, v_x_start + v_width, h_height + v_height)
    return horiz + vert


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "SDFShape",
    "CSGRectangle",
    "CSGCircle",
    "CSGEllipse",
    "CSGPolygon",
    "CSGUnion",
    "CSGIntersection",
    "CSGDifference",
    "lshape",
    "annulus",
    "channel_with_hole",
    "t_junction",
]
