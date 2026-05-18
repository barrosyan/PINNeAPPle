"""Physical domain geometry for physics simulation and world model training.

Provides a composable geometry system supporting:

* **Primitives**: :class:`Rectangle`, :class:`Circle`, :class:`Polygon`,
  :class:`Box3D`, :class:`Sphere`
* **CSG operations**: :class:`Union`, :class:`Difference`, :class:`Intersection`
* **Built-in domains**: :func:`make_cavity`, :func:`make_channel`,
  :func:`make_channel_with_cylinder`, :func:`make_pipe_bend`
* **Sampling**: ``interior_points``, ``boundary_points`` integrate with
  ``pinneapple_data.CollocationSampler`` when available

Geometry objects are used by :class:`~.dataset_factory.PhysicsDatasetFactory`
and :class:`~.specialist_trainer.SpecialistTrainer` to define simulation
domains, boundary condition surfaces, and collocation point distributions.

Quick start::

    from pinneapple_worldmodel.geometry import Rectangle, Circle, Difference

    domain  = Rectangle((0, 0), (2, 1))
    cyl     = Circle(center=(0.5, 0.5), radius=0.1)
    channel = Difference(domain, cyl)   # flow past cylinder

    x_int = channel.interior_points(n=4000)  # (N, 2) collocation points
    x_bc  = channel.boundary_points(n=500)   # (N, 2) boundary points
    n_bc  = channel.normals(x_bc)            # (N, 2) outward normals
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class GeometryBase(ABC):
    """Abstract base class for all geometry objects.

    All geometry objects work in **normalised physical coordinates** and return
    float32 tensors on CPU.  Move to device after sampling.
    """

    @abstractmethod
    def contains(self, points: Tensor) -> Tensor:
        """Return boolean mask ``(N,)`` — True where *points* are inside."""

    @abstractmethod
    def bbox(self) -> Tuple[Tensor, Tensor]:
        """Return ``(lo, hi)`` bounding box tensors, both shape ``(D,)``."""

    @property
    def spatial_dim(self) -> int:
        lo, _ = self.bbox()
        return lo.shape[0]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def interior_points(self, n: int, *, seed: Optional[int] = None) -> Tensor:
        """Sample *n* points inside the geometry using rejection sampling.

        Returns
        -------
        Tensor ``(N, D)`` float32.
        """
        if seed is not None:
            torch.manual_seed(seed)
        lo, hi = self.bbox()
        D = lo.shape[0]
        pts: List[Tensor] = []
        batch = max(n * 4, 1000)
        while sum(p.shape[0] for p in pts) < n:
            cands = lo + (hi - lo) * torch.rand(batch, D)
            mask = self.contains(cands)
            pts.append(cands[mask])
        return torch.cat(pts, dim=0)[:n]

    def boundary_points(self, n: int, *, seed: Optional[int] = None) -> Tensor:
        """Sample *n* points on the boundary (thin shell ε-rejection).

        Uses a thin-shell approach: points just outside the geometry are
        reflected inward.  Override in subclasses for exact sampling.
        """
        if seed is not None:
            torch.manual_seed(seed)
        lo, hi = self.bbox()
        D = lo.shape[0]
        eps = (hi - lo).min().item() * 1e-3
        pts: List[Tensor] = []
        while sum(p.shape[0] for p in pts) < n:
            cands = lo + (hi - lo) * torch.rand(n * 8, D)
            inside  = self.contains(cands)
            perturb = cands + eps * torch.randn_like(cands)
            outside = ~self.contains(perturb)
            on_boundary = inside & outside
            pts.append(cands[on_boundary])
        return torch.cat(pts, dim=0)[:n]

    def normals(self, boundary_pts: Tensor, *, eps: float = 1e-4) -> Tensor:
        """Estimate outward normals at *boundary_pts* via finite differences.

        Returns
        -------
        Tensor ``(N, D)`` normalised outward normal vectors.
        """
        D = boundary_pts.shape[1]
        grads = torch.zeros_like(boundary_pts)
        for d in range(D):
            e = torch.zeros(D)
            e[d] = eps
            pts_fwd = boundary_pts + e
            pts_bwd = boundary_pts - e
            fin = self.contains(pts_fwd).float()
            bin = self.contains(pts_bwd).float()
            grads[:, d] = (fin - bin) / (2 * eps)
        # Outward normal points *away* from the interior
        n = -grads
        norm = n.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return n / norm

    # ------------------------------------------------------------------
    # CSG operators
    # ------------------------------------------------------------------

    def __or__(self, other: "GeometryBase") -> "Union":
        return Union(self, other)

    def __and__(self, other: "GeometryBase") -> "Intersection":
        return Intersection(self, other)

    def __sub__(self, other: "GeometryBase") -> "Difference":
        return Difference(self, other)


# ---------------------------------------------------------------------------
# 2-D Primitives
# ---------------------------------------------------------------------------

class Rectangle(GeometryBase):
    """Axis-aligned rectangle ``[lo_x, hi_x] × [lo_y, hi_y]``.

    Parameters
    ----------
    lo : (x0, y0) — lower-left corner.
    hi : (x1, y1) — upper-right corner.
    """

    def __init__(
        self,
        lo: Sequence[float],
        hi: Sequence[float],
    ) -> None:
        self._lo = torch.tensor(lo, dtype=torch.float32)
        self._hi = torch.tensor(hi, dtype=torch.float32)

    def contains(self, points: Tensor) -> Tensor:
        return ((points >= self._lo) & (points <= self._hi)).all(dim=-1)

    def bbox(self) -> Tuple[Tensor, Tensor]:
        return self._lo.clone(), self._hi.clone()

    def boundary_points(self, n: int, *, seed: Optional[int] = None) -> Tensor:
        """Exact uniform sampling on the four edges."""
        if seed is not None:
            torch.manual_seed(seed)
        x0, y0 = self._lo.tolist()
        x1, y1 = self._hi.tolist()
        w = x1 - x0
        h = y1 - y0
        perim = 2 * (w + h)
        n_w = max(1, int(n * w / perim))
        n_h = max(1, int(n * h / perim))

        t_w = torch.linspace(x0, x1, n_w)
        t_h = torch.linspace(y0, y1, n_h)

        pts = torch.cat([
            torch.stack([t_w, torch.full_like(t_w, y0)], dim=1),
            torch.stack([t_w, torch.full_like(t_w, y1)], dim=1),
            torch.stack([torch.full_like(t_h, x0), t_h], dim=1),
            torch.stack([torch.full_like(t_h, x1), t_h], dim=1),
        ], dim=0)
        idx = torch.randperm(pts.shape[0])[:n]
        return pts[idx]

    def __repr__(self) -> str:
        lo = self._lo.tolist()
        hi = self._hi.tolist()
        return f"Rectangle(lo={lo}, hi={hi})"


class Circle(GeometryBase):
    """2-D filled circle.

    Parameters
    ----------
    center : (cx, cy)
    radius : float
    """

    def __init__(self, center: Sequence[float], radius: float) -> None:
        self._c = torch.tensor(center, dtype=torch.float32)
        self._r = float(radius)

    def contains(self, points: Tensor) -> Tensor:
        return ((points - self._c) ** 2).sum(dim=-1) <= self._r ** 2

    def bbox(self) -> Tuple[Tensor, Tensor]:
        return self._c - self._r, self._c + self._r

    def boundary_points(self, n: int, *, seed: Optional[int] = None) -> Tensor:
        """Exact uniform sampling on the circumference."""
        if seed is not None:
            torch.manual_seed(seed)
        theta = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
        x = self._c[0] + self._r * torch.cos(theta)
        y = self._c[1] + self._r * torch.sin(theta)
        return torch.stack([x, y], dim=1)

    def __repr__(self) -> str:
        return f"Circle(center={self._c.tolist()}, radius={self._r})"


class Polygon(GeometryBase):
    """2-D filled polygon defined by ordered vertices.

    Parameters
    ----------
    vertices : list of (x, y) pairs (will be closed automatically).
    """

    def __init__(self, vertices: Sequence[Sequence[float]]) -> None:
        self._verts = torch.tensor(vertices, dtype=torch.float32)

    def contains(self, points: Tensor) -> Tensor:
        """Ray-casting algorithm for point-in-polygon test."""
        verts = self._verts
        n_v = verts.shape[0]
        x, y = points[:, 0], points[:, 1]
        inside = torch.zeros(points.shape[0], dtype=torch.bool)
        j = n_v - 1
        for i in range(n_v):
            xi, yi = verts[i, 0], verts[i, 1]
            xj, yj = verts[j, 0], verts[j, 1]
            cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
            inside = inside ^ cond
            j = i
        return inside

    def bbox(self) -> Tuple[Tensor, Tensor]:
        return self._verts.min(dim=0).values, self._verts.max(dim=0).values

    def __repr__(self) -> str:
        return f"Polygon(n_vertices={self._verts.shape[0]})"


# ---------------------------------------------------------------------------
# 3-D Primitives
# ---------------------------------------------------------------------------

class Box3D(GeometryBase):
    """Axis-aligned 3-D box."""

    def __init__(self, lo: Sequence[float], hi: Sequence[float]) -> None:
        self._lo = torch.tensor(lo, dtype=torch.float32)
        self._hi = torch.tensor(hi, dtype=torch.float32)

    def contains(self, points: Tensor) -> Tensor:
        return ((points >= self._lo) & (points <= self._hi)).all(dim=-1)

    def bbox(self) -> Tuple[Tensor, Tensor]:
        return self._lo.clone(), self._hi.clone()


class Sphere(GeometryBase):
    """3-D filled sphere."""

    def __init__(self, center: Sequence[float], radius: float) -> None:
        self._c = torch.tensor(center, dtype=torch.float32)
        self._r = float(radius)

    def contains(self, points: Tensor) -> Tensor:
        return ((points - self._c) ** 2).sum(dim=-1) <= self._r ** 2

    def bbox(self) -> Tuple[Tensor, Tensor]:
        return self._c - self._r, self._c + self._r


# ---------------------------------------------------------------------------
# CSG Operations
# ---------------------------------------------------------------------------

class Union(GeometryBase):
    """CSG union: points inside *a* OR *b*."""

    def __init__(self, a: GeometryBase, b: GeometryBase) -> None:
        self.a, self.b = a, b

    def contains(self, points: Tensor) -> Tensor:
        return self.a.contains(points) | self.b.contains(points)

    def bbox(self) -> Tuple[Tensor, Tensor]:
        la, ha = self.a.bbox()
        lb, hb = self.b.bbox()
        return torch.minimum(la, lb), torch.maximum(ha, hb)


class Intersection(GeometryBase):
    """CSG intersection: points inside *a* AND *b*."""

    def __init__(self, a: GeometryBase, b: GeometryBase) -> None:
        self.a, self.b = a, b

    def contains(self, points: Tensor) -> Tensor:
        return self.a.contains(points) & self.b.contains(points)

    def bbox(self) -> Tuple[Tensor, Tensor]:
        la, ha = self.a.bbox()
        lb, hb = self.b.bbox()
        return torch.maximum(la, lb), torch.minimum(ha, hb)


class Difference(GeometryBase):
    """CSG difference: points inside *a* but NOT inside *b*."""

    def __init__(self, a: GeometryBase, b: GeometryBase) -> None:
        self.a, self.b = a, b

    def contains(self, points: Tensor) -> Tensor:
        return self.a.contains(points) & ~self.b.contains(points)

    def bbox(self) -> Tuple[Tensor, Tensor]:
        return self.a.bbox()


# ---------------------------------------------------------------------------
# Named boundary surfaces
# ---------------------------------------------------------------------------

@dataclass
class BoundaryRegion:
    """A named subset of the domain boundary with its own BC type.

    Parameters
    ----------
    name : str — e.g. ``"inlet"``, ``"wall"``, ``"outlet"``.
    bc_type : str — ``"dirichlet"``, ``"neumann"``, ``"periodic"``, ``"slip"``.
    value : float or callable — boundary value (0 for homogeneous).
    sampler : callable ``(n) → Tensor`` — returns *n* boundary points.
    """
    name: str
    bc_type: str
    value: float = 0.0
    sampler: Optional[object] = None  # callable (n) -> Tensor

    def sample(self, n: int) -> Optional[Tensor]:
        if self.sampler is not None:
            return self.sampler(n)
        return None


@dataclass
class PhysicsDomain:
    """Complete domain specification: geometry + boundary regions + metadata.

    Parameters
    ----------
    geometry : GeometryBase
    boundaries : list of BoundaryRegion
    name : str
    spatial_dim : int — 2 or 3.
    tags : list of str
    """
    geometry: GeometryBase
    boundaries: List[BoundaryRegion] = field(default_factory=list)
    name: str = "domain"
    tags: List[str] = field(default_factory=list)

    def interior_points(self, n: int, *, seed: Optional[int] = None) -> Tensor:
        return self.geometry.interior_points(n, seed=seed)

    def boundary_points(
        self, n: int, region: Optional[str] = None, *, seed: Optional[int] = None
    ) -> Tensor:
        if region is not None:
            for reg in self.boundaries:
                if reg.name == region and reg.sampler is not None:
                    return reg.sampler(n)
        return self.geometry.boundary_points(n, seed=seed)

    def to_collocation_dict(
        self,
        n_interior: int = 4000,
        n_boundary: int = 500,
        *,
        seed: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """Return a dict compatible with ``pinneapple_data.CollocationSampler``.

        Keys: ``"interior"``, ``"boundary"``, and one per named region.
        """
        out: Dict[str, Tensor] = {
            "interior": self.interior_points(n_interior, seed=seed),
            "boundary": self.boundary_points(n_boundary, seed=seed),
        }
        for reg in self.boundaries:
            pts = reg.sample(n_boundary // max(len(self.boundaries), 1))
            if pts is not None:
                out[reg.name] = pts
        return out


# ---------------------------------------------------------------------------
# Built-in factory functions
# ---------------------------------------------------------------------------

def make_unit_square(*, with_boundary_labels: bool = True) -> PhysicsDomain:
    """Unit square [0,1]² — standard benchmark domain."""
    rect = Rectangle((0.0, 0.0), (1.0, 1.0))

    boundaries = []
    if with_boundary_labels:
        boundaries = [
            BoundaryRegion("bottom", "dirichlet", 0.0,
                           lambda n: _rect_edge(0, 0, 1, 0, n)),
            BoundaryRegion("top",    "dirichlet", 0.0,
                           lambda n: _rect_edge(0, 1, 1, 1, n)),
            BoundaryRegion("left",   "dirichlet", 0.0,
                           lambda n: _rect_edge(0, 0, 0, 1, n)),
            BoundaryRegion("right",  "dirichlet", 0.0,
                           lambda n: _rect_edge(1, 0, 1, 1, n)),
        ]
    return PhysicsDomain(rect, boundaries, name="unit_square", tags=["2d", "simple"])


def make_cavity(L: float = 1.0) -> PhysicsDomain:
    """Lid-driven cavity: unit square with moving lid (y=L, u=1)."""
    rect = Rectangle((0.0, 0.0), (L, L))
    boundaries = [
        BoundaryRegion("walls", "dirichlet", 0.0,
                       lambda n: _rect_walls(0, 0, L, L, n, exclude_top=True)),
        BoundaryRegion("lid",   "dirichlet", 1.0,
                       lambda n: _rect_edge(0, L, L, L, n)),
    ]
    return PhysicsDomain(rect, boundaries, name="cavity", tags=["2d", "fluid", "navier-stokes"])


def make_channel(
    length: float = 4.0,
    height: float = 1.0,
    *,
    inlet_velocity: float = 1.0,
) -> PhysicsDomain:
    """Poiseuille channel flow domain."""
    rect = Rectangle((0.0, 0.0), (length, height))
    boundaries = [
        BoundaryRegion("inlet",  "dirichlet", inlet_velocity,
                       lambda n: _rect_edge(0, 0, 0, height, n)),
        BoundaryRegion("outlet", "neumann",   0.0,
                       lambda n: _rect_edge(length, 0, length, height, n)),
        BoundaryRegion("walls",  "dirichlet", 0.0,
                       lambda n: torch.cat([
                           _rect_edge(0, 0, length, 0, n // 2),
                           _rect_edge(0, height, length, height, n // 2),
                       ], dim=0)),
    ]
    return PhysicsDomain(rect, boundaries, name="channel", tags=["2d", "fluid", "parabolic"])


def make_channel_with_cylinder(
    length: float = 4.0,
    height: float = 1.0,
    cyl_center: Tuple[float, float] = (1.0, 0.5),
    cyl_radius: float = 0.1,
) -> PhysicsDomain:
    """2-D channel with circular obstacle — classic benchmark."""
    domain  = Rectangle((0.0, 0.0), (length, height))
    cyl     = Circle(cyl_center, cyl_radius)
    geom    = Difference(domain, cyl)
    boundaries = [
        BoundaryRegion("inlet",    "dirichlet", 1.0,
                       lambda n: _rect_edge(0, 0, 0, height, n)),
        BoundaryRegion("outlet",   "neumann",   0.0,
                       lambda n: _rect_edge(length, 0, length, height, n)),
        BoundaryRegion("walls",    "dirichlet", 0.0,
                       lambda n: torch.cat([
                           _rect_edge(0, 0, length, 0, n // 2),
                           _rect_edge(0, height, length, height, n // 2),
                       ], dim=0)),
        BoundaryRegion("cylinder", "dirichlet", 0.0,
                       lambda n: cyl.boundary_points(n)),
    ]
    return PhysicsDomain(
        geom, boundaries,
        name="channel_with_cylinder",
        tags=["2d", "fluid", "obstacle", "navier-stokes"],
    )


def make_l_shaped() -> PhysicsDomain:
    """L-shaped domain (stress concentration benchmark)."""
    square = Rectangle((0.0, 0.0), (1.0, 1.0))
    cutout = Rectangle((0.5, 0.5), (1.0, 1.0))
    geom = Difference(square, cutout)
    return PhysicsDomain(geom, name="l_shaped", tags=["2d", "elasticity", "re-entrant"])


def make_annulus(
    inner_r: float = 0.3,
    outer_r: float = 1.0,
    center: Tuple[float, float] = (0.0, 0.0),
) -> PhysicsDomain:
    """Annular domain — useful for heat/EM problems in cylindrical geometry."""
    outer = Circle(center, outer_r)
    inner = Circle(center, inner_r)
    geom  = Difference(outer, inner)
    boundaries = [
        BoundaryRegion("inner_wall", "dirichlet", 1.0, lambda n: inner.boundary_points(n)),
        BoundaryRegion("outer_wall", "dirichlet", 0.0, lambda n: outer.boundary_points(n)),
    ]
    return PhysicsDomain(geom, boundaries, name="annulus",
                         tags=["2d", "heat", "cylindrical"])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rect_edge(x0, y0, x1, y1, n: int) -> Tensor:
    """n points on the line segment from (x0,y0) to (x1,y1)."""
    t = torch.linspace(0, 1, n)
    x = x0 + (x1 - x0) * t
    y = y0 + (y1 - y0) * t
    return torch.stack([x, y], dim=1)


def _rect_walls(x0, y0, x1, y1, n: int, *, exclude_top: bool = False) -> Tensor:
    parts = [
        _rect_edge(x0, y0, x1, y0, n // 4),   # bottom
        _rect_edge(x0, y0, x0, y1, n // 4),   # left
        _rect_edge(x1, y0, x1, y1, n // 4),   # right
    ]
    if not exclude_top:
        parts.append(_rect_edge(x0, y1, x1, y1, n // 4))
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Domain catalog
# ---------------------------------------------------------------------------

BUILTIN_DOMAINS: Dict[str, object] = {
    "unit_square":             make_unit_square,
    "cavity":                  make_cavity,
    "channel":                 make_channel,
    "channel_with_cylinder":   make_channel_with_cylinder,
    "l_shaped":                make_l_shaped,
    "annulus":                 make_annulus,
}
