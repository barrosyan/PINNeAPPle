"""Pre-built physics domains for PINN training.

Each domain provides:
- SDF for interior/exterior classification
- Named boundary groups (inlet, outlet, walls, obstacle, etc.)
- PINN-ready sampling (collocation + boundary points with region tags)
- Domain metadata compatible with pinneaple_arena bundle format

Example usage::

    domain = ChannelWithObstacleDomain2D(
        length=2.0, height=1.0,
        obstacle_center=(0.5, 0.5), obstacle_radius=0.1
    )
    batch = domain.get_pinn_batch(n_col=4096, n_bc_per_region=512)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Base domain
# ---------------------------------------------------------------------------

@dataclass
class BoundaryRegion:
    """A named boundary region with an associated SDF-like sampler."""
    name: str
    kind: str  # "dirichlet" | "neumann" | "robin" | "periodic" | "symmetry"
    conditions: Dict[str, Any] = field(default_factory=dict)


class PhysicsDomain2D:
    """Base class for 2D physics domains.

    Subclasses implement:
      - ``sdf(p)``   : (N,2) -> (N,) signed distance, negative inside
      - ``sample_boundary_region(region_name, n, seed)`` : (N,2) points on a boundary region
      - ``boundary_regions`` : list of BoundaryRegion

    The base class provides ``sample_interior``, ``get_pinn_batch``, and
    export helpers.
    """

    bounds_min: Tuple[float, float] = (0.0, 0.0)
    bounds_max: Tuple[float, float] = (1.0, 1.0)
    boundary_regions: List[BoundaryRegion] = []

    def sdf(self, p: np.ndarray) -> np.ndarray:
        """Signed distance. Override in subclass."""
        raise NotImplementedError

    def is_inside(self, p: np.ndarray) -> np.ndarray:
        """Boolean mask: True if point is inside domain."""
        return self.sdf(p.astype(np.float64)) <= 0.0

    def sample_interior(self, n: int, *, seed: int = 0) -> np.ndarray:
        """Sample n interior collocation points by rejection."""
        rng = np.random.default_rng(seed)
        bmin = np.asarray(self.bounds_min)
        bmax = np.asarray(self.bounds_max)
        collected = []
        while sum(len(c) for c in collected) < n:
            cands = bmin + rng.random((n * 5, 2)) * (bmax - bmin)
            mask = self.is_inside(cands)
            pts = cands[mask]
            if len(pts) > 0:
                collected.append(pts)
        return np.concatenate(collected, axis=0)[:n].astype(np.float32)

    def sample_boundary_region(self, region_name: str, n: int, *, seed: int = 0) -> np.ndarray:
        """Sample n points on the named boundary region. Override in subclass."""
        raise NotImplementedError(f"sample_boundary_region not implemented for region '{region_name}'")

    def get_region_names(self) -> List[str]:
        return [r.name for r in self.boundary_regions]

    def get_region_conditions(self, region_name: str) -> Dict[str, Any]:
        for r in self.boundary_regions:
            if r.name == region_name:
                return r.conditions
        raise KeyError(f"Region '{region_name}' not found")

    def get_pinn_batch(
        self,
        n_col: int = 4096,
        n_bc_per_region: int = 512,
        n_bc_total: Optional[int] = None,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """Generate a full PINN training batch.

        Returns
        -------
        dict with:
          x_col     : (n_col, 2) interior collocation points
          x_bc      : (N_bc, 2) all boundary points concatenated
          y_bc      : (N_bc, ?) boundary target values (zeros if not specified)
          bc_regions: list of str, region name per boundary point
          ctx       : domain metadata dict
        """
        x_col = self.sample_interior(n_col, seed=seed)

        x_bc_parts = []
        region_labels = []

        for i, region in enumerate(self.boundary_regions):
            n_pts = n_bc_total // max(1, len(self.boundary_regions)) if n_bc_total else n_bc_per_region
            pts = self.sample_boundary_region(region.name, n_pts, seed=seed + i)
            if len(pts) > 0:
                x_bc_parts.append(pts)
                region_labels.extend([region.name] * len(pts))

        x_bc = np.concatenate(x_bc_parts, axis=0).astype(np.float32) if x_bc_parts else np.zeros((0, 2), dtype=np.float32)
        y_bc = np.zeros((len(x_bc), 1), dtype=np.float32)

        ctx = {
            "domain_class": type(self).__name__,
            "bounds_min": list(self.bounds_min),
            "bounds_max": list(self.bounds_max),
            "regions": [{"name": r.name, "kind": r.kind, "conditions": r.conditions} for r in self.boundary_regions],
        }

        return {
            "x_col": x_col,
            "x_bc": x_bc,
            "y_bc": y_bc,
            "bc_regions": region_labels,
            "ctx": ctx,
        }

    def to_bundle_conditions(self) -> Dict[str, Any]:
        """Export region conditions in Arena bundle format."""
        cond = {}
        for r in self.boundary_regions:
            entry = {"type": r.kind}
            entry.update(r.conditions)
            cond[r.name] = entry
        cond["_meta"] = {"domain_class": type(self).__name__}
        return cond

    def to_bundle_manifest(self, problem_id: str = "custom", fields: List[str] = None) -> Dict[str, Any]:
        """Export domain manifest in Arena bundle format."""
        return {
            "problem_id": problem_id,
            "domain": {
                "type": type(self).__name__,
                "bounds_min": list(self.bounds_min),
                "bounds_max": list(self.bounds_max),
            },
            "fields": fields or [],
            "regions": [r.name for r in self.boundary_regions],
        }


# ---------------------------------------------------------------------------
# 2D Rectangular channel
# ---------------------------------------------------------------------------

class ChannelDomain2D(PhysicsDomain2D):
    """Rectangular channel: flow from left (inlet) to right (outlet).

    Boundary regions:
    - inlet  (x = x_min): Dirichlet velocity inlet
    - outlet (x = x_max): Neumann pressure outlet
    - walls  (y = y_min or y = y_max): no-slip
    """

    def __init__(
        self,
        length: float = 2.0,
        height: float = 1.0,
        origin: Tuple[float, float] = (0.0, 0.0),
        inlet_velocity: float = 1.0,
    ):
        self.length = float(length)
        self.height = float(height)
        self.origin = tuple(float(v) for v in origin)
        self.inlet_velocity = float(inlet_velocity)

        ox, oy = self.origin
        self.bounds_min = (ox, oy)
        self.bounds_max = (ox + self.length, oy + self.height)

        self.boundary_regions = [
            BoundaryRegion("inlet",  "dirichlet", {"u": inlet_velocity, "v": 0.0}),
            BoundaryRegion("outlet", "neumann",   {"p": 0.0}),
            BoundaryRegion("walls",  "no_slip",   {"u": 0.0, "v": 0.0}),
        ]

    def sdf(self, p: np.ndarray) -> np.ndarray:
        from .sdf_shapes import sdf2d_rectangle
        cx = (self.bounds_min[0] + self.bounds_max[0]) * 0.5
        cy = (self.bounds_min[1] + self.bounds_max[1]) * 0.5
        hx = (self.bounds_max[0] - self.bounds_min[0]) * 0.5
        hy = (self.bounds_max[1] - self.bounds_min[1]) * 0.5
        return sdf2d_rectangle(p.astype(np.float64), (cx, cy), (hx, hy))

    def _rng(self, seed):
        return np.random.default_rng(seed)

    def sample_boundary_region(self, region_name: str, n: int, *, seed: int = 0) -> np.ndarray:
        rng = self._rng(seed)
        ox, oy = self.bounds_min
        ex, ey = self.bounds_max

        if region_name == "inlet":
            y = rng.uniform(oy, ey, n).astype(np.float32)
            x = np.full(n, ox, dtype=np.float32)
            return np.stack([x, y], axis=1)
        elif region_name == "outlet":
            y = rng.uniform(oy, ey, n).astype(np.float32)
            x = np.full(n, ex, dtype=np.float32)
            return np.stack([x, y], axis=1)
        elif region_name == "walls":
            n_bot, n_top = n // 2, n - n // 2
            x_b = rng.uniform(ox, ex, n_bot).astype(np.float32)
            x_t = rng.uniform(ox, ex, n_top).astype(np.float32)
            pts_b = np.stack([x_b, np.full(n_bot, oy, dtype=np.float32)], axis=1)
            pts_t = np.stack([x_t, np.full(n_top, ey, dtype=np.float32)], axis=1)
            return np.concatenate([pts_b, pts_t], axis=0)
        raise KeyError(f"Unknown region '{region_name}'")


# ---------------------------------------------------------------------------
# 2D Channel with circular obstacle (Navier-Stokes benchmark)
# ---------------------------------------------------------------------------

class ChannelWithObstacleDomain2D(PhysicsDomain2D):
    """Rectangular channel with a circular obstacle.

    Boundary regions:
    - inlet    (x = x_min): Dirichlet velocity
    - outlet   (x = x_max): Neumann p = 0
    - walls    (y = y_min or y = y_max): no-slip
    - obstacle (circle boundary): no-slip
    """

    def __init__(
        self,
        length: float = 2.0,
        height: float = 1.0,
        origin: Tuple[float, float] = (0.0, 0.0),
        obstacle_center: Tuple[float, float] = (0.5, 0.5),
        obstacle_radius: float = 0.1,
        inlet_velocity: float = 1.0,
    ):
        self.length = float(length)
        self.height = float(height)
        self.origin = tuple(float(v) for v in origin)
        self.obstacle_center = tuple(float(v) for v in obstacle_center)
        self.obstacle_radius = float(obstacle_radius)
        self.inlet_velocity = float(inlet_velocity)

        ox, oy = self.origin
        self.bounds_min = (ox, oy)
        self.bounds_max = (ox + self.length, oy + self.height)

        self.boundary_regions = [
            BoundaryRegion("inlet",    "dirichlet", {"u": inlet_velocity, "v": 0.0}),
            BoundaryRegion("outlet",   "neumann",   {"p": 0.0}),
            BoundaryRegion("walls",    "no_slip",   {"u": 0.0, "v": 0.0}),
            BoundaryRegion("obstacle", "no_slip",   {"u": 0.0, "v": 0.0}),
        ]

    def sdf(self, p: np.ndarray) -> np.ndarray:
        from .sdf_shapes import sdf2d_rectangle, sdf2d_circle, sdf_difference
        cx = (self.bounds_min[0] + self.bounds_max[0]) * 0.5
        cy = (self.bounds_min[1] + self.bounds_max[1]) * 0.5
        hx = (self.bounds_max[0] - self.bounds_min[0]) * 0.5
        hy = (self.bounds_max[1] - self.bounds_min[1]) * 0.5
        p64 = p.astype(np.float64)
        d_box = sdf2d_rectangle(p64, (cx, cy), (hx, hy))
        d_obs = sdf2d_circle(p64, self.obstacle_center, self.obstacle_radius)
        return sdf_difference(d_box, d_obs)

    def sample_boundary_region(self, region_name: str, n: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        ox, oy = self.bounds_min
        ex, ey = self.bounds_max

        if region_name == "inlet":
            y = rng.uniform(oy, ey, n).astype(np.float32)
            return np.stack([np.full(n, ox, dtype=np.float32), y], axis=1)
        elif region_name == "outlet":
            y = rng.uniform(oy, ey, n).astype(np.float32)
            return np.stack([np.full(n, ex, dtype=np.float32), y], axis=1)
        elif region_name == "walls":
            n_bot, n_top = n // 2, n - n // 2
            x_b = rng.uniform(ox, ex, n_bot).astype(np.float32)
            x_t = rng.uniform(ox, ex, n_top).astype(np.float32)
            return np.concatenate([
                np.stack([x_b, np.full(n_bot, oy, dtype=np.float32)], axis=1),
                np.stack([x_t, np.full(n_top, ey, dtype=np.float32)], axis=1),
            ], axis=0)
        elif region_name == "obstacle":
            theta = rng.uniform(0.0, 2 * math.pi, n).astype(np.float32)
            r = float(self.obstacle_radius)
            cx, cy = self.obstacle_center
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            return np.stack([x, y], axis=1)
        raise KeyError(f"Unknown region '{region_name}'")


# ---------------------------------------------------------------------------
# Lid-driven cavity
# ---------------------------------------------------------------------------

class LidDrivenCavityDomain2D(PhysicsDomain2D):
    """Unit square cavity with moving lid.

    Boundary regions:
    - lid    (y = y_max): moving wall (u = U_lid, v = 0)
    - walls  (other 3 sides): no-slip
    """

    def __init__(
        self,
        size: float = 1.0,
        origin: Tuple[float, float] = (0.0, 0.0),
        lid_velocity: float = 1.0,
    ):
        self.size = float(size)
        self.origin = tuple(float(v) for v in origin)
        self.lid_velocity = float(lid_velocity)

        ox, oy = self.origin
        self.bounds_min = (ox, oy)
        self.bounds_max = (ox + self.size, oy + self.size)

        self.boundary_regions = [
            BoundaryRegion("lid",   "dirichlet", {"u": lid_velocity, "v": 0.0}),
            BoundaryRegion("walls", "no_slip",   {"u": 0.0, "v": 0.0}),
        ]

    def sdf(self, p: np.ndarray) -> np.ndarray:
        from .sdf_shapes import sdf2d_rectangle
        cx = (self.bounds_min[0] + self.bounds_max[0]) * 0.5
        cy = (self.bounds_min[1] + self.bounds_max[1]) * 0.5
        h = self.size * 0.5
        return sdf2d_rectangle(p.astype(np.float64), (cx, cy), (h, h))

    def sample_boundary_region(self, region_name: str, n: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        ox, oy = self.bounds_min
        ex, ey = self.bounds_max

        if region_name == "lid":
            x = rng.uniform(ox, ex, n).astype(np.float32)
            return np.stack([x, np.full(n, ey, dtype=np.float32)], axis=1)
        elif region_name == "walls":
            # Bottom + left + right walls
            n3 = n // 3
            x_b = rng.uniform(ox, ex, n3).astype(np.float32)
            y_l = rng.uniform(oy, ey, n3).astype(np.float32)
            y_r = rng.uniform(oy, ey, n - 2 * n3).astype(np.float32)
            return np.concatenate([
                np.stack([x_b, np.full(n3, oy, np.float32)], axis=1),
                np.stack([np.full(n3, ox, np.float32), y_l], axis=1),
                np.stack([np.full(n - 2*n3, ex, np.float32), y_r], axis=1),
            ], axis=0)
        raise KeyError(f"Unknown region '{region_name}'")


# ---------------------------------------------------------------------------
# L-shape domain
# ---------------------------------------------------------------------------

class LShapeDomain2D(PhysicsDomain2D):
    """L-shaped 2D domain for heat/Laplace problems.

    The L-shape is a unit square with the top-right quarter removed.

    Boundary regions:
    - boundary : all outer boundary (Dirichlet)
    - reentrant_corner : near the reentrant corner (optional refinement region)
    """

    def __init__(
        self,
        size: float = 2.0,
        origin: Tuple[float, float] = (-1.0, -1.0),
    ):
        self.size = float(size)
        self.origin = tuple(float(v) for v in origin)
        s = self.size
        ox, oy = self.origin
        self.bounds_min = (ox, oy)
        self.bounds_max = (ox + s, oy + s)
        # The cut-out is the top-right quarter
        self.cutout_min = (ox + s * 0.5, oy + s * 0.5)
        self.cutout_max = (ox + s, oy + s)

        self.boundary_regions = [
            BoundaryRegion("boundary", "dirichlet", {"u": 0.0}),
        ]

    def sdf(self, p: np.ndarray) -> np.ndarray:
        from .sdf_shapes import sdf2d_rectangle, sdf_difference
        p64 = p.astype(np.float64)
        cx = (self.bounds_min[0] + self.bounds_max[0]) * 0.5
        cy = (self.bounds_min[1] + self.bounds_max[1]) * 0.5
        hx = (self.bounds_max[0] - self.bounds_min[0]) * 0.5
        hy = (self.bounds_max[1] - self.bounds_min[1]) * 0.5
        d_full = sdf2d_rectangle(p64, (cx, cy), (hx, hy))
        cx2 = (self.cutout_min[0] + self.cutout_max[0]) * 0.5
        cy2 = (self.cutout_min[1] + self.cutout_max[1]) * 0.5
        hx2 = (self.cutout_max[0] - self.cutout_min[0]) * 0.5
        hy2 = (self.cutout_max[1] - self.cutout_min[1]) * 0.5
        d_cut = sdf2d_rectangle(p64, (cx2, cy2), (hx2, hy2))
        return sdf_difference(d_full, d_cut)

    def sample_boundary_region(self, region_name: str, n: int, *, seed: int = 0) -> np.ndarray:
        """Sample on the L-shape boundary using SDF rejection near zero-level."""
        rng = np.random.default_rng(seed)
        tol = 0.03
        bmin = np.asarray(self.bounds_min)
        bmax = np.asarray(self.bounds_max)
        collected = []
        while sum(len(c) for c in collected) < n:
            cands = bmin + rng.random((n * 30, 2)) * (bmax - bmin)
            d = np.abs(self.sdf(cands))
            pts = cands[d < tol]
            if len(pts) > 0:
                collected.append(pts)
        all_pts = np.concatenate(collected, axis=0)
        idx = rng.choice(len(all_pts), min(n, len(all_pts)), replace=False)
        return all_pts[idx].astype(np.float32)


# ---------------------------------------------------------------------------
# Annular domain (concentric circles)
# ---------------------------------------------------------------------------

class AnnularDomain2D(PhysicsDomain2D):
    """Annular (ring) domain between inner and outer circles.

    Boundary regions:
    - inner  : inner circle (e.g. heated wall, no-slip)
    - outer  : outer circle (e.g. cooled wall, far-field)
    """

    def __init__(
        self,
        center: Tuple[float, float] = (0.0, 0.0),
        r_inner: float = 0.2,
        r_outer: float = 1.0,
        inner_condition: Dict[str, Any] = None,
        outer_condition: Dict[str, Any] = None,
    ):
        self.center = tuple(float(v) for v in center)
        self.r_inner = float(r_inner)
        self.r_outer = float(r_outer)
        cx, cy = self.center
        self.bounds_min = (cx - r_outer, cy - r_outer)
        self.bounds_max = (cx + r_outer, cy + r_outer)

        self.boundary_regions = [
            BoundaryRegion("inner", "dirichlet", inner_condition or {"u": 1.0}),
            BoundaryRegion("outer", "dirichlet", outer_condition or {"u": 0.0}),
        ]

    def sdf(self, p: np.ndarray) -> np.ndarray:
        from .sdf_shapes import sdf2d_annulus
        return sdf2d_annulus(p.astype(np.float64), self.center, self.r_inner, self.r_outer)

    def sample_boundary_region(self, region_name: str, n: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        cx, cy = self.center
        if region_name == "inner":
            theta = rng.uniform(0.0, 2 * math.pi, n).astype(np.float32)
            return np.stack([cx + self.r_inner * np.cos(theta), cy + self.r_inner * np.sin(theta)], axis=1)
        elif region_name == "outer":
            theta = rng.uniform(0.0, 2 * math.pi, n).astype(np.float32)
            return np.stack([cx + self.r_outer * np.cos(theta), cy + self.r_outer * np.sin(theta)], axis=1)
        raise KeyError(f"Unknown region '{region_name}'")


# ---------------------------------------------------------------------------
# Multi-obstacle domain
# ---------------------------------------------------------------------------

class MultiObstacleDomain2D(PhysicsDomain2D):
    """Rectangular channel with multiple arbitrary circular obstacles.

    Boundary regions: inlet, outlet, walls, obstacle_{i} for each obstacle.
    """

    def __init__(
        self,
        length: float = 3.0,
        height: float = 1.0,
        origin: Tuple[float, float] = (0.0, 0.0),
        obstacles: List[Tuple[Tuple[float, float], float]] = None,
        inlet_velocity: float = 1.0,
    ):
        """
        Parameters
        ----------
        obstacles : list of ((cx, cy), radius) tuples
        """
        self.length = float(length)
        self.height = float(height)
        self.origin = tuple(float(v) for v in origin)
        self.obstacles = obstacles or [((1.0, 0.5), 0.15), ((2.0, 0.5), 0.15)]
        self.inlet_velocity = float(inlet_velocity)

        ox, oy = self.origin
        self.bounds_min = (ox, oy)
        self.bounds_max = (ox + self.length, oy + self.height)

        self.boundary_regions = [
            BoundaryRegion("inlet",  "dirichlet", {"u": inlet_velocity, "v": 0.0}),
            BoundaryRegion("outlet", "neumann",   {"p": 0.0}),
            BoundaryRegion("walls",  "no_slip",   {"u": 0.0, "v": 0.0}),
        ]
        for i in range(len(self.obstacles)):
            self.boundary_regions.append(
                BoundaryRegion(f"obstacle_{i}", "no_slip", {"u": 0.0, "v": 0.0})
            )

    def sdf(self, p: np.ndarray) -> np.ndarray:
        from .sdf_shapes import sdf2d_rectangle, sdf2d_circle, sdf_difference
        p64 = p.astype(np.float64)
        cx = (self.bounds_min[0] + self.bounds_max[0]) * 0.5
        cy = (self.bounds_min[1] + self.bounds_max[1]) * 0.5
        hx = (self.bounds_max[0] - self.bounds_min[0]) * 0.5
        hy = (self.bounds_max[1] - self.bounds_min[1]) * 0.5
        d = sdf2d_rectangle(p64, (cx, cy), (hx, hy))
        for (oc, r) in self.obstacles:
            d_obs = sdf2d_circle(p64, oc, r)
            d = sdf_difference(d, d_obs)
        return d

    def sample_boundary_region(self, region_name: str, n: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        ox, oy = self.bounds_min
        ex, ey = self.bounds_max

        if region_name == "inlet":
            y = rng.uniform(oy, ey, n).astype(np.float32)
            return np.stack([np.full(n, ox, np.float32), y], axis=1)
        elif region_name == "outlet":
            y = rng.uniform(oy, ey, n).astype(np.float32)
            return np.stack([np.full(n, ex, np.float32), y], axis=1)
        elif region_name == "walls":
            n2 = n // 2
            x_b = rng.uniform(ox, ex, n2).astype(np.float32)
            x_t = rng.uniform(ox, ex, n - n2).astype(np.float32)
            return np.concatenate([
                np.stack([x_b, np.full(n2, oy, np.float32)], axis=1),
                np.stack([x_t, np.full(n - n2, ey, np.float32)], axis=1),
            ], axis=0)
        elif region_name.startswith("obstacle_"):
            idx = int(region_name.split("_")[1])
            (oc, r) = self.obstacles[idx]
            theta = rng.uniform(0.0, 2 * math.pi, n).astype(np.float32)
            return np.stack([oc[0] + r * np.cos(theta), oc[1] + r * np.sin(theta)], axis=1)
        raise KeyError(f"Unknown region '{region_name}'")


# ---------------------------------------------------------------------------
# T-junction domain
# ---------------------------------------------------------------------------

class TJunctionDomain2D(PhysicsDomain2D):
    """T-junction: horizontal main channel + vertical branch from the bottom.

    Boundary regions:
    - inlet_main   (left of main channel)
    - outlet_main  (right of main channel)
    - outlet_branch (bottom of branch)
    - walls        (all solid walls)
    """

    def __init__(
        self,
        main_length: float = 3.0,
        main_height: float = 0.5,
        branch_width: float = 0.5,
        branch_length: float = 1.0,
        origin: Tuple[float, float] = (0.0, 0.0),
        inlet_velocity: float = 1.0,
    ):
        self.main_length = float(main_length)
        self.main_height = float(main_height)
        self.branch_width = float(branch_width)
        self.branch_length = float(branch_length)
        self.origin = tuple(float(v) for v in origin)
        self.inlet_velocity = float(inlet_velocity)

        ox, oy = self.origin
        self.bounds_min = (ox, oy - branch_length)
        self.bounds_max = (ox + main_length, oy + main_height)

        self.boundary_regions = [
            BoundaryRegion("inlet_main",    "dirichlet", {"u": inlet_velocity, "v": 0.0}),
            BoundaryRegion("outlet_main",   "neumann",   {"p": 0.0}),
            BoundaryRegion("outlet_branch", "neumann",   {"p": 0.0}),
            BoundaryRegion("walls",         "no_slip",   {"u": 0.0, "v": 0.0}),
        ]

        # Store geometry for SDF
        self._main_rect = {
            "center": (ox + main_length * 0.5, oy + main_height * 0.5),
            "half": (main_length * 0.5, main_height * 0.5),
        }
        branch_cx = ox + main_length * 0.5
        self._branch_rect = {
            "center": (branch_cx, oy - branch_length * 0.5),
            "half": (branch_width * 0.5, branch_length * 0.5),
        }

    def sdf(self, p: np.ndarray) -> np.ndarray:
        from .sdf_shapes import sdf2d_rectangle, sdf_union
        p64 = p.astype(np.float64)
        d_main = sdf2d_rectangle(p64, self._main_rect["center"], self._main_rect["half"])
        d_branch = sdf2d_rectangle(p64, self._branch_rect["center"], self._branch_rect["half"])
        return sdf_union(d_main, d_branch)

    def sample_boundary_region(self, region_name: str, n: int, *, seed: int = 0) -> np.ndarray:
        """Sample via SDF near-zero rejection with region filtering."""
        rng = np.random.default_rng(seed)
        ox, oy = self.origin
        bmin = np.asarray(self.bounds_min)
        bmax = np.asarray(self.bounds_max)

        if region_name == "inlet_main":
            y = rng.uniform(oy, oy + self.main_height, n).astype(np.float32)
            return np.stack([np.full(n, ox, np.float32), y], axis=1)
        elif region_name == "outlet_main":
            y = rng.uniform(oy, oy + self.main_height, n).astype(np.float32)
            return np.stack([np.full(n, ox + self.main_length, np.float32), y], axis=1)
        elif region_name == "outlet_branch":
            branch_cx = ox + self.main_length * 0.5
            hw = self.branch_width * 0.5
            x = rng.uniform(branch_cx - hw, branch_cx + hw, n).astype(np.float32)
            return np.stack([x, np.full(n, oy - self.branch_length, np.float32)], axis=1)
        elif region_name == "walls":
            # Sample near all walls
            tol = 0.02
            collected = []
            while sum(len(c) for c in collected) < n:
                cands = bmin + rng.random((n * 30, 2)) * (bmax - bmin)
                d = np.abs(self.sdf(cands.astype(np.float64)))
                pts = cands[d < tol]
                if len(pts) > 0:
                    collected.append(pts.astype(np.float32))
            all_pts = np.concatenate(collected, axis=0)
            idx = rng.choice(len(all_pts), min(n, len(all_pts)), replace=False)
            return all_pts[idx]
        raise KeyError(f"Unknown region '{region_name}'")


# ---------------------------------------------------------------------------
# SDFDomain2D: build a domain from any SDF callable
# ---------------------------------------------------------------------------

class SDFDomain2D(PhysicsDomain2D):
    """Flexible domain from any SDF callable and named sampler dict.

    Usage::

        from pinneaple_geom.gen.sdf_shapes import circle, rectangle, SDF
        sdf = rectangle(center=(0.5, 0.5), half_extents=(0.5, 0.5)) - circle(center=(0.3, 0.5), radius=0.1)

        domain = SDFDomain2D(
            sdf_fn=sdf,
            bounds_min=(0.0, 0.0),
            bounds_max=(1.0, 1.0),
            boundary_samplers={
                "inlet":    lambda n, rng: ...,
                "obstacle": lambda n, rng: ...,
            },
            boundary_conditions={
                "inlet":    {"kind": "dirichlet", "u": 1.0, "v": 0.0},
                "obstacle": {"kind": "no_slip"},
            },
        )
    """

    def __init__(
        self,
        sdf_fn,
        *,
        bounds_min: Tuple[float, float],
        bounds_max: Tuple[float, float],
        boundary_samplers: Dict[str, Any] = None,
        boundary_conditions: Dict[str, Any] = None,
    ):
        self._sdf_fn = sdf_fn
        self.bounds_min = tuple(float(v) for v in bounds_min)
        self.bounds_max = tuple(float(v) for v in bounds_max)
        self._samplers = boundary_samplers or {}
        self._conditions = boundary_conditions or {}

        self.boundary_regions = [
            BoundaryRegion(
                name=k,
                kind=self._conditions.get(k, {}).get("kind", "dirichlet"),
                conditions={kk: vv for kk, vv in self._conditions.get(k, {}).items() if kk != "kind"},
            )
            for k in self._samplers
        ]

    def sdf(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(self._sdf_fn(p.astype(np.float64)), dtype=np.float64)

    def sample_boundary_region(self, region_name: str, n: int, *, seed: int = 0) -> np.ndarray:
        if region_name not in self._samplers:
            raise KeyError(f"No sampler registered for region '{region_name}'")
        rng = np.random.default_rng(seed)
        return self._samplers[region_name](n, rng)


# ---------------------------------------------------------------------------
# Registry of built-in domains
# ---------------------------------------------------------------------------

_DOMAIN_REGISTRY: Dict[str, type] = {
    "channel_2d": ChannelDomain2D,
    "channel_with_obstacle_2d": ChannelWithObstacleDomain2D,
    "lid_driven_cavity_2d": LidDrivenCavityDomain2D,
    "l_shape_2d": LShapeDomain2D,
    "annular_2d": AnnularDomain2D,
    "multi_obstacle_2d": MultiObstacleDomain2D,
    "t_junction_2d": TJunctionDomain2D,
}


def get_domain(name: str, **kwargs) -> PhysicsDomain2D:
    """Instantiate a built-in domain by name.

    Parameters
    ----------
    name : domain identifier (e.g. "channel_2d", "channel_with_obstacle_2d")
    **kwargs : constructor arguments
    """
    key = name.lower().strip()
    if key not in _DOMAIN_REGISTRY:
        raise KeyError(f"Unknown domain '{name}'. Available: {sorted(_DOMAIN_REGISTRY.keys())}")
    return _DOMAIN_REGISTRY[key](**kwargs)


def list_domains() -> List[str]:
    """List all registered built-in domain names."""
    return sorted(_DOMAIN_REGISTRY.keys())
