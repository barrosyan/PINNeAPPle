"""Pre-built 3D physics domains for PINN training.

Each domain provides:
- SDF for interior/exterior classification
- Named boundary groups (inlet, outlet, walls, lid, wall, etc.)
- PINN-ready sampling (collocation + boundary points with region tags)
- Domain metadata compatible with pinneaple_arena bundle format

Example usage::

    domain = LidDrivenCavityDomain3D(size=1.0, lid_velocity=1.0)
    batch = domain.get_pinn_batch(n_col=4096, n_bc_per_region=512)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .domains import BoundaryRegion


# ---------------------------------------------------------------------------
# Base 3D domain
# ---------------------------------------------------------------------------

class PhysicsDomain3D:
    """Base class for 3D physics domains.

    Subclasses implement:
      - ``sdf(p)``   : (N,3) -> (N,) signed distance, negative inside
      - ``sample_boundary_region(region_name, n, seed)`` : (N,3) points on a boundary region
      - ``boundary_regions`` : list of BoundaryRegion

    The base class provides ``sample_interior``, ``get_pinn_batch``, and
    export helpers.
    """

    bounds_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounds_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    boundary_regions: List[BoundaryRegion] = []

    def sdf(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def is_inside(self, p: np.ndarray) -> np.ndarray:
        return self.sdf(p.astype(np.float64)) <= 0.0

    def sample_interior(self, n: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        bmin = np.asarray(self.bounds_min, dtype=np.float64)
        bmax = np.asarray(self.bounds_max, dtype=np.float64)
        collected: List[np.ndarray] = []
        while sum(len(c) for c in collected) < n:
            cands = bmin + rng.random((n * 5, 3)) * (bmax - bmin)
            mask = self.is_inside(cands)
            pts = cands[mask]
            if len(pts) > 0:
                collected.append(pts)
        return np.concatenate(collected, axis=0)[:n].astype(np.float32)

    def sample_boundary_region(self, region_name: str, n: int, *, seed: int = 0) -> np.ndarray:
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
          x_col     : (n_col, 3) interior collocation points
          x_bc      : (N_bc, 3) all boundary points concatenated
          y_bc      : (N_bc, 1) boundary target values (zeros placeholder)
          bc_regions: list of str, region name per boundary point
          ctx       : domain metadata dict
        """
        x_col = self.sample_interior(n_col, seed=seed)

        x_bc_parts: List[np.ndarray] = []
        region_labels: List[str] = []

        for i, region in enumerate(self.boundary_regions):
            n_pts = n_bc_total // max(1, len(self.boundary_regions)) if n_bc_total else n_bc_per_region
            pts = self.sample_boundary_region(region.name, n_pts, seed=seed + i)
            if len(pts) > 0:
                x_bc_parts.append(pts)
                region_labels.extend([region.name] * len(pts))

        x_bc = np.concatenate(x_bc_parts, axis=0).astype(np.float32) if x_bc_parts else np.zeros((0, 3), dtype=np.float32)
        y_bc = np.zeros((len(x_bc), 1), dtype=np.float32)

        ctx: Dict[str, Any] = {
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
        cond: Dict[str, Any] = {}
        for r in self.boundary_regions:
            entry: Dict[str, Any] = {"type": r.kind}
            entry.update(r.conditions)
            cond[r.name] = entry
        cond["_meta"] = {"domain_class": type(self).__name__}
        return cond

    def sample_structured_grid(self, nx: int, ny: int, nz: int) -> np.ndarray:
        """Return interior points from a structured (nx, ny, nz) meshgrid.

        For generic non-box domains an is_inside filter is applied.
        For box domains subclasses override this to skip the filter.
        """
        xs = np.linspace(self.bounds_min[0], self.bounds_max[0], nx, dtype=np.float64)
        ys = np.linspace(self.bounds_min[1], self.bounds_max[1], ny, dtype=np.float64)
        zs = np.linspace(self.bounds_min[2], self.bounds_max[2], nz, dtype=np.float64)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        mask = self.is_inside(pts)
        return pts[mask].astype(np.float32)


# ---------------------------------------------------------------------------
# Lid-driven cavity 3D
# ---------------------------------------------------------------------------

class LidDrivenCavityDomain3D(PhysicsDomain3D):
    """Unit cube [ox, ox+L]^3 with a moving lid on the top face (z = oz+Lz).

    Boundary regions:
    - lid   (z = oz+L): Dirichlet moving wall
    - walls (other 5 faces): no-slip
    """

    def __init__(
        self,
        size: float = 1.0,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        lid_velocity: float = 1.0,
        lid_axis: str = "x",
    ):
        self.size = float(size)
        self.origin = tuple(float(v) for v in origin)
        self.lid_velocity = float(lid_velocity)
        self.lid_axis = lid_axis.lower()

        ox, oy, oz = self.origin
        L = self.size
        self.bounds_min = (ox, oy, oz)
        self.bounds_max = (ox + L, oy + L, oz + L)

        lid_cond: Dict[str, Any] = {"w": 0.0}
        if self.lid_axis == "x":
            lid_cond["u"] = self.lid_velocity
            lid_cond["v"] = 0.0
        else:
            lid_cond["u"] = 0.0
            lid_cond["v"] = self.lid_velocity

        self.boundary_regions = [
            BoundaryRegion("lid",   "dirichlet", lid_cond),
            BoundaryRegion("walls", "no_slip",   {"u": 0.0, "v": 0.0, "w": 0.0}),
        ]

    def sdf(self, p: np.ndarray) -> np.ndarray:
        from .sdf_shapes import sdf3d_box
        cx = (self.bounds_min[0] + self.bounds_max[0]) * 0.5
        cy = (self.bounds_min[1] + self.bounds_max[1]) * 0.5
        cz = (self.bounds_min[2] + self.bounds_max[2]) * 0.5
        h = self.size * 0.5
        return sdf3d_box(p.astype(np.float64), (cx, cy, cz), (h, h, h))

    def sample_boundary_region(self, region_name: str, n: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        ox, oy, oz = self.bounds_min
        ex, ey, ez = self.bounds_max

        if region_name == "lid":
            x = rng.uniform(ox, ex, n).astype(np.float32)
            y = rng.uniform(oy, ey, n).astype(np.float32)
            z = np.full(n, ez, dtype=np.float32)
            return np.stack([x, y, z], axis=1)

        elif region_name == "walls":
            # 5 faces: bottom (z=oz), front (y=oy), back (y=ey), left (x=ox), right (x=ex)
            Lx = ex - ox
            Ly = ey - oy
            Lz = ez - oz
            area_bottom = Lx * Ly
            area_front  = Lx * Lz
            area_back   = Lx * Lz
            area_left   = Ly * Lz
            area_right  = Ly * Lz
            total_area = area_bottom + area_front + area_back + area_left + area_right

            counts = _split_by_area(
                n,
                [area_bottom, area_front, area_back, area_left, area_right],
                total_area,
            )
            n_bot, n_front, n_back, n_left, n_right = counts

            parts: List[np.ndarray] = []

            if n_bot > 0:
                x = rng.uniform(ox, ex, n_bot).astype(np.float32)
                y = rng.uniform(oy, ey, n_bot).astype(np.float32)
                parts.append(np.stack([x, y, np.full(n_bot, oz, np.float32)], axis=1))

            if n_front > 0:
                x = rng.uniform(ox, ex, n_front).astype(np.float32)
                z = rng.uniform(oz, ez, n_front).astype(np.float32)
                parts.append(np.stack([x, np.full(n_front, oy, np.float32), z], axis=1))

            if n_back > 0:
                x = rng.uniform(ox, ex, n_back).astype(np.float32)
                z = rng.uniform(oz, ez, n_back).astype(np.float32)
                parts.append(np.stack([x, np.full(n_back, ey, np.float32), z], axis=1))

            if n_left > 0:
                y = rng.uniform(oy, ey, n_left).astype(np.float32)
                z = rng.uniform(oz, ez, n_left).astype(np.float32)
                parts.append(np.stack([np.full(n_left, ox, np.float32), y, z], axis=1))

            if n_right > 0:
                y = rng.uniform(oy, ey, n_right).astype(np.float32)
                z = rng.uniform(oz, ez, n_right).astype(np.float32)
                parts.append(np.stack([np.full(n_right, ex, np.float32), y, z], axis=1))

            return np.concatenate(parts, axis=0) if parts else np.zeros((0, 3), dtype=np.float32)

        raise KeyError(f"Unknown region '{region_name}'")

    def sample_structured_grid(self, nx: int, ny: int, nz: int) -> np.ndarray:
        xs = np.linspace(self.bounds_min[0], self.bounds_max[0], nx, dtype=np.float64)
        ys = np.linspace(self.bounds_min[1], self.bounds_max[1], ny, dtype=np.float64)
        zs = np.linspace(self.bounds_min[2], self.bounds_max[2], nz, dtype=np.float64)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# 3D Rectangular channel (duct)
# ---------------------------------------------------------------------------

class ChannelDomain3D(PhysicsDomain3D):
    """Rectangular duct: x in [ox, ox+Lx], y in [oy, oy+Ly], z in [oz, oz+Lz].

    Flow is driven in the x-direction.

    Boundary regions:
    - inlet  (x = ox): Dirichlet velocity
    - outlet (x = ox+Lx): Neumann pressure outlet
    - walls  (y=oy, y=oy+Ly, z=oz, z=oz+Lz): no-slip
    """

    def __init__(
        self,
        length: float = 2.0,
        height: float = 1.0,
        width: float = 1.0,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        inlet_velocity: float = 1.0,
        flow_profile: str = "uniform",
    ):
        self.length = float(length)
        self.height = float(height)
        self.width = float(width)
        self.origin = tuple(float(v) for v in origin)
        self.inlet_velocity = float(inlet_velocity)
        self.flow_profile = flow_profile

        ox, oy, oz = self.origin
        self.bounds_min = (ox, oy, oz)
        self.bounds_max = (ox + self.length, oy + self.height, oz + self.width)

        inlet_cond: Dict[str, Any] = {
            "u": inlet_velocity,
            "v": 0.0,
            "w": 0.0,
            "_profile": flow_profile,
            "_poiseuille": {
                "y_center": oy + self.height * 0.5,
                "z_center": oz + self.width * 0.5,
                "half_height": self.height * 0.5,
                "half_width": self.width * 0.5,
            },
        }

        self.boundary_regions = [
            BoundaryRegion("inlet",  "dirichlet", inlet_cond),
            BoundaryRegion("outlet", "neumann",   {"p": 0.0}),
            BoundaryRegion("walls",  "no_slip",   {"u": 0.0, "v": 0.0, "w": 0.0}),
        ]

    def sdf(self, p: np.ndarray) -> np.ndarray:
        from .sdf_shapes import sdf3d_box
        cx = (self.bounds_min[0] + self.bounds_max[0]) * 0.5
        cy = (self.bounds_min[1] + self.bounds_max[1]) * 0.5
        cz = (self.bounds_min[2] + self.bounds_max[2]) * 0.5
        hx = self.length * 0.5
        hy = self.height * 0.5
        hz = self.width * 0.5
        return sdf3d_box(p.astype(np.float64), (cx, cy, cz), (hx, hy, hz))

    def sample_boundary_region(self, region_name: str, n: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        ox, oy, oz = self.bounds_min
        ex, ey, ez = self.bounds_max

        if region_name == "inlet":
            y = rng.uniform(oy, ey, n).astype(np.float32)
            z = rng.uniform(oz, ez, n).astype(np.float32)
            x = np.full(n, ox, dtype=np.float32)
            return np.stack([x, y, z], axis=1)

        elif region_name == "outlet":
            y = rng.uniform(oy, ey, n).astype(np.float32)
            z = rng.uniform(oz, ez, n).astype(np.float32)
            x = np.full(n, ex, dtype=np.float32)
            return np.stack([x, y, z], axis=1)

        elif region_name == "walls":
            Lx = ex - ox
            Ly = ey - oy
            Lz = ez - oz
            area_bot   = Lx * Lz
            area_top   = Lx * Lz
            area_front = Lx * Ly
            area_back  = Lx * Ly
            total_area = area_bot + area_top + area_front + area_back

            counts = _split_by_area(n, [area_bot, area_top, area_front, area_back], total_area)
            n_bot, n_top, n_front, n_back = counts

            parts: List[np.ndarray] = []

            if n_bot > 0:
                x = rng.uniform(ox, ex, n_bot).astype(np.float32)
                z = rng.uniform(oz, ez, n_bot).astype(np.float32)
                parts.append(np.stack([x, np.full(n_bot, oy, np.float32), z], axis=1))

            if n_top > 0:
                x = rng.uniform(ox, ex, n_top).astype(np.float32)
                z = rng.uniform(oz, ez, n_top).astype(np.float32)
                parts.append(np.stack([x, np.full(n_top, ey, np.float32), z], axis=1))

            if n_front > 0:
                x = rng.uniform(ox, ex, n_front).astype(np.float32)
                y = rng.uniform(oy, ey, n_front).astype(np.float32)
                parts.append(np.stack([x, y, np.full(n_front, oz, np.float32)], axis=1))

            if n_back > 0:
                x = rng.uniform(ox, ex, n_back).astype(np.float32)
                y = rng.uniform(oy, ey, n_back).astype(np.float32)
                parts.append(np.stack([x, y, np.full(n_back, ez, np.float32)], axis=1))

            return np.concatenate(parts, axis=0) if parts else np.zeros((0, 3), dtype=np.float32)

        raise KeyError(f"Unknown region '{region_name}'")

    def sample_structured_grid(self, nx: int, ny: int, nz: int) -> np.ndarray:
        xs = np.linspace(self.bounds_min[0], self.bounds_max[0], nx, dtype=np.float64)
        ys = np.linspace(self.bounds_min[1], self.bounds_max[1], ny, dtype=np.float64)
        zs = np.linspace(self.bounds_min[2], self.bounds_max[2], nz, dtype=np.float64)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Pipe flow 3D (circular cross-section)
# ---------------------------------------------------------------------------

class PipeFlowDomain3D(PhysicsDomain3D):
    """Circular cross-section pipe, flow in the x-direction.

    The pipe axis is aligned with x.  The circle center in the yz-plane is
    at (y_c, z_c) = (oy + radius, oz + radius) so that the bounding box
    starts at (ox, oy, oz) and the circle fits exactly inside.

    Boundary regions:
    - inlet  (x = ox face, inside circle): Dirichlet Poiseuille profile
    - outlet (x = ox+L face, inside circle): Neumann
    - wall   (cylindrical surface): no-slip
    """

    def __init__(
        self,
        radius: float = 0.5,
        length: float = 2.0,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        inlet_velocity: float = 1.0,
    ):
        self.radius = float(radius)
        self.length = float(length)
        self.origin = tuple(float(v) for v in origin)
        self.inlet_velocity = float(inlet_velocity)

        ox, oy, oz = self.origin
        R = self.radius
        L = self.length

        self.y_center = oy + R
        self.z_center = oz + R
        self.x_start  = ox
        self.x_end    = ox + L

        self.bounds_min = (ox,     oy,     oz)
        self.bounds_max = (ox + L, oy + R * 2.0, oz + R * 2.0)

        self.boundary_regions = [
            BoundaryRegion(
                "inlet",
                "dirichlet",
                {
                    "u": inlet_velocity,
                    "v": 0.0,
                    "w": 0.0,
                    "_profile": "poiseuille",
                    "_poiseuille": {
                        "y_center": self.y_center,
                        "z_center": self.z_center,
                        "radius": R,
                    },
                },
            ),
            BoundaryRegion("outlet", "neumann",  {"p": 0.0}),
            BoundaryRegion("wall",   "no_slip",  {"u": 0.0, "v": 0.0, "w": 0.0}),
        ]

    def sdf(self, p: np.ndarray) -> np.ndarray:
        p64 = p.astype(np.float64)
        cx = (self.x_start + self.x_end) * 0.5
        cy = self.y_center
        cz = self.z_center
        R  = self.radius
        L  = self.length

        dy = p64[:, 1] - cy
        dz = p64[:, 2] - cz
        dx = p64[:, 0] - cx

        d_r = np.sqrt(dy ** 2 + dz ** 2) - R
        d_a = np.abs(dx) - L * 0.5

        d2 = np.minimum(np.maximum(d_r, d_a), 0.0) + np.sqrt(
            np.maximum(d_r, 0.0) ** 2 + np.maximum(d_a, 0.0) ** 2
        )
        return d2

    def _is_in_circle(self, p: np.ndarray) -> np.ndarray:
        dy = p[:, 1] - self.y_center
        dz = p[:, 2] - self.z_center
        return (dy ** 2 + dz ** 2) <= self.radius ** 2

    def sample_interior(self, n: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        R  = self.radius
        ox = self.x_start
        L  = self.length
        cy = self.y_center
        cz = self.z_center

        collected: List[np.ndarray] = []
        while sum(len(c) for c in collected) < n:
            batch = n * 5
            x  = rng.uniform(ox, ox + L, batch)
            r  = R * np.sqrt(rng.random(batch))
            th = rng.uniform(0.0, 2.0 * math.pi, batch)
            y  = cy + r * np.cos(th)
            z  = cz + r * np.sin(th)
            pts = np.stack([x, y, z], axis=1)
            collected.append(pts)

        return np.concatenate(collected, axis=0)[:n].astype(np.float32)

    def sample_boundary_region(self, region_name: str, n: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        R  = self.radius
        cy = self.y_center
        cz = self.z_center
        ox = self.x_start

        if region_name in ("inlet", "outlet"):
            x_val = ox if region_name == "inlet" else self.x_end
            r  = R * np.sqrt(rng.random(n)).astype(np.float32)
            th = rng.uniform(0.0, 2.0 * math.pi, n).astype(np.float32)
            y  = (cy + r * np.cos(th)).astype(np.float32)
            z  = (cz + r * np.sin(th)).astype(np.float32)
            x  = np.full(n, x_val, dtype=np.float32)
            return np.stack([x, y, z], axis=1)

        elif region_name == "wall":
            t  = rng.random(n).astype(np.float32)
            x  = (ox + self.length * t).astype(np.float32)
            th = rng.uniform(0.0, 2.0 * math.pi, n).astype(np.float32)
            y  = (cy + R * np.cos(th)).astype(np.float32)
            z  = (cz + R * np.sin(th)).astype(np.float32)
            return np.stack([x, y, z], axis=1)

        raise KeyError(f"Unknown region '{region_name}'")

    def sample_structured_grid(self, nx: int, ny: int, nz: int) -> np.ndarray:
        xs = np.linspace(self.bounds_min[0], self.bounds_max[0], nx, dtype=np.float64)
        ys = np.linspace(self.bounds_min[1], self.bounds_max[1], ny, dtype=np.float64)
        zs = np.linspace(self.bounds_min[2], self.bounds_max[2], nz, dtype=np.float64)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        mask = self.is_inside(pts)
        return pts[mask].astype(np.float32)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_by_area(n: int, areas: List[float], total: float) -> List[int]:
    """Distribute n points across faces proportionally to area."""
    if total <= 0.0:
        k = len(areas)
        base = n // k
        counts = [base] * k
        counts[-1] += n - base * k
        return counts

    fracs = [a / total for a in areas]
    counts = [int(math.floor(f * n)) for f in fracs]
    remainder = n - sum(counts)
    for i in range(remainder):
        counts[i % len(counts)] += 1
    return counts


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_DOMAIN_REGISTRY_3D: Dict[str, type] = {
    "lid_driven_cavity_3d": LidDrivenCavityDomain3D,
    "channel_3d": ChannelDomain3D,
    "pipe_flow_3d": PipeFlowDomain3D,
}


def get_domain_3d(name: str, **kwargs) -> PhysicsDomain3D:
    """Instantiate a built-in 3D domain by name.

    Parameters
    ----------
    name : domain identifier (e.g. "channel_3d", "lid_driven_cavity_3d")
    **kwargs : constructor arguments
    """
    key = name.lower().strip()
    if key not in _DOMAIN_REGISTRY_3D:
        raise KeyError(f"Unknown 3D domain '{name}'. Available: {sorted(_DOMAIN_REGISTRY_3D.keys())}")
    return _DOMAIN_REGISTRY_3D[key](**kwargs)


def list_domains_3d() -> List[str]:
    """List all registered built-in 3D domain names."""
    return sorted(_DOMAIN_REGISTRY_3D.keys())


__all__ = [
    "PhysicsDomain3D",
    "LidDrivenCavityDomain3D",
    "ChannelDomain3D",
    "PipeFlowDomain3D",
    "_DOMAIN_REGISTRY_3D",
    "get_domain_3d",
    "list_domains_3d",
]
