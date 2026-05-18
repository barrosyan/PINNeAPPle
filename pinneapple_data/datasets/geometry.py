"""Pre-built geometry datasets for PINNeAPPle benchmarks.

Provides boundary and interior point sets for common engineering geometries.
All datasets are analytically generated — no external mesh files required.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .registry import DatasetInfo, DatasetRegistry


def _uniform_circle_boundary(cx: float, cy: float, r: float,
                              n: int) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return cx + r * np.cos(theta), cy + r * np.sin(theta)


def _rejection_interior(xlo, xhi, ylo, yhi, sdf_fn, n: int,
                         seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sample interior points using rejection sampling against an SDF."""
    rng = np.random.default_rng(seed)
    pts: list = []
    while len(pts) < n:
        batch = rng.uniform([xlo, ylo], [xhi, yhi], (n * 4, 2))
        mask = sdf_fn(batch[:, 0], batch[:, 1]) < 0.0
        pts.extend(batch[mask].tolist())
    arr = np.array(pts[:n])
    return arr[:, 0], arr[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# 1. NACA 0012 airfoil
#    Surface coordinates from the 4-digit NACA formula.
#    Includes surface points, normals, and a structured far-field grid.
# ─────────────────────────────────────────────────────────────────────────────

def _naca4_thickness(x: np.ndarray, t: float) -> np.ndarray:
    """NACA 4-digit thickness distribution."""
    return 5*t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
                  + 0.2843*x**3 - 0.1015*x**4)


def _load_naca0012(n_surface: int = 200,
                   n_interior: int = 2000,
                   chord: float = 1.0,
                   aoa_deg: float = 0.0) -> Dict[str, np.ndarray]:
    # Surface coordinates (upper + lower)
    x_c = np.linspace(0.0, chord, n_surface // 2)
    yt = _naca4_thickness(x_c / chord, t=0.12) * chord
    x_upper = x_c
    y_upper = yt
    x_lower = x_c[::-1]
    y_lower = -yt[::-1]
    x_surf = np.concatenate([x_upper, x_lower])
    y_surf = np.concatenate([y_upper, y_lower])

    # Outward normals (approximate via finite differences)
    dx = np.gradient(x_surf)
    dy = np.gradient(y_surf)
    length = np.sqrt(dx**2 + dy**2) + 1e-12
    nx = dy / length
    ny = -dx / length

    # Angle of attack rotation
    aoa = math.radians(aoa_deg)
    cos_a, sin_a = math.cos(aoa), math.sin(aoa)
    x_rot = x_surf * cos_a - y_surf * sin_a
    y_rot = x_surf * sin_a + y_surf * cos_a

    # Interior collocation points (bounding box [-0.5,1.5]×[-1,1], outside airfoil)
    rng = np.random.default_rng(42)
    x_int, y_int = [], []
    while len(x_int) < n_interior:
        xr = rng.uniform(-0.5, 1.5, n_interior * 3)
        yr = rng.uniform(-1.0, 1.0, n_interior * 3)
        # keep points outside airfoil (crude SDF: y outside yt at that x)
        for xi, yi in zip(xr, yr):
            if 0 <= xi <= chord:
                thickness = _naca4_thickness(xi / chord, 0.12) * chord
                if abs(yi) > thickness * 1.1:
                    x_int.append(xi); y_int.append(yi)
            else:
                x_int.append(xi); y_int.append(yi)
            if len(x_int) >= n_interior:
                break

    return {
        "surface_x": x_rot,
        "surface_y": y_rot,
        "normal_x": nx,
        "normal_y": ny,
        "interior_x": np.array(x_int[:n_interior]),
        "interior_y": np.array(y_int[:n_interior]),
        "chord": np.float64(chord),
        "aoa_deg": np.float64(aoa_deg),
        "description": "NACA 0012 airfoil — surface + exterior collocation points",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cylinder 2D
#    Circle of radius r centered at origin, inside a [-D,D]×[-D,D] box.
# ─────────────────────────────────────────────────────────────────────────────

def _load_cylinder_2d(r: float = 0.5, D: float = 2.5,
                      n_surface: int = 200,
                      n_interior: int = 2000) -> Dict[str, np.ndarray]:
    bx, by = _uniform_circle_boundary(0.0, 0.0, r, n_surface)

    # Domain boundary (box)
    side = n_surface // 4
    t = np.linspace(-D, D, side)
    box_x = np.concatenate([t, np.full(side, D), t[::-1], np.full(side, -D)])
    box_y = np.concatenate([np.full(side, -D), t, np.full(side, D), t[::-1]])

    # Interior: inside box, outside cylinder
    def _sdf(x, y):
        return np.sqrt(x**2 + y**2) - r   # >0 outside cylinder

    ix, iy = _rejection_interior(-D, D, -D, D,
                                  lambda x, y: -_sdf(x, y),   # negative inside cylinder
                                  n_interior)
    return {
        "cylinder_x": bx,
        "cylinder_y": by,
        "box_x": box_x,
        "box_y": box_y,
        "interior_x": ix,
        "interior_y": iy,
        "radius": np.float64(r),
        "domain_size": np.float64(D),
        "description": "2D cylinder domain: surface + box boundary + exterior collocation",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. L-Shape domain
#    [0,2]×[0,2] minus [1,2]×[1,2]  (classic singularity benchmark)
# ─────────────────────────────────────────────────────────────────────────────

def _load_lshape_2d(n_boundary: int = 300,
                    n_interior: int = 2000) -> Dict[str, np.ndarray]:
    # 6-segment boundary (ccw)
    segs = [
        (np.linspace(0, 2, n_boundary//6), np.full(n_boundary//6, 0.0)),
        (np.full(n_boundary//6, 2.0), np.linspace(0, 1, n_boundary//6)),
        (np.linspace(2, 1, n_boundary//6), np.full(n_boundary//6, 1.0)),
        (np.full(n_boundary//6, 1.0), np.linspace(1, 2, n_boundary//6)),
        (np.linspace(1, 0, n_boundary//6), np.full(n_boundary//6, 2.0)),
        (np.full(n_boundary//6, 0.0), np.linspace(2, 0, n_boundary//6)),
    ]
    bx = np.concatenate([s[0] for s in segs])
    by = np.concatenate([s[1] for s in segs])

    # Interior via rejection
    rng = np.random.default_rng(42)
    pts: list = []
    while len(pts) < n_interior:
        batch = rng.uniform([0, 0], [2, 2], (n_interior * 3, 2))
        mask = ~((batch[:, 0] > 1.0) & (batch[:, 1] > 1.0))
        pts.extend(batch[mask].tolist())
    arr = np.array(pts[:n_interior])
    return {
        "boundary_x": bx,
        "boundary_y": by,
        "interior_x": arr[:, 0],
        "interior_y": arr[:, 1],
        "description": "L-shaped domain [0,2]²\\[1,2]²  — classical singularity benchmark",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Channel with circular obstacle
# ─────────────────────────────────────────────────────────────────────────────

def _load_channel_with_obstacle(L: float = 4.0, H: float = 1.0,
                                 r: float = 0.15, cx: float = 1.0,
                                 cy: float = 0.5,
                                 n_pts: int = 3000) -> Dict[str, np.ndarray]:
    # Interior points: inside channel, outside cylinder
    rng = np.random.default_rng(0)
    pts: list = []
    while len(pts) < n_pts:
        batch = rng.uniform([0, 0], [L, H], (n_pts * 4, 2))
        d_cyl = np.sqrt((batch[:, 0] - cx)**2 + (batch[:, 1] - cy)**2)
        mask = d_cyl > r * 1.05
        pts.extend(batch[mask].tolist())
    arr = np.array(pts[:n_pts])

    # Inlet (x=0)
    y_in = np.linspace(0, H, 50)
    u_in = 4.0 * y_in * (H - y_in) / H**2   # parabolic Poiseuille

    # Cylinder surface
    theta = np.linspace(0, 2*math.pi, 100, endpoint=False)
    cyl_x = cx + r * np.cos(theta)
    cyl_y = cy + r * np.sin(theta)

    return {
        "interior_x": arr[:, 0],
        "interior_y": arr[:, 1],
        "inlet_y": y_in,
        "inlet_u_profile": u_in,
        "cylinder_x": cyl_x,
        "cylinder_y": cyl_y,
        "L": np.float64(L),
        "H": np.float64(H),
        "r": np.float64(r),
        "description": "Channel flow with circular obstacle — PINN collocation geometry",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

DatasetRegistry.register(
    DatasetInfo(
        id="naca0012",
        name="NACA 0012 Airfoil",
        category="geometry",
        description="NACA 0012 surface + exterior collocation points. Adjust aoa_deg for angle of attack.",
        fields=["surface_x", "surface_y", "normal_x", "normal_y", "interior_x", "interior_y"],
        tags=["aerodynamics", "airfoil", "2d", "cfd"],
        reference="Abbott & Von Doenhoff 1959",
    ),
    _load_naca0012,
)

DatasetRegistry.register(
    DatasetInfo(
        id="cylinder_2d",
        name="Cylinder 2D",
        category="geometry",
        description="2D circular cylinder with box domain — classic bluff-body CFD geometry.",
        fields=["cylinder_x", "cylinder_y", "box_x", "box_y", "interior_x", "interior_y"],
        tags=["cylinder", "bluff-body", "2d", "cfd"],
    ),
    _load_cylinder_2d,
)

DatasetRegistry.register(
    DatasetInfo(
        id="lshape_2d",
        name="L-Shape 2D",
        category="geometry",
        description="L-shaped domain with re-entrant corner — benchmark for singularity-aware PINNs.",
        fields=["boundary_x", "boundary_y", "interior_x", "interior_y"],
        tags=["lshape", "singularity", "2d", "structural"],
    ),
    _load_lshape_2d,
)

DatasetRegistry.register(
    DatasetInfo(
        id="channel_with_obstacle",
        name="Channel with Circular Obstacle",
        category="geometry",
        description="2D channel flow geometry with a circular obstacle and parabolic inlet.",
        fields=["interior_x", "interior_y", "inlet_y", "inlet_u_profile", "cylinder_x", "cylinder_y"],
        tags=["channel", "obstacle", "2d", "cfd", "poiseuille"],
    ),
    _load_channel_with_obstacle,
)
