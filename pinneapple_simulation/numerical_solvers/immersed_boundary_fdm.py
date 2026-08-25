"""Immersed-boundary-method (IBM) finite-difference solver for incompressible
Navier-Stokes on an arbitrary 3D geometry.

The ONLY geometry input is a surface point cloud (an (N,3) array of points
sampled on the wall/boundary of some solid, e.g. exported from an STL mesh)
plus the bounding box of the domain to mesh. No mesh connectivity, CAD
kernel, or problem-specific metadata is required.

Method
------
1. A regular Cartesian grid is laid over the domain bounding box (cell-
   centered, uniform spacing per axis).
2. **IBM wall mask**: a `scipy.spatial.cKDTree` built on the wall point cloud
   is queried for nearest-surface distance at every grid cell; cells within
   ``~1.5 * min(dx,dy,dz)`` of the surface are flagged "wall" and have
   no-slip (zero velocity) enforced every step.
3. **Exterior/interior mask**: a `scipy.spatial.Delaunay` triangulation of a
   jittered subsample of the wall points is used as an inside/outside test
   (`find_simplex`), so that cells outside the solid's convex/concave hull
   implied by the point cloud are excluded from the fluid domain. Jitter is
   used to make the triangulation robust to near-degenerate/coplanar wall
   samples.
4. **Time integration**: explicit predictor-corrector (Chorin-style
   fractional-step / projection method):
     - Predictor: first-order upwind convection + explicit viscous diffusion
       (second-order Adams-Bashforth in time once two steps are available),
       plus the previous pressure gradient.
     - Pressure correction: either (a) a fast multi-step divergence-penalty
       relaxation (``solve_ibm_internal_flow``) or (b) a Jacobi/SOR pressure-
       Poisson solve of ``div(u*)`` (``solve_ibm_external_flow``), each
       driving the predicted velocity field toward zero divergence.
     - The IBM wall mask and exterior mask are re-applied after every
       sub-step so solid/outside cells stay at rest.

Two boundary-condition topologies are provided, matching two common flow
classes:

- ``solve_ibm_internal_flow``: the geometry bounding box IS the fluid
  domain (an internal / enclosed flow). Two sub-modes:
    * "lid_driven" -- classic lid-driven-cavity benchmark: one bounding
      face moves at a fixed tangential velocity, all other faces (and any
      interior wall point cloud) are no-slip.
    * "channel" -- automatic inlet/outlet detection: the 6 bounding faces
      are scanned for open (non-wall-masked) cross-sections; the two with
      the most open fluid cells are taken as the inlet/outlet pair (an
      optional hint point picks which of the two is the inlet), and a
      uniform normal inflow velocity / convective outflow condition is
      applied there. This generalizes to bent/curved conduits, not just
      straight channels.
- ``solve_ibm_external_flow``: the geometry is treated as a solid immersed
  in a free stream inside a larger padded "wind tunnel" domain built around
  the geometry bounding box -- the generic setup for external/aerodynamic
  flow around an arbitrary immersed body.

Both functions take only plain arrays / scalars (wall point cloud, bbox,
Reynolds number, inlet velocity, density, grid resolution, iteration count)
-- no coupling to any particular problem-instance data format.
"""
from __future__ import annotations

import time as _time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry

BBox = Tuple[float, float, float, float, float, float]


# ═══════════════════════════════════════════════════════════════════════════
#  Shared FDM building blocks
# ═══════════════════════════════════════════════════════════════════════════

def _build_ibm_masks(
    wall_points: np.ndarray,
    coords_grid: np.ndarray,
    grid_shape: Tuple[int, int, int],
    h: float,
    wall_thresh_h: float = 1.5,
    hull_sample: int = 3000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Build the IBM wall mask and exterior (outside-geometry) mask.

    Returns (wall_mask3d, exterior_mask3d, dist_to_wall, kdtree, delaunay_or_none).
    ``dist_to_wall`` is the flat (nx*ny*nz,) nearest-surface distance at
    every grid cell, reusable for other distance-based masks (e.g. a
    thicker "solid body" mask for external flow). ``kdtree`` is the
    `cKDTree` built on `wall_points`, reusable for further distance queries.
    """
    from scipy.spatial import Delaunay, cKDTree

    tree = cKDTree(wall_points)
    dist_w, _ = tree.query(coords_grid, k=1)
    wall_mask3d = (dist_w < h * wall_thresh_h).reshape(grid_shape)

    n_hull = min(len(wall_points), hull_sample)
    rng_h = np.random.default_rng(seed)
    idx_h = (rng_h.choice(len(wall_points), n_hull, replace=False)
             if len(wall_points) > n_hull else np.arange(len(wall_points)))
    hull = None
    for jitter in (0.0, 1e-4, 5e-4):
        try:
            pts_j = wall_points[idx_h] + (
                rng_h.uniform(-jitter, jitter, (len(idx_h), 3)) if jitter > 0 else 0.0
            )
            hull = Delaunay(pts_j)
            break
        except Exception:
            continue

    if hull is not None:
        exterior_mask3d = (hull.find_simplex(coords_grid) < 0).reshape(grid_shape)
    else:
        exterior_mask3d = np.zeros(grid_shape, bool)

    return wall_mask3d, exterior_mask3d, dist_w, tree, hull


def _grid_operators(dx: float, dy: float, dz: float):
    """Return (laplacian, divergence, upwind_convection) closures for a
    uniform grid with periodic-index (np.roll-based) central differences --
    valid away from the domain boundary since walls/exterior are masked."""

    def laplacian(f: np.ndarray) -> np.ndarray:
        return ((np.roll(f, -1, 0) - 2 * f + np.roll(f, 1, 0)) / dx ** 2 +
                 (np.roll(f, -1, 1) - 2 * f + np.roll(f, 1, 1)) / dy ** 2 +
                 (np.roll(f, -1, 2) - 2 * f + np.roll(f, 1, 2)) / dz ** 2)

    def divergence(u_: np.ndarray, v_: np.ndarray, w_: np.ndarray) -> np.ndarray:
        return ((np.roll(u_, -1, 0) - np.roll(u_, 1, 0)) / (2 * dx) +
                 (np.roll(v_, -1, 1) - np.roll(v_, 1, 1)) / (2 * dy) +
                 (np.roll(w_, -1, 2) - np.roll(w_, 1, 2)) / (2 * dz))

    def upwind_convection(u_, v_, w_, f_):
        """First-order upwind advection of scalar field f_ by velocity (u_,v_,w_)."""
        cx = np.where(u_ >= 0, u_ * (f_ - np.roll(f_, 1, 0)) / dx,
                      u_ * (np.roll(f_, -1, 0) - f_) / dx)
        cy = np.where(v_ >= 0, v_ * (f_ - np.roll(f_, 1, 1)) / dy,
                      v_ * (np.roll(f_, -1, 1) - f_) / dy)
        cz = np.where(w_ >= 0, w_ * (f_ - np.roll(f_, 1, 2)) / dz,
                      w_ * (np.roll(f_, -1, 2) - f_) / dz)
        return cx + cy + cz

    return laplacian, divergence, upwind_convection


def field_divergence(u: np.ndarray, v: np.ndarray, w: np.ndarray,
                      dx: float, dy: float, dz: float) -> np.ndarray:
    """Central-difference divergence of a velocity field -- exposed standalone
    for verifying the near-zero-divergence property of a projected field."""
    _, divergence, _ = _grid_operators(dx, dy, dz)
    return divergence(u, v, w)


def interpolate_to_points(
    grid_axes: Tuple[np.ndarray, np.ndarray, np.ndarray],
    field: np.ndarray,
    query_points: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Trilinear interpolation of a (nx,ny,nz) grid field onto arbitrary
    query points, via `scipy.interpolate.RegularGridInterpolator`."""
    from scipy.interpolate import RegularGridInterpolator

    fn = RegularGridInterpolator(grid_axes, field, method="linear",
                                  bounds_error=False, fill_value=fill_value)
    return fn(query_points).astype(np.float32)


def _sample_interior_points(
    bbox: BBox,
    tree: Any,
    hull: Any,
    h: float,
    n_target: int,
    seed: int = 42,
) -> np.ndarray:
    """Rejection-sample points strictly inside the fluid interior (inside the
    Delaunay hull of the wall points, and not within the near-wall IBM band)
    -- used as default visualization/query points when the caller doesn't
    supply its own."""
    x0, y0, z0, x1, y1, z1 = bbox
    rng = np.random.default_rng(seed)
    out = np.empty((0, 3), dtype=np.float32)
    for mult in (8, 20, 40):
        cands = np.column_stack([
            rng.uniform(x0, x1, n_target * mult),
            rng.uniform(y0, y1, n_target * mult),
            rng.uniform(z0, z1, n_target * mult),
        ]).astype(np.float32)
        inside = (hull.find_simplex(cands) >= 0 if hull is not None
                  else np.ones(len(cands), bool))
        d_w, _ = tree.query(cands[inside])
        inner = cands[inside][d_w > h * 1.8]
        if len(inner) >= n_target:
            out = inner[:n_target]
            break
        out = inner
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Internal (enclosed) flow: lid-driven cavity / channel with auto BC faces
# ═══════════════════════════════════════════════════════════════════════════

def solve_ibm_internal_flow(
    wall_points: np.ndarray,
    bbox: BBox,
    *,
    Re: float = 100.0,
    U_in: float = 1.0,
    rho: float = 1.0,
    nx: int = 32,
    ny: int = 32,
    nz: int = 32,
    n_iter: int = 500,
    n_frames: int = 0,
    bc_mode: str = "lid_driven",
    lid_dir: str = "z+",
    channel_dir: str = "x+",
    inlet_hint_point: Optional[np.ndarray] = None,
    query_points: Optional[np.ndarray] = None,
    log_every: Optional[int] = None,
    verbose: bool = False,
    seed: int = 0,
) -> Dict[str, Any]:
    """IBM/FDM incompressible Navier-Stokes on an enclosed domain whose
    bounding box IS the fluid domain.

    Parameters
    ----------
    wall_points : (N,3) array
        Point cloud sampled on the solid wall/surface geometry. This is the
        only geometry input.
    bbox : (x0,y0,z0,x1,y1,z1)
        Bounding box of the fluid domain to mesh.
    Re, U_in, rho : Reynolds number, characteristic (lid/inlet) velocity,
        density. Kinematic viscosity is derived as ``nu = U_in*L_char/Re``
        with ``L_char = max(Lx,Ly,Lz)``.
    nx, ny, nz : grid resolution.
    n_iter : number of explicit time steps.
    n_frames : number of evenly-spaced convergence snapshots to record
        (0 disables snapshotting).
    bc_mode : "lid_driven" (default) or "channel".
      - "lid_driven": one bounding face (`lid_dir`) moves tangentially at
        `U_in`; all other boundary/wall cells are no-slip.
      - "channel": the 6 bounding faces are scanned for open (non-wall)
        cross-sections; the two with the most open fluid cells become the
        inlet/outlet pair (uniform normal inflow + convective outflow).
        `inlet_hint_point` (a physical-space point near the desired inlet)
        disambiguates which of the two detected faces is the inlet; falls
        back to `channel_dir`-based axis selection if fewer than two open
        faces are found (e.g. a fully enclosed geometry).
    query_points : (M,3) array, optional
        Points to interpolate the final velocity/pressure field onto.
        Defaults to a rejection-sampled cloud of interior fluid points.

    Returns
    -------
    dict with keys: coords, u, v, w, p, speed, divergence_rms,
    surf_p, surf_speed, grid_axes, wall_mask_fraction, dt, nu, frames (list,
    possibly empty), bc_info.
    """
    from scipy.spatial import cKDTree

    wall_points = np.asarray(wall_points, dtype=np.float64)
    x0, y0, z0, x1, y1, z1 = bbox
    Lx, Ly, Lz = x1 - x0, y1 - y0, z1 - z0
    L_char = max(Lx, Ly, Lz)
    nu = U_in * L_char / Re

    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
    xg = np.linspace(x0 + dx / 2, x1 - dx / 2, nx)
    yg = np.linspace(y0 + dy / 2, y1 - dy / 2, ny)
    zg = np.linspace(z0 + dz / 2, z1 - dz / 2, nz)
    XX, YY, ZZ = np.meshgrid(xg, yg, zg, indexing="ij")
    coords_grid = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    h = min(dx, dy, dz)

    wall_mask3d, exterior_mask3d, dist_w, tree, hull = _build_ibm_masks(
        wall_points, coords_grid, (nx, ny, nz), h, seed=seed,
    )

    # -- Lid mask (used only for bc_mode == "lid_driven") --
    if lid_dir == "z+":
        lid_mask3d = ZZ > z1 - dz * 2
    elif lid_dir == "z-":
        lid_mask3d = ZZ < z0 + dz * 2
    elif lid_dir == "y+":
        lid_mask3d = YY > y1 - dy * 2
    elif lid_dir == "y-":
        lid_mask3d = YY < y0 + dy * 2
    elif lid_dir == "x+":
        lid_mask3d = XX > x1 - dx * 2
    else:
        lid_mask3d = XX < x0 + dx * 2

    bc_info: Dict[str, Any] = {"bc_mode": bc_mode}
    inlet_mask3d = outlet_mask3d = None
    inlet_face = outlet_face = None

    if bc_mode == "channel":
        fluid_tmp = ~wall_mask3d & ~exterior_mask3d
        faces = {
            "x-": fluid_tmp[:2, :, :].sum(), "x+": fluid_tmp[-2:, :, :].sum(),
            "y-": fluid_tmp[:, :2, :].sum(), "y+": fluid_tmp[:, -2:, :].sum(),
            "z-": fluid_tmp[:, :, :2].sum(), "z+": fluid_tmp[:, :, -2:].sum(),
        }

        def _face_mask(face_name: str) -> np.ndarray:
            m = np.zeros((nx, ny, nz), bool)
            if face_name == "x-": m[:2, :, :] = True
            elif face_name == "x+": m[-2:, :, :] = True
            elif face_name == "y-": m[:, :2, :] = True
            elif face_name == "y+": m[:, -2:, :] = True
            elif face_name == "z-": m[:, :, :2] = True
            elif face_name == "z+": m[:, :, -2:] = True
            return m

        def _face_centroid(face_name: str) -> np.ndarray:
            cx, cy, cz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
            return {
                "x-": np.array([x0, cy, cz]), "x+": np.array([x1, cy, cz]),
                "y-": np.array([cx, y0, cz]), "y+": np.array([cx, y1, cz]),
                "z-": np.array([cx, cy, z0]), "z+": np.array([cx, cy, z1]),
            }[face_name]

        open_faces = sorted(faces.items(), key=lambda kv: -kv[1])
        pipe_ends = [(k, v) for k, v in open_faces if v > 5][:2]

        if len(pipe_ends) >= 2:
            if inlet_hint_point is not None:
                hint = np.asarray(inlet_hint_point, dtype=float).reshape(3)
                dists = {k: float(np.linalg.norm(_face_centroid(k) - hint)) for k, _ in pipe_ends}
                inlet_face = min(dists, key=dists.get)
                outlet_face = [k for k, _ in pipe_ends if k != inlet_face][0]
            else:
                inlet_face, outlet_face = pipe_ends[0][0], pipe_ends[1][0]
            inlet_mask3d = _face_mask(inlet_face)
            outlet_mask3d = _face_mask(outlet_face)
        else:
            if channel_dir.startswith("x"):
                inlet_face, outlet_face = "x-", "x+"
                inlet_mask3d, outlet_mask3d = XX < x0 + dx * 2, XX > x1 - dx * 2
            elif channel_dir.startswith("y"):
                inlet_face, outlet_face = "y-", "y+"
                inlet_mask3d, outlet_mask3d = YY < y0 + dy * 2, YY > y1 - dy * 2
            else:
                inlet_face, outlet_face = "z-", "z+"
                inlet_mask3d, outlet_mask3d = ZZ < z0 + dz * 2, ZZ > z1 - dz * 2
        bc_info.update(inlet_face=inlet_face, outlet_face=outlet_face)

    fluid_mask = ~exterior_mask3d & ~wall_mask3d
    u = np.zeros((nx, ny, nz))
    v = np.zeros_like(u)
    w = np.zeros_like(u)
    p = np.zeros_like(u)

    if bc_mode == "channel":
        inlet_fl = inlet_mask3d & fluid_mask
        outlet_fl = outlet_mask3d & fluid_mask

        def _apply_inlet_vel(u_, v_, w_):
            sign = -1.0 if "+" in inlet_face else 1.0
            if inlet_face in ("x-", "x+"):
                u_[inlet_fl] = U_in * sign; v_[inlet_fl] = 0.0; w_[inlet_fl] = 0.0
            elif inlet_face in ("y-", "y+"):
                v_[inlet_fl] = U_in * sign; u_[inlet_fl] = 0.0; w_[inlet_fl] = 0.0
            else:
                w_[inlet_fl] = U_in * sign; u_[inlet_fl] = 0.0; v_[inlet_fl] = 0.0

        def _apply_outlet_bc(u_, v_, w_):
            """Convective (zero-gradient) outlet: copy the adjacent interior cell."""
            if outlet_face == "x+": u_[-1, :, :] = u_[-2, :, :]; v_[-1, :, :] = v_[-2, :, :]; w_[-1, :, :] = w_[-2, :, :]
            elif outlet_face == "x-": u_[0, :, :] = u_[1, :, :]; v_[0, :, :] = v_[1, :, :]; w_[0, :, :] = w_[1, :, :]
            elif outlet_face == "y+": u_[:, -1, :] = u_[:, -2, :]; v_[:, -1, :] = v_[:, -2, :]; w_[:, -1, :] = w_[:, -2, :]
            elif outlet_face == "y-": u_[:, 0, :] = u_[:, 1, :]; v_[:, 0, :] = v_[:, 1, :]; w_[:, 0, :] = w_[:, 1, :]
            elif outlet_face == "z+": u_[:, :, -1] = u_[:, :, -2]; v_[:, :, -1] = v_[:, :, -2]; w_[:, :, -1] = w_[:, :, -2]
            else: u_[:, :, 0] = u_[:, :, 1]; v_[:, :, 0] = v_[:, :, 1]; w_[:, :, 0] = w_[:, :, 1]

        _apply_inlet_vel(u, v, w)
    else:
        u[lid_mask3d] = U_in

    dt_conv = 0.05 * h / (U_in + 1e-12)
    dt_diff = 0.05 * h ** 2 / (nu + 1e-12)
    dt = min(dt_conv, dt_diff)
    U_clip = 10.0 * max(U_in, 1.0)

    laplacian, divergence, upwind_convection = _grid_operators(dx, dy, dz)

    def _apply_pressure_penalty(u_, v_, w_, p_, n_sub: int = 3):
        """Multi-step divergence-penalty pressure correction: repeatedly
        estimates the local pressure increment needed to null the local
        divergence (an explicit local-Poisson relaxation), applies it to
        both the pressure field and the velocity field via the discrete
        gradient. Cheaper and more crash-resistant per step than a full
        Poisson solve, at the cost of needing more outer time steps."""
        for _ in range(n_sub):
            div = (np.gradient(u_, dx, axis=0) + np.gradient(v_, dy, axis=1)
                   + np.gradient(w_, dz, axis=2))
            dp = -(rho / (dt + 1e-30)) * div * (h ** 2 / 6.0)
            p_ = p_ + dp
            u_ = u_ - (dt / rho) * np.gradient(dp, dx, axis=0)
            v_ = v_ - (dt / rho) * np.gradient(dp, dy, axis=1)
            w_ = w_ - (dt / rho) * np.gradient(dp, dz, axis=2)
            p_[wall_mask3d] = 0.0
            if bc_mode == "channel":
                p_[outlet_mask3d] = 0.0
        return u_, v_, w_, p_

    snap_iters = ({int(round(n_iter * (fi + 1) / n_frames)) - 1 for fi in range(n_frames)}
                  if n_frames > 0 else set())
    if n_iter > 0:
        snap_iters.add(n_iter - 1)
    snapshots: List[Dict[str, np.ndarray]] = []

    rhs_u_prev = rhs_v_prev = rhs_w_prev = None
    div_rms_final = float("nan")

    for it in range(n_iter):
        u_max_cur = max(float(np.max(np.abs(u))), U_in, 1e-12)
        dt = min(0.20 * h / u_max_cur, dt_diff)

        dp_dx = (np.roll(p, -1, 0) - np.roll(p, 1, 0)) / (2 * dx)
        dp_dy = (np.roll(p, -1, 1) - np.roll(p, 1, 1)) / (2 * dy)
        dp_dz = (np.roll(p, -1, 2) - np.roll(p, 1, 2)) / (2 * dz)

        rhs_u = -upwind_convection(u, v, w, u) - dp_dx / rho + nu * laplacian(u)
        rhs_v = -upwind_convection(u, v, w, v) - dp_dy / rho + nu * laplacian(v)
        rhs_w = -upwind_convection(u, v, w, w) - dp_dz / rho + nu * laplacian(w)

        if rhs_u_prev is not None:
            u_star = u + dt * (1.5 * rhs_u - 0.5 * rhs_u_prev)
            v_star = v + dt * (1.5 * rhs_v - 0.5 * rhs_v_prev)
            w_star = w + dt * (1.5 * rhs_w - 0.5 * rhs_w_prev)
        else:
            u_star = u + dt * rhs_u
            v_star = v + dt * rhs_v
            w_star = w + dt * rhs_w
        rhs_u_prev, rhs_v_prev, rhs_w_prev = rhs_u, rhs_v, rhs_w

        u_star[wall_mask3d] = v_star[wall_mask3d] = w_star[wall_mask3d] = 0.0
        u_star[exterior_mask3d] = v_star[exterior_mask3d] = w_star[exterior_mask3d] = 0.0
        if bc_mode != "channel":
            u_star[lid_mask3d] = U_in; v_star[lid_mask3d] = 0.0; w_star[lid_mask3d] = 0.0
        else:
            _apply_inlet_vel(u_star, v_star, w_star)
            _apply_outlet_bc(u_star, v_star, w_star)

        u_new, v_new, w_new, p = _apply_pressure_penalty(u_star, v_star, w_star, p)

        u_new[wall_mask3d] = v_new[wall_mask3d] = w_new[wall_mask3d] = 0.0
        u_new[exterior_mask3d] = v_new[exterior_mask3d] = w_new[exterior_mask3d] = 0.0

        u = np.clip(u_new, -U_clip, U_clip)
        v = np.clip(v_new, -U_clip, U_clip)
        w = np.clip(w_new, -U_clip, U_clip)

        if not np.isfinite(u).all():
            u = np.nan_to_num(u, nan=0.0, posinf=U_clip, neginf=-U_clip)
            v = np.nan_to_num(v, nan=0.0, posinf=U_clip, neginf=-U_clip)
            w = np.nan_to_num(w, nan=0.0, posinf=U_clip, neginf=-U_clip)
            p = np.nan_to_num(p, nan=0.0)
            rhs_u_prev = rhs_v_prev = rhs_w_prev = None

        if it in snap_iters:
            snapshots.append({"it": it, "u": u.copy(), "v": v.copy(), "w": w.copy(), "p": p.copy()})

        eff_log_every = log_every if log_every is not None else max(1, n_iter // 8)
        if verbose and (it + 1) % eff_log_every == 0:
            div = divergence(u, v, w)
            div_rms_final = float(np.sqrt(np.mean(div ** 2)))
            print(f"  [ibm_internal] iter {it+1:4d}/{n_iter}  div_rms={div_rms_final:.3e}  "
                  f"u_max={float(np.max(np.abs(u))):.4f}")

    div_final = divergence(u, v, w)
    div_rms_final = float(np.sqrt(np.mean(div_final ** 2)))
    speed = np.sqrt(u ** 2 + v ** 2 + w ** 2)

    grid_axes = (xg, yg, zg)
    surf_p = interpolate_to_points(grid_axes, p, wall_points)
    surf_speed = interpolate_to_points(grid_axes, speed, wall_points)

    if query_points is None:
        n_vis = max(6000, int(fluid_mask.sum()) // 4)
        query_points = _sample_interior_points(bbox, tree, hull, h, n_vis)
        if len(query_points) < 100:
            query_points = coords_grid[fluid_mask.ravel()].astype(np.float32)
    query_points = np.asarray(query_points, dtype=np.float32)

    frames = []
    for snap in sorted(snapshots, key=lambda s: s["it"]):
        u_s = interpolate_to_points(grid_axes, snap["u"], query_points)
        v_s = interpolate_to_points(grid_axes, snap["v"], query_points)
        w_s = interpolate_to_points(grid_axes, snap["w"], query_points)
        p_s = interpolate_to_points(grid_axes, snap["p"], query_points)
        frames.append({"it": snap["it"], "u": u_s, "v": v_s, "w": w_s, "p": p_s})

    return {
        "coords": query_points,
        "u": interpolate_to_points(grid_axes, u, query_points),
        "v": interpolate_to_points(grid_axes, v, query_points),
        "w": interpolate_to_points(grid_axes, w, query_points),
        "p": interpolate_to_points(grid_axes, p, query_points),
        "speed": interpolate_to_points(grid_axes, speed, query_points),
        "surf_coords": wall_points,
        "surf_p": surf_p,
        "surf_speed": surf_speed,
        "divergence_rms": div_rms_final,
        "grid_axes": grid_axes,
        "grid_fields": {"u": u, "v": v, "w": w, "p": p,
                         "wall_mask": wall_mask3d, "exterior_mask": exterior_mask3d},
        "wall_mask_fraction": float(wall_mask3d.mean()),
        "dt": dt,
        "nu": nu,
        "bbox": list(bbox),
        "frames": frames,
        "bc_info": bc_info,
        "Re": Re,
        "U_in": U_in,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  External (immersed-body) flow: padded wind-tunnel domain around an
#  arbitrary geometry, free-stream inflow, IBM no-slip on the body surface
# ═══════════════════════════════════════════════════════════════════════════

def solve_ibm_external_flow(
    wall_points: np.ndarray,
    bbox: BBox,
    *,
    Re: float = 100.0,
    U_in: float = 1.0,
    rho: float = 1200.0,
    nx: int = 32,
    ny: int = 32,
    nz: int = 32,
    n_iter: int = 1000,
    conv_tol: float = 1e-4,
    pad_upstream: float = 1.5,
    pad_downstream: float = 3.5,
    pad_lateral: float = 1.0,
    poisson_iters: int = 80,
    poisson_omega: float = 0.667,
    query_points: Optional[np.ndarray] = None,
    n_vis_grid: Tuple[int, int, int] = (40, 16, 16),
    log_every: Optional[int] = None,
    verbose: bool = False,
    seed: int = 0,
) -> Dict[str, Any]:
    """IBM/FDM incompressible Navier-Stokes for external (free-stream)
    flow around an arbitrary solid immersed body.

    Unlike `solve_ibm_internal_flow`, the fluid domain is NOT the geometry
    bounding box: a padded "wind tunnel" box is built around it
    (`pad_upstream`/`pad_downstream` multiples of the body's streamwise
    extent, `pad_lateral` multiples of the body's largest cross-section),
    free-stream flows in along +x, and cells inside/near the body (per the
    IBM distance mask) are held at rest.

    Parameters mirror `solve_ibm_internal_flow` where applicable; additional
    parameters:

    poisson_iters, poisson_omega : Jacobi-with-relaxation iteration count
        and relaxation factor used for the pressure-Poisson solve each step
        (as opposed to the divergence-penalty scheme used for internal flow).
    conv_tol : early-exit threshold on RMS divergence, checked periodically.
    query_points : (M,3), optional. Defaults to a stratified grid over the
        full wind-tunnel domain (`n_vis_grid` resolution) with solid-body
        points removed.

    Returns
    -------
    dict with keys: coords, u, v, w, p, speed, divergence_rms, surf_p,
    surf_speed, grid_axes, wt_bbox, geom_bbox, dt, nu, Re, U_in.
    """
    from scipy.spatial import cKDTree

    wall_points = np.asarray(wall_points, dtype=np.float64)
    x0, y0, z0, x1, y1, z1 = bbox
    Lx, Ly, Lz = x1 - x0, y1 - y0, z1 - z0
    L_char = max(Lx, Ly, Lz)
    nu = U_in * L_char / Re

    pad_lat = pad_lateral * max(Ly, Lz)
    x0_wt, x1_wt = x0 - pad_upstream * Lx, x1 + pad_downstream * Lx
    y0_wt, y1_wt = y0 - pad_lat, y1 + pad_lat
    z0_wt, z1_wt = z0 - pad_lat, z1 + pad_lat
    Lx_wt, Ly_wt, Lz_wt = x1_wt - x0_wt, y1_wt - y0_wt, z1_wt - z0_wt

    dx, dy, dz = Lx_wt / nx, Ly_wt / ny, Lz_wt / nz
    xg = np.linspace(x0_wt + dx / 2, x1_wt - dx / 2, nx)
    yg = np.linspace(y0_wt + dy / 2, y1_wt - dy / 2, ny)
    zg = np.linspace(z0_wt + dz / 2, z1_wt - dz / 2, nz)
    XX, YY, ZZ = np.meshgrid(xg, yg, zg, indexing="ij")
    coords_grid = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    h = min(dx, dy, dz)

    tree = cKDTree(wall_points)
    dist_w, _ = tree.query(coords_grid, k=1)
    wall_mask3d = (dist_w < h * 1.5).reshape(nx, ny, nz)
    inlet_mask3d = XX < x0_wt + dx * 2.0
    outlet_mask3d = XX > x1_wt - dx * 2.0

    # Solid-body mask: a slightly thicker IBM distance band than the pure
    # no-slip wall mask. A convex-hull inside/outside test is deliberately
    # NOT used here (unlike the internal-flow solver): it fails for
    # non-convex bodies (through-holes, concave pockets), incorrectly
    # filling concavities as solid and blocking flow that should pass
    # through them. The distance-based mask handles arbitrary topology.
    solid_mask3d = (dist_w < h * 4.0).reshape(nx, ny, nz)

    # Initialize near the expected steady state with the closed-form
    # potential-flow solution for a sphere of equivalent radius in a
    # uniform stream -- converges in far fewer FDM iterations than a
    # cold (zero) start.
    gx, gy, gz = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1)
    R_eff = 0.45 * max(Lx, Ly, Lz)
    Xr, Yr, Zr = XX - gx, YY - gy, ZZ - gz
    R2 = Xr ** 2 + Yr ** 2 + Zr ** 2
    R2s = np.maximum(R2, (R_eff * 0.05) ** 2)
    Rs3, Rs5 = R2s ** 1.5, R2s ** 2.5
    R3 = R_eff ** 3
    u_p = U_in * (1.0 + R3 / (2.0 * Rs3) - 1.5 * R3 * Xr ** 2 / Rs5)
    v_p = -1.5 * U_in * R3 * Xr * Yr / Rs5
    w_p = -1.5 * U_in * R3 * Xr * Zr / Rs5
    p_p = 0.5 * rho * (U_in ** 2 - (u_p ** 2 + v_p ** 2 + w_p ** 2))
    blend = np.clip(dist_w.reshape(nx, ny, nz) / (h * 2.0), 0.0, 1.0)
    u = np.clip(u_p * blend, -U_in * 2, U_in * 3)
    v = np.clip(v_p * blend, -U_in * 2, U_in * 2)
    w = np.clip(w_p * blend, -U_in * 2, U_in * 2)
    p = p_p * blend
    u[solid_mask3d] = v[solid_mask3d] = w[solid_mask3d] = 0.0
    u[wall_mask3d] = v[wall_mask3d] = w[wall_mask3d] = 0.0
    u[inlet_mask3d] = U_in; v[inlet_mask3d] = 0.0; w[inlet_mask3d] = 0.0

    def laplacian(f):
        return (np.gradient(np.gradient(f, dx, axis=0), dx, axis=0) +
                np.gradient(np.gradient(f, dy, axis=1), dy, axis=1) +
                np.gradient(np.gradient(f, dz, axis=2), dz, axis=2))

    def divergence(u_, v_, w_):
        return (np.gradient(u_, dx, axis=0) + np.gradient(v_, dy, axis=1)
                 + np.gradient(w_, dz, axis=2))

    dt_diff = 0.25 * h ** 2 / (nu + 1e-12)
    dt = min(0.25 * h / (U_in + 1e-12), dt_diff)
    U_clip = 5.0 * max(U_in, 1.0)
    poisson_coeff = 2.0 / dx ** 2 + 2.0 / dy ** 2 + 2.0 / dz ** 2
    p_clip = 5.0 * rho * U_in ** 2 * max(Lx_wt, Ly_wt, Lz_wt)

    div_rms_final = float("nan")
    log_freq = log_every if log_every is not None else max(1, n_iter // 10)

    for it in range(n_iter):
        u_max_cur = max(float(np.max(np.abs(u))), U_in, 1e-12)
        dt = min(0.25 * h / u_max_cur, dt_diff)

        conv_u = u * np.gradient(u, dx, axis=0) + v * np.gradient(u, dy, axis=1) + w * np.gradient(u, dz, axis=2)
        conv_v = u * np.gradient(v, dx, axis=0) + v * np.gradient(v, dy, axis=1) + w * np.gradient(v, dz, axis=2)
        conv_w = u * np.gradient(w, dx, axis=0) + v * np.gradient(w, dy, axis=1) + w * np.gradient(w, dz, axis=2)

        u_star = u + dt * (-conv_u + nu * laplacian(u))
        v_star = v + dt * (-conv_v + nu * laplacian(v))
        w_star = w + dt * (-conv_w + nu * laplacian(w))

        u_star[wall_mask3d] = v_star[wall_mask3d] = w_star[wall_mask3d] = 0.0
        u_star[solid_mask3d] = v_star[solid_mask3d] = w_star[solid_mask3d] = 0.0
        u_star[inlet_mask3d] = U_in; v_star[inlet_mask3d] = 0.0; w_star[inlet_mask3d] = 0.0
        u_star = np.clip(u_star, -U_clip, U_clip)
        v_star = np.clip(v_star, -U_clip, U_clip)
        w_star = np.clip(w_star, -U_clip, U_clip)

        div_star = divergence(u_star, v_star, w_star)
        rhs_p = (rho / (dt + 1e-30)) * div_star

        phi = np.zeros_like(p)
        for _ in range(poisson_iters):
            phi_new = (
                (phi[2:, 1:-1, 1:-1] + phi[:-2, 1:-1, 1:-1]) / dx ** 2 +
                (phi[1:-1, 2:, 1:-1] + phi[1:-1, :-2, 1:-1]) / dy ** 2 +
                (phi[1:-1, 1:-1, 2:] + phi[1:-1, 1:-1, :-2]) / dz ** 2 -
                rhs_p[1:-1, 1:-1, 1:-1]
            ) / poisson_coeff
            phi[1:-1, 1:-1, 1:-1] = (poisson_omega * phi_new
                                      + (1.0 - poisson_omega) * phi[1:-1, 1:-1, 1:-1])
            phi[outlet_mask3d] = 0.0
            phi[solid_mask3d] = 0.0
            phi[inlet_mask3d] = 0.0

        p = np.clip(p + phi, -p_clip, p_clip)
        p[outlet_mask3d] = 0.0
        p[solid_mask3d] = 0.0

        u_new = u_star - (dt / rho) * np.gradient(phi, dx, axis=0)
        v_new = v_star - (dt / rho) * np.gradient(phi, dy, axis=1)
        w_new = w_star - (dt / rho) * np.gradient(phi, dz, axis=2)

        u_new[wall_mask3d] = v_new[wall_mask3d] = w_new[wall_mask3d] = 0.0
        u_new[solid_mask3d] = v_new[solid_mask3d] = w_new[solid_mask3d] = 0.0
        u_new[inlet_mask3d] = U_in; v_new[inlet_mask3d] = 0.0; w_new[inlet_mask3d] = 0.0

        u = np.clip(u_new, -U_clip, U_clip)
        v = np.clip(v_new, -U_clip, U_clip)
        w = np.clip(w_new, -U_clip, U_clip)

        if not np.isfinite(u).all():
            u = np.nan_to_num(u, nan=0.0, posinf=U_clip, neginf=-U_clip)
            v = np.nan_to_num(v, nan=0.0, posinf=U_clip, neginf=-U_clip)
            w = np.nan_to_num(w, nan=0.0, posinf=U_clip, neginf=-U_clip)
            p = np.nan_to_num(p, nan=0.0, posinf=p_clip, neginf=-p_clip)

        if (it + 1) % log_freq == 0 or it == 0:
            div = divergence(u, v, w)
            div_rms_final = float(np.sqrt(np.mean(div ** 2)))
            if verbose:
                print(f"  [ibm_external] iter {it+1:4d}/{n_iter}  div_rms={div_rms_final:.3e}  "
                      f"u_max={float(np.max(np.abs(u))):.3f}")
            if div_rms_final < conv_tol and it > n_iter // 10:
                break

    div_final = divergence(u, v, w)
    div_rms_final = float(np.sqrt(np.mean(div_final ** 2)))
    speed = np.sqrt(u ** 2 + v ** 2 + w ** 2)
    grid_axes = (xg, yg, zg)

    surf_p = interpolate_to_points(grid_axes, p, wall_points)
    surf_speed = interpolate_to_points(grid_axes, speed, wall_points)

    if query_points is None:
        nv_x, nv_y, nv_z = n_vis_grid
        xv = np.linspace(x0_wt + dx / 2, x1_wt - dx / 2, nv_x, dtype=np.float32)
        yv = np.linspace(y0_wt + dy / 2, y1_wt - dy / 2, nv_y, dtype=np.float32)
        zv = np.linspace(z0_wt + dz / 2, z1_wt - dz / 2, nv_z, dtype=np.float32)
        XVV, YVV, ZVV = np.meshgrid(xv, yv, zv, indexing="ij")
        cands = np.column_stack([XVV.ravel(), YVV.ravel(), ZVV.ravel()])
        d_vis, _ = tree.query(cands, k=1)
        query_points = cands[d_vis >= h * 3.0].astype(np.float32)
    query_points = np.asarray(query_points, dtype=np.float32)

    wt_bbox = [float(x0_wt), float(y0_wt), float(z0_wt), float(x1_wt), float(y1_wt), float(z1_wt)]

    return {
        "coords": query_points,
        "u": interpolate_to_points(grid_axes, u, query_points),
        "v": interpolate_to_points(grid_axes, v, query_points),
        "w": interpolate_to_points(grid_axes, w, query_points),
        "p": interpolate_to_points(grid_axes, p, query_points),
        "speed": interpolate_to_points(grid_axes, speed, query_points),
        "surf_coords": wall_points,
        "surf_p": surf_p,
        "surf_speed": surf_speed,
        "divergence_rms": div_rms_final,
        "grid_axes": grid_axes,
        "grid_fields": {"u": u, "v": v, "w": w, "p": p,
                         "wall_mask": wall_mask3d, "solid_mask": solid_mask3d},
        "dt": dt,
        "nu": nu,
        "wt_bbox": wt_bbox,
        "geom_bbox": list(bbox),
        "Re": Re,
        "U_in": U_in,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Dispatcher + registry wrapper
# ═══════════════════════════════════════════════════════════════════════════

def solve_ibm_navier_stokes(
    wall_points: np.ndarray,
    bbox: BBox,
    flow_type: str = "internal",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Dispatch to `solve_ibm_internal_flow` (`flow_type="internal"`) or
    `solve_ibm_external_flow` (`flow_type="external"`). All other keyword
    arguments are forwarded to the selected solver."""
    if flow_type == "external":
        return solve_ibm_external_flow(wall_points, bbox, **kwargs)
    if flow_type == "internal":
        return solve_ibm_internal_flow(wall_points, bbox, **kwargs)
    raise ValueError(f"Unknown flow_type '{flow_type}'. Use 'internal' or 'external'.")


@SolverRegistry.register(
    name="immersed_boundary_fdm",
    family="pde",
    description="IBM (immersed-boundary-method) + FDM incompressible Navier-Stokes on an arbitrary "
                "3D geometry defined only by a wall point cloud -- internal (lid-driven / auto-detected "
                "channel) and external (free-stream around an immersed body) flow topologies, Chorin-style "
                "fractional-step time integration.",
    tags=["fdm", "ibm", "navier-stokes", "incompressible", "cfd", "3d", "projection-method"],
)
class ImmersedBoundaryFDMSolver(SolverBase):
    """Thin `SolverBase`/registry wrapper. The functional API
    (`solve_ibm_internal_flow`, `solve_ibm_external_flow`,
    `solve_ibm_navier_stokes`) is the primary entry point and can be used
    directly without this wrapper."""

    def __init__(self, flow_type: str = "internal", **solver_kwargs: Any):
        super().__init__()
        self.flow_type = flow_type
        self.solver_kwargs = solver_kwargs

    def forward(self, wall_points: np.ndarray, bbox: BBox, **kwargs: Any) -> SolverOutput:
        merged = {**self.solver_kwargs, **kwargs}
        sol = solve_ibm_navier_stokes(wall_points, bbox, flow_type=self.flow_type, **merged)
        uvwp = np.stack([sol["u"], sol["v"], sol["w"], sol["p"]], axis=-1).astype(np.float32)
        return SolverOutput(
            result=torch.from_numpy(uvwp),
            losses={"divergence_rms": torch.tensor(float(sol["divergence_rms"]))},
            extras={k: v for k, v in sol.items() if k not in ("u", "v", "w", "p")},
        )
