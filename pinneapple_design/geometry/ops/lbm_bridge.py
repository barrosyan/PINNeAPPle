"""Bridge from a surface mesh (STL / trimesh) to a boolean LBM obstacle mask.

Composes three pieces of existing geometry infrastructure that previously had
no glue between them:

  * mesh loading -- ``TrimeshBridge`` / ``pinneapple_design.geometry.io.stl``
    (returns :class:`~pinneapple_design.geometry.core.mesh.MeshData`, itself
    a thin numpy wrapper the caller can build a raw ``trimesh.Trimesh`` from)
  * point-in-mesh / voxel-grid machinery --
    :mod:`pinneapple_design.geometry.ops.voxelize` (``VoxelGrid``,
    ``voxelize_sdf``)
  * the angle-of-attack rotation convention already used for the 2D NACA
    obstacle path -- ``airfoil_naca_mask`` in
    ``pinneapple_simulation/numerical_solvers/lbm.py``

into "load an STL -> get a boolean 3D occupancy grid I can hand to an LBM
solver as its solid mask", with the same AoA sign meaning a caller would
already know from the 2D NACA path.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple, Union

import numpy as np
import torch

from pinneapple_design.geometry.core.mesh import MeshData
from pinneapple_design.geometry.ops.voxelize import voxelize_sdf

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


# ---------------------------------------------------------------------------
# AoA rotation -- mirrors airfoil_naca_mask's convention
# ---------------------------------------------------------------------------

def _aoa_rotation_matrix(aoa_deg: float, rotation_axis: str) -> np.ndarray:
    """3x3 rotation matrix for angle-of-attack ``aoa_deg`` about ``rotation_axis``.

    Mirrors, exactly, the in-plane rotation used by ``airfoil_naca_mask()``
    in ``pinneapple_simulation/numerical_solvers/lbm.py``::

        alpha = math.radians(aoa_deg)
        xr =  dX*cos(alpha) + dY*sin(alpha)
        yr = -dX*sin(alpha) + dY*cos(alpha)

    That function only ever rotates a 2D (x, y) grid, i.e. implicitly "about
    z". This generalizes the same formula to 3D by picking the two
    non-``rotation_axis`` axes, in ``("x", "y", "z")`` order, as the
    "primary" (playing the role of the 2D code's ``X``) and "secondary"
    (playing the role of the 2D code's ``Y``) axes of the rotation plane,
    and applying the identical ``(primary, secondary) -> (xr, yr)`` formula
    above. Concretely:

      * ``rotation_axis="z"`` reproduces ``airfoil_naca_mask`` verbatim
        (primary=x, secondary=y) -- a 2D-mesh-in-3D user gets bit-identical
        sign behaviour.
      * ``rotation_axis="y"`` (the default here) rotates in the x-z plane
        (primary=x=streamwise, secondary=z=vertical/"up"), i.e. the standard
        aircraft pitch convention (rotation about the spanwise axis) for a
        3D wing/body sitting with its span along y -- the natural 3D
        generalization of the 2D case's (streamwise, vertical) pair.
      * ``rotation_axis="x"`` rotates in the y-z plane.

    Returns
    -------
    (3, 3) numpy rotation matrix ``R`` such that ``R @ v`` applies the
    rotation to a displacement vector ``v`` (e.g. a vertex position already
    relative to the rotation center).
    """
    if rotation_axis not in _AXIS_INDEX:
        raise ValueError(
            f"rotation_axis must be one of 'x', 'y', 'z'; got {rotation_axis!r}"
        )
    alpha = math.radians(aoa_deg)
    c, s = math.cos(alpha), math.sin(alpha)
    primary, secondary = [a for a in ("x", "y", "z") if a != rotation_axis]
    ip, isec = _AXIS_INDEX[primary], _AXIS_INDEX[secondary]

    R = np.eye(3, dtype=np.float64)
    R[ip, ip] = c
    R[ip, isec] = s
    R[isec, ip] = -s
    R[isec, isec] = c
    return R


# ---------------------------------------------------------------------------
# Mesh adapter
# ---------------------------------------------------------------------------

def _to_trimesh(mesh):
    """Coerce a MeshData / trimesh.Trimesh / mesh-like object to a
    ``trimesh.Trimesh`` (mirrors the conversion already done in
    ``MeshCollocator._get_trimesh``)."""
    import trimesh  # local import: trimesh is an optional geometry extra

    if isinstance(mesh, trimesh.Trimesh):
        return mesh.copy()
    if isinstance(mesh, MeshData):
        return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        return trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=False
        )
    raise TypeError(
        "mesh_to_obstacle_mask expects a trimesh.Trimesh, a "
        "pinneapple_design.geometry.core.mesh.MeshData (as returned by "
        "TrimeshBridge.load() / pinneapple_design.geometry.io.stl.load_stl()), "
        f"or an object exposing '.vertices'/'.faces' arrays; got {type(mesh).__name__}"
    )


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def mesh_to_obstacle_mask(
    mesh,
    bounds: Dict[str, Tuple[float, float]],
    resolution: Union[int, Tuple[int, int, int]],
    *,
    aoa_deg: float = 0.0,
    rotation_axis: str = "y",
    oracle_supersample: int = 2,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Voxelize a surface mesh into a boolean 3D LBM obstacle/solid mask.

    Parameters
    ----------
    mesh : trimesh.Trimesh | MeshData | mesh-like
        Surface mesh, e.g. from ``TrimeshBridge().load(path)`` or
        ``pinneapple_design.geometry.io.stl.load_stl(path)`` (both return
        :class:`MeshData`), or a raw ``trimesh.Trimesh``.
    bounds : {"x": (x0, x1), "y": (y0, y1), "z": (z0, z1)}
        World-space extent of the LBM domain to voxelize over (same format
        as :func:`pinneapple_design.geometry.ops.voxelize.voxelize_sdf`).
    resolution : int | (nx, ny, nz)
        Output grid resolution (isotropic if a single int).
    aoa_deg : float
        Angle of attack in degrees. The mesh is rotated by this amount about
        its own centroid, about ``rotation_axis``, before voxelizing -- see
        :func:`_aoa_rotation_matrix` for the exact sign/axis convention
        (mirrors ``airfoil_naca_mask`` in
        ``pinneapple_simulation/numerical_solvers/lbm.py``).
    rotation_axis : {"x", "y", "z"}
        Axis to rotate about. Default "y" (pitch about the spanwise axis,
        i.e. rotate in the streamwise/x - vertical/z plane) -- the natural
        3D generalization of the 2D NACA path's (x, y) rotation.
    oracle_supersample : int
        The internal point-in-mesh oracle (see implementation note below)
        is built at a pitch finer than the output grid's own voxel size by
        this factor, to reduce aliasing at the mesh surface. Default 2.
    device : torch.device
        Device for the returned/intermediate grid computation.

    Returns
    -------
    np.ndarray of shape ``resolution`` (or the (nx,ny,nz) tuple), dtype
    bool. ``True`` = solid/occupied (inside the mesh) -- directly usable as
    an LBM ``obstacle_mask`` (matching the ``obstacle_mask: bool tensor``
    convention documented on ``LBMSolver``/``LBMSolver3D`` in
    ``pinneapple_simulation/numerical_solvers/lbm.py``, modulo needing a
    ``torch.from_numpy(...)`` there since this module stays numpy-only).

    Implementation note -- why ``voxelize_sdf`` and not ``voxelize_pointcloud``
    -----------------------------------------------------------------------
    Both were viable per the audit: ``voxelize_pointcloud`` could rasterize
    points sampled on the mesh *surface*, and ``voxelize_sdf`` evaluates an
    arbitrary inside/outside predicate on a dense grid of cell-center query
    points. ``voxelize_sdf`` was chosen because it maps directly onto "solid
    occupancy volume" semantics: a surface point cloud only rasterizes a
    thin shell (it says nothing about the interior) and would need its own
    separate flood-fill step to become a solid mask, duplicating exactly
    what ``voxelize_sdf`` already gets for free from evaluating a
    containment predicate at every grid cell.

    That still leaves "what containment predicate": the obvious choice,
    ``trimesh.Trimesh.contains()`` (ray casting), requires an rtree/pyembree
    broad-phase index that is NOT guaranteed to be installed -- confirmed by
    actually calling it against a real trimesh 5.1.0 install in this repo's
    own ``.venv``, which raises ``ModuleNotFoundError: No module named
    'rtree'``, not by reading trimesh's docs. Instead this uses
    ``trimesh.Trimesh.voxelized(pitch).fill()`` (the same
    voxel-occupancy approach already used elsewhere in this repo, see
    ``stl_domain_batch_builder._inside_voxel_occupancy``), which rasterizes
    the surface and flood-fills the interior via scipy (already a hard
    dependency elsewhere in this package, e.g. ``MeshCollocator``'s
    ``ConvexHull`` path) without touching the ray/rtree path at all. That
    gives a plain boolean ``is_filled(points)`` oracle, which is then wired
    into ``voxelize_sdf`` as a signed-distance-like function (``-1`` inside,
    ``+1`` outside, thresholded at 0) so the final grid still goes through
    this module's shared ``VoxelGrid`` machinery instead of reimplementing
    grid math here.
    """
    tm = _to_trimesh(mesh)

    if aoa_deg != 0.0:
        R = _aoa_rotation_matrix(aoa_deg, rotation_axis)
        center = np.asarray(tm.centroid, dtype=np.float64).copy()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = center - R @ center  # rotate about centroid, not the origin
        tm.apply_transform(T)

    axes = [k for k in ("x", "y", "z") if k in bounds]
    if len(axes) != 3:
        raise ValueError(
            "mesh_to_obstacle_mask requires 3D bounds with 'x', 'y' and 'z' "
            f"keys; got {sorted(bounds.keys())}"
        )

    if isinstance(resolution, int):
        res = (resolution, resolution, resolution)
    else:
        res = tuple(int(r) for r in resolution)
        if len(res) != 3:
            raise ValueError(f"resolution must have 3 entries for a 3D mask; got {res}")

    lo = np.array([bounds[a][0] for a in axes], dtype=np.float64)
    hi = np.array([bounds[a][1] for a in axes], dtype=np.float64)
    cell = (hi - lo) / np.array(res, dtype=np.float64)
    pitch = float(np.min(cell)) / max(1, int(oracle_supersample))
    pitch = max(pitch, 1e-9)

    try:
        surf_vox = tm.voxelized(pitch=pitch)
        solid_vox = surf_vox.fill()
    except Exception as exc:  # pragma: no cover -- depends on native/optional libs
        raise RuntimeError(
            "mesh_to_obstacle_mask: trimesh voxelization/fill failed; the "
            "mesh may not be closed/watertight enough for a solid interior "
            f"to be well-defined (original error: {exc})"
        ) from exc

    def _inside_sdf(pts: torch.Tensor) -> torch.Tensor:
        pts_np = pts.detach().cpu().numpy().astype(np.float64)
        inside = np.asarray(solid_vox.is_filled(pts_np), dtype=bool)
        vals = np.where(inside, -1.0, 1.0).astype(np.float32)
        return torch.from_numpy(vals).to(pts.device)

    grid = voxelize_sdf(_inside_sdf, bounds, res, threshold=0.0, device=device)
    return grid.data.detach().cpu().numpy().astype(bool)


__all__ = ["mesh_to_obstacle_mask"]
