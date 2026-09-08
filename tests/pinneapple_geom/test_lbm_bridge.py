"""Tests for pinneapple_design.geometry.ops.lbm_bridge.mesh_to_obstacle_mask."""
import math

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from pinneapple_design.geometry.ops.lbm_bridge import (
    _aoa_rotation_matrix,
    mesh_to_obstacle_mask,
)


# ---------------------------------------------------------------------------
# AoA rotation convention
# ---------------------------------------------------------------------------

def test_aoa_rotation_matches_airfoil_naca_mask_2d_convention():
    """rotation_axis='z' must reproduce airfoil_naca_mask's exact per-point
    formula from pinneapple_simulation/numerical_solvers/lbm.py::

        xr =  dX*cos(alpha) + dY*sin(alpha)
        yr = -dX*sin(alpha) + dY*cos(alpha)
    """
    aoa_deg = 15.0
    R = _aoa_rotation_matrix(aoa_deg, rotation_axis="z")

    dX, dY = 3.0, -2.0
    rotated = R @ np.array([dX, dY, 0.0])

    alpha = math.radians(aoa_deg)
    xr_expected = dX * math.cos(alpha) + dY * math.sin(alpha)
    yr_expected = -dX * math.sin(alpha) + dY * math.cos(alpha)

    assert rotated[0] == pytest.approx(xr_expected)
    assert rotated[1] == pytest.approx(yr_expected)
    assert rotated[2] == pytest.approx(0.0, abs=1e-12)


def test_aoa_rotation_invalid_axis_raises():
    with pytest.raises(ValueError):
        _aoa_rotation_matrix(10.0, rotation_axis="w")


# ---------------------------------------------------------------------------
# mesh_to_obstacle_mask -- box primitive, no rotation
# ---------------------------------------------------------------------------

def test_mesh_to_obstacle_mask_box_no_rotation():
    box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))  # centered at origin
    bounds = {"x": (-3.0, 3.0), "y": (-3.0, 3.0), "z": (-3.0, 3.0)}
    resolution = 30

    mask = mesh_to_obstacle_mask(box, bounds, resolution, aoa_deg=0.0)

    assert mask.shape == (30, 30, 30)
    assert mask.dtype == bool

    domain_vol = 6.0 ** 3
    cell_vol = domain_vol / (30 ** 3)
    box_vol = 2.0 ** 3
    expected_count = box_vol / cell_vol
    # Voxelization tolerance: allow +/-25% around the analytic count.
    assert mask.sum() == pytest.approx(expected_count, rel=0.25)

    # Occupied-voxel centroid should sit near the box's own centroid (origin).
    idx = np.argwhere(mask)
    lo = np.array([-3.0, -3.0, -3.0])
    cell = np.array([6.0 / 30, 6.0 / 30, 6.0 / 30])
    world = lo + (idx + 0.5) * cell
    centroid = world.mean(axis=0)
    np.testing.assert_allclose(centroid, [0.0, 0.0, 0.0], atol=0.3)


def test_mesh_to_obstacle_mask_box_rotation_swaps_extent():
    # Elongated along x; rotating 90deg about y should make it elongated
    # along z instead (new_x = old_z, new_z = -old_x for this convention).
    box = trimesh.creation.box(extents=(4.0, 1.0, 1.0))
    bounds = {"x": (-3.0, 3.0), "y": (-3.0, 3.0), "z": (-3.0, 3.0)}
    resolution = 30

    mask0 = mesh_to_obstacle_mask(box, bounds, resolution, aoa_deg=0.0, rotation_axis="y")
    mask90 = mesh_to_obstacle_mask(box, bounds, resolution, aoa_deg=90.0, rotation_axis="y")

    def extent(mask, axis):
        idx = np.argwhere(mask)
        return idx[:, axis].max() - idx[:, axis].min()

    # Unrotated: long along x (axis 0), short along z (axis 2).
    assert extent(mask0, 0) > extent(mask0, 2)
    # Rotated 90deg about y: long along z, short along x.
    assert extent(mask90, 2) > extent(mask90, 0)


def test_mesh_to_obstacle_mask_sphere_shape_and_count():
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    bounds = {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "z": (-2.0, 2.0)}
    resolution = (32, 32, 32)

    mask = mesh_to_obstacle_mask(sphere, bounds, resolution)

    assert mask.shape == (32, 32, 32)
    domain_vol = 4.0 ** 3
    cell_vol = domain_vol / (32 ** 3)
    sphere_vol = 4.0 / 3.0 * math.pi * 1.0 ** 3
    expected_count = sphere_vol / cell_vol
    assert mask.sum() == pytest.approx(expected_count, rel=0.3)


def test_mesh_to_obstacle_mask_rejects_non_3d_bounds():
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        mesh_to_obstacle_mask(box, {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}, 16)


def test_mesh_to_obstacle_mask_accepts_meshdata():
    from pinneapple_design.geometry.io.trimesh_bridge import TrimeshBridge

    box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    mesh_data = TrimeshBridge().from_trimesh(box)

    bounds = {"x": (-3.0, 3.0), "y": (-3.0, 3.0), "z": (-3.0, 3.0)}
    mask = mesh_to_obstacle_mask(mesh_data, bounds, 24)
    assert mask.dtype == bool
    assert mask.sum() > 0
