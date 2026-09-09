"""Real geometric tests for direct OCC feature operations (fillet/chamfer/shell).

These build a real cadquery solid (a unit box, matching the primitive used
throughout ``pinneapple_design/geometry/gen/cadquery_gen.py``), apply the
wrappers in ``pinneapple_design/geometry/ops/occ_features.py``, and check
the resulting solid via the real OCP/OCC kernel's own ``Volume()`` /
``BoundingBox()`` queries -- not guessed numbers.
"""
from __future__ import annotations

import math

import pytest

from pinneapple_design.geometry.gen.cadquery_gen import cadquery_available


pytestmark = pytest.mark.skipif(
    not cadquery_available(),
    reason="cadquery (OCP/OpenCASCADE backend) is not installed",
)


def _box():
    import cadquery as cq

    return cq.Workplane("XY").box(1.0, 1.0, 1.0, centered=True)


def test_fillet_edges_reduces_volume_below_box():
    from pinneapple_design.geometry.ops.occ_features import fillet_edges

    box = _box()
    v0 = box.val().Volume()
    assert math.isclose(v0, 1.0, rel_tol=1e-6)

    filleted = fillet_edges(box, 0.1)
    v1 = filleted.val().Volume()

    # A constant-radius edge fillet on all 12 edges of a unit cube rounds
    # off material at every edge, so the resulting solid must have strictly
    # less volume than the original box, but still be a substantial solid
    # (not degenerate).
    assert v1 < v0
    assert v1 > 0.9 * v0
    # Known-good real OCP result observed for this exact call (r=0.1 on a
    # unit box): volume ~= 0.97558701...
    assert math.isclose(v1, 0.9755870138909416, rel_tol=1e-6)


def test_fillet_edges_with_selector_removes_less_material():
    from pinneapple_design.geometry.ops.occ_features import fillet_edges

    box = _box()
    v0 = box.val().Volume()

    all_edges = fillet_edges(box, 0.1).val().Volume()
    only_vertical = fillet_edges(_box(), 0.1, edge_selector="|Z").val().Volume()

    # Filleting only the 4 vertical edges removes strictly less material
    # than filleting all 12 edges of the box.
    assert only_vertical < v0
    assert only_vertical > all_edges


def test_chamfer_edges_reduces_volume_below_box():
    from pinneapple_design.geometry.ops.occ_features import chamfer_edges

    box = _box()
    v0 = box.val().Volume()

    chamfered = chamfer_edges(box, 0.1)
    v2 = chamfered.val().Volume()

    assert v2 < v0
    assert v2 > 0.9 * v0
    # A symmetric 45-degree chamfer of length L cuts a right-triangular
    # prism of cross-section area L^2/2 off each of the 12 edges (to
    # leading order, ignoring corner double-counting); known-good real OCP
    # result for L=0.1 on a unit box: volume ~= 0.94533333...
    assert math.isclose(v2, 0.9453333333333336, rel_tol=1e-6)


def test_fillet_and_chamfer_reject_non_positive_size():
    from pinneapple_design.geometry.ops.occ_features import chamfer_edges, fillet_edges

    box = _box()
    with pytest.raises(ValueError):
        fillet_edges(box, 0.0)
    with pytest.raises(ValueError):
        chamfer_edges(box, -0.1)


def test_shell_solid_creates_internal_cavity():
    from pinneapple_design.geometry.ops.occ_features import shell_solid

    box = _box()
    v0 = box.val().Volume()
    bb0 = box.val().BoundingBox()

    shelled = shell_solid(box, 0.1)
    shell = shelled.val()
    v_shell = shell.Volume()
    bb1 = shell.BoundingBox()

    # No faces were removed, so hollow() takes the "fully enclosed" branch:
    # the solid is offset outward by the wall thickness (bounding box grows
    # by 2*thickness per axis) but the resulting *material* volume is only
    # the thin wall shell, i.e. strictly less than the volume of its own
    # outer bounding box -- proof of an interior cavity.
    outer_extent = 1.0 + 2 * 0.1
    assert math.isclose(bb1.xlen, outer_extent, rel_tol=1e-6)
    assert math.isclose(bb1.ylen, outer_extent, rel_tol=1e-6)
    assert math.isclose(bb1.zlen, outer_extent, rel_tol=1e-6)
    assert v_shell < bb1.xlen * bb1.ylen * bb1.zlen
    # Known-good real OCP result for thickness=0.1 outward shell of a unit
    # box with no faces removed: volume ~= 0.69843657...
    assert math.isclose(v_shell, 0.698436573732035, rel_tol=1e-6)
    assert v_shell != v0
    assert bb1.xlen > bb0.xlen


def test_shell_solid_with_open_face_removes_material_and_stays_open():
    from pinneapple_design.geometry.ops.occ_features import shell_solid

    box = _box()
    v0 = box.val().Volume()

    opened = shell_solid(box, 0.1, faces_to_remove=">Z")
    v_open = opened.val().Volume()

    # Removing a face before shelling produces an open-topped hollow box:
    # a distinct (smaller) material volume than the closed-cavity shell.
    closed_shell_volume = shell_solid(_box(), 0.1).val().Volume()
    assert v_open < v0
    assert v_open != closed_shell_volume


def test_shell_solid_rejects_zero_thickness():
    from pinneapple_design.geometry.ops.occ_features import shell_solid

    box = _box()
    with pytest.raises(ValueError):
        shell_solid(box, 0.0)


def test_accepts_raw_shape_not_just_workplane():
    from pinneapple_design.geometry.ops.occ_features import fillet_edges

    box_wp = _box()
    raw_solid = box_wp.val()  # a cadquery Shape/Solid, not a Workplane

    filleted = fillet_edges(raw_solid, 0.1)
    # Result is still a Workplane wrapping the filleted solid.
    import cadquery as cq

    assert isinstance(filleted, cq.Workplane)
    assert math.isclose(filleted.val().Volume(), 0.9755870138909416, rel_tol=1e-6)


def test_rejects_unsupported_input_type():
    from pinneapple_design.geometry.ops.occ_features import fillet_edges

    with pytest.raises(TypeError):
        fillet_edges("not-a-solid", 0.1)
