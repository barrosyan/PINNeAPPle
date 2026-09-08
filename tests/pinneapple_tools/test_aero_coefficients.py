"""Tests for pinneapple_tools.aero_coefficients.

Analytical case: an idealized zero-thickness flat plate (chord c, span b)
at angle of attack alpha, with uniform-but-different Cp on its top and
bottom faces. Since the net force this produces is, by construction, purely
normal to the plate, rotating it from body axes (axial/normal) into wind
axes (drag/lift) by the very same alpha used to tilt the plate must give
zero drag and CL = delta_Cp exactly -- the pressure-only ("inviscid")
prediction for a flat plate is the textbook d'Alembert's-paradox result
(zero drag), so this is also a physically meaningful sanity check, not just
an algebraic identity.
"""
import math

import numpy as np
import pytest

from pinneapple_tools.aero_coefficients import (
    AeroCoefficients,
    compute_cp,
    force_to_coefficients,
    integrate_surface_forces,
)


# ---------------------------------------------------------------------------
# compute_cp
# ---------------------------------------------------------------------------

def test_compute_cp_basic():
    p = np.array([100.0, 150.0, 50.0])
    p_inf = 100.0
    q_inf = 50.0
    cp = compute_cp(p, p_inf, q_inf)
    np.testing.assert_allclose(cp, [0.0, 1.0, -1.0])


def test_compute_cp_zero_q_inf_raises():
    with pytest.raises(ValueError):
        compute_cp(np.array([1.0]), p_inf=0.0, q_inf=0.0)


# ---------------------------------------------------------------------------
# integrate_surface_forces -- single axis-aligned panel sanity check
# ---------------------------------------------------------------------------

def test_integrate_surface_forces_single_panel():
    # Unit square in the z=0 plane, outward normal +z, uniform pressure p=3.
    # Pressure acts opposite the outward normal -> force = (0, 0, -3).
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    pressure = np.array([3.0, 3.0])

    force, moment = integrate_surface_forces(vertices, faces, normals, pressure)

    np.testing.assert_allclose(force, [0.0, 0.0, -3.0], atol=1e-10)
    # Resultant of a uniform load over the unit square acts at its centroid
    # (0.5, 0.5, 0) -> moment about the origin = (0.5,0.5,0) x (0,0,-3).
    expected_moment = np.cross([0.5, 0.5, 0.0], [0.0, 0.0, -3.0])
    np.testing.assert_allclose(moment, expected_moment, atol=1e-10)


def test_integrate_surface_forces_with_shear():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    pressure = np.array([0.0, 0.0])
    shear = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # uniform tangential traction

    force, _ = integrate_surface_forces(vertices, faces, normals, pressure, face_shear=shear)
    np.testing.assert_allclose(force, [1.0, 0.0, 0.0], atol=1e-10)


def test_integrate_surface_forces_mismatched_shapes_raise():
    vertices = np.zeros((3, 3))
    faces = np.array([[0, 1, 2]])
    with pytest.raises(ValueError):
        integrate_surface_forces(vertices, faces, np.zeros((2, 3)), np.zeros(1))
    with pytest.raises(ValueError):
        integrate_surface_forces(vertices, faces, np.zeros((1, 3)), np.zeros(2))


# ---------------------------------------------------------------------------
# Full pipeline: flat plate at AoA -> CL, CD, CM
# ---------------------------------------------------------------------------

def _flat_plate_mesh(chord: float, span: float, alpha_deg: float):
    """Zero-thickness rectangular plate tilted by alpha_deg about y, with
    duplicate top/bottom triangle sets so each side can carry its own
    (uniform) pressure -- see module docstring."""
    alpha = math.radians(alpha_deg)
    t = np.array([math.cos(alpha), 0.0, math.sin(alpha)])   # chordwise (downstream)
    s = np.array([0.0, 1.0, 0.0])                            # spanwise
    n = np.array([-math.sin(alpha), 0.0, math.cos(alpha)])   # plate normal ("top" side)

    p0 = np.zeros(3)
    p1 = chord * t
    p2 = chord * t + span * s
    p3 = span * s
    vertices = np.stack([p0, p1, p2, p3])

    # Two triangles, duplicated for the top (+n) and bottom (-n) sides.
    tri = np.array([[0, 1, 2], [0, 2, 3]])
    faces = np.concatenate([tri, tri], axis=0)
    normals = np.array([n, n, -n, -n])
    return vertices, faces, normals, n


@pytest.mark.parametrize("alpha_deg", [0.0, 5.0, 10.0, 25.0])
def test_flat_plate_zero_drag_and_known_cl_cm(alpha_deg):
    chord, span = 1.0, 1.0
    q_inf, p_inf = 2.0, 0.0
    cp_top, cp_bot = -0.5, 0.5  # delta_Cp = 1.0

    vertices, faces, normals, _n = _flat_plate_mesh(chord, span, alpha_deg)

    p_top = cp_top * q_inf + p_inf
    p_bot = cp_bot * q_inf + p_inf
    # sanity: compute_cp inverts this exactly
    assert compute_cp(np.array([p_top]), p_inf, q_inf)[0] == pytest.approx(cp_top)
    assert compute_cp(np.array([p_bot]), p_inf, q_inf)[0] == pytest.approx(cp_bot)

    face_pressure = np.array([p_top, p_top, p_bot, p_bot])

    force, moment = integrate_surface_forces(vertices, faces, normals, face_pressure)

    ref_area = chord * span
    ref_length = chord
    coeffs = force_to_coefficients(
        force, moment, q_inf, ref_area, ref_length, alpha_deg,
        drag_axis="x", lift_axis="z",
    )

    assert isinstance(coeffs, AeroCoefficients)
    # Pressure-only (inviscid) prediction for a flat plate: zero drag
    # (d'Alembert's paradox), independent of alpha.
    assert coeffs.CD == pytest.approx(0.0, abs=1e-9)
    # CL equals the top/bottom Cp difference exactly, independent of alpha.
    assert coeffs.CL == pytest.approx(1.0, abs=1e-9)
    # Pitching moment about the leading edge is exactly -0.5 for this
    # uniformly-loaded plate (resultant acts at the geometric centroid,
    # 0.5*chord back from the leading edge), independent of alpha.
    assert coeffs.CM == pytest.approx(-0.5, abs=1e-9)
    # No sideforce: everything lives in the x-z plane.
    assert coeffs.CY == pytest.approx(0.0, abs=1e-9)


def test_force_to_coefficients_input_validation():
    force = np.array([1.0, 0.0, 1.0])
    moment = np.zeros(3)
    with pytest.raises(ValueError):
        force_to_coefficients(force, moment, q_inf=0.0, ref_area=1.0, ref_length=1.0, alpha_deg=0.0)
    with pytest.raises(ValueError):
        force_to_coefficients(force, moment, q_inf=1.0, ref_area=0.0, ref_length=1.0, alpha_deg=0.0)
    with pytest.raises(ValueError):
        force_to_coefficients(force, moment, q_inf=1.0, ref_area=1.0, ref_length=0.0, alpha_deg=0.0)
    with pytest.raises(ValueError):
        force_to_coefficients(
            force, moment, q_inf=1.0, ref_area=1.0, ref_length=1.0, alpha_deg=0.0,
            drag_axis="x", lift_axis="x",
        )
