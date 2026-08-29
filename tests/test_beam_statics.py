"""Validates pinneapple_systems.process_components.beam_statics against
independent textbook closed-form identities, boundary conditions, and
internal consistency checks (partial-load formulas reducing to full-load
formulas at their limiting case, continuity at piecewise cutoffs,
load-symmetry)."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_systems.process_components.beam_statics import (
    rectangular_section_properties,
    solve_beam,
    von_mises_stress_rectangular_section,
)

L, E = 10.0, 2.1e11
I, Q = rectangular_section_properties(width_m=0.1, height_m=0.2)
q0 = 500.0


def test_cantilever_uniform_tip_deflection_matches_textbook_qL4_over_8EI():
    x = np.array([L])
    r = solve_beam("cantilever_uniform", x, L, E, I, q0)
    assert r.deflection[0] == pytest.approx(q0 * L ** 4 / (8 * E * I), rel=1e-10)


def test_simply_supported_uniform_midspan_deflection_matches_textbook_5qL4_over_384EI():
    x = np.array([L / 2])
    r = solve_beam("simply_supported_uniform", x, L, E, I, q0)
    assert r.deflection[0] == pytest.approx(5 * q0 * L ** 4 / (384 * E * I), rel=1e-10)


def test_cantilever_boundary_conditions_zero_deflection_and_slope_at_fixed_end():
    x = np.linspace(0, L, 50)
    r = solve_beam("cantilever_uniform", x, L, E, I, q0)
    assert r.deflection[0] == pytest.approx(0.0, abs=1e-9)
    assert r.slope[0] == pytest.approx(0.0, abs=1e-9)


def test_simply_supported_boundary_conditions_zero_deflection_at_both_supports():
    x = np.linspace(0, L, 50)
    r = solve_beam("simply_supported_uniform", x, L, E, I, q0)
    assert r.deflection[0] == pytest.approx(0.0, abs=1e-9)
    assert r.deflection[-1] == pytest.approx(0.0, abs=1e-6)


def test_cantilever_partial_load_reduces_to_full_load_when_extent_equals_length():
    x = np.linspace(0, L, 25)
    full = solve_beam("cantilever_uniform", x, L, E, I, q0)
    partial_full_extent = solve_beam("cantilever_partial_uniform", x, L, E, I, q0, load_extent=L)
    assert partial_full_extent.deflection == pytest.approx(full.deflection, rel=1e-9)
    assert partial_full_extent.bending_moment == pytest.approx(full.bending_moment, rel=1e-9)


def test_simply_supported_partial_load_reduces_to_full_load_when_extent_equals_length():
    x = np.linspace(0, L, 25)
    full = solve_beam("simply_supported_uniform", x, L, E, I, q0)
    partial_full_extent = solve_beam("simply_supported_partial_uniform", x, L, E, I, q0, load_extent=L)
    assert partial_full_extent.deflection == pytest.approx(full.deflection, rel=1e-9)


def test_partial_uniform_load_case_requires_load_extent():
    x = np.linspace(0, L, 10)
    with pytest.raises(ValueError):
        solve_beam("cantilever_partial_uniform", x, L, E, I, q0, load_extent=None)


def test_cantilever_partial_uniform_load_deflection_continuous_at_cutoff():
    a = 4.0
    eps = 1e-4
    x = np.array([a - eps, a + eps])
    r = solve_beam("cantilever_partial_uniform", x, L, E, I, q0, load_extent=a)
    assert r.deflection[0] == pytest.approx(r.deflection[1], abs=1e-6)


def test_triangular_midspan_load_deflection_is_symmetric_about_midspan():
    x = np.linspace(0, L, 41)
    r = solve_beam("simply_supported_triangular_midspan_max", x, L, E, I, q0)
    assert r.deflection == pytest.approx(r.deflection[::-1], rel=1e-6)


def test_von_mises_stress_matches_pure_bending_limit_when_shear_is_zero():
    M = np.array([1000.0])
    V = np.array([0.0])
    sigma = von_mises_stress_rectangular_section(M, V, I, Q, width_m=0.1, height_m=0.2)
    sigma_bending = M * (0.2 / 2) / I
    assert sigma[0] == pytest.approx(abs(sigma_bending[0]), rel=1e-9)


def test_unknown_load_case_raises():
    x = np.linspace(0, L, 5)
    with pytest.raises(ValueError):
        solve_beam("not_a_real_case", x, L, E, I, q0)  # type: ignore[arg-type]
