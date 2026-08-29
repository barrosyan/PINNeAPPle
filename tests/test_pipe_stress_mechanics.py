"""Validates pinneapple_systems.process_components.pipe_stress_mechanics
against independent textbook identities: the thin-wall pressure-vessel
limit of the Lame equations, the Von Mises hydrostatic-stress invariant,
its reduction to the biaxial form, and the Euler/Timoshenko column-
buckling closed forms."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_systems.process_components.pipe_stress_mechanics import (
    beam_column_moment_amplification_factor,
    classify_buckling_mode,
    constrained_rod_buckling_load,
    euler_critical_buckling_load,
    lame_hoop_stress_inner,
    lame_hoop_stress_outer,
    rotating_bending_stress_cycle,
    torsional_shear_stress,
    von_mises_triaxial,
)


def test_lame_hoop_stress_matches_thin_wall_pressure_vessel_formula_at_small_t_over_r():
    p, r_mean, t = 1.0e6, 1.0, 0.001  # t/r = 1e-3, thin-wall regime
    ro, ri = r_mean + t / 2, r_mean - t / 2
    sigma_thin_wall = p * r_mean / t
    sigma_outer = lame_hoop_stress_outer(p, ro, ri)
    sigma_inner = lame_hoop_stress_inner(p, ro, ri)
    assert sigma_outer == pytest.approx(sigma_thin_wall, rel=2e-3)
    assert sigma_inner == pytest.approx(sigma_thin_wall, rel=2e-3)


def test_lame_hoop_inner_exceeds_outer_for_a_thick_wall_under_internal_pressure():
    p, ro, ri = 2.0e7, 0.1, 0.06
    assert lame_hoop_stress_inner(p, ro, ri) > lame_hoop_stress_outer(p, ro, ri)


def test_torsional_shear_matches_T_c_over_J_identity():
    T, ro, J = 5000.0, 0.05, 3.0e-6
    assert torsional_shear_stress(T, ro, J) == pytest.approx(T * ro / J, rel=1e-12)


def test_von_mises_triaxial_is_zero_under_a_purely_hydrostatic_stress_state():
    """Classic invariant: equal principal stresses in all three directions
    (no shear) carry no deviatoric/Von-Mises stress at all."""
    p = 5.0e7
    sigma_vm = von_mises_triaxial(sigma_axial=p, sigma_hoop=p, sigma_radial=p, tau=0.0)
    assert sigma_vm == pytest.approx(0.0, abs=1e-6)


def test_von_mises_triaxial_reduces_to_biaxial_formula_when_radial_is_zero():
    sigma_L, sigma_H, tau = 3.0e7, -1.0e7, 5.0e6
    vm_triaxial = von_mises_triaxial(sigma_L, sigma_H, sigma_radial=0.0, tau=tau)
    vm_biaxial = np.sqrt(sigma_L ** 2 + sigma_H ** 2 - sigma_L * sigma_H + 3.0 * tau ** 2)
    assert vm_triaxial == pytest.approx(vm_biaxial, rel=1e-10)


def test_von_mises_pure_uniaxial_tension_equals_the_applied_stress():
    sigma = 2.5e8
    assert von_mises_triaxial(sigma, 0.0, 0.0, 0.0) == pytest.approx(sigma, rel=1e-10)


def test_euler_buckling_matches_textbook_pinned_pinned_closed_form():
    E, I, L = 200e9, 8.0e-6, 5.0
    P_cr = euler_critical_buckling_load(E, I, L, end_condition_factor=1.0)
    assert P_cr == pytest.approx(np.pi ** 2 * E * I / L ** 2, rel=1e-10)


def test_euler_buckling_fixed_free_cantilever_is_four_times_weaker_than_pinned_pinned():
    E, I, L = 200e9, 8.0e-6, 5.0
    P_pinned = euler_critical_buckling_load(E, I, L, end_condition_factor=1.0)
    P_cantilever = euler_critical_buckling_load(E, I, L, end_condition_factor=2.0)
    assert P_pinned == pytest.approx(4.0 * P_cantilever, rel=1e-9)


def test_constrained_rod_helical_critical_load_is_exactly_double_sinusoidal():
    result = constrained_rod_buckling_load(E=200e9, I=5e-6, lateral_load_per_length=50.0, radial_clearance=0.02)
    assert result.F_critical_helical == pytest.approx(2.0 * result.F_critical_sinusoidal, rel=1e-12)
    assert result.mode == "applicable"


def test_constrained_rod_buckling_not_applicable_without_lateral_load():
    result = constrained_rod_buckling_load(E=200e9, I=5e-6, lateral_load_per_length=0.0, radial_clearance=0.02)
    assert result.mode == "not_applicable"
    assert result.F_critical_sinusoidal == float("inf")


def test_classify_buckling_mode_transitions_straight_sinusoidal_helical():
    result = constrained_rod_buckling_load(E=200e9, I=5e-6, lateral_load_per_length=50.0, radial_clearance=0.02)
    F = np.array([0.5 * result.F_critical_sinusoidal, 1.5 * result.F_critical_sinusoidal, 1.5 * result.F_critical_helical])
    modes = classify_buckling_mode(F, result)
    assert list(modes) == [0, 1, 2]


def test_beam_column_amplification_factor_is_one_at_zero_axial_load():
    E, I, L = 200e9, 8e-6, 5.0
    AF = beam_column_moment_amplification_factor(compressive_force=0.0, E=E, I=I, length=L)
    assert AF == pytest.approx(1.0, rel=1e-9)


def test_beam_column_amplification_factor_matches_1_over_1_minus_ratio_below_the_clip():
    E, I, L = 200e9, 8e-6, 5.0
    P_cr = euler_critical_buckling_load(E, I, L)
    P = 0.5 * P_cr
    AF = beam_column_moment_amplification_factor(compressive_force=P, E=E, I=I, length=L)
    assert AF == pytest.approx(1.0 / (1.0 - 0.5), rel=1e-9)


def test_beam_column_amplification_factor_stays_finite_at_the_critical_load():
    E, I, L = 200e9, 8e-6, 5.0
    P_cr = euler_critical_buckling_load(E, I, L)
    AF = beam_column_moment_amplification_factor(compressive_force=P_cr, E=E, I=I, length=L)
    assert np.isfinite(AF)
    assert AF > 10.0


def test_rotating_bending_cycle_alt_and_mean_match_pure_bending_case():
    """With no hoop/torsion, sigma_total(theta) = sigma_axial +
    sigma_bend*cos(theta) is exactly the applied stress, so the Von
    Mises envelope over the cycle should recover sigma_alt=sigma_bend,
    sigma_mean=sigma_axial directly."""
    sigma_axial, sigma_bend = 5.0e7, 2.0e7
    result = rotating_bending_stress_cycle(sigma_axial, sigma_bend, sigma_hoop=0.0, tau=0.0)
    assert result.sigma_alternating == pytest.approx(sigma_bend, rel=1e-3)
    assert result.sigma_mean == pytest.approx(sigma_axial, rel=1e-3)


def test_rotating_bending_cycle_with_zero_bending_amplitude_has_zero_alternating_stress():
    result = rotating_bending_stress_cycle(sigma_axial=1.0e8, sigma_bending_amplitude=0.0)
    assert result.sigma_alternating == pytest.approx(0.0, abs=1e-6)
