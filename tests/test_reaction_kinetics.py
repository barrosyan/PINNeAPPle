"""Validates pinneapple_systems.process_components.reaction_kinetics
against independent analytical checks (equilibrium ratios, mass
conservation) using a small GENERIC example network (A &lt;-&gt; B), not any
specific real chemistry -- this module is domain-agnostic and should be
validated as such."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_systems.process_components.reaction_kinetics import (
    AdvectionDispersionReactionSolver,
    Reaction,
    ReactionNetwork,
    acid_fraction,
    arrhenius_rate_constant,
    base_fraction,
    diprotic_fractions,
    integrate_network,
    linear_combination_rate_constant,
    mass_action_rate,
    quadratic_in_T,
)


def _reversible_A_B_network(k1: float, k2: float) -> ReactionNetwork:
    forward = Reaction("A->B", mass_action_rate(k1, {"A": 1}), {"A": -1.0, "B": 1.0})
    backward = Reaction("B->A", mass_action_rate(k2, {"B": 1}), {"A": 1.0, "B": -1.0})
    return ReactionNetwork(species=("A", "B"), reactions=(forward, backward))


def test_reversible_reaction_reaches_the_correct_equilibrium_ratio():
    k1, k2 = 2.0, 0.5
    network = _reversible_A_B_network(k1, k2)
    C0 = np.array([1.0, 0.0])
    t_eval = np.linspace(0, 50, 200)
    result = integrate_network(network, C0, t_eval)
    A_final, B_final = result.C[0, -1], result.C[1, -1]
    # At equilibrium, k1*[A] = k2*[B] -> [B]/[A] = k1/k2
    assert B_final / A_final == pytest.approx(k1 / k2, rel=1e-3)


def test_reaction_network_conserves_total_mass_for_a_closed_system():
    network = _reversible_A_B_network(1.5, 0.7)
    C0 = np.array([2.0, 1.0])
    t_eval = np.linspace(0, 20, 50)
    result = integrate_network(network, C0, t_eval)
    totals = result.C[0, :] + result.C[1, :]
    assert totals == pytest.approx(totals[0], rel=1e-6)


def test_integrate_network_clamps_negative_overshoot_and_stays_nonnegative():
    # A fast, stiff-ish decay that could overshoot into negative territory
    # without clamping.
    network = ReactionNetwork(
        species=("A",),
        reactions=(Reaction("decay", mass_action_rate(50.0, {"A": 1}), {"A": -1.0}),),
    )
    C0 = np.array([1.0])
    t_eval = np.linspace(0, 5, 500)
    result = integrate_network(network, C0, t_eval)
    assert np.all(result.C >= -1e-9)


def test_arrhenius_rate_constant_increases_with_temperature_for_positive_activation_energy():
    k = arrhenius_rate_constant(A=1.0e10, Ea_over_R=5000.0)
    assert k(T_K=350.0) > k(T_K=300.0)


def test_arrhenius_requires_temperature():
    k = arrhenius_rate_constant(A=1.0, Ea_over_R=100.0)
    with pytest.raises(ValueError):
        k()


def test_quadratic_in_T_matches_direct_evaluation():
    f = quadratic_in_T(1.18e-4, -7.86e-2, 20.5)
    T = 298.15
    assert f(T) == pytest.approx(1.18e-4 * T ** 2 - 7.86e-2 * T + 20.5)


def test_acid_and_base_fraction_sum_to_one_and_have_correct_limits():
    pH, pKa = 7.0, 7.5
    assert acid_fraction(pH, pKa) + base_fraction(pH, pKa) == pytest.approx(1.0)
    # far below pKa: almost entirely acid form
    assert acid_fraction(pKa - 4, pKa) > 0.999
    # far above pKa: almost entirely base form
    assert base_fraction(pKa + 4, pKa) > 0.999


def test_diprotic_fractions_sum_to_one():
    a0, a1, a2 = diprotic_fractions(pH=8.3, pKa1=6.3, pKa2=10.3)
    assert a0 + a1 + a2 == pytest.approx(1.0)
    assert all(0.0 <= a <= 1.0 for a in (a0, a1, a2))


def test_linear_combination_rate_constant():
    k_eff = linear_combination_rate_constant([(2.0, "cat_A"), (5.0, "cat_B")])
    C = {"cat_A": 1.0e-3, "cat_B": 4.0e-4}
    assert k_eff(C=C) == pytest.approx(2.0 * 1.0e-3 + 5.0 * 4.0e-4)


def test_catalyzed_rate_constant_plugs_into_mass_action_rate():
    k_eff = linear_combination_rate_constant([(1.0e5, "H")])
    rate_fn = mass_action_rate(k_eff, {"NH2Cl": 2})
    C = {"NH2Cl": 1.0e-4, "H": 1.0e-7}
    expected = (1.0e5 * 1.0e-7) * (1.0e-4 ** 2)
    assert rate_fn(C) == pytest.approx(expected)


# --- transport -------------------------------------------------------------
def test_transport_with_no_reaction_and_no_source_conserves_total_mass_on_periodic_grid():
    inert_network = ReactionNetwork(species=("A",), reactions=())
    solver = AdvectionDispersionReactionSolver(inert_network, n_grid=40, length_m=100.0, velocity_m_s=0.5, dispersion_m2_s=0.1)
    C0 = np.zeros((1, 40))
    C0[0, 10:15] = 1.0  # a localized pulse
    t_eval = np.linspace(0, 30, 10)
    result = solver.integrate(C0, t_eval)
    totals = result.C[0].sum(axis=0) * solver.dx
    assert totals == pytest.approx(totals[0], rel=1e-3)


def test_transport_pulse_advects_downstream():
    inert_network = ReactionNetwork(species=("A",), reactions=())
    solver = AdvectionDispersionReactionSolver(inert_network, n_grid=40, length_m=100.0, velocity_m_s=1.0, dispersion_m2_s=0.01)
    C0 = np.zeros((1, 40))
    C0[0, 5] = 1.0
    t_eval = np.array([0.0, 10.0])
    result = solver.integrate(C0, t_eval)
    profile_0 = result.C[0, :, 0]
    profile_t = result.C[0, :, -1]
    centroid_0 = np.sum(np.arange(40) * profile_0) / np.sum(profile_0)
    centroid_t = np.sum(np.arange(40) * profile_t) / np.sum(profile_t)
    assert centroid_t > centroid_0  # moved downstream (+x) under u > 0


def test_transport_reaction_rhs_reuses_the_same_network_as_0d():
    # The whole point of building transport on top of ReactionNetwork:
    # a species with a pure first-order decay reaction should decay at
    # grid points with no advection/dispersion coupling needed to see
    # the SAME kinetics as the plain 0D integrate_network call.
    network = ReactionNetwork(
        species=("A",),
        reactions=(Reaction("decay", mass_action_rate(0.2, {"A": 1}), {"A": -1.0}),),
    )
    solver = AdvectionDispersionReactionSolver(network, n_grid=5, length_m=10.0, velocity_m_s=0.0, dispersion_m2_s=0.0)
    C0 = np.ones((1, 5))
    t_eval = np.linspace(0, 10, 20)
    transport_result = solver.integrate(C0, t_eval)

    zerod_result = integrate_network(network, np.array([1.0]), t_eval)
    assert transport_result.C[0, 0, :] == pytest.approx(zerod_result.C[0, :], rel=1e-3)
