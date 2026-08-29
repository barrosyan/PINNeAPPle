"""Validates pinneapple_systems.process_components.non_newtonian_pipe_flow
against independent closed-form identities: the Newtonian limit (n=1),
the Hagen-Poiseuille laminar friction-factor identity, and a hand-
computed single-step Euler pressure integration."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_systems.process_components.non_newtonian_pipe_flow import (
    effective_viscosity,
    generalized_reynolds_number,
    herschel_bulkley_stress,
    integrate_pressure_profile,
    metzner_reed_friction_factor,
    pressure_gradient,
)


def test_herschel_bulkley_reduces_to_newtonian_at_zero_yield_stress_and_n_equal_1():
    mu = 0.05
    gamma_dot = np.array([1.0, 10.0, 100.0])
    tau = herschel_bulkley_stress(tau_y=0.0, K=mu, n=1.0, gamma_dot=gamma_dot)
    assert tau == pytest.approx(mu * gamma_dot, rel=1e-10)


def test_effective_viscosity_reduces_to_constant_mu_for_newtonian_fluid():
    mu = 0.03
    gamma_dot = np.array([1.0, 50.0, 500.0])
    mu_eff = effective_viscosity(tau_y=0.0, K=mu, n=1.0, gamma_dot=gamma_dot)
    assert mu_eff == pytest.approx(mu, rel=1e-9)


def test_generalized_reynolds_number_reduces_to_newtonian_definition_at_n_equal_1():
    rho, v, D, mu = 1000.0, 2.0, 0.1, 0.001
    Re_gen = generalized_reynolds_number(rho, v, D, K=mu, n=1.0)
    Re_newtonian = rho * v * D / mu
    assert Re_gen == pytest.approx(Re_newtonian, rel=1e-6)


def test_laminar_friction_factor_matches_hagen_poiseuille_16_over_Re():
    Re = np.array([100.0, 500.0, 2000.0])
    f = metzner_reed_friction_factor(Re)
    assert f == pytest.approx(16.0 / Re, rel=1e-4)


def test_turbulent_friction_factor_matches_blasius_correlation():
    Re = np.array([5000.0, 50000.0])
    f = metzner_reed_friction_factor(Re)
    assert f == pytest.approx(0.0791 * Re ** (-0.25), rel=1e-4)


def test_pressure_gradient_pure_hydrostatic_when_velocity_is_zero():
    rho, g = 1200.0, 9.81
    dPds = pressure_gradient(rho=rho, v=0.0, D_h=0.1, f=0.02, inclination_rad=np.pi / 2, g=g)
    assert dPds == pytest.approx(rho * g, rel=1e-9)


def test_pressure_gradient_zero_hydrostatic_component_for_horizontal_path():
    rho, v, D_h, f = 1200.0, 1.5, 0.1, 0.02
    dPds = pressure_gradient(rho=rho, v=v, D_h=D_h, f=f, inclination_rad=0.0)
    expected_friction_only = f * rho * v ** 2 / (2.0 * D_h)
    assert dPds == pytest.approx(expected_friction_only, rel=1e-9)


def test_vertical_pressure_integration_matches_hand_computed_single_step():
    rho, v, D_h, K, n, P0, g = 1200.0, 1.0, 0.2, 0.05, 1.0, 1.0e5, 9.81
    length = 10.0
    profile = integrate_pressure_profile(
        length_m=length, v_m_s=v, D_h_m=D_h, K_Pa_sn=K, n_flow_index=n,
        P_inlet_Pa=P0, rho_profile=rho, inclination_deg_profile=90.0, g=g, n_steps=1,
    )
    Re = generalized_reynolds_number(rho, v, D_h, K, n)
    f = metzner_reed_friction_factor(Re)
    dPds_expected = pressure_gradient(rho, v, D_h, f, np.pi / 2, g)
    assert profile.P_Pa[-1] == pytest.approx(P0 + dPds_expected * length, rel=1e-9)
    assert profile.tvd_m[-1] == pytest.approx(length, rel=1e-9)


def test_equivalent_density_of_a_static_vertical_column_recovers_its_own_density():
    """A fluid column at rest (v=0) under only its own hydrostatic head
    has equivalent density exactly equal to its own density -- the
    textbook check for any hydrostatic-equivalent-density formula."""
    rho = 1450.0
    profile = integrate_pressure_profile(
        length_m=500.0, v_m_s=0.0, D_h_m=0.2, K_Pa_sn=0.05, n_flow_index=1.0,
        P_inlet_Pa=0.0, rho_profile=rho, inclination_deg_profile=90.0, n_steps=200,
    )
    assert profile.rho_eq_kg_m3[-1] == pytest.approx(rho, rel=1e-3)


def test_horizontal_path_never_accumulates_true_vertical_depth():
    profile = integrate_pressure_profile(
        length_m=100.0, v_m_s=1.0, D_h_m=0.15, K_Pa_sn=0.05, n_flow_index=0.7,
        P_inlet_Pa=1.0e5, rho_profile=1100.0, inclination_deg_profile=0.0, n_steps=50,
    )
    assert profile.tvd_m[-1] == pytest.approx(0.0, abs=1e-9)
