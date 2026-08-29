"""Validates pinneapple_systems.process_components.torsional_stickslip:
the Stribeck friction curve's own limiting behavior, and end-to-end
smoke/physical-sanity checks on the coupled wave-equation simulation
(zero friction differential -> smooth rotation, large friction
differential -> pronounced stick-slip, matching the well-known physical
mechanism the model is built to reproduce)."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_systems.process_components.torsional_stickslip import (
    simulate_torsional_stickslip,
    stribeck_friction_torque,
)


def test_stribeck_friction_at_zero_velocity_equals_static_torque():
    T = stribeck_friction_torque(omega=1e-15, T_static=1000.0, T_kinetic=400.0, omega_breakout=1.5)
    assert abs(T) == pytest.approx(1000.0, rel=1e-6)


def test_stribeck_friction_at_high_velocity_approaches_kinetic_torque():
    T = stribeck_friction_torque(omega=1000.0, T_static=1000.0, T_kinetic=400.0, omega_breakout=1.5)
    assert abs(T) == pytest.approx(400.0, rel=1e-3)


def test_stribeck_friction_opposes_the_direction_of_motion():
    T_pos = stribeck_friction_torque(omega=5.0, T_static=1000.0, T_kinetic=400.0, omega_breakout=1.5)
    T_neg = stribeck_friction_torque(omega=-5.0, T_static=1000.0, T_kinetic=400.0, omega_breakout=1.5)
    assert T_pos < 0
    assert T_neg > 0


def test_larger_static_kinetic_gap_produces_more_severe_stick_slip_near_breakout_speed():
    """The Stribeck mechanism only destabilizes rotation when the
    operating speed is comparable to (or below) the breakout velocity --
    well above it the friction curve has already flattened onto its
    kinetic plateau and any startup wave transient simply decays with
    time regardless of the static/kinetic gap (verified separately: at
    10x the breakout speed, even a large gap settles to SSI ~1e-2 given
    enough run time). At a speed near the breakout velocity, a
    negligible gap settles to smooth rotation while a large gap sustains
    a genuine sustained (non-decaying) stick-slip limit cycle."""
    common = dict(
        length_m=3000.0, G_Pa=80e9, J_m4=6e-6, rho_kg_m3=7850.0,
        omega_set_rad_s=2.0, omega_breakout_rad_s=1.5, end_inertia_kg_m2=50.0,
        n_nodes=30, n_steps=30000, save_every=20, damping_coeff=0.05, ssi_window_fraction=0.1,
    )
    negligible_gap = simulate_torsional_stickslip(T_static_N_m=500.0, T_kinetic_N_m=500.0, **common)
    large_gap = simulate_torsional_stickslip(T_static_N_m=2000.0, T_kinetic_N_m=300.0, **common)
    assert large_gap.stick_slip_index > 5.0 * negligible_gap.stick_slip_index


def test_result_shapes_are_internally_consistent():
    n_nodes, n_steps, save_every = 20, 800, 20
    result = simulate_torsional_stickslip(
        length_m=1000.0, G_Pa=80e9, J_m4=4e-6, rho_kg_m3=7850.0,
        omega_set_rad_s=5.0, T_static_N_m=800.0, T_kinetic_N_m=500.0,
        omega_breakout_rad_s=1.5, end_inertia_kg_m2=20.0,
        n_nodes=n_nodes, n_steps=n_steps, save_every=save_every,
    )
    n_saved = n_steps // save_every
    assert result.omega_field_rad_s.shape == (n_saved, n_nodes)
    assert result.z_m.shape == (n_nodes,)
    assert result.t_s.shape == (n_saved,)
    assert result.end_omega_ts_rad_s.shape == (n_saved,)


def test_n_nodes_below_three_is_rejected():
    with pytest.raises(ValueError):
        simulate_torsional_stickslip(
            length_m=100.0, G_Pa=80e9, J_m4=1e-6, rho_kg_m3=7850.0,
            omega_set_rad_s=1.0, T_static_N_m=1.0, T_kinetic_N_m=1.0,
            omega_breakout_rad_s=1.0, end_inertia_kg_m2=1.0, n_nodes=2,
        )
