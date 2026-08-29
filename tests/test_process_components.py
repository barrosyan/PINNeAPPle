"""Validates pinneapple_systems.process_components against independent
analytical cross-checks (ideal-gas closed forms, textbook Moody-chart
points, IEC worked-example-style checks) -- not just internal
self-consistency."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_systems.process_components import (
    GasComposition,
    HeatExchangerSpec,
    PipeSpec,
    TransientPipe,
    ValveSpec,
    colebrook_white_f,
    compressible_mass_flow,
    heat_exchanger_steady_state,
    installed_cv,
    rapid_steady_state_profile,
    solve_path_from_pressure_ratio,
    state_from_PT,
)

PURE_METHANE = GasComposition(components=("Methane",), mole_fractions=(1.0,))
NATURAL_GAS = GasComposition(
    components=("Methane", "Ethane", "Propane", "n-Butane", "Nitrogen", "CarbonDioxide"),
    mole_fractions=(0.900, 0.045, 0.015, 0.005, 0.020, 0.015),
)


# --- real_gas_eos ------------------------------------------------------
def test_ideal_gas_limit_near_standard_conditions():
    s = state_from_PT(NATURAL_GAS, 101_325.0, 288.15)
    assert 0.97 < s.Z < 1.01


def test_methane_density_matches_published_reference_order_of_magnitude():
    s = state_from_PT(PURE_METHANE, 100e5, 300.0)
    assert 60.0 < s.rho_kg_m3 < 78.0
    assert 0.83 < s.Z < 0.93


# --- polytropic_path -----------------------------------------------------
def test_compression_matches_ideal_gas_closed_form_at_low_pressure():
    P1_Pa, T1_K = 2.0e5, 300.0
    inlet = state_from_PT(PURE_METHANE, P1_Pa, T1_K)
    eta_p = 0.80
    R_specific = 8.314462618 / inlet.molar_mass_kg_mol
    k = inlet.k_isentropic
    n = 1.0 / (1.0 - (1.0 - 1.0 / k) / eta_p)
    P2_Pa = P1_Pa * 2.0
    W_ideal = (n / (n - 1.0)) * R_specific * T1_K * ((P2_Pa / P1_Pa) ** ((n - 1.0) / n) - 1.0)

    result = solve_path_from_pressure_ratio(inlet, PURE_METHANE, P2_Pa, eta_p)
    assert result.polytropic_work_J_kg == pytest.approx(W_ideal, rel=0.02)


def test_compression_discharge_temperature_rises_above_isentropic():
    inlet = state_from_PT(NATURAL_GAS, 55e5, 290.0)
    result = solve_path_from_pressure_ratio(inlet, NATURAL_GAS, 55e5 * 2.0, 0.78)
    assert result.outlet.T_K > inlet.T_K
    assert result.actual_enthalpy_change_J_kg > result.isentropic_work_J_kg


def test_compression_isentropic_efficiency_below_polytropic_efficiency():
    # Real-machine reheat effect: for PR > 1, isentropic efficiency <= polytropic efficiency.
    inlet = state_from_PT(NATURAL_GAS, 55e5, 290.0)
    result = solve_path_from_pressure_ratio(inlet, NATURAL_GAS, 55e5 * 2.5, 0.78)
    assert result.isentropic_efficiency < result.polytropic_efficiency
    assert 0.0 < result.isentropic_efficiency < 1.0


def test_expansion_direction_reduces_temperature_and_efficiency_convention_is_bounded():
    # A turboexpander case (P2 < P1): this exercises the branch that was
    # found and fixed to need a DIFFERENT isentropic-efficiency formula
    # than compression (see polytropic_path.py's _finalize docstring).
    inlet = state_from_PT(NATURAL_GAS, 100e5, 320.0)
    result = solve_path_from_pressure_ratio(inlet, NATURAL_GAS, 50e5, 0.80)
    assert result.pressure_ratio < 1.0
    assert result.polytropic_work_J_kg < 0.0
    assert result.outlet.T_K < inlet.T_K
    # The real expander must extract LESS work than the ideal isentropic
    # path for the same pressure drop -> smaller-magnitude enthalpy drop.
    assert abs(result.actual_enthalpy_change_J_kg) < abs(result.isentropic_work_J_kg)
    # And the efficiency, with the corrected convention, must land in (0, 1] -- NOT > 1.
    assert 0.0 < result.isentropic_efficiency <= 1.0


def test_expansion_matches_ideal_gas_closed_form_at_low_pressure():
    # NOTE on what "polytropic_work_J_kg" (W) actually is: W = integral of
    # v dP along the ACTUAL path (not the isentropic path, and not simply
    # eta_p times the isentropic work -- those are three different
    # quantities). For an ideal gas with dh = eta_p*v*dP (the expansion
    # branch), substituting v=R*T/P and T(P)=T1*(P/P1)^(eta_p*m) (itself
    # obtained by integrating the SAME relation) gives, after integrating
    # W's own defining ODE dW/dP=v along that path:
    #   W = (R*T1 / (eta_p*m)) * ((P2/P1)^(eta_p*m) - 1)
    # An earlier version of this test compared W against `eta_p *
    # isentropic_work` instead -- numerically close in magnitude but the
    # WRONG target quantity (off by ~18% here), caught by cross-checking
    # against a fine-step manual Euler integration of the module's own
    # ODE before concluding the closed form (not the module) was wrong.
    P1_Pa, T1_K = 4.0e5, 320.0
    inlet = state_from_PT(PURE_METHANE, P1_Pa, T1_K)
    eta_p = 0.85
    R_specific = 8.314462618 / inlet.molar_mass_kg_mol
    k = inlet.k_isentropic
    P2_Pa = P1_Pa * 0.5
    m = (k - 1.0) / k
    em = eta_p * m
    r = P2_Pa / P1_Pa
    W_ideal = (R_specific * T1_K / em) * (r ** em - 1.0)

    result = solve_path_from_pressure_ratio(inlet, PURE_METHANE, P2_Pa, eta_p)
    assert result.polytropic_work_J_kg == pytest.approx(W_ideal, rel=0.02)


def test_expansion_and_compression_share_the_same_exact_work_efficiency_identity():
    # By construction, dh = eta_p*v*dP (expansion) or v*dP/eta_p
    # (compression), and dW = v*dP in both branches -- so actual
    # enthalpy change and polytropic work must satisfy an EXACT
    # (floating-point-precision) algebraic identity, independent of the
    # gas model. This is a stronger, model-independent check than any
    # closed-form comparison.
    inlet_c = state_from_PT(NATURAL_GAS, 55e5, 290.0)
    comp = solve_path_from_pressure_ratio(inlet_c, NATURAL_GAS, 55e5 * 2.0, 0.80)
    assert comp.actual_enthalpy_change_J_kg == pytest.approx(comp.polytropic_work_J_kg / 0.80, rel=1e-9)

    inlet_e = state_from_PT(NATURAL_GAS, 100e5, 320.0)
    exp = solve_path_from_pressure_ratio(inlet_e, NATURAL_GAS, 50e5, 0.80)
    assert exp.actual_enthalpy_change_J_kg == pytest.approx(exp.polytropic_work_J_kg * 0.80, rel=1e-9)


# --- control_valve ---------------------------------------------------------
SPEC = ValveSpec(name="v", Cv_max_us=800.0, xT=0.60)


def test_installed_cv_monotonic_and_bounded():
    travels = [0.0, 0.3, 0.6, 1.0]
    cvs = [installed_cv(t, SPEC) for t in travels]
    assert all(b > a for a, b in zip(cvs, cvs[1:]))
    assert cvs[-1] == pytest.approx(SPEC.Cv_max_us, rel=1e-9)


def test_valve_flow_saturates_once_choked():
    P1, rho1, k, xT = 80e5, 60.0, 1.3, 0.60
    x_choke = k / 1.40 * xT
    P2_at_choke = P1 * (1 - x_choke)
    r_at = compressible_mass_flow(200.0, P1, P2_at_choke, rho1, k, xT)
    r_beyond = compressible_mass_flow(200.0, P1, P2_at_choke * 0.5, rho1, k, xT)
    assert r_beyond.choked
    assert r_beyond.mass_flow_kg_s == pytest.approx(r_at.mass_flow_kg_s, rel=1e-6)


def test_valve_flow_scales_linearly_with_cv_unchoked():
    P1, P2, rho1, k = 80e5, 76e5, 60.0, 1.3
    low = compressible_mass_flow(100.0, P1, P2, rho1, k, 0.60)
    high = compressible_mass_flow(200.0, P1, P2, rho1, k, 0.60)
    assert high.mass_flow_kg_s == pytest.approx(2.0 * low.mass_flow_kg_s, rel=1e-9)


# --- heat_exchanger ----------------------------------------------------
HX_SPEC = HeatExchangerSpec(
    name="hx", UA_design_W_K=250_000.0, design_dP_Pa=40_000.0, design_m_dot_hot_kg_s=100.0,
    design_rho_hot_kg_m3=40.0, thermal_mass_J_K=5.0e6, cold_side_capacity_rate_W_K=400_000.0,
)


def test_heat_exchanger_energy_balance_closes():
    r = heat_exchanger_steady_state(HX_SPEC, 100.0, 400.0, 2400.0, 305.0, 40.0)
    Q_hot_side = 100.0 * 2400.0 * (400.0 - r.T_hot_out_K)
    Q_cold_side = HX_SPEC.cold_side_capacity_rate_W_K * (r.T_cold_out_K - 305.0)
    assert Q_hot_side == pytest.approx(r.Q_W, rel=1e-9)
    assert Q_cold_side == pytest.approx(r.Q_W, rel=1e-9)


def test_heat_exchanger_effectiveness_bounded():
    r = heat_exchanger_steady_state(HX_SPEC, 100.0, 400.0, 2400.0, 305.0, 40.0)
    assert 0.0 < r.effectiveness < 1.0


# --- pipe_network_1d -----------------------------------------------------
PIPE_SPEC = PipeSpec(name="p", length_m=50_000.0, diameter_m=0.6, roughness_m=4.5e-5, n_cells=12)


def test_colebrook_matches_known_moody_chart_region():
    f = colebrook_white_f(1.0e5, 0.001)
    assert 0.018 < f < 0.023


def test_colebrook_laminar_matches_64_over_re():
    f = colebrook_white_f(1500.0, 0.0005)
    assert f == pytest.approx(64.0 / 1500.0, rel=1e-9)


def test_pipe_steady_profile_pressure_drops_with_flow():
    profile = rapid_steady_state_profile(PIPE_SPEC, NATURAL_GAS, 150.0, 90e5, 310.0)
    pressures = [p.P_Pa for p in profile]
    assert all(a >= b for a, b in zip(pressures, pressures[1:]))
    assert pressures[-1] < pressures[0]


def test_transient_pipe_conserves_mass_with_matched_boundary_flow():
    pipe = TransientPipe(PIPE_SPEC, NATURAL_GAS)
    state0 = pipe.initialize_from_steady_state(150.0, 90e5, 310.0)
    mass0 = pipe.total_mass_kg(state0)
    P_out = pipe.steady_outlet_pressure(150.0, 90e5, 310.0)

    sol = pipe.simulate(
        state0, (0.0, 1800.0), m_dot_in_fn=lambda t: 150.0, T_in_fn=lambda t: 310.0,
        P_out_fn=lambda t: P_out, t_eval=np.linspace(0, 1800, 4),
    )
    assert sol.success
    n = PIPE_SPEC.n_cells
    for k in range(sol.y.shape[1]):
        state_k = type(state0)(sol.y[:n, k], sol.y[n:, k])
        assert pipe.total_mass_kg(state_k) == pytest.approx(mass0, rel=0.01)


def test_rapid_profile_solves_a_large_pressure_drop_case():
    # Regression: an earlier bracket [0.30*P, 0.999999*P] for the
    # per-cell momentum closure was too tight once a few cells
    # downstream have already dropped enough pressure that density (and
    # thus velocity, for fixed mass flux) has changed enough to need a
    # larger fractional drop per cell than 70% -- this case (inlet
    # pressure roughly halving over the pipe) used to raise scipy's raw
    # "f(a) and f(b) must have different signs" instead of solving.
    profile = rapid_steady_state_profile(PIPE_SPEC, NATURAL_GAS, m_dot_kg_s=150.0, P_in_Pa=60e5, T_in_K=313.0)
    assert profile[-1].P_Pa < profile[0].P_Pa
    assert all(p.P_Pa > 0 for p in profile)


def test_genuine_choking_raises_a_clear_actionable_error():
    undersized_spec = PipeSpec(name="undersized", length_m=50_000.0, diameter_m=0.3, roughness_m=4.5e-5, n_cells=15)
    with pytest.raises(ValueError, match="chokes before reaching the end of the pipe"):
        rapid_steady_state_profile(undersized_spec, NATURAL_GAS, m_dot_kg_s=175.0, P_in_Pa=90e5, T_in_K=310.0)
