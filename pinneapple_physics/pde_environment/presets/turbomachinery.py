"""Turbomachinery problem presets.

Covers axial compressor physics for use with TurboDesigner as a data/constraint source:
  - axial_compressor_meanline   : 1D mean-line (Euler work + continuity + energy)
  - axial_compressor_cascade_2d : 2D blade-to-blade cascade (compressible Euler)
  - axial_compressor_stage_3d   : 3D single-stage in cylindrical coords (rotating Euler)

The ``solver_spec["name"] = "turbodesigner"`` signals that
``pinneapple_simulation.external_solvers.turbodesigner`` can generate training
data for these problems from analytical mean-line solutions.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..conditions import DirichletBC, NeumannBC
from ..scales import ScaleSpec
from ..spec import PDETermSpec, ProblemSpec
from .registry import register_preset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _polytropic_temp_ratio(pressure_ratio: float, gamma: float, eta: float) -> float:
    """Total temperature ratio for polytropic compression with isentropic efficiency eta."""
    return 1.0 + (pressure_ratio ** ((gamma - 1.0) / gamma) - 1.0) / eta


# ===========================================================================
# PRESET 1 — 1D Mean-Line Thermodynamic Analysis
# ===========================================================================

@register_preset("axial_compressor_meanline")
def axial_compressor_meanline(
    num_stages: int = 5,
    pressure_ratio: float = 3.0,
    mass_flow_rate: float = 4.37,
    rpm: float = 10_000.0,
    inlet_total_pressure: float = 101_325.0,
    inlet_total_temperature: float = 288.15,
    isentropic_efficiency: float = 0.878,
    hub_to_tip_ratio: float = 0.5,
    axial_velocity: float = 136.0,
    gamma: float = 1.4,
    R_gas: float = 287.0,
) -> ProblemSpec:
    """1D mean-line thermodynamic analysis of a multi-stage axial compressor.

    Solves Euler work equation, continuity, and isentropic relations along
    the streamwise coordinate s ∈ [0, 1] (inlet to outlet, normalised).

    PDE residuals (stage-averaged, kind="compressor_meanline_1d"):
      - Energy    : c_p * d(T_t)/ds = W_stage / L_machine
      - Continuity: d(rho * u * A)/ds = 0
      - State     : p_t = rho * R_gas * T_t  (ideal gas)

    Fields  : T_t (K), p_t (Pa), rho (kg/m³), u (m/s), c_theta (m/s)
    Coords  : s  (dimensionless streamwise, 0 = inlet, 1 = outlet)

    Parameters
    ----------
    num_stages              : number of compressor stages
    pressure_ratio          : overall total pressure ratio (p_tout / p_tin)
    mass_flow_rate          : kg/s
    rpm                     : shaft speed in rev/min
    inlet_total_pressure    : Pa
    inlet_total_temperature : K
    isentropic_efficiency   : stage polytropic efficiency (0–1)
    hub_to_tip_ratio        : h/t ratio at inlet
    axial_velocity          : mean axial velocity (m/s), used as inlet BC
    gamma                   : specific heat ratio
    R_gas                   : gas constant J/(kg·K)
    """
    c_p = gamma * R_gas / (gamma - 1.0)
    omega = rpm * 2.0 * math.pi / 60.0

    T_t_out = inlet_total_temperature * _polytropic_temp_ratio(
        pressure_ratio, gamma, isentropic_efficiency
    )
    delta_T_stage = (T_t_out - inlet_total_temperature) / num_stages
    p_t_out = inlet_total_pressure * pressure_ratio

    coords = ("s",)
    fields = ("T_t", "p_t", "rho", "u", "c_theta")

    pde = PDETermSpec(
        kind="compressor_meanline_1d",
        fields=fields,
        coords=coords,
        params={
            "gamma": gamma,
            "R_gas": R_gas,
            "c_p": c_p,
            "omega": omega,
            "num_stages": num_stages,
            "delta_T_stage": delta_T_stage,
            "W_stage_per_unit_length": c_p * delta_T_stage,
            "mass_flow_rate": mass_flow_rate,
            "hub_to_tip_ratio": hub_to_tip_ratio,
            "isentropic_efficiency": isentropic_efficiency,
        },
    )

    _T_t0 = float(inlet_total_temperature)
    _p_t0 = float(inlet_total_pressure)
    _u0 = float(axial_velocity)
    _p_t1 = float(p_t_out)

    inlet_bc = DirichletBC(
        "inlet",
        fields=("T_t", "p_t", "u"),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 0], 0.0),
        value_fn=lambda X, ctx: np.column_stack([
            np.full(X.shape[0], _T_t0, dtype=np.float32),
            np.full(X.shape[0], _p_t0, dtype=np.float32),
            np.full(X.shape[0], _u0, dtype=np.float32),
        ]),
        weight=10.0,
    )
    outlet_bc = DirichletBC(
        "outlet",
        fields=("p_t",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 0], 1.0),
        value_fn=lambda X, ctx: np.full((X.shape[0], 1), _p_t1, dtype=np.float32),
        weight=5.0,
    )

    return ProblemSpec(
        name="axial_compressor_meanline",
        dim=1,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(inlet_bc, outlet_bc),
        domain_bounds={"s": (0.0, 1.0)},
        field_ranges={
            "T_t": (inlet_total_temperature * 0.98, T_t_out * 1.05),
            "p_t": (inlet_total_pressure * 0.98, p_t_out * 1.05),
            "rho": (0.5, 6.0),
            "u": (50.0, 300.0),
            "c_theta": (-250.0, 250.0),
        },
        scales=ScaleSpec(L=1.0, U=axial_velocity),
        sample_defaults={"n_col": 10_000, "n_bc": 500},
        solver_spec={
            "name": "turbodesigner",
            "machine_type": "axial",
            "configuration": "compressor",
        },
        meta={
            "description": "Axial compressor mean-line thermodynamic analysis (1D PINN)",
            "num_stages": num_stages,
            "pressure_ratio": pressure_ratio,
            "rpm": rpm,
            "T_t_outlet_K": T_t_out,
            "p_t_outlet_Pa": p_t_out,
            "delta_T_stage_K": delta_T_stage,
            "digital_twin_fields": ["T_t", "p_t", "u"],
            "turbodesigner_definition": {
                "gamma": gamma,
                "axial_velocity": axial_velocity,
                "rpm": rpm,
                "gas_constant": R_gas,
                "mass_flow_rate": mass_flow_rate,
                "pressure_ratio": pressure_ratio,
                "inlet_total_pressure": inlet_total_pressure,
                "inlet_total_temperature": inlet_total_temperature,
                "isentropic_efficiency": isentropic_efficiency,
                "num_stages": num_stages,
                "hub_to_tip_ratio": hub_to_tip_ratio,
            },
        },
    )


# ===========================================================================
# PRESET 2 — 2D Blade-to-Blade Cascade Flow
# ===========================================================================

@register_preset("axial_compressor_cascade_2d")
def axial_compressor_cascade_2d(
    inlet_mach: float = 0.5,
    flow_angle_in_deg: float = 50.0,
    flow_angle_out_deg: float = 20.0,
    chord: float = 0.1,
    pitch_to_chord: float = 0.8,
    gamma: float = 1.4,
    R_gas: float = 287.0,
    p_inlet: float = 101_325.0,
    T_inlet: float = 288.15,
) -> ProblemSpec:
    """2D compressible Euler flow through an axial compressor blade cascade.

    Models the blade-to-blade passage between adjacent blades.  The domain is
    a 2D channel (x = axial, y = pitch) with periodic BC in y implied by the
    blade row symmetry.

    PDE: steady 2D compressible Euler (kind="compressible_euler_2d").
    Fields : rho, u (axial), v (tangential), p, T
    Coords : (x, y)

    Parameters
    ----------
    inlet_mach        : inlet absolute Mach number
    flow_angle_in_deg : inlet flow angle from axial (deg), positive = swirl
    flow_angle_out_deg: design outlet flow angle from axial (deg)
    chord             : blade chord length (m)
    pitch_to_chord    : blade pitch / chord
    gamma, R_gas      : thermodynamic constants
    p_inlet, T_inlet  : inlet total conditions (Pa, K)
    """
    c_sound_in = math.sqrt(gamma * R_gas * T_inlet)
    U_in = inlet_mach * c_sound_in
    u_in = U_in * math.cos(math.radians(flow_angle_in_deg))
    v_in = U_in * math.sin(math.radians(flow_angle_in_deg))
    rho_in = p_inlet / (R_gas * T_inlet)
    pitch = pitch_to_chord * chord
    domain_x = 2.5 * chord

    coords = ("x", "y")
    fields = ("rho", "u", "v", "p", "T")

    pde = PDETermSpec(
        kind="compressible_euler_2d",
        fields=fields,
        coords=coords,
        params={
            "gamma": gamma,
            "R_gas": R_gas,
            "inlet_mach": inlet_mach,
        },
    )

    _rho_in = float(rho_in)
    _u_in = float(u_in)
    _v_in = float(v_in)
    _T_in = float(T_inlet)
    _domain_x = float(domain_x)

    inlet_bc = DirichletBC(
        "cascade_inlet",
        fields=("rho", "u", "v", "T"),
        selector_type="callable",
        selector=lambda X, ctx: X[:, 0] < 1e-9,
        value_fn=lambda X, ctx: np.column_stack([
            np.full(X.shape[0], _rho_in, dtype=np.float32),
            np.full(X.shape[0], _u_in, dtype=np.float32),
            np.full(X.shape[0], _v_in, dtype=np.float32),
            np.full(X.shape[0], _T_in, dtype=np.float32),
        ]),
        weight=10.0,
    )
    outlet_bc = NeumannBC(
        "cascade_outlet",
        fields=("rho", "u", "v", "T"),
        selector_type="callable",
        selector=lambda X, ctx: X[:, 0] > _domain_x - 1e-9,
        weight=1.0,
    )
    blade_bc = DirichletBC(
        "blade_surface",
        fields=("u", "v"),
        selector_type="tag",
        selector={"tag": "blade"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 2), dtype=np.float32),
        weight=20.0,
    )

    return ProblemSpec(
        name="axial_compressor_cascade_2d",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(inlet_bc, outlet_bc, blade_bc),
        domain_bounds={
            "x": (0.0, domain_x),
            "y": (0.0, pitch),
        },
        field_ranges={
            "rho": (0.3 * rho_in, 2.5 * rho_in),
            "u": (0.0, U_in * 1.5),
            "v": (-U_in, U_in),
            "p": (0.5 * p_inlet, 2.5 * p_inlet),
            "T": (0.75 * T_inlet, 1.5 * T_inlet),
        },
        scales=ScaleSpec(L=chord, U=U_in),
        sample_defaults={"n_col": 80_000, "n_bc": 8_000},
        solver_spec={
            "name": "turbodesigner",
            "machine_type": "axial",
            "configuration": "compressor",
        },
        meta={
            "description": "2D axial compressor blade cascade — compressible Euler PINN",
            "inlet_mach": inlet_mach,
            "flow_angle_in_deg": flow_angle_in_deg,
            "flow_angle_out_deg": flow_angle_out_deg,
            "chord": chord,
            "pitch": pitch,
            "digital_twin_fields": ["p", "rho", "u", "v"],
        },
    )


# ===========================================================================
# PRESET 3 — 3D Single-Stage Compressor (Cylindrical, Rotating Frame)
# ===========================================================================

@register_preset("axial_compressor_stage_3d")
def axial_compressor_stage_3d(
    rpm: float = 10_000.0,
    pressure_ratio_stage: float = 1.4,
    mass_flow_rate: float = 4.37,
    hub_radius: float = 0.10,
    tip_radius: float = 0.20,
    axial_length: float = 0.15,
    inlet_total_pressure: float = 101_325.0,
    inlet_total_temperature: float = 288.15,
    isentropic_efficiency: float = 0.88,
    gamma: float = 1.4,
    R_gas: float = 287.0,
) -> ProblemSpec:
    """3D single compressor stage in cylindrical coordinates (r, theta, z).

    Models the rotor passage using steady 3D compressible Euler equations in
    a rotating reference frame (rotating at ``rpm``).  The PINN learns the
    full 3D flow field including spanwise redistribution and tip effects.

    PDE: steady 3D compressible Euler, rotating frame (kind="compressible_euler_rotating_3d").
    Fields : rho, u_r, u_theta, u_z, p, T
    Coords : (r, theta, z)

    Parameters
    ----------
    rpm                     : rotor speed (rev/min)
    pressure_ratio_stage    : stage total-to-total pressure ratio
    mass_flow_rate          : kg/s
    hub_radius              : hub radius at rotor inlet (m)
    tip_radius              : tip radius at rotor inlet (m)
    axial_length            : rotor axial chord (m)
    inlet_total_pressure    : Pa
    inlet_total_temperature : K
    isentropic_efficiency   : stage isentropic efficiency (0–1)
    gamma, R_gas            : thermodynamic constants
    """
    omega = rpm * 2.0 * math.pi / 60.0
    U_tip = omega * tip_radius
    mean_radius = 0.5 * (hub_radius + tip_radius)
    annulus_area = math.pi * (tip_radius ** 2 - hub_radius ** 2)
    rho_in = inlet_total_pressure / (R_gas * inlet_total_temperature)
    u_z_in = mass_flow_rate / (rho_in * annulus_area)

    T_t_out = inlet_total_temperature * _polytropic_temp_ratio(
        pressure_ratio_stage, gamma, isentropic_efficiency
    )
    p_t_out = inlet_total_pressure * pressure_ratio_stage

    coords = ("r", "theta", "z")
    fields = ("rho", "u_r", "u_theta", "u_z", "p", "T")

    pde = PDETermSpec(
        kind="compressible_euler_rotating_3d",
        fields=fields,
        coords=coords,
        params={
            "gamma": gamma,
            "R_gas": R_gas,
            "omega": omega,
            "mean_radius": mean_radius,
            "U_tip": U_tip,
        },
    )

    _rho_in = float(rho_in)
    _u_z_in = float(u_z_in)
    _T_in = float(inlet_total_temperature)
    _p_t_out = float(p_t_out)
    _hub_r = float(hub_radius)
    _tip_r = float(tip_radius)
    _ax_len = float(axial_length)

    inlet_bc = DirichletBC(
        "stage_inlet",
        fields=("rho", "u_z", "T"),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 2], 0.0),
        value_fn=lambda X, ctx: np.column_stack([
            np.full(X.shape[0], _rho_in, dtype=np.float32),
            np.full(X.shape[0], _u_z_in, dtype=np.float32),
            np.full(X.shape[0], _T_in, dtype=np.float32),
        ]),
        weight=10.0,
    )
    outlet_bc = DirichletBC(
        "stage_outlet",
        fields=("p",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 2], _ax_len),
        value_fn=lambda X, ctx: np.full((X.shape[0], 1), _p_t_out, dtype=np.float32),
        weight=5.0,
    )
    hub_wall = DirichletBC(
        "hub_wall",
        fields=("u_r",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 0], _hub_r, atol=1e-4),
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=10.0,
    )
    tip_wall = DirichletBC(
        "tip_wall",
        fields=("u_r",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 0], _tip_r, atol=1e-4),
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=10.0,
    )

    return ProblemSpec(
        name="axial_compressor_stage_3d",
        dim=3,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(inlet_bc, outlet_bc, hub_wall, tip_wall),
        domain_bounds={
            "r": (hub_radius, tip_radius),
            "theta": (0.0, 2.0 * math.pi),
            "z": (0.0, axial_length),
        },
        field_ranges={
            "rho": (0.5 * rho_in, 3.0 * rho_in),
            "u_r": (-50.0, 50.0),
            "u_theta": (-U_tip, U_tip),
            "u_z": (0.0, 350.0),
            "p": (0.5 * inlet_total_pressure, p_t_out * 1.2),
            "T": (0.8 * inlet_total_temperature, T_t_out * 1.1),
        },
        scales=ScaleSpec(L=tip_radius, U=U_tip),
        sample_defaults={"n_col": 200_000, "n_bc": 20_000},
        solver_spec={
            "name": "turbodesigner",
            "machine_type": "axial",
            "configuration": "compressor",
        },
        meta={
            "description": "3D single axial compressor stage — rotating-frame Euler PINN",
            "rpm": rpm,
            "pressure_ratio_stage": pressure_ratio_stage,
            "hub_radius": hub_radius,
            "tip_radius": tip_radius,
            "mean_radius": mean_radius,
            "U_tip_m_s": U_tip,
            "T_t_outlet_K": T_t_out,
            "p_t_outlet_Pa": p_t_out,
            "digital_twin_fields": ["p", "T", "u_z", "u_theta"],
            "turbodesigner_definition": {
                "gamma": gamma,
                "rpm": rpm,
                "gas_constant": R_gas,
                "mass_flow_rate": mass_flow_rate,
                "pressure_ratio": pressure_ratio_stage,
                "inlet_total_pressure": inlet_total_pressure,
                "inlet_total_temperature": inlet_total_temperature,
                "isentropic_efficiency": isentropic_efficiency,
                "num_stages": 1,
                "hub_to_tip_ratio": hub_radius / tip_radius,
            },
        },
    )
