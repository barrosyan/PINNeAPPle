"""Engineering problem presets.

Covers multi-physics problems in aerospace, automotive, PC thermal management,
industrial furnaces, refractories, and datacenter cooling.

Each preset returns a ``ProblemSpec`` ready for use with the pinneapple
training pipeline, data generation, and digital twin modules.

Named regions (inlet, outlet, wall, ...) are resolved via ``selector_type="tag"``
conditions — the caller's geometry/mesh pipeline is expected to supply
``ctx["tag_masks"]`` for each tag referenced below.

Domains
-------
Aerospace
  - rocket_nozzle_cfd           : 2D axisymmetric compressible flow (nozzle)
  - rocket_structural           : rocket casing under internal pressure + thermal
  - aircraft_wing_aerodynamics  : 2D transonic RANS-simplified airfoil
  - aircraft_wing_structural    : composite wing spar (plane stress)

Automotive
  - car_external_aero           : external 2D car body aerodynamics
  - car_brake_thermal           : transient brake disc heating
  - car_suspension_fatigue      : cyclic stress in suspension wishbone

PC / Electronics Cooling
  - cpu_heatsink_thermal        : 3D conduction/convection in CPU heatsink
  - pcb_thermal                 : 2D PCB heat spreading with component hotspots
  - fan_cooler_cfd              : radial fan flow (2D)

Industrial Furnace / Refractory
  - industrial_furnace_thermal  : 3D steady-state furnace wall conduction
  - refractory_lining           : refractory wall with multi-layer conductivity
  - furnace_combustion_zone     : simplified combustion + heat release

Datacenter
  - datacenter_airflow_2d       : server rack row cooling (2D channel flow)
  - datacenter_server_thermal   : server board conduction/convection
  - datacenter_cfd_3d           : simplified 3D hot-aisle/cold-aisle
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from ..spec import PDETermSpec, ProblemSpec
from ..conditions import DirichletBC, NeumannBC, InitialCondition
from ..scales import ScaleSpec
from ..environment_typing import CoordNames
from .registry import register_preset
from ._utils import _lame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reynolds(rho: float, U: float, L: float, mu: float) -> float:
    return rho * U * L / mu


def _const_value_fn(values: Dict[str, float]):
    """Return a value_fn emitting the constant values (in dict order) for every selected point."""
    vals = [float(v) for v in values.values()]

    def _fn(X: np.ndarray, ctx: Dict[str, Any], _vals=vals) -> np.ndarray:
        out = np.zeros((X.shape[0], len(_vals)), dtype=np.float32)
        for i, v in enumerate(_vals):
            out[:, i] = v
        return out

    return _fn


def _tagged_dirichlet(tag: str, values: Dict[str, float], weight: float = 10.0):
    """Dirichlet condition applied to points carrying ``tag`` in ctx["tag_masks"]."""
    return DirichletBC(
        name=tag,
        fields=tuple(values.keys()),
        selector_type="tag",
        selector={"tag": tag},
        value_fn=_const_value_fn(values),
        weight=weight,
    )


def _tagged_neumann(tag: str, values: Dict[str, float], weight: float = 5.0):
    """Neumann condition applied to points carrying ``tag`` in ctx["tag_masks"]."""
    return NeumannBC(
        name=tag,
        fields=tuple(values.keys()),
        selector_type="tag",
        selector={"tag": tag},
        value_fn=_const_value_fn(values),
        weight=weight,
    )


# ===========================================================================
# AEROSPACE
# ===========================================================================

@register_preset("rocket_nozzle_cfd")
def rocket_nozzle_cfd(
    gamma: float = 1.4,
    R_gas: float = 287.0,
    T_inlet: float = 3500.0,
    p_inlet: float = 10e6,
    throat_radius: float = 0.1,
    exit_radius: float = 0.2,
    nozzle_length: float = 0.5,
) -> ProblemSpec:
    """
    2D axisymmetric compressible flow through a convergent-divergent rocket nozzle.

    PDE: Euler equations (compressible, inviscid) in axisymmetric coordinates.
    Fields: rho (density), u_axial, u_radial, p (pressure), T (temperature).

    Parameters
    ----------
    gamma       : specific heat ratio (default 1.4 for air; ~1.2-1.3 for combustion gas)
    R_gas       : gas constant J/(kg·K)
    T_inlet     : stagnation temperature at inlet (K)
    p_inlet     : stagnation pressure at inlet (Pa)
    throat_radius : nozzle throat radius (m)
    exit_radius   : nozzle exit radius (m)
    nozzle_length : axial length of nozzle (m)
    """
    coords: CoordNames = ("r", "z")
    fields = ("rho", "u", "v", "p", "T")

    a_exit = math.pi * exit_radius ** 2
    a_throat = math.pi * throat_radius ** 2
    area_ratio = a_exit / a_throat

    pde = PDETermSpec(
        kind="compressible_euler_axisymmetric",
        fields=fields,
        coords=coords,
        params={
            "gamma": gamma,
            "R_gas": R_gas,
            "area_ratio": area_ratio,
        },
    )

    p_exit = p_inlet * (2 / (gamma + 1)) ** (gamma / (gamma - 1))

    conditions = (
        _tagged_dirichlet("inlet", {"p": p_inlet, "T": T_inlet}, weight=20.0),
        _tagged_neumann("outlet", {"p": 0.0}, weight=5.0),
        _tagged_dirichlet("wall", {"u": 0.0, "v": 0.0}, weight=20.0),  # no-slip on wall
        _tagged_neumann("axis", {"v": 0.0}, weight=5.0),               # symmetry axis: zero radial vel
    )

    return ProblemSpec(
        name="rocket_nozzle_cfd",
        problem_id="rocket_nozzle_cfd",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=conditions,
        domain_bounds={"r": (0.0, exit_radius * 1.1), "z": (0.0, nozzle_length)},
        solver_spec={
            "name": "openfoam",
            "solver": "rhoCentralFoam",
            "mesh": "structured",
        },
        scales=ScaleSpec(L=nozzle_length, U=math.sqrt(gamma * R_gas * T_inlet)),
        meta={
            "description": "Compressible axisymmetric rocket nozzle flow",
            "throat_radius": throat_radius,
            "exit_radius": exit_radius,
            "area_ratio": area_ratio,
            "p_exit_estimate": p_exit,
            "digital_twin_fields": ["p", "T", "u"],
        },
    )


@register_preset("rocket_structural")
def rocket_structural(
    E: float = 200e9,
    nu: float = 0.3,
    p_internal: float = 10e6,
    T_inner: float = 800.0,
    T_outer: float = 293.0,
    alpha_T: float = 12e-6,
    inner_radius: float = 0.2,
    outer_radius: float = 0.22,
) -> ProblemSpec:
    """
    Rocket motor casing under combined internal pressure + thermal gradient.

    Thin-walled cylindrical shell (plane strain 2D cross-section).
    PDE: thermoelasticity (linear).
    Fields: ux, uy, T.
    """
    coords: CoordNames = ("x", "y")
    fields = ("ux", "uy", "T")
    lam, mu = _lame(E, nu)

    pde = PDETermSpec(
        kind="thermoelasticity_2d",
        fields=fields,
        coords=coords,
        params={"E": E, "nu": nu, "alpha_T": alpha_T, "lam": lam, "mu": mu},
    )

    conditions = (
        _tagged_neumann("inner_wall", {"p_normal": p_internal}, weight=10.0),
        _tagged_dirichlet("outer_wall", {"ux": 0.0, "uy": 0.0}, weight=20.0),
        _tagged_dirichlet("T_inner", {"T": T_inner}, weight=20.0),
        _tagged_dirichlet("T_outer", {"T": T_outer}, weight=20.0),
    )

    return ProblemSpec(
        name="rocket_structural",
        problem_id="rocket_structural",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=conditions,
        domain_bounds={"x": (-outer_radius, outer_radius), "y": (-outer_radius, outer_radius)},
        solver_spec={"name": "fenics", "formulation": "thermoelasticity_plane_strain"},
        meta={
            "description": "Rocket casing thermoelasticity",
            "von_mises_formula": "sqrt(sxx^2 - sxx*syy + syy^2 + 3*sxy^2)",
            "inner_radius": inner_radius,
            "outer_radius": outer_radius,
            "digital_twin_fields": ["T", "ux", "uy"],
        },
    )


@register_preset("aircraft_wing_aerodynamics")
def aircraft_wing_aerodynamics(
    Re: float = 5e6,
    Ma: float = 0.3,
    alpha_deg: float = 5.0,
    chord: float = 1.0,
    rho_inf: float = 1.225,
    U_inf: float = 102.0,
    nu_air: float = 1.5e-5,
) -> ProblemSpec:
    """
    2D incompressible/low-Ma airfoil aerodynamics (NACA-like profile).

    Simplified RANS: steady incompressible Navier-Stokes with turbulent
    viscosity constant (nu_t) as additional parameter.

    Fields: u, v, p
    Regions: farfield, airfoil_surface, wake_outlet
    """
    coords: CoordNames = ("x", "y")
    fields = ("u", "v", "p")

    pde = PDETermSpec(
        kind="incompressible_navier_stokes_2d",
        fields=fields,
        coords=coords,
        params={"nu": nu_air, "Re": Re},
    )

    conditions = (
        _tagged_dirichlet("farfield_inlet", {
            "u": U_inf * math.cos(math.radians(alpha_deg)),
            "v": U_inf * math.sin(math.radians(alpha_deg)),
        }, weight=20.0),
        _tagged_neumann("farfield_outlet", {"p": 0.0}, weight=5.0),
        _tagged_dirichlet("airfoil", {"u": 0.0, "v": 0.0}, weight=20.0),   # no-slip
        _tagged_neumann("wake_outlet", {"u": 0.0, "v": 0.0}, weight=5.0),
    )

    return ProblemSpec(
        name="aircraft_wing_aerodynamics",
        problem_id="aircraft_wing_aerodynamics",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=conditions,
        domain_bounds={"x": (-5 * chord, 15 * chord), "y": (-5 * chord, 5 * chord)},
        solver_spec={"name": "openfoam", "solver": "simpleFoam", "turbulence": "kOmegaSST"},
        scales=ScaleSpec(L=chord, U=U_inf),
        meta={
            "description": "2D airfoil aerodynamics",
            "alpha_deg": alpha_deg,
            "Ma": Ma,
            "Re": Re,
            "digital_twin_fields": ["u", "v", "p"],
        },
    )


@register_preset("aircraft_wing_structural")
def aircraft_wing_structural(
    E: float = 70e9,
    nu: float = 0.33,
    lift_load: float = 50000.0,
    span: float = 5.0,
    thickness: float = 0.1,
) -> ProblemSpec:
    """
    2D plane stress model of an aircraft wing spar under lift bending load.

    Material: Aluminium alloy (default E=70 GPa, nu=0.33).
    Fields: ux (spanwise displacement), uy (vertical displacement).
    """
    coords: CoordNames = ("x", "y")
    fields = ("ux", "uy")
    lam, mu = _lame(E, nu)

    pde = PDETermSpec(
        kind="linear_elasticity_plane_stress",
        fields=fields,
        coords=coords,
        params={"E": E, "nu": nu, "lam": lam, "mu": mu},
    )

    conditions = (
        _tagged_dirichlet("root_fixed", {"ux": 0.0, "uy": 0.0}, weight=20.0),
        _tagged_neumann("tip_load", {"ty": -lift_load}, weight=10.0),
        _tagged_neumann("free_surface", {"tx": 0.0, "ty": 0.0}, weight=5.0),
    )

    return ProblemSpec(
        name="aircraft_wing_structural",
        problem_id="aircraft_wing_structural",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=conditions,
        domain_bounds={"x": (0.0, span), "y": (-thickness / 2, thickness / 2)},
        solver_spec={"name": "fenics", "formulation": "plane_stress"},
        meta={
            "description": "Aircraft wing spar plane stress",
            "von_mises_formula": "sqrt(sxx^2 - sxx*syy + syy^2 + 3*sxy^2)",
            "sigma_y_aluminium": 276e6,
            "digital_twin_fields": ["ux", "uy"],
        },
    )


# ===========================================================================
# AUTOMOTIVE
# ===========================================================================

@register_preset("car_external_aero")
def car_external_aero(
    U_inf: float = 33.3,       # ~120 km/h
    rho: float = 1.225,
    nu_air: float = 1.5e-5,
    car_length: float = 4.5,
    car_height: float = 1.4,
) -> ProblemSpec:
    """
    2D external aerodynamics around a simplified car body.

    Steady incompressible Navier-Stokes.  The car is represented as a
    bluff body; the drag coefficient Cd and lift Cl can be derived from the
    pressure/velocity fields.

    Fields: u, v, p
    """
    coords: CoordNames = ("x", "y")
    fields = ("u", "v", "p")

    Re = _reynolds(rho, U_inf, car_length, nu_air * rho)

    pde = PDETermSpec(
        kind="incompressible_navier_stokes_2d",
        fields=fields,
        coords=coords,
        params={"nu": nu_air, "Re": Re},
    )

    conditions = (
        _tagged_dirichlet("inlet", {"u": U_inf, "v": 0.0}, weight=20.0),
        _tagged_neumann("outlet", {"p": 0.0}, weight=5.0),
        _tagged_dirichlet("ground", {"u": U_inf, "v": 0.0}, weight=20.0),  # moving ground
        _tagged_dirichlet("car_body", {"u": 0.0, "v": 0.0}, weight=20.0),
        _tagged_neumann("top", {"v": 0.0}, weight=5.0),
    )

    return ProblemSpec(
        name="car_external_aero",
        problem_id="car_external_aero",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=conditions,
        domain_bounds={"x": (-2 * car_length, 5 * car_length), "y": (0.0, 5 * car_height)},
        solver_spec={"name": "openfoam", "solver": "simpleFoam", "turbulence": "kEpsilon"},
        scales=ScaleSpec(L=car_length, U=U_inf),
        meta={
            "description": "2D car external aerodynamics",
            "Re": Re,
            "U_inf_kmh": U_inf * 3.6,
            "digital_twin_fields": ["u", "v", "p"],
        },
    )


@register_preset("car_brake_thermal")
def car_brake_thermal(
    k_disc: float = 55.0,       # W/(m·K) cast iron
    rho_disc: float = 7100.0,
    cp_disc: float = 500.0,
    q_friction: float = 2e6,    # surface heat flux W/m²
    T_ambient: float = 293.0,
    h_conv: float = 80.0,       # convection coefficient W/(m²·K)
    disc_radius: float = 0.16,
    disc_thickness: float = 0.025,
    t_braking: float = 5.0,     # braking duration (s)
) -> ProblemSpec:
    """
    Transient heat conduction in a brake disc during emergency braking.

    PDE: Fourier heat equation ρ cp ∂T/∂t = ∇·(k ∇T) + q
    Fields: T (temperature)
    Regions: friction_surface (heat flux), cooling_surface (convection)
    """
    coords: CoordNames = ("r", "z", "t")
    fields = ("T",)

    alpha = k_disc / (rho_disc * cp_disc)

    pde = PDETermSpec(
        kind="heat_equation_transient",
        fields=fields,
        coords=coords,
        params={
            "k": k_disc,
            "rho": rho_disc,
            "cp": cp_disc,
            "alpha": alpha,
        },
    )

    initial = InitialCondition(
        name="initial",
        fields=("T",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 2], 0.0),
        value_fn=lambda X, ctx, _t=float(T_ambient): np.full((X.shape[0], 1), _t, dtype=np.float32),
        weight=10.0,
    )

    conditions = (
        _tagged_neumann("friction_surface", {"q_heat": q_friction}, weight=10.0),
        _tagged_neumann("cooling_surface", {"h": h_conv, "T_ref": T_ambient}, weight=5.0),
        initial,
    )

    return ProblemSpec(
        name="car_brake_thermal",
        problem_id="car_brake_thermal",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=conditions,
        domain_bounds={
            "r": (0.0, disc_radius),
            "z": (0.0, disc_thickness),
            "t": (0.0, t_braking),
        },
        solver_spec={"name": "openfoam", "solver": "chtMultiRegionFoam"},
        meta={
            "description": "Transient brake disc thermal analysis",
            "alpha_m2_s": alpha,
            "T_max_estimate": T_ambient + q_friction * disc_thickness / k_disc,
            "digital_twin_fields": ["T"],
        },
    )


@register_preset("car_suspension_fatigue")
def car_suspension_fatigue(
    E: float = 210e9,
    nu: float = 0.3,
    F_vertical: float = 8000.0,
    F_lateral: float = 3000.0,
    arm_length: float = 0.35,
    arm_width: float = 0.04,
) -> ProblemSpec:
    """
    Plane stress fatigue analysis of a suspension wishbone arm.

    PDE: linear elasticity plane stress.
    Fields: ux, uy (displacements) → von Mises stress for fatigue.
    """
    coords: CoordNames = ("x", "y")
    fields = ("ux", "uy")
    lam, mu = _lame(E, nu)

    pde = PDETermSpec(
        kind="linear_elasticity_plane_stress",
        fields=fields,
        coords=coords,
        params={"E": E, "nu": nu, "lam": lam, "mu": mu},
    )

    conditions = (
        _tagged_dirichlet("mounting_fixed", {"ux": 0.0, "uy": 0.0}, weight=20.0),
        _tagged_neumann("wheel_hub_load", {"ty": -F_vertical, "tx": F_lateral}, weight=10.0),
        _tagged_neumann("free_edges", {"tx": 0.0, "ty": 0.0}, weight=5.0),
    )

    return ProblemSpec(
        name="car_suspension_fatigue",
        problem_id="car_suspension_fatigue",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=conditions,
        domain_bounds={"x": (0.0, arm_length), "y": (-arm_width / 2, arm_width / 2)},
        solver_spec={"name": "fenics", "formulation": "plane_stress"},
        meta={
            "description": "Suspension wishbone plane stress fatigue",
            "von_mises_formula": "sqrt(sxx^2 - sxx*syy + syy^2 + 3*sxy^2)",
            "sigma_y_steel": 355e6,
            "digital_twin_fields": ["ux", "uy"],
        },
    )


# ===========================================================================
# PC / ELECTRONICS COOLING
# ===========================================================================

@register_preset("cpu_heatsink_thermal")
def cpu_heatsink_thermal(
    k_aluminium: float = 205.0,    # W/(m·K)
    k_copper: float = 400.0,
    q_cpu: float = 200.0,          # CPU heat dissipation (W)
    T_ambient: float = 293.0,
    h_fin: float = 50.0,           # fin convection coefficient W/(m²·K)
    die_area: float = 1e-4,        # CPU die area m² (≈ 1 cm²)
    heatsink_length: float = 0.08,
    heatsink_height: float = 0.05,
    heatsink_width: float = 0.08,
) -> ProblemSpec:
    """
    3D steady-state heat conduction/convection in a CPU heatsink.

    CPU die injects uniform heat flux at the base; fins are cooled by
    forced convection.

    PDE: Laplace/Poisson for temperature (steady state).
    Fields: T
    """
    coords: CoordNames = ("x", "y", "z")
    fields = ("T",)

    q_flux = q_cpu / die_area  # W/m²

    pde = PDETermSpec(
        kind="heat_equation_steady",
        fields=fields,
        coords=coords,
        params={"k": k_aluminium},
    )

    conditions = (
        _tagged_neumann("cpu_base", {"q_heat": q_flux}, weight=10.0),
        _tagged_neumann("fin_surfaces", {"h": h_fin, "T_ref": T_ambient}, weight=5.0),
        _tagged_neumann("insulated_sides", {"q_heat": 0.0}, weight=5.0),
    )

    return ProblemSpec(
        name="cpu_heatsink_thermal",
        problem_id="cpu_heatsink_thermal",
        dim=3,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=conditions,
        domain_bounds={
            "x": (0.0, heatsink_length),
            "y": (0.0, heatsink_width),
            "z": (0.0, heatsink_height),
        },
        solver_spec={"name": "fenics", "formulation": "steady_heat"},
        meta={
            "description": "CPU heatsink thermal management",
            "q_flux_W_m2": q_flux,
            "T_junction_estimate": T_ambient + q_cpu * heatsink_height / (k_aluminium * die_area),
            "digital_twin_fields": ["T"],
        },
    )


@register_preset("pcb_thermal")
def pcb_thermal(
    k_pcb: float = 0.3,            # W/(m·K) — FR4 in-plane
    k_z_pcb: float = 0.25,        # W/(m·K) — FR4 through-plane
    q_components: Dict[str, float] = None,  # {component_id: power_W}
    T_ambient: float = 308.0,     # 35°C ambient
    h_natural: float = 10.0,      # natural convection
    board_length: float = 0.2,
    board_width: float = 0.15,
    board_thickness: float = 0.0016,
) -> ProblemSpec:
    """
    2D PCB heat spreading model with multiple component heat sources.

    PDE: 2D anisotropic heat equation (effective k).
    Fields: T
    Hotspot temperatures are critical for reliability.
    """
    coords: CoordNames = ("x", "y")
    fields = ("T",)

    if q_components is None:
        q_components = {"cpu": 85.0, "gpu": 60.0, "vrm": 15.0}

    total_power = sum(q_components.values())
    q_flux_avg = total_power / (board_length * board_width)

    pde = PDETermSpec(
        kind="heat_equation_steady_anisotropic",
        fields=fields,
        coords=coords,
        params={"k_x": k_pcb, "k_y": k_pcb, "k_z": k_z_pcb},
    )

    conditions = (
        _tagged_neumann("component_hotspots", {"q_heat": q_flux_avg}, weight=10.0),
        _tagged_neumann("board_surface", {"h": h_natural, "T_ref": T_ambient}, weight=5.0),
        _tagged_neumann("board_edges", {"q_heat": 0.0}, weight=5.0),
    )

    return ProblemSpec(
        name="pcb_thermal",
        problem_id="pcb_thermal",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=conditions,
        domain_bounds={"x": (0.0, board_length), "y": (0.0, board_width)},
        solver_spec={"name": "fenics", "formulation": "steady_heat_anisotropic"},
        meta={
            "description": "PCB thermal management",
            "components": q_components,
            "total_power_W": total_power,
            "q_flux_avg": q_flux_avg,
            "T_max_safe": 358.0,  # 85°C
            "digital_twin_fields": ["T"],
        },
    )


@register_preset("fan_cooler_cfd")
def fan_cooler_cfd(
    rpm: float = 2000.0,
    blade_radius: float = 0.05,
    hub_radius: float = 0.01,
    rho: float = 1.225,
    nu_air: float = 1.5e-5,
    n_blades: int = 7,
) -> ProblemSpec:
    """
    2D radial fan flow in a PC cooler.

    Rotating frame: steady incompressible NS with rotation source term.
    Fields: u_r (radial), u_theta (tangential), p.
    """
    coords: CoordNames = ("x", "y")
    fields = ("u", "v", "p")

    omega = 2 * math.pi * rpm / 60.0
    U_tip = omega * blade_radius
    Re = _reynolds(rho, U_tip, blade_radius, nu_air * rho)

    pde = PDETermSpec(
        kind="incompressible_navier_stokes_rotating_frame",
        fields=fields,
        coords=coords,
        params={"nu": nu_air, "omega": omega, "Re": Re},
    )

    conditions = (
        _tagged_dirichlet("inlet_eye", {"u": 0.0, "v": 0.0, "p": 0.0}, weight=20.0),
        _tagged_neumann("outlet_periphery", {"p": 0.0}, weight=5.0),
        _tagged_dirichlet("blade_wall", {"u": 0.0, "v": 0.0}, weight=20.0),
        _tagged_dirichlet("hub_wall", {"u": 0.0, "v": 0.0}, weight=20.0),
    )

    return ProblemSpec(
        name="fan_cooler_cfd",
        problem_id="fan_cooler_cfd",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=conditions,
        domain_bounds={"x": (-blade_radius, blade_radius), "y": (-blade_radius, blade_radius)},
        solver_spec={"name": "openfoam", "solver": "simpleFoam", "MRF": True},
        meta={
            "description": "PC fan cooler CFD",
            "omega_rad_s": omega,
            "U_tip": U_tip,
            "Re": Re,
            "n_blades": n_blades,
            "digital_twin_fields": ["u", "v", "p"],
        },
    )


# ===========================================================================
# INDUSTRIAL FURNACE / REFRACTORY
# ===========================================================================

@register_preset("industrial_furnace_thermal")
def industrial_furnace_thermal(
    T_hot_gas: float = 1600.0,    # K — combustion zone
    T_ambient: float = 300.0,
    k_refractory: float = 1.5,    # W/(m·K) — firebrick
    k_insulation: float = 0.1,    # W/(m·K) — ceramic fibre
    h_hot_gas: float = 50.0,      # convection coeff inside W/(m²·K)
    h_ambient: float = 10.0,      # external natural convection
    eps_wall: float = 0.9,        # emissivity for radiation
    sigma_SB: float = 5.67e-8,    # Stefan-Boltzmann constant
    wall_thickness: float = 0.3,  # total wall thickness (m)
    furnace_height: float = 2.0,
    furnace_width: float = 3.0,
) -> ProblemSpec:
    """
    3D steady-state heat transfer in an industrial furnace wall.

    Combined conduction + convection + radiation.
    PDE: Laplace for T (steady) with radiative BC approximated as
         nonlinear Neumann (q_rad = eps * sigma * (T^4 - T_amb^4)).

    Fields: T
    """
    coords: CoordNames = ("x", "y", "z")
    fields = ("T",)

    pde = PDETermSpec(
        kind="heat_equation_steady",
        fields=fields,
        coords=coords,
        params={"k": k_refractory},
    )

    conditions = (
        _tagged_neumann("hot_inner_surface", {
            "h": h_hot_gas,
            "T_ref": T_hot_gas,
            "radiation_eps": eps_wall,
            "radiation_sigma": sigma_SB,
        }, weight=10.0),
        _tagged_neumann("outer_surface", {"h": h_ambient, "T_ref": T_ambient}, weight=5.0),
        _tagged_neumann("insulation_interface", {"k": k_insulation}, weight=5.0),
    )

    return ProblemSpec(
        name="industrial_furnace_thermal",
        problem_id="industrial_furnace_thermal",
        dim=3,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=conditions,
        domain_bounds={
            "x": (0.0, furnace_width),
            "y": (0.0, furnace_height),
            "z": (0.0, wall_thickness),
        },
        solver_spec={"name": "openfoam", "solver": "chtMultiRegionFoam", "radiation": "P1"},
        meta={
            "description": "Industrial furnace wall heat transfer",
            "T_wall_inner_estimate": T_hot_gas - h_hot_gas * (T_hot_gas - T_ambient) * wall_thickness / k_refractory,
            "digital_twin_fields": ["T"],
            "alert_T_max": 1800.0,  # refractory limit
        },
    )


@register_preset("refractory_lining")
def refractory_lining(
    layers: List[Dict] = None,
    T_hot: float = 1700.0,
    T_cold: float = 350.0,
    total_thickness: float = 0.5,
) -> ProblemSpec:
    """
    Multi-layer refractory wall with different conductivities.

    Typical layers:
      [{"k": 1.8, "thickness": 0.1, "name": "working_lining"},
       {"k": 0.5, "thickness": 0.2, "name": "safety_lining"},
       {"k": 0.08, "thickness": 0.2, "name": "insulation"}]

    PDE: 1D (in z) or 2D steady-state conduction.
    Fields: T
    """
    coords: CoordNames = ("x", "z")
    fields = ("T",)

    if layers is None:
        layers = [
            {"k": 1.8, "thickness": 0.1, "name": "working_lining"},
            {"k": 0.5, "thickness": 0.2, "name": "safety_lining"},
            {"k": 0.08, "thickness": 0.2, "name": "insulation"},
        ]

    k_eff = total_thickness / sum(lay["thickness"] / lay["k"] for lay in layers)

    pde = PDETermSpec(
        kind="heat_equation_steady_multilayer",
        fields=fields,
        coords=coords,
        params={"layers": layers, "k_eff": k_eff},
    )

    conditions = (
        _tagged_dirichlet("hot_face", {"T": T_hot}, weight=20.0),
        _tagged_dirichlet("cold_face", {"T": T_cold}, weight=20.0),
    )

    return ProblemSpec(
        name="refractory_lining",
        problem_id="refractory_lining",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=conditions,
        domain_bounds={"x": (0.0, 1.0), "z": (0.0, total_thickness)},
        solver_spec={"name": "fenics", "formulation": "steady_heat_multilayer"},
        meta={
            "description": "Multi-layer refractory lining",
            "layers": layers,
            "k_eff": k_eff,
            "q_flux_estimate": k_eff * (T_hot - T_cold) / total_thickness,
            "digital_twin_fields": ["T"],
            "alert_layer_interface_T": {lay["name"]: None for lay in layers},
        },
    )


@register_preset("furnace_combustion_zone")
def furnace_combustion_zone(
    T_flame: float = 2200.0,
    T_wall: float = 1600.0,
    rho_flue: float = 0.24,       # kg/m³ at high temperature
    nu_flue: float = 2e-4,        # kinematic viscosity m²/s
    U_inlet: float = 5.0,         # flue gas inlet velocity m/s
    furnace_length: float = 6.0,
    furnace_height: float = 2.0,
    Q_combustion: float = 5e6,    # volumetric heat release W/m³
) -> ProblemSpec:
    """
    Simplified combustion zone: hot gas flow with volumetric heat release.

    PDE: Navier-Stokes + energy equation with source term Q.
    Fields: u, v, p, T
    """
    coords: CoordNames = ("x", "y")
    fields = ("u", "v", "p", "T")

    pde = PDETermSpec(
        kind="navier_stokes_energy_2d",
        fields=fields,
        coords=coords,
        params={
            "nu": nu_flue,
            "rho": rho_flue,
            "Q_source": Q_combustion,
        },
    )

    conditions = (
        _tagged_dirichlet("fuel_inlet", {"u": U_inlet, "v": 0.0, "T": T_flame}, weight=20.0),
        _tagged_neumann("flue_outlet", {"p": 0.0}, weight=5.0),
        _tagged_dirichlet("refractory_wall", {"u": 0.0, "v": 0.0, "T": T_wall}, weight=20.0),
    )

    return ProblemSpec(
        name="furnace_combustion_zone",
        problem_id="furnace_combustion_zone",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=conditions,
        domain_bounds={"x": (0.0, furnace_length), "y": (0.0, furnace_height)},
        solver_spec={"name": "openfoam", "solver": "buoyantSimpleFoam", "radiation": "P1"},
        meta={
            "description": "Furnace combustion zone flow and heat",
            "digital_twin_fields": ["T", "u", "p"],
        },
    )


# ===========================================================================
# DATACENTER
# ===========================================================================

@register_preset("datacenter_airflow_2d")
def datacenter_airflow_2d(
    U_cold_aisle: float = 2.5,    # m/s cold aisle inlet velocity
    T_cold_air: float = 291.0,    # 18°C supply
    T_hot_max: float = 318.0,     # 45°C hot aisle limit
    Q_rack: float = 20000.0,      # W per rack
    rack_height: float = 2.0,
    rack_depth: float = 1.0,
    aisle_width: float = 1.2,
    n_racks: int = 10,
    rho_air: float = 1.2,
    nu_air: float = 1.5e-5,
    cp_air: float = 1006.0,       # J/(kg·K)
) -> ProblemSpec:
    """
    2D server rack row cooling (cold-aisle/hot-aisle containment).

    Airflow channel between racks modelled as 2D NS + energy equation.
    Fields: u, v, p, T

    This is the canonical datacenter digital twin problem.
    """
    coords: CoordNames = ("x", "y")
    fields = ("u", "v", "p", "T")

    Re = _reynolds(rho_air, U_cold_aisle, aisle_width, nu_air * rho_air)
    q_flux = Q_rack / (rack_height * rack_depth)

    pde = PDETermSpec(
        kind="incompressible_navier_stokes_energy_2d",
        fields=fields,
        coords=coords,
        params={
            "nu": nu_air,
            "rho": rho_air,
            "cp": cp_air,
            "Re": Re,
        },
    )

    conditions = (
        _tagged_dirichlet("cold_aisle_inlet", {"u": U_cold_aisle, "v": 0.0, "T": T_cold_air}, weight=20.0),
        _tagged_neumann("hot_aisle_outlet", {"p": 0.0}, weight=5.0),
        _tagged_neumann("server_surfaces", {"q_heat": q_flux}, weight=10.0),
        _tagged_neumann("floor_ceiling", {"u": 0.0, "v": 0.0}, weight=5.0),
    )

    return ProblemSpec(
        name="datacenter_airflow_2d",
        problem_id="datacenter_airflow_2d",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=conditions,
        domain_bounds={"x": (0.0, n_racks * (rack_depth + aisle_width)), "y": (0.0, rack_height)},
        solver_spec={"name": "openfoam", "solver": "buoyantSimpleFoam"},
        meta={
            "description": "Datacenter hot-aisle/cold-aisle cooling",
            "Re": Re,
            "n_racks": n_racks,
            "total_IT_load_kW": Q_rack * n_racks / 1000,
            "PUE_target": 1.3,
            "digital_twin_fields": ["T", "u", "v", "p"],
            "alert_T_max": T_hot_max,
        },
    )


@register_preset("datacenter_server_thermal")
def datacenter_server_thermal(
    Q_cpu: float = 250.0,
    Q_gpu: float = 400.0,
    Q_ram: float = 20.0,
    T_inlet_air: float = 295.0,
    U_fan: float = 3.0,
    h_forced: float = 120.0,      # forced convection coeff
    k_board: float = 0.3,
    server_length: float = 0.6,
    server_width: float = 0.45,
) -> ProblemSpec:
    """
    2D server board conduction + forced convection from fans.

    Models the thermal distribution across a server 1U board.
    Fields: T (temperature map of the board).
    """
    coords: CoordNames = ("x", "y")
    fields = ("T",)

    total_Q = Q_cpu + Q_gpu + Q_ram
    q_avg = total_Q / (server_length * server_width)

    pde = PDETermSpec(
        kind="heat_equation_steady",
        fields=fields,
        coords=coords,
        params={"k": k_board},
    )

    conditions = (
        _tagged_neumann("cpu_zone", {"q_heat": Q_cpu / (0.04 * 0.04)}, weight=10.0),   # ~4cm die
        _tagged_neumann("gpu_zone", {"q_heat": Q_gpu / (0.06 * 0.06)}, weight=10.0),
        _tagged_neumann("ram_zone", {"q_heat": Q_ram / (0.1 * 0.01)}, weight=10.0),
        _tagged_neumann("board_surface", {"h": h_forced, "T_ref": T_inlet_air}, weight=5.0),
        _tagged_neumann("board_edges", {"q_heat": 0.0}, weight=5.0),
    )

    return ProblemSpec(
        name="datacenter_server_thermal",
        problem_id="datacenter_server_thermal",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=conditions,
        domain_bounds={"x": (0.0, server_length), "y": (0.0, server_width)},
        solver_spec={"name": "fenics", "formulation": "steady_heat"},
        meta={
            "description": "Server board thermal map",
            "total_power_W": total_Q,
            "T_cpu_max_safe": 368.0,   # 95°C
            "T_gpu_max_safe": 373.0,   # 100°C
            "digital_twin_fields": ["T"],
            "alert_T_max": 368.0,
        },
    )


@register_preset("datacenter_cfd_3d")
def datacenter_cfd_3d(
    room_length: float = 20.0,
    room_width: float = 10.0,
    room_height: float = 3.0,
    n_racks: int = 20,
    Q_per_rack: float = 15000.0,
    T_supply: float = 291.0,
    U_supply: float = 1.5,
    rho_air: float = 1.2,
    nu_air: float = 1.5e-5,
    cp_air: float = 1006.0,
) -> ProblemSpec:
    """
    Simplified 3D datacenter room CFD for hot-spot prediction.

    Large-scale problem — typically solved with OpenFOAM.
    Fields: u, v, w, p, T
    """
    coords: CoordNames = ("x", "y", "z")
    fields = ("u", "v", "w", "p", "T")

    total_load = Q_per_rack * n_racks
    Re = _reynolds(rho_air, U_supply, room_height, nu_air * rho_air)

    pde = PDETermSpec(
        kind="incompressible_navier_stokes_energy_3d",
        fields=fields,
        coords=coords,
        params={"nu": nu_air, "rho": rho_air, "cp": cp_air, "Re": Re},
    )

    conditions = (
        _tagged_dirichlet("crac_supply", {"u": 0.0, "v": U_supply, "w": 0.0, "T": T_supply}, weight=20.0),
        _tagged_neumann("return_air", {"p": 0.0}, weight=5.0),
        _tagged_neumann("rack_surfaces", {"q_heat": Q_per_rack / (2.0 * 0.6 * 2.0)}, weight=10.0),
        _tagged_dirichlet("room_walls", {"u": 0.0, "v": 0.0, "w": 0.0}, weight=20.0),
    )

    return ProblemSpec(
        name="datacenter_cfd_3d",
        problem_id="datacenter_cfd_3d",
        dim=3,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=conditions,
        domain_bounds={
            "x": (0.0, room_length),
            "y": (0.0, room_height),
            "z": (0.0, room_width),
        },
        solver_spec={"name": "openfoam", "solver": "buoyantSimpleFoam", "turbulence": "kEpsilon"},
        scales=ScaleSpec(L=room_height, U=U_supply),
        meta={
            "description": "3D datacenter room CFD",
            "total_IT_load_kW": total_load / 1000,
            "Re": Re,
            "digital_twin_fields": ["T", "u", "v", "w", "p"],
            "alert_T_max": 318.0,   # 45°C hot aisle limit
        },
    )
