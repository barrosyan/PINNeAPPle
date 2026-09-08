"""Phenomenon -> governing-equation -> typical-parameters -> assumptions
knowledge base for problem design.

Every entry below is *sourced* (not invented) from an actual, working
preset in ``pinneapple_physics.pde_environment.presets`` -- its governing
equation, default/typical parameter values, and assumptions are transcribed
from that preset function's own docstring, ``PDETermSpec.meta``, and default
keyword arguments. ``preset_module`` / ``preset_function`` point back at the
real preset so every entry is traceable and independently verifiable by
reading the cited module.

This is deliberately a representative subset (not an exhaustive catalog of
every preset in the repo) spanning four physics domains: CFD, general
engineering (thermal/structural), solid mechanics, and astrophysics.

Usage
-----
>>> from pinneapple_problemdesign.knowledge.physics_knowledge import lookup_phenomenon
>>> hits = lookup_phenomenon("poiseuille")
>>> hits[0].governing_equation
'...'
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class PhenomenonEntry:
    """One physics-knowledge record, traceable back to a real preset.

    Attributes
    ----------
    phenomenon:
        Short human name of the physical scenario.
    governing_equation:
        The PDE/ODE (or algebraic closed-form relation) actually used by
        the source preset, transcribed from its docstring / ``meta``.
    typical_parameters:
        Mapping of parameter name -> typical value/range with units, taken
        from the source preset function's default keyword arguments or
        docstring.
    assumptions:
        Modeling assumptions / validity notes documented in the source
        preset (Newtonian fluid, small-strain, weak-field, etc.).
    preset_module:
        Dotted, importable module path of the real preset this entry was
        sourced from.
    preset_function:
        Name of the preset factory function within ``preset_module``
        (also its ``@register_preset`` key, where registered).
    """

    phenomenon: str
    governing_equation: str
    typical_parameters: Dict[str, str]
    assumptions: Tuple[str, ...]
    preset_module: str
    preset_function: str


_CFD = "pinneapple_physics.pde_environment.presets.cfd"
_ENGINEERING = "pinneapple_physics.pde_environment.presets.engineering"
_SOLID_MECHANICS = "pinneapple_physics.pde_environment.presets.solid_mechanics"
_ASTROPHYSICS = "pinneapple_physics.pde_environment.presets.astrophysics"


PHENOMENON_KNOWLEDGE_BASE: Tuple[PhenomenonEntry, ...] = (
    # ------------------------------------------------------------------
    # CFD  (pinneapple_physics/pde_environment/presets/cfd.py)
    # ------------------------------------------------------------------
    PhenomenonEntry(
        phenomenon="Incompressible viscous channel flow (2D)",
        governing_equation=(
            "Incompressible Navier-Stokes: div(u)=0; "
            "du/dt + (u.grad)u = -grad(p)/rho + nu*lap(u)  "
            "(PDETermSpec.kind='navier_stokes_incompressible')"
        ),
        typical_parameters={
            "Re": "100.0 (default channel Reynolds number)",
            "Umax": "1.0 (characteristic inlet centerline velocity)",
        },
        assumptions=(
            "Newtonian, incompressible fluid",
            "No-slip walls (Dirichlet u=v=0)",
            "Parabolic (Poiseuille-like) inlet velocity profile",
            "Outlet: dp/dn=0 (Neumann), more stable than hard outlet pressure",
        ),
        preset_module=_CFD,
        preset_function="ns_incompressible_2d_default",
    ),
    PhenomenonEntry(
        phenomenon="Incompressible viscous duct flow (3D)",
        governing_equation=(
            "Incompressible Navier-Stokes in 3D (u,v,w,p); "
            "PDETermSpec.kind='navier_stokes_incompressible'"
        ),
        typical_parameters={
            "Re": "100.0 (default duct Reynolds number)",
            "Umax": "1.0 (centerline velocity)",
        },
        assumptions=(
            "Newtonian, incompressible fluid",
            "No-slip on all duct walls",
            "Poiseuille-like separable inlet profile over (y,z)",
            "Outlet: dp/dn=0",
        ),
        preset_module=_CFD,
        preset_function="ns_incompressible_3d_default",
    ),
    PhenomenonEntry(
        phenomenon="Lid-driven cavity flow (3D benchmark)",
        governing_equation=(
            "Steady incompressible Navier-Stokes, PDETermSpec.kind="
            "'navier_stokes_incompressible'"
        ),
        typical_parameters={
            "Re": "100.0 (= U_lid * size / nu)",
            "size": "1.0 (unit cube side length)",
            "lid_velocity": "1.0 (moving-lid speed)",
        },
        assumptions=(
            "Steady state",
            "Lid face moves with prescribed tangential velocity, no-slip",
            "Remaining five faces are stationary no-slip walls",
            "Standard benchmark: Ghia et al. 1982 (3D extension)",
        ),
        preset_module=_CFD,
        preset_function="lid_driven_cavity_3d",
    ),
    PhenomenonEntry(
        phenomenon="Pressure-driven rectangular channel (Poiseuille) flow",
        governing_equation="Incompressible Navier-Stokes, 3D rectangular duct",
        typical_parameters={
            "Re": "100.0",
            "inlet_profile": "u = 16*Umax*y*(H-y)*z*(W-z)/(H^2*W^2)",
        },
        assumptions=(
            "Fully-developed Poiseuille inlet profile for a rectangular duct",
            "No-slip on the four lateral walls",
            "Outlet dp/dn=0",
        ),
        preset_module=_CFD,
        preset_function="channel_flow_3d",
    ),
    PhenomenonEntry(
        phenomenon="Hagen-Poiseuille pipe flow (circular cylinder)",
        governing_equation=(
            "Incompressible Navier-Stokes in a cylindrical pipe; exact "
            "laminar profile u = 2*Umax*(1 - r^2/R^2)"
        ),
        typical_parameters={
            "Re": "100.0 (= 2*R*Umax/nu)",
            "radius": "0.5 (pipe radius R)",
        },
        assumptions=(
            "Fully-developed laminar (Hagen-Poiseuille) inlet profile",
            "No-slip on the cylindrical wall",
            "Outlet dp/dn=0",
        ),
        preset_module=_CFD,
        preset_function="pipe_flow_3d",
    ),
    # ------------------------------------------------------------------
    # Engineering (pinneapple_physics/pde_environment/presets/engineering.py)
    # ------------------------------------------------------------------
    PhenomenonEntry(
        phenomenon="Compressible convergent-divergent nozzle flow (rocket)",
        governing_equation=(
            "Compressible Euler equations (inviscid), axisymmetric "
            "(PDETermSpec.kind='compressible_euler_axisymmetric')"
        ),
        typical_parameters={
            "gamma": "1.4 (air default; ~1.2-1.3 for combustion gas)",
            "R_gas": "287.0 J/(kg*K)",
            "T_inlet": "3500.0 K (stagnation temperature)",
            "p_inlet": "10e6 Pa (stagnation pressure)",
        },
        assumptions=(
            "Inviscid (Euler, no viscous wall boundary layer)",
            "Axisymmetric geometry",
            "No-slip enforced only weakly on 'wall' tag; symmetry axis dv/dn=0",
        ),
        preset_module=_ENGINEERING,
        preset_function="rocket_nozzle_cfd",
    ),
    PhenomenonEntry(
        phenomenon="Rocket motor casing under pressure + thermal gradient",
        governing_equation=(
            "Linear thermoelasticity, plane strain "
            "(PDETermSpec.kind='thermoelasticity_2d')"
        ),
        typical_parameters={
            "E": "200e9 Pa",
            "nu": "0.3",
            "alpha_T": "12e-6 1/K (thermal expansion coefficient)",
            "T_inner/T_outer": "800.0 K / 293.0 K",
        },
        assumptions=(
            "Thin-walled cylindrical shell reduced to a 2D cross-section",
            "Linear (small-strain) thermoelasticity",
            "Von Mises stress: sqrt(sxx^2 - sxx*syy + syy^2 + 3*sxy^2)",
        ),
        preset_module=_ENGINEERING,
        preset_function="rocket_structural",
    ),
    PhenomenonEntry(
        phenomenon="2D airfoil aerodynamics (low-Mach, RANS-simplified)",
        governing_equation=(
            "Steady incompressible Navier-Stokes with constant turbulent "
            "viscosity augmentation (PDETermSpec.kind="
            "'incompressible_navier_stokes_2d')"
        ),
        typical_parameters={
            "Re": "5e6",
            "Ma": "0.3",
            "alpha_deg": "5.0 (angle of attack)",
        },
        assumptions=(
            "Low-Mach / near-incompressible flow",
            "No-slip on airfoil surface",
            "Farfield Dirichlet inlet, farfield/wake Neumann outlet",
        ),
        preset_module=_ENGINEERING,
        preset_function="aircraft_wing_aerodynamics",
    ),
    PhenomenonEntry(
        phenomenon="Aircraft wing spar under lift bending load",
        governing_equation=(
            "Linear elasticity, plane stress "
            "(PDETermSpec.kind='linear_elasticity_plane_stress')"
        ),
        typical_parameters={
            "E": "70e9 Pa (aluminium alloy)",
            "nu": "0.33",
            "sigma_y_aluminium": "276e6 Pa",
        },
        assumptions=(
            "Plane stress (thin spar cross-section)",
            "Linear elastic material",
            "Root fixed (Dirichlet), tip loaded (Neumann traction)",
        ),
        preset_module=_ENGINEERING,
        preset_function="aircraft_wing_structural",
    ),
    PhenomenonEntry(
        phenomenon="External car-body aerodynamics (bluff body)",
        governing_equation=(
            "Steady incompressible Navier-Stokes "
            "(PDETermSpec.kind='incompressible_navier_stokes_2d')"
        ),
        typical_parameters={
            "U_inf": "33.3 m/s (~120 km/h)",
            "car_length": "4.5 m",
        },
        assumptions=(
            "2D bluff-body approximation of a 3D car",
            "Moving-ground boundary condition",
            "No-slip on car body",
        ),
        preset_module=_ENGINEERING,
        preset_function="car_external_aero",
    ),
    PhenomenonEntry(
        phenomenon="Transient brake-disc heating during emergency braking",
        governing_equation=(
            "Transient Fourier heat conduction with source: "
            "rho*cp*dT/dt = div(k*grad(T)) + q "
            "(PDETermSpec.kind='heat_equation_transient')"
        ),
        typical_parameters={
            "k_disc": "55.0 W/(m*K) (cast iron)",
            "rho_disc": "7100.0 kg/m^3",
            "cp_disc": "500.0 J/(kg*K)",
            "q_friction": "2e6 W/m^2 (friction surface heat flux)",
        },
        assumptions=(
            "Pure conduction (no explicit convective transport term)",
            "Convective cooling BC on non-friction surface (Neumann with h, T_ref)",
            "Uniform friction heat flux over braking duration",
        ),
        preset_module=_ENGINEERING,
        preset_function="car_brake_thermal",
    ),
    PhenomenonEntry(
        phenomenon="Suspension wishbone fatigue stress analysis",
        governing_equation=(
            "Linear elasticity, plane stress "
            "(PDETermSpec.kind='linear_elasticity_plane_stress')"
        ),
        typical_parameters={
            "E": "210e9 Pa (steel)",
            "nu": "0.3",
            "sigma_y_steel": "355e6 Pa",
        },
        assumptions=(
            "Plane stress",
            "Linear elastic (fatigue assessed post-hoc from von Mises stress)",
            "Mounting fixed (Dirichlet), wheel-hub load applied (Neumann)",
        ),
        preset_module=_ENGINEERING,
        preset_function="car_suspension_fatigue",
    ),
    PhenomenonEntry(
        phenomenon="CPU heatsink conduction/convection cooling",
        governing_equation=(
            "Steady-state heat conduction (Laplace/Poisson for T); "
            "PDETermSpec.kind='heat_equation_steady'"
        ),
        typical_parameters={
            "k_aluminium": "205.0 W/(m*K)",
            "q_cpu": "200.0 W (die heat dissipation)",
            "die_area": "1e-4 m^2 (~1 cm^2)",
        },
        assumptions=(
            "Steady state (no transient term)",
            "Uniform heat flux injected at CPU base",
            "Forced-convection Neumann BC (h, T_ref) on fin surfaces",
        ),
        preset_module=_ENGINEERING,
        preset_function="cpu_heatsink_thermal",
    ),
    PhenomenonEntry(
        phenomenon="PCB heat spreading with component hotspots",
        governing_equation=(
            "2D anisotropic steady heat equation; "
            "PDETermSpec.kind='heat_equation_steady_anisotropic'"
        ),
        typical_parameters={
            "k_pcb": "0.3 W/(m*K) (FR4, in-plane)",
            "k_z_pcb": "0.25 W/(m*K) (FR4, through-plane)",
            "T_ambient": "308.0 K (35 C)",
        },
        assumptions=(
            "Anisotropic effective conductivity (in-plane vs through-plane)",
            "Natural-convection Neumann BC on board surface",
            "Insulated (zero-flux) board edges",
        ),
        preset_module=_ENGINEERING,
        preset_function="pcb_thermal",
    ),
    PhenomenonEntry(
        phenomenon="Industrial furnace wall conduction/convection/radiation",
        governing_equation=(
            "Steady heat conduction (Laplace for T) with a nonlinear "
            "radiative Neumann BC q_rad = eps*sigma*(T^4 - T_amb^4); "
            "PDETermSpec.kind='heat_equation_steady'"
        ),
        typical_parameters={
            "T_hot_gas": "1600.0 K (combustion zone)",
            "k_refractory": "1.5 W/(m*K) (firebrick)",
            "eps_wall": "0.9 (emissivity)",
        },
        assumptions=(
            "Combined conduction + convection + radiation at boundaries",
            "Radiation linearized/approximated as a nonlinear Neumann term",
            "Refractory temperature limit ~1800 K (alert threshold)",
        ),
        preset_module=_ENGINEERING,
        preset_function="industrial_furnace_thermal",
    ),
    PhenomenonEntry(
        phenomenon="Multi-layer refractory lining conduction",
        governing_equation=(
            "1D/2D steady-state conduction across layered media; "
            "effective conductivity k_eff = total_thickness / "
            "sum(thickness_i / k_i)"
        ),
        typical_parameters={
            "layers": (
                "working_lining k=1.8, safety_lining k=0.5, "
                "insulation k=0.08 W/(m*K)"
            ),
            "T_hot/T_cold": "1700.0 K / 350.0 K",
        },
        assumptions=(
            "Series-resistance (1D) approximation across layers",
            "Perfect thermal contact between layers",
            "Fixed hot-face / cold-face Dirichlet temperatures",
        ),
        preset_module=_ENGINEERING,
        preset_function="refractory_lining",
    ),
    PhenomenonEntry(
        phenomenon="Datacenter hot-aisle/cold-aisle rack cooling",
        governing_equation=(
            "Incompressible Navier-Stokes coupled with the energy equation; "
            "PDETermSpec.kind='incompressible_navier_stokes_energy_2d'"
        ),
        typical_parameters={
            "U_cold_aisle": "2.5 m/s",
            "T_cold_air": "291.0 K (18 C supply)",
            "Q_rack": "20000.0 W per rack",
        },
        assumptions=(
            "2D channel approximation of rack-row airflow",
            "Server surfaces modeled as prescribed heat-flux (Neumann)",
            "Canonical datacenter digital-twin problem (per module docstring)",
        ),
        preset_module=_ENGINEERING,
        preset_function="datacenter_airflow_2d",
    ),
    # ------------------------------------------------------------------
    # Solid mechanics (pinneapple_physics/pde_environment/presets/solid_mechanics.py)
    # ------------------------------------------------------------------
    PhenomenonEntry(
        phenomenon="General axisymmetric linear elasticity (r,z)",
        governing_equation=(
            "Axisymmetric equilibrium div(sigma)=0 in cylindrical coords: "
            "d(s_rr)/dr + d(s_rz)/dz + (s_rr-s_tt)/r = 0; "
            "d(s_rz)/dr + d(s_zz)/dz + s_rz/r = 0 "
            "(PDETermSpec.kind='axisymmetric_linear_elasticity')"
        ),
        typical_parameters={
            "E": "2.1e11 Pa",
            "nu": "0.3",
            "r_min/r_max": "10.0 mm / 50.0 mm",
        },
        assumptions=(
            "Isotropic, small-strain linear elasticity",
            "Axisymmetry (no theta dependence)",
            "Applications: pressure vessels, gun barrels, rotating shafts, "
            "valve seats, bearing races, bolted flanges",
        ),
        preset_module=_SOLID_MECHANICS,
        preset_function="axisymmetric_linear_elasticity_2d_default",
    ),
    PhenomenonEntry(
        phenomenon="Thick-walled cylinder under internal/external pressure",
        governing_equation=(
            "Lamé closed-form solution: sigma_rr(r), sigma_tt(r), u_r(r) "
            "for a thick-walled hollow cylinder"
        ),
        typical_parameters={
            "a (inner radius)": "20.0 mm",
            "b (outer radius)": "60.0 mm",
            "p_a (internal pressure)": "100e6 Pa",
        },
        assumptions=(
            "Plane-strain-like axisymmetric elasticity",
            "Analytical (Lamé) solution available -- standard PINN elasticity "
            "benchmark",
            "Applications: pressure vessels, gun barrels, hydraulic cylinders",
        ),
        preset_module=_SOLID_MECHANICS,
        preset_function="thick_walled_cylinder_lame_default",
    ),
    PhenomenonEntry(
        phenomenon="Threaded coupling under combined pressure/axial/torque load",
        governing_equation=(
            "Axisymmetric linear elasticity (u_r,u_z) decoupled from the "
            "torsional Navier equation d^2(u_th)/dr^2 + (1/r)d(u_th)/dr "
            "- u_th/r^2 + d^2(u_th)/dz^2 = 0"
        ),
        typical_parameters={
            "E": "2.1e11 Pa (AISI 4145H steel)",
            "p_inner": "20e6 Pa",
            "T_torque": "40e3 N*m",
        },
        assumptions=(
            "Linear elasticity: meridional and torsional problems decouple",
            "Quasi-static (inertia neglected)",
            "Full 6-component von Mises: sqrt(1/2[(s_rr-s_zz)^2+"
            "(s_zz-s_tt)^2+(s_tt-s_rr)^2+6(t_rz^2+t_rth^2+t_thz^2)])",
        ),
        preset_module=_SOLID_MECHANICS,
        preset_function="threaded_coupling_tc50_rotating_default",
    ),
    # ------------------------------------------------------------------
    # Astrophysics (pinneapple_physics/pde_environment/presets/astrophysics.py)
    # ------------------------------------------------------------------
    PhenomenonEntry(
        phenomenon="Restricted two-body (Kepler) orbit",
        governing_equation=(
            "Newton's law of gravitation, reduced two-body ODE system: "
            "dx/dt=vx; dy/dt=vy; dvx/dt=-mu*x/r^3; dvy/dt=-mu*y/r^3, "
            "r=sqrt(x^2+y^2)"
        ),
        typical_parameters={
            "mu": "398600.4418 km^3/s^2 (Earth GM)",
            "a": "8000.0 km (semi-major axis)",
            "e": "0.15 (eccentricity)",
        },
        assumptions=(
            "Restricted two-body problem (point masses, no perturbations)",
            "Initial condition: perigee passage at t=0, tangential velocity "
            "from vis-viva",
        ),
        preset_module=_ASTROPHYSICS,
        preset_function="kepler_two_body_orbit",
    ),
    PhenomenonEntry(
        phenomenon="Self-gravitating polytropic stellar structure (Lane-Emden)",
        governing_equation=(
            "Lane-Emden equation as a first-order system with phi=dtheta/dxi: "
            "dtheta/dxi = phi; dphi/dxi = -theta^n - (2/xi)*phi"
        ),
        typical_parameters={
            "n": "1.0 (polytropic index; default has closed form "
            "theta=sin(xi)/xi)",
            "xi_min": "1e-3 (avoids the 2/xi origin singularity)",
        },
        assumptions=(
            "Hydrostatic equilibrium + polytropic EOS P = K*rho^(1+1/n)",
            "n=1.5 models a non-relativistic degenerate star (white-dwarf "
            "core); n=3 is the Eddington standard model",
            "Domain starts at xi_min>0, not exactly 0",
        ),
        preset_module=_ASTROPHYSICS,
        preset_function="lane_emden_polytrope",
    ),
    PhenomenonEntry(
        phenomenon="1D compressible Euler shock tube (Sod problem)",
        governing_equation=(
            "1D compressible Euler equations, conservative ideal-gas form: "
            "d(rho)/dt + d(rho*u)/dx = 0; "
            "d(rho*u)/dt + d(rho*u^2+p)/dx = 0; "
            "d(E)/dt + d((E+p)*u)/dx = 0, p=(gamma-1)(E-0.5*rho*u^2)"
        ),
        typical_parameters={
            "gamma": "1.4 (ideal diatomic gas)",
            "left state (rho,u,p)": "(1, 0, 1)",
            "right state (rho,u,p)": "(0.125, 0, 0.1)",
        },
        assumptions=(
            "Ideal gas, gamma-law equation of state",
            "Inviscid compressible flow",
            "Standard code-validation case (Sod 1978); has exact Riemann "
            "solution",
        ),
        preset_module=_ASTROPHYSICS,
        preset_function="sod_shock_tube_astro",
    ),
    PhenomenonEntry(
        phenomenon="Weak-field gravitational light bending (Schwarzschild)",
        governing_equation=(
            "Null-geodesic ('photon orbit') equation in u=1/r: "
            "d^2u/dphi^2 + u = 3*m*u^2, m := GM/c^2"
        ),
        typical_parameters={
            "GM": "1.32712440018e20 m^3/s^2 (Sun)",
            "b": "6.957e8 m (impact parameter = solar radius, grazing)",
        },
        assumptions=(
            "Weak-field limit (m/b << 1); NOT valid near the photon sphere "
            "(r = 1.5 * Schwarzschild radius)",
            "Closed-form solution used is a first-order perturbative "
            "approximation of the exact ODE",
            "Reproduces Einstein's 1.7515 arcsec Sun-grazing deflection",
        ),
        preset_module=_ASTROPHYSICS,
        preset_function="schwarzschild_light_bending_weak_field",
    ),
    PhenomenonEntry(
        phenomenon="Steady thin alpha-disk accretion (Shakura-Sunyaev)",
        governing_equation=(
            "Angular-momentum/torque-balance equation for a steady, "
            "geometrically-thin, optically-thick disk; radial flux "
            "F(r) = (3*G*M*Mdot)/(8*pi*r^3) * [1 - sqrt(R_in/r)], "
            "T_eff(r) = (F(r)/sigma_SB)^(1/4)"
        ),
        typical_parameters={
            "M_bh_solar": "10.0 solar masses (stellar-mass BH X-ray binary)",
            "Mdot_edd_frac": "0.1 (fraction of Eddington accretion rate)",
            "eta": "0.1 (radiative efficiency)",
        },
        assumptions=(
            "Geometrically thin, optically thick disk (Shakura & Sunyaev "
            "1973)",
            "Zero-torque inner boundary condition: F(R_in)=0",
            "Steady-state (time-independent) radial structure",
        ),
        preset_module=_ASTROPHYSICS,
        preset_function="shakura_sunyaev_accretion_disk",
    ),
)


def lookup_phenomenon(name_or_keyword: str) -> List[PhenomenonEntry]:
    """Look up knowledge-base entries by a case-insensitive keyword.

    Matches against ``phenomenon``, ``governing_equation``, and
    ``preset_function`` so both a physical-scenario name (e.g. "poiseuille",
    "shock tube") and an equation/PDE-kind keyword (e.g. "navier_stokes",
    "lane_emden") work as queries.

    Parameters
    ----------
    name_or_keyword:
        Free-text keyword or phrase to search for.

    Returns
    -------
    List[PhenomenonEntry]
        All matching entries (possibly empty). Order follows
        ``PHENOMENON_KNOWLEDGE_BASE``.
    """
    if not name_or_keyword:
        return []
    kw = name_or_keyword.strip().lower()
    hits: List[PhenomenonEntry] = []
    for entry in PHENOMENON_KNOWLEDGE_BASE:
        haystack = " ".join([
            entry.phenomenon,
            entry.governing_equation,
            entry.preset_function,
        ]).lower()
        if kw in haystack:
            hits.append(entry)
    return hits


__all__ = ["PhenomenonEntry", "PHENOMENON_KNOWLEDGE_BASE", "lookup_phenomenon"]
