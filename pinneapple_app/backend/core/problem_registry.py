"""Problem registry: metadata, solver recommendations, and model suggestions per preset."""
from __future__ import annotations
from typing import Dict, Any

# ── Problem metadata catalogue ────────────────────────────────────────────
# key = preset name in pinneapple_environment
# value = dict with: family, time_dependent, recommended_solver,
#                    recommended_models, collocation_strategy, description

_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── Fluid dynamics ─────────────────────────────────────────────────────
    "burgers_1d": {
        "family": "fluid", "time_dependent": True,
        "recommended_solver": "fdm_1d",
        "recommended_models": ["siren", "vanilla_pinn", "modified_mlp", "pinnsformer"],
        "collocation_strategy": "lhs",
        "description": "1-D Burgers equation — nonlinear advection-diffusion with shock formation.",
        "tags": ["1D", "nonlinear", "time-dependent"],
    },
    "ns_incompressible_2d": {
        "family": "fluid", "time_dependent": True,
        "recommended_solver": "ns_fdm_2d",
        "recommended_models": ["fno", "siren", "mesh_graph_net", "vanilla_pinn"],
        "collocation_strategy": "lhs",
        "description": "2-D incompressible Navier-Stokes.",
        "tags": ["2D", "CFD", "incompressible"],
    },
    "lid_driven_cavity_3d": {
        "family": "fluid", "time_dependent": False,
        "recommended_solver": "lid_cavity_fdm",
        "recommended_models": ["fno", "mesh_graph_net", "siren", "modified_mlp"],
        "collocation_strategy": "sobol",
        "description": "3-D lid-driven cavity flow benchmark.",
        "tags": ["3D", "CFD", "steady"],
    },
    "channel_flow_3d": {
        "family": "fluid", "time_dependent": False,
        "recommended_solver": "ns_fdm_3d",
        "recommended_models": ["fno", "mesh_graph_net"],
        "collocation_strategy": "sobol",
        "description": "3-D pressure-driven channel flow (Poiseuille).",
        "tags": ["3D", "CFD", "steady"],
    },
    "pipe_flow_3d": {
        "family": "fluid", "time_dependent": False,
        "recommended_solver": "ns_fdm_3d",
        "recommended_models": ["fno", "mesh_graph_net", "siren"],
        "collocation_strategy": "sobol",
        "description": "3-D pipe flow with cylindrical geometry.",
        "tags": ["3D", "CFD", "geometry"],
    },
    "darcy_pressure_only_3d": {
        "family": "fluid", "time_dependent": False,
        "recommended_solver": "fdm_3d_generic",
        "recommended_models": ["vanilla_pinn", "siren", "modified_mlp"],
        "collocation_strategy": "lhs",
        "description": "Darcy porous-media pressure equation.",
        "tags": ["3D", "porous", "steady"],
    },
    "rocket_nozzle_cfd": {
        "family": "fluid", "time_dependent": False,
        "recommended_solver": "ns_fdm_3d",
        "recommended_models": ["fno", "mesh_graph_net"],
        "collocation_strategy": "sobol",
        "description": "Rocket nozzle compressible CFD.",
        "tags": ["3D", "compressible", "aerospace"],
    },
    "car_external_aero": {
        "family": "fluid", "time_dependent": False,
        "recommended_solver": "ns_fdm_3d",
        "recommended_models": ["mesh_graph_net", "fno"],
        "collocation_strategy": "sobol",
        "description": "External aerodynamics around a vehicle body.",
        "tags": ["3D", "CFD", "automotive"],
    },
    "datacenter_airflow_2d": {
        "family": "fluid", "time_dependent": False,
        "recommended_solver": "ns_fdm_2d",
        "recommended_models": ["fno", "siren", "modified_mlp"],
        "collocation_strategy": "lhs",
        "description": "2-D airflow inside a server room.",
        "tags": ["2D", "CFD", "thermal-fluid"],
    },
    "fan_cooler_cfd": {
        "family": "fluid", "time_dependent": False,
        "recommended_solver": "ns_fdm_3d",
        "recommended_models": ["fno", "mesh_graph_net"],
        "collocation_strategy": "sobol",
        "description": "CFD of a fan cooler geometry.",
        "tags": ["3D", "cooling", "CFD"],
    },
    # ── Thermal ────────────────────────────────────────────────────────────
    "steady_heat_conduction_3d": {
        "family": "thermal", "time_dependent": False,
        "recommended_solver": "heat_fdm_3d",
        "recommended_models": ["siren", "modified_mlp", "vanilla_pinn", "hash_grid_mlp"],
        "collocation_strategy": "lhs",
        "description": "Steady-state 3-D heat conduction (Laplace/Poisson).",
        "tags": ["3D", "thermal", "steady"],
    },
    "transient_heat_3d": {
        "family": "thermal", "time_dependent": True,
        "recommended_solver": "heat_fdm_3d",
        "recommended_models": ["siren", "modified_mlp", "vanilla_pinn"],
        "collocation_strategy": "lhs",
        "description": "Transient 3-D heat equation.",
        "tags": ["3D", "thermal", "time-dependent"],
    },
    "laplace_2d": {
        "family": "thermal", "time_dependent": False,
        "recommended_solver": "fdm_2d_generic",
        "recommended_models": ["vanilla_pinn", "siren", "modified_mlp"],
        "collocation_strategy": "lhs",
        "description": "2-D Laplace equation (steady heat / electrostatics).",
        "tags": ["2D", "elliptic", "steady"],
    },
    "poisson_2d": {
        "family": "thermal", "time_dependent": False,
        "recommended_solver": "fdm_2d_generic",
        "recommended_models": ["vanilla_pinn", "siren", "modified_mlp"],
        "collocation_strategy": "lhs",
        "description": "2-D Poisson equation with source term.",
        "tags": ["2D", "elliptic", "source"],
    },
    "pcb_thermal": {
        "family": "thermal", "time_dependent": False,
        "recommended_solver": "heat_fdm_3d",
        "recommended_models": ["siren", "modified_mlp"],
        "collocation_strategy": "sobol",
        "description": "PCB printed circuit board steady-state thermal.",
        "tags": ["3D", "thermal", "electronics"],
    },
    "cpu_heatsink_thermal": {
        "family": "thermal", "time_dependent": False,
        "recommended_solver": "heat_fdm_3d",
        "recommended_models": ["siren", "modified_mlp", "fno"],
        "collocation_strategy": "sobol",
        "description": "CPU heatsink thermal analysis.",
        "tags": ["3D", "thermal", "electronics"],
    },
    "car_brake_thermal": {
        "family": "thermal", "time_dependent": True,
        "recommended_solver": "heat_fdm_3d",
        "recommended_models": ["siren", "vanilla_pinn"],
        "collocation_strategy": "lhs",
        "description": "Transient brake disc thermal problem.",
        "tags": ["3D", "thermal", "automotive"],
    },
    "industrial_furnace_thermal": {
        "family": "thermal", "time_dependent": True,
        "recommended_solver": "heat_fdm_3d",
        "recommended_models": ["siren", "modified_mlp"],
        "collocation_strategy": "lhs",
        "description": "Industrial furnace transient thermal problem.",
        "tags": ["3D", "thermal", "industrial"],
    },
    "datacenter_server_thermal": {
        "family": "thermal", "time_dependent": False,
        "recommended_solver": "heat_fdm_3d",
        "recommended_models": ["siren", "modified_mlp"],
        "collocation_strategy": "sobol",
        "description": "Server blade steady-state thermal.",
        "tags": ["3D", "thermal", "electronics"],
    },
    "refractory_lining": {
        "family": "thermal", "time_dependent": False,
        "recommended_solver": "heat_fdm_3d",
        "recommended_models": ["siren", "modified_mlp"],
        "collocation_strategy": "lhs",
        "description": "Refractory lining thermal analysis.",
        "tags": ["3D", "thermal", "industrial"],
    },
    "reaction_diffusion_2d": {
        "family": "diffusion", "time_dependent": True,
        "recommended_solver": "fdm_2d_generic",
        "recommended_models": ["siren", "vanilla_pinn", "modified_mlp"],
        "collocation_strategy": "lhs",
        "description": "2-D reaction-diffusion (Turing patterns, combustion).",
        "tags": ["2D", "diffusion", "time-dependent"],
    },
    # ── Wave / acoustics ────────────────────────────────────────────────────
    "helmholtz_acoustics_3d": {
        "family": "wave", "time_dependent": False,
        "recommended_solver": "fdm_3d_generic",
        "recommended_models": ["siren", "vanilla_pinn"],
        "collocation_strategy": "sobol",
        "description": "Helmholtz equation for frequency-domain acoustics.",
        "tags": ["3D", "acoustics", "frequency-domain"],
    },
    "wave_ultrasound_3d": {
        "family": "wave", "time_dependent": True,
        "recommended_solver": "elastic_fdm_3d",
        "recommended_models": ["siren", "pinnsformer"],
        "collocation_strategy": "sobol",
        "description": "3-D ultrasound / elastic wave propagation.",
        "tags": ["3D", "wave", "time-dependent"],
    },
    "crystal_phonon": {
        "family": "wave", "time_dependent": True,
        "recommended_solver": "elastic_fdm_3d",
        "recommended_models": ["siren", "vanilla_pinn"],
        "collocation_strategy": "sobol",
        "description": "Crystal phonon / lattice dynamics.",
        "tags": ["3D", "wave", "materials"],
    },
    # ── Structural ──────────────────────────────────────────────────────────
    "linear_elasticity_3d": {
        "family": "structural", "time_dependent": False,
        "recommended_solver": "fdm_3d_generic",
        "recommended_models": ["modified_mlp", "mesh_graph_net", "vanilla_pinn"],
        "collocation_strategy": "sobol",
        "description": "3-D linear elasticity (Navier equations).",
        "tags": ["3D", "structural", "steady"],
    },
    "plane_stress_2d": {
        "family": "structural", "time_dependent": False,
        "recommended_solver": "fdm_2d_generic",
        "recommended_models": ["modified_mlp", "vanilla_pinn", "siren"],
        "collocation_strategy": "lhs",
        "description": "2-D plane-stress linear elasticity.",
        "tags": ["2D", "structural", "steady"],
    },
    "plane_strain_2d": {
        "family": "structural", "time_dependent": False,
        "recommended_solver": "fdm_2d_generic",
        "recommended_models": ["modified_mlp", "vanilla_pinn", "siren"],
        "collocation_strategy": "lhs",
        "description": "2-D plane-strain linear elasticity.",
        "tags": ["2D", "structural", "steady"],
    },
    "von_mises_2d": {
        "family": "structural", "time_dependent": False,
        "recommended_solver": "fdm_2d_generic",
        "recommended_models": ["modified_mlp", "vanilla_pinn"],
        "collocation_strategy": "lhs",
        "description": "2-D von Mises stress criterion.",
        "tags": ["2D", "structural", "failure"],
    },
    "thermoelasticity_2d": {
        "family": "structural", "time_dependent": False,
        "recommended_solver": "fdm_2d_generic",
        "recommended_models": ["modified_mlp", "vanilla_pinn"],
        "collocation_strategy": "lhs",
        "description": "2-D coupled thermo-elastic problem.",
        "tags": ["2D", "structural", "thermal-coupling"],
    },
    "material_fracture_2d": {
        "family": "structural", "time_dependent": False,
        "recommended_solver": "fdm_2d_generic",
        "recommended_models": ["mesh_graph_net", "modified_mlp"],
        "collocation_strategy": "sobol",
        "description": "2-D fracture mechanics / phase-field.",
        "tags": ["2D", "structural", "fracture"],
    },
    "car_suspension_fatigue": {
        "family": "structural", "time_dependent": True,
        "recommended_solver": "fdm_3d_generic",
        "recommended_models": ["modified_mlp", "vanilla_pinn"],
        "collocation_strategy": "lhs",
        "description": "Fatigue analysis of a car suspension component.",
        "tags": ["3D", "structural", "automotive"],
    },
    "axisymmetric_linear_elasticity_2d": {
        "family": "structural", "time_dependent": False,
        "recommended_solver": "fdm_2d_generic",
        "recommended_models": ["modified_mlp", "vanilla_pinn"],
        "collocation_strategy": "lhs",
        "description": "Axisymmetric linear elasticity.",
        "tags": ["2D", "structural", "axisymmetric"],
    },
    "thick_walled_cylinder_lame": {
        "family": "structural", "time_dependent": False,
        "recommended_solver": "fdm_2d_generic",
        "recommended_models": ["vanilla_pinn", "siren"],
        "collocation_strategy": "lhs",
        "description": "Thick-walled cylinder Lamé stress solution.",
        "tags": ["2D", "structural", "pressure-vessel"],
    },
    "rocket_structural": {
        "family": "structural", "time_dependent": False,
        "recommended_solver": "fdm_3d_generic",
        "recommended_models": ["modified_mlp", "mesh_graph_net"],
        "collocation_strategy": "sobol",
        "description": "Structural analysis of a rocket nozzle.",
        "tags": ["3D", "structural", "aerospace"],
    },
    # ── Finance ─────────────────────────────────────────────────────────────
    "black_scholes_1d": {
        "family": "finance", "time_dependent": True,
        "recommended_solver": "fdm_1d",
        "recommended_models": ["vanilla_pinn", "siren"],
        "collocation_strategy": "lhs",
        "description": "Black-Scholes PDE for option pricing.",
        "tags": ["1D", "finance", "time-dependent"],
    },
    "heston_pde_2d": {
        "family": "finance", "time_dependent": True,
        "recommended_solver": "fdm_2d_generic",
        "recommended_models": ["vanilla_pinn", "siren", "modified_mlp"],
        "collocation_strategy": "lhs",
        "description": "Heston stochastic volatility PDE.",
        "tags": ["2D", "finance", "time-dependent"],
    },
    # ── Biological ─────────────────────────────────────────────────────────
    "sir_epidemic": {
        "family": "biological", "time_dependent": True,
        "recommended_solver": "fdm_1d",
        "recommended_models": ["vanilla_pinn", "siren", "neural_ode"],
        "collocation_strategy": "uniform",
        "description": "SIR epidemic ODE model.",
        "tags": ["ODE", "biological", "time-dependent"],
    },
    "drug_diffusion_tissue": {
        "family": "biological", "time_dependent": True,
        "recommended_solver": "fdm_2d_generic",
        "recommended_models": ["vanilla_pinn", "siren"],
        "collocation_strategy": "lhs",
        "description": "Drug diffusion through biological tissue.",
        "tags": ["2D", "biological", "diffusion"],
    },
    "pk_two_compartment": {
        "family": "biological", "time_dependent": True,
        "recommended_solver": "fdm_1d",
        "recommended_models": ["vanilla_pinn", "neural_ode"],
        "collocation_strategy": "uniform",
        "description": "Two-compartment pharmacokinetic model.",
        "tags": ["ODE", "biological", "pharmacokinetics"],
    },
    # ── Climate / environment ───────────────────────────────────────────────
    "climate_atmosphere_2d": {
        "family": "fluid", "time_dependent": True,
        "recommended_solver": "ns_fdm_2d",
        "recommended_models": ["fno", "siren", "modified_mlp"],
        "collocation_strategy": "sobol",
        "description": "2-D atmospheric fluid dynamics.",
        "tags": ["2D", "climate", "fluid"],
    },
    "climate_ocean_gyre": {
        "family": "fluid", "time_dependent": True,
        "recommended_solver": "ns_fdm_2d",
        "recommended_models": ["fno", "mesh_graph_net"],
        "collocation_strategy": "sobol",
        "description": "Ocean gyre circulation (barotropic vorticity).",
        "tags": ["2D", "climate", "fluid"],
    },
    "opinion_dynamics_2d": {
        "family": "diffusion", "time_dependent": True,
        "recommended_solver": "fdm_2d_generic",
        "recommended_models": ["vanilla_pinn", "siren"],
        "collocation_strategy": "lhs",
        "description": "Opinion dynamics social PDE.",
        "tags": ["2D", "social", "diffusion"],
    },
    # ── Aerospace ───────────────────────────────────────────────────────────
    "aircraft_wing_aerodynamics": {
        "family": "fluid", "time_dependent": False,
        "recommended_solver": "ns_fdm_3d",
        "recommended_models": ["mesh_graph_net", "fno"],
        "collocation_strategy": "sobol",
        "description": "External aerodynamics over an aircraft wing.",
        "tags": ["3D", "aerospace", "CFD"],
    },
    "aircraft_wing_structural": {
        "family": "structural", "time_dependent": False,
        "recommended_solver": "fdm_3d_generic",
        "recommended_models": ["modified_mlp", "mesh_graph_net"],
        "collocation_strategy": "sobol",
        "description": "Structural analysis of an aircraft wing under load.",
        "tags": ["3D", "aerospace", "structural"],
    },
    # ── Industrial ──────────────────────────────────────────────────────────
    "furnace_combustion_zone": {
        "family": "fluid", "time_dependent": False,
        "recommended_solver": "ns_fdm_3d",
        "recommended_models": ["fno", "siren"],
        "collocation_strategy": "sobol",
        "description": "Reactive flow in a combustion furnace zone.",
        "tags": ["3D", "combustion", "industrial"],
    },
    "datacenter_cfd_3d": {
        "family": "fluid", "time_dependent": False,
        "recommended_solver": "ns_fdm_3d",
        "recommended_models": ["fno", "mesh_graph_net"],
        "collocation_strategy": "sobol",
        "description": "3-D airflow + thermal in a data center.",
        "tags": ["3D", "CFD", "thermal-fluid"],
    },
    "rotary_coupling_torsion": {
        "family": "structural", "time_dependent": False,
        "recommended_solver": "fdm_3d_generic",
        "recommended_models": ["modified_mlp", "vanilla_pinn"],
        "collocation_strategy": "lhs",
        "description": "Torsional stress in a rotary shaft coupling.",
        "tags": ["3D", "structural", "mechanical"],
    },
    "threaded_coupling_tc50_box": {
        "family": "structural", "time_dependent": False,
        "recommended_solver": "fdm_3d_generic",
        "recommended_models": ["modified_mlp", "mesh_graph_net"],
        "collocation_strategy": "sobol",
        "description": "TC-50 box-thread structural analysis.",
        "tags": ["3D", "structural", "oilfield"],
    },
    "threaded_coupling_tc50_pin": {
        "family": "structural", "time_dependent": False,
        "recommended_solver": "fdm_3d_generic",
        "recommended_models": ["modified_mlp", "mesh_graph_net"],
        "collocation_strategy": "sobol",
        "description": "TC-50 pin-thread structural analysis.",
        "tags": ["3D", "structural", "oilfield"],
    },
    "threaded_coupling_tc50_rotating": {
        "family": "structural", "time_dependent": True,
        "recommended_solver": "fdm_3d_generic",
        "recommended_models": ["modified_mlp", "mesh_graph_net"],
        "collocation_strategy": "sobol",
        "description": "TC-50 rotating thread contact mechanics.",
        "tags": ["3D", "structural", "oilfield"],
    },
}

# ── Solver-to-pinneapple_solvers mapping ───────────────────────────────────
SOLVER_MAP = {
    "fdm_1d":          {"class": "HeatConduction3D",        "dim": 1},
    "fdm_2d_generic":  {"class": "HeatConduction3D",        "dim": 2},
    "fdm_3d_generic":  {"class": "HeatConduction3D",        "dim": 3},
    "heat_fdm_3d":     {"class": "HeatConduction3D",        "dim": 3},
    "ns_fdm_2d":       {"class": "NavierStokes3D",          "dim": 2},
    "ns_fdm_3d":       {"class": "NavierStokes3D",          "dim": 3},
    "lid_cavity_fdm":  {"class": "LidDrivenCavitySolver3D", "dim": 3},
    "elastic_fdm_3d":  {"class": "ElasticWave3D",           "dim": 3},
    "uniform":         {"class": None, "dim": None},          # collocation only
}


def get_problem_meta(preset_name: str) -> Dict[str, Any]:
    """Return metadata dict for a preset, with sensible defaults for unknowns."""
    base = _REGISTRY.get(preset_name, {})
    if not base:
        # try prefix matching (e.g. "ns_incompressible_2d_default" → "ns_incompressible_2d")
        for key in _REGISTRY:
            if preset_name.startswith(key) or key.startswith(preset_name.split("_default")[0]):
                base = _REGISTRY[key]
                break
    return {
        "family":               base.get("family", "generic"),
        "time_dependent":       base.get("time_dependent", False),
        "recommended_solver":   base.get("recommended_solver", "fdm_2d_generic"),
        "recommended_models":   base.get("recommended_models", ["vanilla_pinn", "siren"]),
        "collocation_strategy": base.get("collocation_strategy", "lhs"),
        "description":          base.get("description", ""),
        "tags":                 base.get("tags", []),
    }


def recommend_models(preset_name: str) -> list:
    meta = get_problem_meta(preset_name)
    return meta["recommended_models"]


def recommended_solver(preset_name: str) -> str:
    meta = get_problem_meta(preset_name)
    return meta["recommended_solver"]


def all_problems() -> list:
    from pinneapple_physics.pde_environment import list_presets
    return sorted(list_presets())
