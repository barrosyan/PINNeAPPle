"""pinneapple_simulation — Physics simulation back-ends and data generation.

Sub-modules
-----------
numerical_solvers  (was pinneapple_solvers)
    PDE solvers (FDM, FEM, CFD): HeatConduction3D, NavierStokes3D,
    LidDrivenCavitySolver3D, ChannelFlowSolver3D, ElasticWave3D.
    Optional bridges to OpenFOAM and FEniCS. SolverRegistry, generate_pinn_dataset.
    CAD-to-CFD pipeline (CADToCFDPipeline).

particle_dynamics  (was pinneapple_dynamics)
    Differentiable particle / continuum dynamics: rigid-body (RigidBody,
    RigidBodySystem), MPM (MPMSimulator), SPH (SPHParticles). All in pure
    PyTorch for autograd compatibility.

external_solvers   (was pinneapple_integrations)
    External tool bridges: OpenFOAM case staging & field extraction, MATLAB
    engine wrapper, Modelica / FMU simulation (via fmpy / OMPython), FEniCS
    workflow. All optional — graceful degradation when tool not installed.

Integration helpers
-------------------
``simulate(scenario, ...)``
    Dispatch function: routes to the appropriate solver/simulator based on
    scenario string (``"heat_3d"``, ``"ns_3d"``, ``"mpm"``, ``"sph"``, …).
``generate_data(scenario, n_samples, ...)``
    Generates a dataset of trajectories using the matching solver.

Usage
-----
>>> from pinneapple_simulation import simulate, generate_data, HeatConduction3D
>>> output = simulate("heat_3d", t_end=1.0, nx=32, ny=32, nz=32)
>>> dataset = generate_data("ns_3d", n_samples=50)
"""
from __future__ import annotations
from typing import Any

# ── sub-modules (new descriptive names) ───────────────────────────────────────
from . import numerical_solvers
from . import particle_dynamics
from . import external_solvers

# backward-compat aliases
solvers      = numerical_solvers
dynamics     = particle_dynamics
integrations = external_solvers

# ── numerical_solvers re-exports ──────────────────────────────────────────────
from .numerical_solvers import (
    SolverBase, SolverOutput,
    SolverRegistry, SolverSpec, register_all, SolverCatalog,
    generate_pinn_dataset,
    openfoam_available, fenics_available,
    CFDMesh, NSFlowSolver, CADToCFDPipeline,
    SolverOutput3D,
    HeatConfig3D, HeatConduction3D,
    NavierStokesConfig3D, NavierStokes3D,
    ElasticWaveConfig3D, ElasticWave3D,
    LidDrivenCavityConfig3D, LidDrivenCavitySolver3D,
    ChannelFlowConfig3D, ChannelFlowSolver3D,
)

try:
    from .numerical_solvers import OpenFOAMBridge, generate_case, run_openfoam, extract_fields, openfoam_to_dataset
except ImportError:
    pass

try:
    from .numerical_solvers import FEnicsBridge
except ImportError:
    pass

# ── particle_dynamics re-exports ──────────────────────────────────────────────
from .particle_dynamics import (
    RigidBody, RigidBodyState, RigidBodySystem,
    MPMSimulator, MPMState,
    ParticleSystem, SPHParticles,
)

# ── external_solvers re-exports ───────────────────────────────────────────────
from .external_solvers import (
    # OpenFOAM
    OpenFOAMCaseTemplate, stage_case_for_scenario,
    OpenFOAMRunConfig, run_openfoam_case,
    write_sample_dict_cloud, run_sampling,
    read_sampled_scalar_field, export_bundle, openfoam_case_to_upd,
    # Optional (may be None)
    MATLABEngine, run_matlab_script, run_matlab_function,
    load_mat, save_mat, mat_to_upd,
    FMUSimConfig, simulate_fmu, fmu_to_upd,
    OMSimConfig, simulate_openmodelica, om_to_upd, modelica_result_to_upd,
    FEniCSConfig, solve_and_package, dof_to_upd, FEniCSWorkflow,
)

# ── Scenario registry ──────────────────────────────────────────────────────────
_SCENARIO_MAP: dict[str, Any] = {
    "heat_3d":    HeatConduction3D,
    "ns_3d":      NavierStokes3D,
    "elastic_3d": ElasticWave3D,
    "lid_cavity": LidDrivenCavitySolver3D,
    "channel_3d": ChannelFlowSolver3D,
    "mpm":        MPMSimulator,
    "sph":        SPHParticles,
    "rigid":      RigidBodySystem,
}


def simulate(scenario: str, **kwargs) -> Any:
    """Run a named physics scenario and return its output.

    Parameters
    ----------
    scenario : str — one of ``"heat_3d"``, ``"ns_3d"``, ``"elastic_3d"``,
               ``"lid_cavity"``, ``"channel_3d"``, ``"mpm"``, ``"sph"``, ``"rigid"``
    """
    if scenario not in _SCENARIO_MAP:
        raise ValueError(f"Unknown scenario '{scenario}'. Available: {list(_SCENARIO_MAP)}")
    cls = _SCENARIO_MAP[scenario]
    instance = cls(**{k: v for k, v in kwargs.items() if not k.startswith("run_")})
    if hasattr(instance, "run"):
        return instance.run(**{k[4:]: v for k, v in kwargs.items() if k.startswith("run_")})
    return instance


def generate_data(scenario: str, n_samples: int = 10, **kwargs) -> list:
    """Generate a list of simulation outputs for training data."""
    return [simulate(scenario, **kwargs) for _ in range(n_samples)]


__all__ = [
    # Sub-modules (new names)
    "numerical_solvers", "particle_dynamics", "external_solvers",
    # Sub-modules (old aliases — backward compat)
    "solvers", "dynamics", "integrations",
    # Integration
    "simulate", "generate_data",
    # numerical_solvers
    "SolverBase", "SolverOutput",
    "SolverRegistry", "SolverSpec", "register_all", "SolverCatalog",
    "generate_pinn_dataset", "openfoam_available", "fenics_available",
    "CFDMesh", "NSFlowSolver", "CADToCFDPipeline",
    "SolverOutput3D",
    "HeatConfig3D", "HeatConduction3D",
    "NavierStokesConfig3D", "NavierStokes3D",
    "ElasticWaveConfig3D", "ElasticWave3D",
    "LidDrivenCavityConfig3D", "LidDrivenCavitySolver3D",
    "ChannelFlowConfig3D", "ChannelFlowSolver3D",
    # particle_dynamics
    "RigidBody", "RigidBodyState", "RigidBodySystem",
    "MPMSimulator", "MPMState",
    "ParticleSystem", "SPHParticles",
    # external_solvers
    "OpenFOAMCaseTemplate", "stage_case_for_scenario",
    "OpenFOAMRunConfig", "run_openfoam_case",
    "write_sample_dict_cloud", "run_sampling",
    "read_sampled_scalar_field", "export_bundle", "openfoam_case_to_upd",
    "MATLABEngine", "run_matlab_script", "run_matlab_function",
    "load_mat", "save_mat", "mat_to_upd",
    "FMUSimConfig", "simulate_fmu", "fmu_to_upd",
    "OMSimConfig", "simulate_openmodelica", "om_to_upd", "modelica_result_to_upd",
    "FEniCSConfig", "solve_and_package", "dof_to_upd", "FEniCSWorkflow",
]
