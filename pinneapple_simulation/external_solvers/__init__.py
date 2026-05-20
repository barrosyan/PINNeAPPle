"""External tool integrations for PINNeAPPle.

Subpackages
-----------
openfoam  — case staging, execution, field sampling, and UPD packaging
matlab    — MATLAB Engine API wrapper, subprocess runner, .mat I/O
modelica  — FMU simulation (fmpy) and OpenModelica (OMPython)
fenics    — workflow layer on top of pinneapple_solvers.fenics_bridge
mujoco    — MuJoCo (google-deepmind/mujoco) trajectory collection + MJX parallel rollouts
genesis   — Genesis AI (Genesis-Embodied-AI/Genesis) multi-physics robotics simulation

All subpackages use lazy / optional imports so that the base package remains
importable even when a specific external tool is not installed.
"""

# OpenFOAM (no optional deps beyond subprocess + pandas)
from .openfoam import (
    OpenFOAMCaseTemplate,
    stage_case_for_scenario,
    OpenFOAMRunConfig,
    run_openfoam_case,
    write_sample_dict_cloud,
    run_sampling,
    read_sampled_scalar_field,
    export_bundle,
    openfoam_case_to_upd,
)

# MATLAB (optional: matlab.engine or scipy)
try:
    from .matlab import (
        MATLABEngine,
        run_matlab_script,
        run_matlab_function,
        load_mat,
        save_mat,
        mat_to_upd,
    )
except Exception:
    MATLABEngine = None          # type: ignore
    run_matlab_script = None     # type: ignore
    run_matlab_function = None   # type: ignore
    load_mat = None              # type: ignore
    save_mat = None              # type: ignore
    mat_to_upd = None            # type: ignore

# Modelica / FMI (optional: fmpy, OMPython)
try:
    from .modelica import (
        FMUSimConfig,
        simulate_fmu,
        fmu_to_upd,
        OMSimConfig,
        simulate_openmodelica,
        om_to_upd,
        modelica_result_to_upd,
    )
except Exception:
    FMUSimConfig = None              # type: ignore
    simulate_fmu = None              # type: ignore
    fmu_to_upd = None                # type: ignore
    OMSimConfig = None               # type: ignore
    simulate_openmodelica = None     # type: ignore
    om_to_upd = None                 # type: ignore
    modelica_result_to_upd = None    # type: ignore

# FEniCS (optional: pinneapple_solvers.fenics_bridge + dolfinx / legacy FEniCS)
try:
    from .fenics import (
        FEniCSConfig,
        solve_and_package,
        dof_to_upd,
        FEniCSWorkflow,
    )
except Exception:
    FEniCSConfig = None       # type: ignore
    solve_and_package = None  # type: ignore
    dof_to_upd = None         # type: ignore
    FEniCSWorkflow = None     # type: ignore

# MuJoCo (optional: mujoco >= 3.0)
try:
    from .mujoco import (
        MuJoCoConfig,
        load_model,
        make_data,
        MuJoCoRunner,
        trajectory_to_upd,
        trajectories_to_upd,
        MJXParallelRunner,
    )
except Exception:
    MuJoCoConfig = None           # type: ignore
    load_model = None             # type: ignore
    make_data = None              # type: ignore
    MuJoCoRunner = None           # type: ignore
    trajectory_to_upd = None      # type: ignore
    trajectories_to_upd = None    # type: ignore
    MJXParallelRunner = None      # type: ignore

# TurboDesigner (optional: turbodesigner; falls back to built-in analytical solver)
try:
    from .turbodesigner import (
        TurboDesignerConfig,
        TurboDesignerWorkflow,
        run_turbodesigner,
        turbodesigner_to_upd,
    )
except Exception:
    TurboDesignerConfig = None       # type: ignore
    TurboDesignerWorkflow = None     # type: ignore
    run_turbodesigner = None         # type: ignore
    turbodesigner_to_upd = None      # type: ignore

# Genesis AI (optional: genesis-world)
try:
    from .genesis import (
        GenesisConfig,
        EntitySpec,
        build_scene,
        GenesisRunner,
        genesis_traj_to_upd,
        genesis_trajs_to_upd,
    )
except Exception:
    GenesisConfig = None          # type: ignore
    EntitySpec = None             # type: ignore
    build_scene = None            # type: ignore
    GenesisRunner = None          # type: ignore
    genesis_traj_to_upd = None    # type: ignore
    genesis_trajs_to_upd = None   # type: ignore

__all__ = [
    # OpenFOAM
    "OpenFOAMCaseTemplate",
    "stage_case_for_scenario",
    "OpenFOAMRunConfig",
    "run_openfoam_case",
    "write_sample_dict_cloud",
    "run_sampling",
    "read_sampled_scalar_field",
    "export_bundle",
    "openfoam_case_to_upd",
    # MATLAB
    "MATLABEngine",
    "run_matlab_script",
    "run_matlab_function",
    "load_mat",
    "save_mat",
    "mat_to_upd",
    # Modelica / FMI
    "FMUSimConfig",
    "simulate_fmu",
    "fmu_to_upd",
    "OMSimConfig",
    "simulate_openmodelica",
    "om_to_upd",
    "modelica_result_to_upd",
    # FEniCS
    "FEniCSConfig",
    "solve_and_package",
    "dof_to_upd",
    "FEniCSWorkflow",
    # MuJoCo
    "MuJoCoConfig",
    "load_model",
    "make_data",
    "MuJoCoRunner",
    "trajectory_to_upd",
    "trajectories_to_upd",
    "MJXParallelRunner",
    # Genesis AI
    "GenesisConfig",
    "EntitySpec",
    "build_scene",
    "GenesisRunner",
    "genesis_traj_to_upd",
    "genesis_trajs_to_upd",
    # TurboDesigner
    "TurboDesignerConfig",
    "TurboDesignerWorkflow",
    "run_turbodesigner",
    "turbodesigner_to_upd",
]
