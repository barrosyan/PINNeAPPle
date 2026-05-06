"""External tool integrations for PINNeAPPle.

Subpackages
-----------
openfoam  — case staging, execution, field sampling, and UPD packaging
matlab    — MATLAB Engine API wrapper, subprocess runner, .mat I/O
modelica  — FMU simulation (fmpy) and OpenModelica (OMPython)
fenics    — workflow layer on top of pinneaple_solvers.fenics_bridge

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

# FEniCS (optional: pinneaple_solvers.fenics_bridge + dolfinx / legacy FEniCS)
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
]
