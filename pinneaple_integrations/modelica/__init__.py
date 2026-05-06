"""Modelica integration for PINNeAPPle.

Two execution backends:
  fmi_runner  — FMU simulation via fmpy (portable, works with any FMI-compliant tool)
  om_runner   — OpenModelica simulation via OMPython (requires omc on PATH)

Common utilities:
  result_reader — convert simulation result dicts to UPD PhysicalSample

Typical workflow
----------------
>>> from pinneaple_integrations.modelica import FMUSimConfig, fmu_to_upd
>>> cfg = FMUSimConfig(stop_time=10.0, parameters={"k": 2.0}, output_variables=["x", "v"])
>>> sample = fmu_to_upd("MyModel.fmu", cfg)
"""
from .fmi_runner import FMUSimConfig, simulate_fmu, fmu_to_upd
from .om_runner import OMSimConfig, simulate_openmodelica, om_to_upd
from .result_reader import modelica_result_to_upd

__all__ = [
    # FMI / FMU
    "FMUSimConfig",
    "simulate_fmu",
    "fmu_to_upd",
    # OpenModelica
    "OMSimConfig",
    "simulate_openmodelica",
    "om_to_upd",
    # Shared utilities
    "modelica_result_to_upd",
]
