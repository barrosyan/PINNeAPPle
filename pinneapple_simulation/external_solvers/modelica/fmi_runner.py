"""FMI/FMU simulation via fmpy.

FMI (Functional Mock-up Interface) is the open standard for model exchange
across tools (OpenModelica, Dymola, Simulink, Modelon, etc.).

Requires: pip install fmpy
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np


@dataclass
class FMUSimConfig:
    """Configuration for a single FMU simulation run.

    Parameters
    ----------
    start_time : simulation start time
    stop_time : simulation end time
    step_size : fixed communication step size; None lets the FMU solver decide
    output_interval : output recording interval; defaults to step_size
    parameters : initial/override parameter values  {name: value}
    output_variables : variables to record; None = all FMU outputs
    """
    start_time: float = 0.0
    stop_time: float = 1.0
    step_size: Optional[float] = None
    output_interval: Optional[float] = None
    parameters: Dict[str, float] = field(default_factory=dict)
    output_variables: Optional[List[str]] = None


def simulate_fmu(
    fmu_path: Union[str, Path],
    config: FMUSimConfig,
) -> Dict[str, np.ndarray]:
    """Simulate an FMU and return output variables as numpy arrays.

    Parameters
    ----------
    fmu_path : path to the .fmu archive
    config : simulation configuration

    Returns
    -------
    dict mapping variable name → 1-D numpy array (includes "time")
    """
    try:
        from fmpy import simulate_fmu as _sim
    except ImportError:
        raise ImportError("fmpy is required for FMU simulation: pip install fmpy")

    result = _sim(
        str(Path(fmu_path).expanduser().resolve()),
        start_time=config.start_time,
        stop_time=config.stop_time,
        step_size=config.step_size,
        output_interval=config.output_interval,
        start_values=config.parameters or None,
        output=config.output_variables,
        validate=False,
    )
    # fmpy returns a numpy structured array
    return {name: np.asarray(result[name]) for name in result.dtype.names}


def fmu_to_upd(
    fmu_path: Union[str, Path],
    config: FMUSimConfig,
):
    """Simulate an FMU and package results as a UPD PhysicalSample.

    Returns
    -------
    PhysicalSample with fields = output signals, coords = {"time": ...}
    """
    import torch
    from pinneapple_data.physical_sample import PhysicalSample

    results = simulate_fmu(fmu_path, config)
    time = results.pop("time", None)
    coords = {"time": time} if time is not None else {}
    fields = {
        k: torch.as_tensor(v.copy(), dtype=torch.float32)
        for k, v in results.items()
    }
    return PhysicalSample(
        state=fields,
        domain={"type": "grid", "coords": coords},
        provenance={
            "version": "0.1",
            "source": "modelica_fmu",
            "fmu_path": str(Path(fmu_path).resolve()),
            "start_time": config.start_time,
            "stop_time": config.stop_time,
        },
        schema={"units": {}},
    )
