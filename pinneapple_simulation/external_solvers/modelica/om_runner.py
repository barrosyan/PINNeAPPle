"""OpenModelica simulation via OMPython.

Requires: pip install OMPython
and a working OpenModelica installation (omc on PATH).
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class OMSimConfig:
    """Configuration for an OpenModelica simulation.

    Parameters
    ----------
    model_name : fully-qualified Modelica model name (e.g. "Modelica.Thermal.HeatTransfer.Examples.TwoMasses")
    start_time : simulation start time
    stop_time : simulation end time
    number_of_intervals : number of output intervals
    tolerance : solver tolerance
    parameters : dict of parameter overrides {name: value}
    output_variables : variables to extract; None = all
    """
    model_name: str
    start_time: float = 0.0
    stop_time: float = 1.0
    number_of_intervals: int = 500
    tolerance: float = 1e-6
    parameters: Dict[str, Any] = field(default_factory=dict)
    output_variables: Optional[List[str]] = None


def simulate_openmodelica(
    model_file: Union[str, Path],
    config: OMSimConfig,
    *,
    working_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, np.ndarray]:
    """Load and simulate a Modelica model using OpenModelica.

    Parameters
    ----------
    model_file : path to the .mo source file
    config : simulation configuration
    working_dir : directory for intermediate files; defaults to a temp dir

    Returns
    -------
    dict mapping variable name → 1-D numpy array (includes "time")
    """
    try:
        from OMPython import OMCSessionZMQ
    except ImportError:
        raise ImportError(
            "OMPython is required: pip install OMPython\n"
            "OpenModelica (omc) must also be on PATH."
        )

    work = str(working_dir or tempfile.mkdtemp(prefix="om_"))
    omc = OMCSessionZMQ()

    try:
        omc.sendExpression(f'cd("{work}")')
        omc.sendExpression(f'loadFile("{Path(model_file).resolve()}")')

        for k, v in config.parameters.items():
            omc.sendExpression(
                f'setParameterValue({config.model_name}, {k}, {v})'
            )

        omc.sendExpression(
            f"simulate({config.model_name}, "
            f"startTime={config.start_time}, "
            f"stopTime={config.stop_time}, "
            f"numberOfIntervals={config.number_of_intervals}, "
            f"tolerance={config.tolerance})"
        )

        result_path = os.path.join(work, f"{config.model_name}_res.mat")
        return _read_om_result(result_path, config.output_variables)
    finally:
        omc.sendExpression("quit()")


def _read_om_result(
    mat_path: str,
    variables: Optional[List[str]],
) -> Dict[str, np.ndarray]:
    """Parse an OpenModelica result .mat file (DSres format)."""
    try:
        import scipy.io as sio
    except ImportError:
        raise ImportError("scipy is required to read OM results: pip install scipy")

    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    # OM DSres format: 'name' (char array rows) + 'data_2' (float matrix, rows = vars)
    raw_names = data.get("name", [])
    names = [str(n).strip() for n in raw_names] if len(raw_names) else []
    mat = data.get("data_2", data.get("data_1", np.zeros((0, 0))))

    out: Dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        if i < mat.shape[0] and (variables is None or name in variables):
            out[name] = mat[i]
    return out


def om_to_upd(
    model_file: Union[str, Path],
    config: OMSimConfig,
    **sim_kwargs,
):
    """Simulate a Modelica model with OpenModelica and return a UPD PhysicalSample."""
    import torch
    from pinneapple_data.physical_sample import PhysicalSample

    results = simulate_openmodelica(model_file, config, **sim_kwargs)
    time = results.pop("time", None)
    coords = {"time": time} if time is not None else {}
    fields = {
        k: torch.as_tensor(np.asarray(v), dtype=torch.float32)
        for k, v in results.items()
    }
    return PhysicalSample(
        fields=fields,
        coords=coords,
        meta={
            "upd": {"version": "0.1", "source": "openmodelica"},
            "provenance": {
                "model_file": str(Path(model_file).resolve()),
                "model_name": config.model_name,
            },
            "units": {},
        },
    )
