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


def _row_to_str(row: np.ndarray) -> str:
    chars: List[str] = []
    for c in np.asarray(row).ravel():
        if isinstance(c, (bytes, np.bytes_)):
            ch = c.decode("latin-1")
        elif isinstance(c, (str, np.str_)):
            ch = str(c)
        else:
            code = int(c)
            ch = chr(code) if code != 0 else ""
        chars.append(ch)
    return "".join(chars).strip("\x00 ").strip()


def _read_om_result(
    mat_path: str,
    variables: Optional[List[str]],
) -> Dict[str, np.ndarray]:
    """Parse an OpenModelica result .mat file (DSres format).

    Variable name -> data lookup goes through 'dataInfo', not positional
    indexing into 'name': 'name' enumerates ALL variables (parameters/time
    in data_1, time-varying signals in data_2) while data_2 only holds the
    latter, in its own order. Each dataInfo row gives (dataset, signed
    1-based row) for the corresponding name; a negative row means the
    trajectory is the negative of that row's data. 'Aclass[3]' ("binNormal"
    vs "binTrans") says whether name/dataInfo/data_1/data_2 need transposing
    before this row-per-variable indexing applies.
    """
    try:
        import scipy.io as sio
    except ImportError:
        raise ImportError("scipy is required to read OM results: pip install scipy")

    data = sio.loadmat(
        mat_path, chars_as_strings=False, struct_as_record=False, squeeze_me=False
    )

    aclass = data.get("Aclass")
    trans = False
    if aclass is not None:
        flat = "".join(_row_to_str(row) for row in np.atleast_2d(np.asarray(aclass)))
        trans = "binTrans" in flat

    def _norm(arr, swap: bool) -> np.ndarray:
        # swapaxes(0, 1) rather than a full .T: 'name' can carry an extra
        # trailing char-code axis (from chars_as_strings=False), which a
        # full transpose would also reverse and corrupt.
        arr = np.asarray(arr) if arr is not None else np.zeros((0, 0))
        if arr.ndim < 2:
            arr = np.atleast_2d(arr)
        return np.swapaxes(arr, 0, 1) if swap else arr

    # 'name'/'dataInfo' are stored row-per-variable when Aclass says
    # "binNormal" and need transposing back to that shape when "binTrans".
    # The data_N blocks follow the OPPOSITE convention: they are already
    # row-per-variable under "binTrans" (no transpose needed) and stored
    # row-per-timestep (need transposing) under "binNormal". This matches
    # the reference DyMat reader (DyMatFile.__init__): dataInfo is indexed
    # as dataInfo[0][i]/[1][i] (columns=vars) under binTrans but
    # dataInfo[i][0]/[i][1] (rows=vars) under binNormal, while data_N is
    # used directly under binTrans and explicitly .transpose()'d under
    # binNormal.
    name_arr = _norm(data.get("name"), trans)
    names = [_row_to_str(row) for row in name_arr]

    data_info = _norm(data.get("dataInfo"), trans)
    data_1 = _norm(data.get("data_1", np.zeros((0, 0))), not trans)
    data_2 = _norm(data.get("data_2", np.zeros((0, 0))), not trans)

    n_time = data_2.shape[1] if data_2.ndim == 2 and data_2.shape[0] else 0

    out: Dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        if variables is not None and name not in variables:
            continue
        if i >= data_info.shape[0]:
            continue
        dataset = int(data_info[i][0])
        signed_idx = int(data_info[i][1])
        sign = -1.0 if signed_idx < 0 else 1.0
        row_idx = abs(signed_idx) - 1
        if dataset == 2 and 0 <= row_idx < data_2.shape[0]:
            out[name] = sign * data_2[row_idx]
        elif dataset == 1 and 0 <= row_idx < data_1.shape[0]:
            value = sign * data_1[row_idx][-1]
            out[name] = np.full(n_time, value) if n_time else np.asarray([value])
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
        state=fields,
        domain={"type": "grid", "coords": coords},
        provenance={
            "version": "0.1",
            "source": "openmodelica",
            "model_file": str(Path(model_file).resolve()),
            "model_name": config.model_name,
        },
        schema={"units": {}},
    )
