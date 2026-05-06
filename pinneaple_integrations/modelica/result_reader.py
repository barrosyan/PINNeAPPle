"""Convert Modelica/FMI simulation result dicts to UPD PhysicalSample."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def modelica_result_to_upd(
    result: Dict[str, np.ndarray],
    *,
    time_key: str = "time",
    field_keys: Optional[Sequence[str]] = None,
    units: Optional[Dict[str, str]] = None,
    source: str = "modelica",
):
    """Package a Modelica simulation result dict as a UPD PhysicalSample.

    Parameters
    ----------
    result : dict[str, ndarray] as returned by simulate_fmu or simulate_openmodelica
    time_key : key for the time axis in *result*
    field_keys : subset of keys to include as physical fields;
        if None all keys except *time_key* are used
    units : optional dict mapping field name → unit string
    source : provenance tag ("modelica_fmu", "openmodelica", ...)

    Returns
    -------
    PhysicalSample
    """
    import torch
    from pinneaple_data.physical_sample import PhysicalSample

    time = result.get(time_key)
    coords = {time_key: np.asarray(time)} if time is not None else {}

    keys = list(field_keys) if field_keys is not None else [
        k for k in result if k != time_key
    ]
    fields = {
        k: torch.as_tensor(np.asarray(result[k]), dtype=torch.float32)
        for k in keys
        if k in result
    }

    return PhysicalSample(
        fields=fields,
        coords=coords,
        meta={
            "upd": {"version": "0.1", "source": source},
            "provenance": {},
            "units": units or {},
        },
    )
