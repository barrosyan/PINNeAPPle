"""Convert MATLAB solver output (.mat file) to UPD PhysicalSample."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np

from .mat_io import load_mat


def mat_to_upd(
    mat_path: Union[str, Path],
    *,
    field_keys: Optional[Sequence[str]] = None,
    coord_keys: Optional[Sequence[str]] = None,
    units: Optional[Dict[str, str]] = None,
):
    """Load a .mat file and package its arrays as a UPD PhysicalSample.

    Parameters
    ----------
    mat_path : path to the .mat file
    field_keys : variable names to treat as physical fields.
        If None, all numeric variables not in coord_keys are used.
    coord_keys : variable names to treat as coordinate arrays (e.g. "x", "t").
    units : optional dict mapping field name → unit string

    Returns
    -------
    PhysicalSample
    """
    import torch
    from pinneapple_data.physical_sample import PhysicalSample

    data = load_mat(mat_path)
    coord_set = set(coord_keys or [])

    fields: Dict[str, "torch.Tensor"] = {}
    coords: Dict[str, np.ndarray] = {}

    for k, v in data.items():
        arr = np.asarray(v)
        if not np.issubdtype(arr.dtype, np.number):
            continue
        if k in coord_set:
            coords[k] = arr.ravel().astype(np.float64)
        elif field_keys is None or k in field_keys:
            fields[k] = torch.as_tensor(arr, dtype=torch.float32)

    return PhysicalSample(
        state=fields,
        domain={"type": "grid", "coords": coords},
        provenance={
            "version": "0.1",
            "source": "matlab",
            "mat_path": str(Path(mat_path).resolve()),
        },
        schema={"units": units or {}},
    )
