"""Adapters from Genesis AI trajectory dicts to PINNeAPPle's UPD format.

genesis_traj_to_upd    — single trajectory dict → PhysicalSample
genesis_trajs_to_upd   — list of trajectory dicts → list[PhysicalSample]
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def genesis_traj_to_upd(
    traj: dict,
    time_array: Optional[np.ndarray] = None,
    field_map: Optional[Dict[str, str]] = None,
    meta_extra: Optional[dict] = None,
):
    """Convert a Genesis AI trajectory dict to a UPD PhysicalSample.

    Parameters
    ----------
    traj : output from GenesisRunner.simulate().
        Keys are field names (e.g. "robot_pos", "robot_vel", "step").
    time_array : (T,) float array of simulation timestamps.
        If None and "step" is present, steps are used as the time coordinate.
    field_map : optional rename map {traj_key: upd_field_name}.
    meta_extra : extra keys merged into the UPD meta dict.

    Returns
    -------
    PhysicalSample  (from pinneapple_data)
    """
    import torch
    from pinneapple_data.physical_sample import PhysicalSample

    fm = field_map or {}
    skip = {"step"}

    fields: Dict[str, "torch.Tensor"] = {}
    for key, arr in traj.items():
        if key in skip:
            continue
        name = fm.get(key, key)
        flat = np.asarray(arr, dtype=np.float32)
        if flat.ndim == 1:
            flat = flat[:, None]
        fields[name] = torch.as_tensor(flat)

    if time_array is not None:
        coords = {"time": np.asarray(time_array, dtype=np.float32)}
    elif "step" in traj:
        coords = {"time": np.asarray(traj["step"], dtype=np.float32)}
    else:
        coords = {}

    meta = {
        "upd": {"version": "0.1", "source": "genesis"},
        "provenance": {"solver": "genesis.scene.step"},
        "units": {},
    }
    if meta_extra:
        meta.update(meta_extra)

    return PhysicalSample(fields=fields, coords=coords, meta=meta)


def genesis_trajs_to_upd(
    trajs: List[dict],
    time_array: Optional[np.ndarray] = None,
    field_map: Optional[Dict[str, str]] = None,
    meta_extra: Optional[dict] = None,
) -> list:
    """Convert a list of Genesis trajectory dicts to PhysicalSamples."""
    return [
        genesis_traj_to_upd(t, time_array=time_array, field_map=field_map, meta_extra=meta_extra)
        for t in trajs
    ]
