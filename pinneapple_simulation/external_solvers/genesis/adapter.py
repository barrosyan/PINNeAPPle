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
    dt: Optional[float] = None,
):
    """Convert a Genesis AI trajectory dict to a UPD PhysicalSample.

    Parameters
    ----------
    traj : output from GenesisRunner.simulate().
        Keys are field names (e.g. "robot_pos", "robot_vel", "step").
    time_array : (T,) float array of simulation timestamps.
        If None and "step" is present, the time coordinate is derived from
        the step index (see ``dt``).
    field_map : optional rename map {traj_key: upd_field_name}.
    meta_extra : extra keys merged into the UPD meta dict.
    dt : simulated seconds per ``scene.step()`` call (``GenesisConfig.dt``).
        Used to convert the raw step index into elapsed sim time
        (``time = step * dt``) when ``time_array`` is not supplied. If not
        given, the step index is exposed under a "step" coordinate instead
        of "time", since a bare step count is not a time value.

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
        step_idx = np.asarray(traj["step"], dtype=np.float32)
        if dt is not None:
            coords = {"time": step_idx * float(dt)}
        else:
            coords = {"step": step_idx}
    else:
        coords = {}

    provenance = {"version": "0.1", "source": "genesis", "solver": "genesis.scene.step"}
    units: Dict[str, str] = {}
    if meta_extra:
        provenance.update(meta_extra.get("provenance", meta_extra))
        units.update(meta_extra.get("units", {}))

    return PhysicalSample(
        state=fields,
        domain={"type": "grid", "coords": coords},
        provenance=provenance,
        schema={"units": units},
    )


def genesis_trajs_to_upd(
    trajs: List[dict],
    time_array: Optional[np.ndarray] = None,
    field_map: Optional[Dict[str, str]] = None,
    meta_extra: Optional[dict] = None,
    dt: Optional[float] = None,
) -> list:
    """Convert a list of Genesis trajectory dicts to PhysicalSamples."""
    return [
        genesis_traj_to_upd(
            t, time_array=time_array, field_map=field_map, meta_extra=meta_extra, dt=dt
        )
        for t in trajs
    ]
