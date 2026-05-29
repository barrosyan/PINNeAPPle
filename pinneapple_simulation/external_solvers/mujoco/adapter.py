"""Adapters from MuJoCo trajectory dicts to PINNeAPPle's UPD format.

trajectory_to_upd   — converts a single trajectory dict → PhysicalSample
trajectories_to_upd — converts a list of trajectories → list[PhysicalSample]
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def trajectory_to_upd(
    traj: dict,
    field_map: Optional[Dict[str, str]] = None,
    meta_extra: Optional[dict] = None,
):
    """Convert a MuJoCo trajectory dict to a UPD PhysicalSample.

    Parameters
    ----------
    traj : output from MuJoCoRunner.simulate() — keys: time, qpos, qvel,
        ctrl, xpos, sensor
    field_map : optional rename map {traj_key: upd_field_name}
        Default fields exposed: qpos, qvel, ctrl, sensor
    meta_extra : extra keys merged into the UPD meta dict

    Returns
    -------
    PhysicalSample  (from pinneapple_data)
    """
    import torch
    from pinneapple_data.physical_sample import PhysicalSample

    default_fields = ["qpos", "qvel", "ctrl", "sensor"]
    fm = field_map or {}

    fields: Dict[str, "torch.Tensor"] = {}
    for key in default_fields:
        arr = traj.get(key)
        if arr is None or arr.size == 0:
            continue
        name = fm.get(key, key)
        fields[name] = torch.as_tensor(np.asarray(arr, dtype=np.float32))

    # xpos: flatten body × 3 into a single field
    if "xpos" in traj and traj["xpos"].size > 0:
        xpos_flat = traj["xpos"].reshape(traj["xpos"].shape[0], -1)
        name = fm.get("xpos", "xpos")
        fields[name] = torch.as_tensor(np.asarray(xpos_flat, dtype=np.float32))

    coords = {"time": traj["time"].astype(np.float32)}

    meta = {
        "upd": {"version": "0.1", "source": "mujoco"},
        "provenance": {"solver": "mujoco.mj_step"},
        "units": {},
    }
    if meta_extra:
        meta.update(meta_extra)

    return PhysicalSample(fields=fields, coords=coords, meta=meta)


def trajectories_to_upd(
    trajs: List[dict],
    field_map: Optional[Dict[str, str]] = None,
    meta_extra: Optional[dict] = None,
) -> list:
    """Convert a list of trajectory dicts to a list of PhysicalSamples."""
    return [
        trajectory_to_upd(t, field_map=field_map, meta_extra=meta_extra)
        for t in trajs
    ]
