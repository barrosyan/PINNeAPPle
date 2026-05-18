"""OpenFOAM field extraction → UPD PhysicalSample.

Moved from pinneapple_geom/io/openfoam.py: field reading is a data concern,
not a geometry concern, and requires pinneapple_data imports.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Dict, Optional, Sequence


def _latest_time_dir(case_dir: str) -> str:
    times = []
    for p in os.listdir(case_dir):
        try:
            float(p)
            times.append(p)
        except Exception:
            pass
    if not times:
        raise FileNotFoundError("No time directories found in OpenFOAM case.")
    return os.path.join(case_dir, sorted(times, key=float)[-1])


def _read_uniform_field(text: str) -> Optional[float]:
    m = re.search(r"uniform\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", text)
    return float(m.group(1)) if m else None


def _read_internal_field(path: str):
    import torch
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()

    uni = _read_uniform_field(txt)
    if uni is not None:
        return torch.tensor([uni], dtype=torch.float32)

    m = re.search(r"nonuniform\s+List<scalar>\s+\d+\s*\((.*?)\)\s*;", txt, flags=re.S)
    if m:
        nums = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", m.group(1))
        return torch.tensor([float(x) for x in nums], dtype=torch.float32)

    raise ValueError(f"Unsupported OpenFOAM field format: {path}")


def openfoam_case_to_upd(
    case_dir: str,
    *,
    time: str | None = None,
    fields: Sequence[str] = ("p", "U"),
):
    """Read OpenFOAM internalField data and package as a UPD PhysicalSample.

    Parameters
    ----------
    case_dir : path to the OpenFOAM case directory
    time : time directory name; uses latest if None
    fields : field names to extract (must exist in the time directory)

    Returns
    -------
    PhysicalSample with fields dict and provenance metadata.
    """
    import torch
    from pinneapple_data.physical_sample import PhysicalSample

    tdir = os.path.join(case_dir, time) if time else _latest_time_dir(case_dir)

    out_fields: Dict[str, "torch.Tensor"] = {}
    for f in fields:
        path = os.path.join(tdir, f)
        if os.path.exists(path):
            out_fields[f] = _read_internal_field(path)

    return PhysicalSample(
        fields=out_fields,
        coords={"time_dir": time or os.path.basename(tdir)},
        meta={
            "upd": {"version": "0.1", "domain": "cfd", "source": "openfoam"},
            "provenance": {
                "case_dir": os.path.abspath(case_dir),
                "time_dir": os.path.basename(tdir),
            },
            "units": {},
        },
    )
