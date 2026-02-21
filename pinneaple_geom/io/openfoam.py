"""OpenFOAM case loading and field extraction for UPD PhysicalSample."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import os
import glob
import re

import torch
from pinneaple_data.physical_sample import PhysicalSample


def _latest_time_dir(case_dir: str) -> str:
    # pick latest numeric time folder
    times = []
    for p in os.listdir(case_dir):
        try:
            float(p)
            times.append(p)
        except Exception:
            pass
    if not times:
        raise FileNotFoundError("No time directories found in OpenFOAM case.")
    times_sorted = sorted(times, key=lambda s: float(s))
    return os.path.join(case_dir, times_sorted[-1])


def _read_uniform_field(text: str) -> Optional[float]:
    # matches: uniform 1.23;
    m = re.search(r"uniform\s+([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)", text)
    if m:
        return float(m.group(1))
    return None


def _read_internal_field(path: str) -> torch.Tensor:
    # MVP parser: supports uniform scalar or list of scalars
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()

    uni = _read_uniform_field(txt)
    if uni is not None:
        return torch.tensor([uni], dtype=torch.float32)

    # attempt: parse nonuniform list "( ... )"
    m = re.search(r"nonuniform\s+List<scalar>\s+\d+\s*\((.*?)\)\s*;", txt, flags=re.S)
    if m:
        body = m.group(1)
        vals = [float(v) for v in re.findall(r"[+-]?\d+(\.\d+)?([eE][+-]?\d+)?", body)]
        # regex returns tuples; rebuild robustly:
        nums = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", body)
        return torch.tensor([float(x) for x in nums], dtype=torch.float32)

    raise ValueError(f"Unsupported OpenFOAM field format: {path}")


def openfoam_case_to_upd(
    case_dir: str,
    *,
    time: str | None = None,
    fields: Sequence[str] = ("p", "U"),
) -> PhysicalSample:
    """
    Very small but practical OpenFOAM bridge:
      - reads internalField for requested fields at one time directory
      - stores as PhysicalSample fields

    Notes:
      - Full OpenFOAM parsing is complex; this MVP is for internalField extraction + provenance.
      - Geometry/mesh can be added later (polyMesh).
    """
    tdir = os.path.join(case_dir, time) if time else _latest_time_dir(case_dir)

    out_fields: Dict[str, torch.Tensor] = {}
    for f in fields:
        path = os.path.join(tdir, f)
        if not os.path.exists(path):
            # common OpenFOAM variants: "p_rgh" etc.
            continue
        out_fields[f] = _read_internal_field(path)

    sample = PhysicalSample(
        fields=out_fields,
        coords={"time_dir": time or os.path.basename(tdir)},
        meta={
            "upd": {"version": "0.1", "domain": "cfd", "source": "openfoam"},
            "provenance": {"case_dir": os.path.abspath(case_dir), "time_dir": os.path.basename(tdir)},
            "units": {},  # OpenFOAM units parsing optional; can be added later
        },
    )
    return sample
