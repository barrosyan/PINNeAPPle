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


_NUM = r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"


def _read_uniform_field(text: str) -> Optional[list]:
    """Match a scalar ('uniform 1;') or vector/tensor ('uniform (1 2 3);') value."""
    m = re.search(rf"uniform\s+\(({_NUM}(?:\s+{_NUM})*)\)", text)
    if m:
        return [float(x) for x in re.findall(_NUM, m.group(1))]
    m = re.search(rf"uniform\s+({_NUM})", text)
    if m:
        return [float(m.group(1))]
    return None


def _read_internal_field(path: str):
    import torch
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()

    # Scope matching to the 'internalField ... ;' statement only. Matching
    # against the whole file is wrong: boundaryField patches routinely carry
    # their own 'value uniform (...);' entries (e.g. a fixedValue patch on an
    # otherwise nonuniform solved field), and a whole-file search would latch
    # onto the first such occurrence instead of the actual internal field
    # data -- silently returning a boundary value in place of the field.
    m_field = re.search(r"internalField\s+(.*?);", txt, flags=re.S)
    field_txt = m_field.group(1) if m_field else txt

    uni = _read_uniform_field(field_txt)
    if uni is not None:
        return torch.tensor([uni] if len(uni) > 1 else uni, dtype=torch.float32)

    m = re.search(r"nonuniform\s+List<scalar>\s+\d+\s*\((.*?)\)", field_txt, flags=re.S)
    if m:
        nums = re.findall(_NUM, m.group(1))
        return torch.tensor([float(x) for x in nums], dtype=torch.float32)

    m = re.search(
        r"nonuniform\s+List<(?:vector|tensor|symmTensor)>\s+\d+\s*\((.*)\)",
        field_txt,
        flags=re.S,
    )
    if m:
        tuples = re.findall(r"\(([^()]*)\)", m.group(1))
        rows = [[float(x) for x in re.findall(_NUM, t)] for t in tuples]
        return torch.tensor(rows, dtype=torch.float32)

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
    time_dir_name = time or os.path.basename(tdir)

    out_fields: Dict[str, "torch.Tensor"] = {}
    for f in fields:
        path = os.path.join(tdir, f)
        if os.path.exists(path):
            out_fields[f] = _read_internal_field(path)

    coords: Dict[str, "torch.Tensor"] = {}
    try:
        t_value = float(time_dir_name)
    except ValueError:
        t_value = None

    # Cell centers ("C") are only present if the case was run through
    # OpenFOAM's writeCellCentres function object; without them there is no
    # way to recover per-cell spatial coordinates from the field files alone.
    c_path = os.path.join(tdir, "C")
    n_cells = next((v.shape[0] for v in out_fields.values() if v.ndim >= 1), None)
    if os.path.exists(c_path):
        centers = _read_internal_field(c_path)
        if centers.ndim == 2 and centers.shape[1] == 3:
            coords["x"] = centers[:, 0]
            coords["y"] = centers[:, 1]
            coords["z"] = centers[:, 2]
            n_cells = centers.shape[0]

    if t_value is not None:
        coords["time"] = (
            torch.full((n_cells,), t_value, dtype=torch.float32)
            if n_cells
            else torch.tensor([t_value], dtype=torch.float32)
        )

    return PhysicalSample(
        state=out_fields,
        domain={"type": "grid", "coords": coords},
        provenance={
            "version": "0.1",
            "physics_domain": "cfd",
            "source": "openfoam",
            "case_dir": os.path.abspath(case_dir),
            "time_dir": time_dir_name,
        },
        schema={"units": {}},
    )
