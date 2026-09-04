"""OpenFOAM field extraction → UPD PhysicalSample.

Moved from pinneapple_geom/io/openfoam.py: field reading is a data concern,
not a geometry concern, and requires pinneapple_data imports.

Internal-field parsing is delegated to ``binary_reader.read_internal_field``,
which handles both ASCII and binary FoamFiles (detected per-file from each
file's own header, since a case's declared ``writeFormat`` and an
individual field file's actual format can differ -- e.g. a hand-written
initial condition is commonly ASCII even when the solver writes binary for
every later time). This replaced an ASCII-only regex parser that raised or
silently mis-parsed on any binary-format field, which is the OpenFOAM
default for most real cases.

When ``constant/polyMesh`` is present, cell centers/sizes are now
reconstructed directly from the mesh (``mesh_reader.load_mesh``) instead of
requiring the case to have been run with ``writeCellCentres`` -- and the
returned ``PhysicalSample`` is tagged ``domain={"type": "mesh"}`` with a
proper ``geometry`` object (a ``.nodes`` array), not ``{"type": "grid"}``.
The previous "grid" tag was wrong for finite-volume cell data (an
unstructured point cloud, not a structured grid) and broke
``pinneapple_data.dataloaders.build_physical_sample_dataloader``'s own
grid-vs-mesh branch, which expects an ``xr.Dataset`` on the grid path and
never got one from this function.
"""
from __future__ import annotations

import os
from typing import Dict, Sequence

from . import binary_reader as _bin
from . import mesh_reader as _mesh


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


def _read_internal_field(path: str, n_cells_hint: int = None):
    """Read one field file's ``internalField`` (ASCII or binary, whichever
    that file's own header says) and return a torch tensor. ``n_cells_hint``
    broadcasts a uniform value to the mesh's actual cell count when known
    (from a reconstructed polyMesh); without it a uniform field stays a
    length-1/length-k tensor, as before this function gained mesh support.
    """
    import numpy as np
    import torch

    with open(path, "rb") as f:
        data = f.read()
    arr, is_uniform, n_components = _bin.read_internal_field(data, path)
    if is_uniform and n_cells_hint:
        if n_components == 1:
            arr = np.full((n_cells_hint,), arr[0], dtype=arr.dtype)
        else:
            arr = np.tile(arr[None, :], (n_cells_hint, 1))
    return torch.as_tensor(arr, dtype=torch.float32)


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
    PhysicalSample with fields dict and provenance metadata. ``domain`` is
    ``{"type": "mesh", ...}`` with a populated ``geometry.nodes`` when
    ``constant/polyMesh`` is present (the common case, and now readable
    whether it is ASCII or binary); it falls back to the previous
    ``{"type": "grid", "coords": {...}}`` shape (populated only if a ``C``
    cell-centers field exists) when no polyMesh is found, unchanged from
    before for callers relying on that path.
    """
    import torch
    from pinneapple_data.physical_sample import PhysicalSample

    tdir = os.path.join(case_dir, time) if time else _latest_time_dir(case_dir)
    time_dir_name = time or os.path.basename(tdir)

    mesh = None
    polymesh_dir = os.path.join(case_dir, "constant", "polyMesh")
    if os.path.isdir(polymesh_dir) and all(
        os.path.exists(os.path.join(polymesh_dir, f)) for f in ("points", "owner", "neighbour", "faces")
    ):
        try:
            mesh = _mesh.load_mesh(case_dir)
        except Exception:
            mesh = None  # fall back to the C-field / no-coords path below

    n_cells_hint = mesh.nodes.shape[0] if mesh is not None else None
    out_fields: Dict[str, "torch.Tensor"] = {}
    for f in fields:
        path = os.path.join(tdir, f)
        if os.path.exists(path):
            out_fields[f] = _read_internal_field(path, n_cells_hint=n_cells_hint)

    try:
        t_value = float(time_dir_name)
    except ValueError:
        t_value = None

    if mesh is not None:
        n_cells = mesh.nodes.shape[0]
        state = {k: v.numpy() for k, v in out_fields.items()}
        if t_value is not None:
            import numpy as np
            state["_time"] = np.full((n_cells,), t_value, dtype=np.float32)
        return PhysicalSample(
            state=state,
            geometry=mesh,
            domain={"type": "mesh", "n_cells": n_cells},
            provenance={
                "version": "0.1",
                "physics_domain": "cfd",
                "source": "openfoam",
                "case_dir": os.path.abspath(case_dir),
                "time_dir": time_dir_name,
                "mesh_source": "constant/polyMesh (binary_reader/mesh_reader)",
            },
            schema={"units": {}},
        )

    # No polyMesh found: fall back to the original C-field-or-nothing path.
    coords: Dict[str, "torch.Tensor"] = {}
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
