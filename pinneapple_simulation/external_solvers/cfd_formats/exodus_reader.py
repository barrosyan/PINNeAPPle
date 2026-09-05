"""Exodus II (``.exo``/``.e``) reader, via the classic-NetCDF layer.

Validation status
------------------
**Validated against a real classic-NetCDF Exodus file.** Written by
``meshio`` 5.3.5's real Exodus writer (itself using the real ``netCDF4``
package to do the actual encoding), forced to classic NetCDF-3 -- read
correctly, node coordinates and two vertex fields matched their
known-exact values exactly. See ``tests/fixtures/cfd_formats/README.md``
and ``tests/test_cfd_format_readers.py`` for the fixture and test.

This process also surfaced a real bug: pointing this reader at the
*other* real, valid Exodus file the same writer produces by default
(NetCDF-4/HDF5-based, no format override -- also genuine Exodus, many
modern solvers default to it for large meshes) raised a confusing raw
``scipy`` ``TypeError`` instead of a clear, actionable error. Fixed --
this now raises ``NotImplementedError`` naming the actual problem (HDF5
container, not classic NetCDF) and the ``netCDF4``/``h5netcdf`` fallback,
still true to the "raise on that failure rather than silently returning
nothing" contract below, just with a message that says why. Element
connectivity is still not read (see "Variables read" below); that part
remains unverified.

Format background
------------------
Exodus II (Sandia National Laboratories) is a schema layered on top of
classic NetCDF -- an ``.exo``/``.e`` file is a NetCDF file with a fixed set
of conventionally-named dimensions and variables. This reader uses
``scipy.io.netcdf_file`` (already a PINNeAPPle dependency via ``scipy``),
which reads the *classic* NetCDF format without any extra dependency; a
64-bit-offset or NetCDF-4/HDF5-based Exodus file (large meshes) needs
``netCDF4`` or ``h5netcdf`` instead and is out of scope here (raise on
that failure rather than silently returning nothing).

Variables read (Exodus II data model): ``coord`` (or ``coordx``/``coordy``/
``coordz`` when split), ``vals_nod_var<k>`` (nodal field ``k``, one array
per time step) and ``name_nod_var`` (their names, as a fixed-width char
array), and ``time_whole`` (the time values). Element-block connectivity
(``connect<k>``) is not read by this module -- only the node cloud and
nodal fields, which is what a PINN training pipeline needs; a caller
wanting cell/element structure can extend this reader with the
``connect<k>``/``eb_prop1`` variables the same way.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence


def _decode_name_array(arr) -> List[str]:
    """Exodus stores variable-name arrays as a NetCDF char matrix
    (n_names, name_strlen); decode each row to a plain, stripped string."""
    names = []
    for row in arr:
        try:
            s = b"".join(bytes(c) if isinstance(c, (bytes, bytearray)) else bytes([c]) for c in row)
        except TypeError:
            s = bytes(row)
        names.append(s.decode("ascii", errors="ignore").strip("\x00").strip())
    return names


def read_exodus(path: str, *, fields: Optional[Sequence[str]] = None, time_index: int = -1):
    """Read node coordinates and nodal fields from a classic-NetCDF Exodus
    II file.

    Parameters
    ----------
    path : path to the .exo/.e file.
    fields : nodal variable names to extract (from Exodus's own
        ``name_nod_var`` list); all are returned if None.
    time_index : which time step's nodal-variable values to read (Exodus
        stores one array per (variable, time step) pair); -1 = last.

    Returns
    -------
    dict with keys ``"coords"`` (N, 3), ``"fields"`` (``{name: (N,)}``),
    and ``"time"`` (the time value read, or None if the file has no
    ``time_whole`` variable).
    """
    import numpy as np
    from scipy.io import netcdf_file

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "rb") as _fh:
        _head = _fh.read(8)
    if _head[:8] == b"\x89HDF\r\n\x1a\n":
        raise NotImplementedError(
            f"{path}: this is an HDF5-based (NetCDF-4) Exodus file -- "
            "scipy.io.netcdf_file only reads the classic NetCDF format (see this module's "
            "docstring). Re-export from a classic-format writer, or read it with `netCDF4` "
            "(`netCDF4.Dataset(path)`) / `h5netcdf` instead; this reader does not support that "
            "variant yet."
        )

    try:
        _nc_ctx = netcdf_file(path, mmap=False)
    except TypeError as e:
        raise NotImplementedError(
            f"{path}: does not look like a classic-NetCDF (Exodus II) file that "
            f"scipy.io.netcdf_file can read ({e}). If this is a 64-bit-offset or NetCDF-4 "
            "variant, see this module's docstring for the netCDF4/h5netcdf fallback."
        ) from e

    with _nc_ctx as nc:
        variables = nc.variables
        if "coord" in variables:
            coord = np.array(variables["coord"][:])  # (num_dim, num_nodes) in Exodus's own layout
            coords = coord.T
        else:
            axes = [variables[k][:] for k in ("coordx", "coordy", "coordz") if k in variables]
            if not axes:
                raise ValueError(f"exodus_reader: no 'coord'/'coordx' variable found in {path}")
            coords = np.stack([np.asarray(a) for a in axes], axis=1)
        if coords.shape[1] < 3:
            coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))

        time_val = None
        if "time_whole" in variables:
            t = np.array(variables["time_whole"][:])
            if t.size:
                time_val = float(t[time_index])

        out_fields: Dict[str, "np.ndarray"] = {}
        if "name_nod_var" in variables:
            names = _decode_name_array(np.array(variables["name_nod_var"][:]))
            for k, name in enumerate(names, start=1):
                if fields is not None and name not in fields:
                    continue
                var = variables.get(f"vals_nod_var{k}")
                if var is None:
                    continue
                arr = np.array(var[:])  # (num_time_steps, num_nodes)
                out_fields[name] = arr[time_index]

    return {"coords": coords, "fields": out_fields, "time": time_val}


def exodus_to_upd(path: str, *, fields: Optional[Sequence[str]] = None, time_index: int = -1):
    """Read an Exodus II file and package it as a UPD ``PhysicalSample``
    (same ``domain={"type": "mesh"}`` / ``geometry.nodes`` contract as
    ``openfoam.field_reader.openfoam_case_to_upd``)."""
    import numpy as np
    from pinneapple_data.physical_sample import PhysicalSample
    from ..openfoam.mesh_reader import MeshGeometry

    result = read_exodus(path, fields=fields, time_index=time_index)
    coords = result["coords"]
    geom = MeshGeometry(
        nodes=coords,
        cell_size=np.zeros_like(coords),
        cell_delta=np.zeros(coords.shape[0]),
        bounds_min=coords.min(axis=0),
        bounds_max=coords.max(axis=0),
        n_points=coords.shape[0],
        n_faces=0,
        n_internal_faces=0,
    )
    state = dict(result["fields"])
    if result["time"] is not None:
        state["_time"] = np.full((coords.shape[0],), result["time"], dtype=np.float32)
    return PhysicalSample(
        state=state,
        geometry=geom,
        domain={"type": "mesh", "n_cells": coords.shape[0]},
        provenance={
            "version": "0.1", "physics_domain": "cfd", "source": "exodus",
            "case_dir": os.path.abspath(path),
            "validation": "validated against a real classic-NetCDF Exodus file (real meshio+"
                           "netCDF4 writer) -- see module docstring; element connectivity "
                           "not read, still unverified",
        },
        schema={"units": {}},
    )
