"""CGNS (CFD General Notation System) reader, via the CGNS/HDF5 mapping.

Validation status
------------------
**Not validated against a real CGNS file** -- unlike this repository's
OpenFOAM binary reader (``openfoam/binary_reader.py``,
``openfoam/mesh_reader.py``), which was checked byte-for-byte against a
real 244 MB LES case (cell centers, cell counts and a y+ value all matched
the case's own published documentation to 3+ significant figures), no CGNS
file was available to validate this module against. It implements the
documented CGNS/HDF5 mapping (SIDS-to-HDF5, as used by modern CGNS ≥ 3.x
built with the HDF5 backend -- the default for essentially every CGNS
writer in general use: CFD General Notation System committee spec,
`https://cgns.github.io/CGNS_docs_current/hdf5/index.html`) as precisely
as that specification allows without a reference file, but treat it as
**unverified** until exercised against real output from a real CGNS
writer (Pointwise, ANSYS Fluent's CGNS export, SU2, CGNS's own reference
`cgns_utils`, ...) and adjusted for whatever real-world deviation surfaces.
Please open an issue with a (even small/synthetic) reproducer file if this
misparses something.

Format background
------------------
A CGNS/HDF5 file is a plain HDF5 file where the CGNS node tree is mapped
onto nested HDF5 groups. Every CGNS node is one HDF5 group carrying two
attributes -- ``label`` (the CGNS/SIDS node type, e.g. ``"CGNSBase_t"``,
``"Zone_t"``, ``"GridCoordinates_t"``, ``"FlowSolution_t"``,
``"DataArray_t"``) and ``type`` (the CGNS data-type code, e.g. ``"R8"``
double, ``"I4"`` 32-bit int, ``"C1"`` char) -- and, for a leaf/data node, one
HDF5 dataset named literally ``" data"`` (with the CGNS-mandated leading
space) holding the actual numeric payload.

This reader walks that tree looking for the standard SIDS layout:
``CGNSBase_t -> Zone_t -> GridCoordinates_t -> DataArray_t`` (node
coordinates, conventionally named ``CoordinateX``/``Y``/``Z``) and
``Zone_t -> FlowSolution_t -> DataArray_t`` (solution fields, at whichever
grid location -- Vertex or CellCenter -- the writer used; both are
returned, the caller's ``fields`` argument selects by name regardless of
location).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence


def _node_label(group) -> str:
    lbl = group.attrs.get("label", b"")
    return lbl.decode() if isinstance(lbl, bytes) else str(lbl)


def _node_data(group):
    """The ``" data"`` dataset a CGNS/HDF5 leaf node carries, if any."""
    import numpy as np
    ds = group.get(" data")
    if ds is None:
        return None
    arr = ds[()]
    return np.asarray(arr)


def _find_children_by_label(group, label: str) -> List:
    return [child for name, child in group.items() if hasattr(child, "attrs") and _node_label(child) == label]


def _first_base_and_zone(root):
    bases = _find_children_by_label(root, "CGNSBase_t")
    if not bases:
        raise ValueError("cgns_reader: no CGNSBase_t node found -- is this a valid CGNS/HDF5 file?")
    base = bases[0]
    zones = _find_children_by_label(base, "Zone_t")
    if not zones:
        raise ValueError(f"cgns_reader: CGNSBase_t '{base.name}' has no Zone_t child.")
    return base, zones[0]


def read_cgns_mesh_and_fields(
    path: str,
    *,
    fields: Optional[Sequence[str]] = None,
    zone_index: int = 0,
):
    """Read node coordinates and solution fields from a CGNS/HDF5 file.

    Parameters
    ----------
    path : path to the .cgns file.
    fields : field (DataArray_t) names to extract from every
        FlowSolution_t container found in the zone; all are returned if
        None.
    zone_index : which Zone_t under the first CGNSBase_t to read (CGNS
        supports multiple zones/blocks per base; multi-zone assembly is
        not attempted here -- read each zone separately and combine
        yourself if you need the whole domain).

    Returns
    -------
    dict with keys ``"coords"`` (N, 3) and ``"fields"``
    (``{name: array}``, each ``(N,)`` or ``(N, k)``, at whatever grid
    location -- vertex or cell-center -- the writer used; the two are not
    distinguished or interpolated onto a common set here).
    """
    import h5py

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        base, zone = _first_base_and_zone(f)
        zones = _find_children_by_label(base, "Zone_t")
        if zone_index >= len(zones):
            raise IndexError(f"cgns_reader: zone_index={zone_index} but base only has {len(zones)} zone(s)")
        zone = zones[zone_index]

        grid_coord_nodes = _find_children_by_label(zone, "GridCoordinates_t")
        if not grid_coord_nodes:
            raise ValueError(f"cgns_reader: zone '{zone.name}' has no GridCoordinates_t node.")
        gc = grid_coord_nodes[0]
        coord_arrays = {}
        for name, child in gc.items():
            if hasattr(child, "attrs") and _node_label(child) == "DataArray_t":
                arr = _node_data(child)
                if arr is not None:
                    coord_arrays[name] = arr.reshape(-1)
        axis_order = [n for n in ("CoordinateX", "CoordinateY", "CoordinateZ") if n in coord_arrays]
        if not axis_order:
            raise ValueError(f"cgns_reader: zone '{zone.name}' GridCoordinates_t has no CoordinateX/Y/Z arrays.")
        import numpy as np
        coords = np.stack([coord_arrays[n] for n in axis_order], axis=1)

        out_fields: Dict[str, "object"] = {}
        for sol in _find_children_by_label(zone, "FlowSolution_t"):
            for name, child in sol.items():
                if not (hasattr(child, "attrs") and _node_label(child) == "DataArray_t"):
                    continue
                if fields is not None and name not in fields:
                    continue
                arr = _node_data(child)
                if arr is not None:
                    out_fields[name] = arr.reshape(-1)

    return {"coords": coords, "fields": out_fields}


def cgns_to_upd(path: str, *, fields: Optional[Sequence[str]] = None, zone_index: int = 0):
    """Read a CGNS/HDF5 file and package it as a UPD ``PhysicalSample``
    (``domain={"type": "mesh"}``, ``geometry.nodes`` = node coordinates --
    same contract ``openfoam.field_reader.openfoam_case_to_upd`` produces,
    see ``mesh_reader.MeshGeometry``)."""
    import numpy as np
    from pinneapple_data.physical_sample import PhysicalSample
    from ..openfoam.mesh_reader import MeshGeometry

    result = read_cgns_mesh_and_fields(path, fields=fields, zone_index=zone_index)
    coords = result["coords"]
    if coords.shape[1] < 3:
        pad = np.zeros((coords.shape[0], 3 - coords.shape[1]), dtype=coords.dtype)
        coords = np.concatenate([coords, pad], axis=1)
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
    return PhysicalSample(
        state=dict(result["fields"]),
        geometry=geom,
        domain={"type": "mesh", "n_cells": coords.shape[0]},
        provenance={
            "version": "0.1", "physics_domain": "cfd", "source": "cgns",
            "case_dir": os.path.abspath(path),
            "validation": "unverified against a real CGNS file -- see module docstring",
        },
        schema={"units": {}},
    )
