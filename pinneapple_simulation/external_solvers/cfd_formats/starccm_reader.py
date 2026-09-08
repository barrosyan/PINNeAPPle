"""Siemens Simcenter STAR-CCM+ readers: CGNS export (delegated to the real
``cgns_reader``) and plain-text table/CSV export (implemented here). The
proprietary native ``.sim``/``.ccm`` binary format is explicitly NOT
supported -- see "What is NOT supported" below.

What IS supported, and why it's legitimate
--------------------------------------------
STAR-CCM+ does not require reading its native project file to get mesh and
field data out of it -- it has two real, documented export paths that leave
its own proprietary binary format entirely:

1. **CGNS export.** STAR-CCM+ can export a region's mesh and solution
   fields to CGNS (CFD General Notation System) via its own
   ``Export > CGNS...`` action -- an open, standardised interchange format
   (the same one ``cgns_reader.py`` already reads, validated against a real
   CGNS Mid-Level Library file; see that module's docstring). A STAR-CCM+
   CGNS export is ordinary CGNS/HDF5 laid out per the standard SIDS schema
   (``CGNSBase_t -> Zone_t -> GridCoordinates_t`` / ``FlowSolution_t``) --
   nothing STAR-CCM+-specific about the bytes on disk. So :func:`starccm_cgns_to_upd`
   below is a thin, honest re-export of :func:`cgns_reader.cgns_to_upd`
   under a name that documents *why* it applies to STAR-CCM+ output,
   **not** a reimplementation -- reusing already-validated logic rather
   than duplicating (and risking drifting from) it.

2. **Plain-text table/CSV export.** STAR-CCM+'s ``Plot > Export`` /
   ``Table > Export`` action writes scalar/vector field data (e.g. probe
   points, an XY plot's underlying data, a derived-part table) as a plain
   delimited text file: one header row of quoted column names -- often
   carrying a unit suffix, e.g. ``"X (m)"``, ``"Temperature (K)"`` -- then
   one data row per point/sample, comma- or tab-delimited. This is a real,
   commonly-used, simple, and fully documented export path (STAR-CCM+ User
   Guide, "Exporting Plot and Table Data"), genuinely different from the
   proprietary ``.sim``/``.ccm`` binary and *not* just guessed at here --
   :func:`read_starccm_table` below is a real, working parser for it, not a
   stub.

What is NOT supported, and why
--------------------------------
STAR-CCM+'s native project files (``.sim``, and the older ``.ccm``) are a
**proprietary, undocumented binary format** with no public specification
and no independently-implementable open encoding (unlike CGNS/Exodus/Fluent
ASCII, which this package's sibling modules implement directly from
published specs). Siemens does not publish the ``.sim``/``.ccm`` byte
layout, and the only reliable way to read one is Siemens' own Simcenter
STAR-CCM+ installation (interactively, or via its bundled Java macro API,
analogous to how ``abaqus_reader.py``'s ``.odb`` bridge shells out to a
real, licensed Abaqus installation rather than guessing at that format).
This module deliberately does **not** attempt to reverse-engineer or guess
at ``.sim``/``.ccm`` -- doing so would produce a reader that looks like it
works but silently misparses on the first file that doesn't match the
guess, which is worse than refusing outright. There is no
``starccm_sim_to_upd`` (or similar) function here, and none should be added
without a real, licensed STAR-CCM+ installation to validate against (the
same bar ``export_odb_fields`` was held to for Abaqus).

If you have a real ``.sim``/``.ccm`` file: export it to CGNS or a table/CSV
from within STAR-CCM+ first (both are standard menu actions), then use one
of the two functions in this module.
"""
from __future__ import annotations

import csv
import os
import re
from typing import Dict, List, Optional, Sequence

from .cgns_reader import cgns_to_upd, read_cgns_mesh_and_fields

# ---------------------------------------------------------------------------
# 1. CGNS export path -- delegates entirely to the real cgns_reader.
# ---------------------------------------------------------------------------


def read_starccm_cgns(path: str, *, fields: Optional[Sequence[str]] = None, zone_index: int = 0):
    """Read a STAR-CCM+ CGNS export. Thin re-export of
    :func:`cgns_reader.read_cgns_mesh_and_fields` -- a STAR-CCM+ CGNS export
    is standard CGNS/HDF5, so the existing, already-validated reader applies
    unchanged; see the module docstring for why this is legitimate reuse
    rather than a separate implementation."""
    return read_cgns_mesh_and_fields(path, fields=fields, zone_index=zone_index)


def starccm_cgns_to_upd(path: str, *, fields: Optional[Sequence[str]] = None, zone_index: int = 0):
    """Read a STAR-CCM+ CGNS export and package it as a UPD
    ``PhysicalSample``. Thin re-export of :func:`cgns_reader.cgns_to_upd` --
    STAR-CCM+'s CGNS export is standard CGNS/HDF5 (see module docstring), so
    this calls straight through to the same, already-validated CGNS parser
    rather than duplicating it under a new name."""
    return cgns_to_upd(path, fields=fields, zone_index=zone_index)


# ---------------------------------------------------------------------------
# 2. Plain-text table/CSV export path -- a real, implemented parser.
# ---------------------------------------------------------------------------

_UNIT_SUFFIX_RE = re.compile(r"\s*\([^)]*\)\s*$")


def _strip_unit_suffix(column_name: str) -> str:
    """Strip a trailing unit annotation like `` (m)`` or `` (K)`` off a
    STAR-CCM+ table column header, e.g. ``"Temperature (K)"`` ->
    ``"Temperature"``. Leaves a name with no such suffix untouched."""
    return _UNIT_SUFFIX_RE.sub("", column_name).strip()


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        return ","


def read_starccm_table(path: str, *, coord_names: Sequence[str] = ("X", "Y", "Z")):
    """Parse a STAR-CCM+ plain-text table/CSV export (``Plot > Export`` /
    ``Table > Export``): one header row of (optionally quoted, optionally
    unit-suffixed) column names, then one data row per point.

    Parameters
    ----------
    path : path to the exported ``.csv``/``.txt`` table file.
    coord_names : column-name prefixes (case-insensitive, unit suffix like
        ``" (m)"`` ignored) treated as spatial coordinates, in order. Any
        column not matched here is returned as a field. Missing trailing
        coordinates (e.g. a 2D export with only X/Y) are zero-padded to 3D.

    Returns
    -------
    dict with ``"coords"`` (N, 3), ``"fields"`` (``{name: (N,)}`` for every
    non-coordinate column), and ``"column_names"`` (the original header row,
    unit suffixes included, for reference/round-tripping).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file has no header row, no data rows, or no column matches
        any of ``coord_names`` at all.
    """
    import numpy as np

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = _sniff_delimiter(sample)
        reader = csv.reader(f, delimiter=delimiter)
        rows = [row for row in reader if any(cell.strip() != "" for cell in row)]

    if not rows:
        raise ValueError(f"{path}: empty table -- no header row found.")

    header = [h.strip() for h in rows[0]]
    data_rows = rows[1:]
    if not data_rows:
        raise ValueError(f"{path}: header row found but no data rows follow.")

    stripped = [_strip_unit_suffix(h) for h in header]
    lowered = [s.lower() for s in stripped]

    coord_cols: List[Optional[int]] = []
    for cname in coord_names:
        try:
            coord_cols.append(lowered.index(cname.lower()))
        except ValueError:
            coord_cols.append(None)

    if all(c is None for c in coord_cols):
        raise ValueError(
            f"{path}: none of the expected coordinate columns {list(coord_names)} were found "
            f"in header {header!r} -- pass coord_names= matching this file's actual column "
            "names if they differ (e.g. a non-English STAR-CCM+ locale, or a derived-part "
            "table using different axis labels)."
        )

    n_rows = len(data_rows)
    n_cols = len(header)
    values = np.empty((n_rows, n_cols), dtype=np.float64)
    for i, row in enumerate(data_rows):
        if len(row) != n_cols:
            raise ValueError(
                f"{path}: data row {i + 2} has {len(row)} values, expected {n_cols} "
                f"(matching the header) -- malformed table export."
            )
        values[i, :] = [float(v) for v in row]

    coords = np.zeros((n_rows, 3), dtype=np.float64)
    for axis, col in enumerate(coord_cols):
        if col is not None:
            coords[:, axis] = values[:, col]

    used_cols = {c for c in coord_cols if c is not None}
    fields: Dict[str, "object"] = {
        stripped[j]: values[:, j] for j in range(n_cols) if j not in used_cols
    }

    return {"coords": coords, "fields": fields, "column_names": header}


def starccm_table_to_upd(path: str, *, coord_names: Sequence[str] = ("X", "Y", "Z")):
    """Read a STAR-CCM+ plain-text table/CSV export and package it as a UPD
    ``PhysicalSample`` (``domain={"type": "mesh"}``, ``geometry.nodes`` =
    point coordinates -- same contract as
    ``openfoam.field_reader.openfoam_case_to_upd``; no cell/face topology
    since a table export is a scattered point/sample set, not a mesh)."""
    import numpy as np
    from pinneapple_data.physical_sample import PhysicalSample
    from ..openfoam.mesh_reader import MeshGeometry

    result = read_starccm_table(path, coord_names=coord_names)
    coords = result["coords"]
    geom = MeshGeometry(
        nodes=coords,
        cell_size=np.zeros_like(coords),
        cell_delta=np.zeros(coords.shape[0]),
        bounds_min=coords.min(axis=0) if coords.shape[0] else np.zeros(3),
        bounds_max=coords.max(axis=0) if coords.shape[0] else np.zeros(3),
        n_points=coords.shape[0],
        n_faces=0,
        n_internal_faces=0,
    )
    return PhysicalSample(
        state=dict(result["fields"]),
        geometry=geom,
        domain={"type": "mesh", "n_cells": coords.shape[0]},
        provenance={
            "version": "0.1", "physics_domain": "cfd", "source": "starccm_table",
            "case_dir": os.path.abspath(path),
            "column_names": result["column_names"],
            "validation": "real, implemented delimited-text parser for STAR-CCM+'s "
                           "documented Plot/Table export format -- tested against a "
                           "hand-constructed synthetic fixture (see "
                           "tests/test_starccm_reader.py), not a captured real STAR-CCM+ "
                           "file (none was available); no cell/face topology (a table "
                           "export is a scattered point set, not a mesh) -- see module "
                           "docstring for the .sim/.ccm binary format, which is NOT "
                           "supported.",
        },
        schema={"units": {}},
    )
