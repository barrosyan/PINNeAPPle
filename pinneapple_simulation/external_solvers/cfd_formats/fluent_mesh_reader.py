"""ANSYS Fluent / Gambit neutral mesh (``.msh``) reader -- node coordinates
only.

Validation status
------------------
**Not validated against a real Fluent .msh file** -- see the same caveat
in ``cgns_reader.py``'s module docstring.

Scope, deliberately narrow
---------------------------
A full Fluent case (``.cas``/``.dat``) is a much larger, partly
undocumented format: dozens of section types (nodes, faces, cells, zones,
periodic-shadow face pairs, boundary condition records, solver settings,
...), an optional binary encoding per section, and no publicly published
authoritative specification (what's known comes from reverse-engineering
by third-party tools, e.g. meshio's Fluent reader, and Gambit/Fluent's own
older public neutral-format docs). Writing a *complete, correct* reader
for that is a multi-week reverse-engineering effort in its own right (the
same scale of work this repository's OpenFOAM binary reader took, and that
one had a real 244 MB reference case to validate every byte-layout
decision against end-to-end -- this format has no equivalent reference
file available here).

What is implemented, and kept deliberately small so it can actually be
trusted: **ASCII-encoded node coordinates only** (section index ``10``,
Fluent/Gambit's mesh-node section) -- the single most immediately useful
piece for a PINN training pipeline that just needs a spatial point cloud,
and the one section whose ASCII grammar is simple enough to parse
correctly without a reference file to check against. Cells (section
``12``), faces (section ``13``), zones and BC records, and any binary-
encoded section, are NOT parsed -- reading one of those raises
``NotImplementedError`` rather than silently returning nothing, so a
caller finds out immediately rather than getting a mesh that looks
plausible but is missing all its topology.

Format background (section 10 only)
------------------------------------
A Fluent/Gambit ``.msh`` file is ASCII text, structured as parenthesised
sections ``(<section-id> ...)``. A node section looks like::

    (10 (zone-id first-index last-index type nd)
    (
    x1 y1 z1
    x2 y2 z2
    ...
    ))

with ``zone-id 0`` used for a "declaration" header section (no coordinate
data follows) and any other ``zone-id`` introducing an actual coordinate
block. ``nd`` (present in 3D files) is the number of spatial dimensions.
"""
from __future__ import annotations

import os
import re
from typing import Optional

_SECTION10_RE = re.compile(
    r"\(10\s*\(\s*([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+(\d+)\s*(\d+)?\s*\)\s*\((.*?)\)\s*\)",
    re.S,
)


def read_fluent_mesh_nodes(path: str):
    """Read the node coordinates out of a Fluent/Gambit ASCII ``.msh``
    file's section-10 blocks.

    Returns
    -------
    (N, 3) numpy array of node coordinates (z filled with 0 for a 2D mesh).

    Raises
    ------
    NotImplementedError
        If the file appears to be binary-encoded (Fluent's binary section
        marker) -- not supported by this reader.
    ValueError
        If no section-10 coordinate block is found at all.
    """
    import numpy as np

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "rb") as f:
        head = f.read(4096)
    if b"\x00" in head:
        raise NotImplementedError(
            f"{path}: appears to be a binary-encoded Fluent mesh (a NUL byte was found in the "
            "first 4KB) -- only the ASCII encoding is supported by this reader."
        )

    with open(path, "r", encoding="ascii", errors="ignore") as f:
        text = f.read()

    all_coords = []
    for m in _SECTION10_RE.finditer(text):
        zone_id = int(m.group(1), 16)
        if zone_id == 0:
            continue  # declaration header, no coordinate payload
        nd = int(m.group(5)) if m.group(5) else 3
        body = m.group(6)
        nums = [float(x) for x in body.split()]
        if len(nums) % nd != 0:
            raise ValueError(
                f"{path}: section-10 zone {zone_id:x} body has {len(nums)} numbers, "
                f"not a multiple of nd={nd} -- malformed or unsupported variant."
            )
        block = np.array(nums, dtype=np.float64).reshape(-1, nd)
        if nd == 2:
            block = np.pad(block, ((0, 0), (0, 1)))
        all_coords.append(block)

    if not all_coords:
        raise ValueError(f"{path}: no section-10 (node) coordinate block found.")
    return np.concatenate(all_coords, axis=0)


def fluent_mesh_to_upd(path: str):
    """Read a Fluent/Gambit ASCII ``.msh`` file's node coordinates and
    package them as a UPD ``PhysicalSample`` with an empty field set (this
    reader extracts geometry only -- see the module docstring) --
    ``domain={"type": "mesh"}`` / ``geometry.nodes``, same contract as
    ``openfoam.field_reader.openfoam_case_to_upd``."""
    import numpy as np
    from pinneapple_data.physical_sample import PhysicalSample
    from ..openfoam.mesh_reader import MeshGeometry

    coords = read_fluent_mesh_nodes(path)
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
        state={},
        geometry=geom,
        domain={"type": "mesh", "n_cells": coords.shape[0]},
        provenance={
            "version": "0.1", "physics_domain": "cfd", "source": "fluent_msh",
            "case_dir": os.path.abspath(path),
            "validation": "unverified against a real Fluent file; geometry (nodes) only, "
                           "no cells/faces/zones/fields -- see module docstring",
        },
        schema={"units": {}},
    )
