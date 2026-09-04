"""Abaqus mesh (``.inp``) reader, and an ``.odb`` results bridge.

Two genuinely different problems, handled two different ways
---------------------------------------------------------------
**``.inp`` (input deck: mesh, materials, BCs, step definitions)** is a
plain, stable, publicly-documented text keyword format (Abaqus Keywords
Reference). ``read_abaqus_inp_mesh`` below parses the ``*NODE`` and
``*ELEMENT`` blocks directly -- genuinely implementable and testable
without any Abaqus installation, same as this repository's other
open-format readers.

**``.odb`` (output database: the actual solved result fields)** is a
proprietary binary format with no public specification at all, and
reading it requires Abaqus's own bundled Python and its ``odbAccess``
module -- there is no independent, open way to parse an ``.odb`` file
(unlike CGNS/Exodus, which are open formats this repository can implement
directly; unlike OpenFOAM's binary format, whose byte layout is
documented and was reverse-engineerable). Writing a from-scratch binary
``.odb`` parser here would be guessing at an undocumented, versioned,
proprietary format with zero ability to validate the guess -- exactly the
kind of unverifiable code this repository is trying to avoid, not add.

So ``.odb`` support here is a **bridge**, not a parser: :func:`export_odb_fields`
shells out to a real local Abaqus installation's own Python interpreter
(``abaqus python <script>``, i.e. the standard way any external tool
reads odb data) running a small script (installed alongside this module,
``_odb_export_script.py``) that uses the genuine ``odbAccess`` API to dump
the requested step/frame/field data to a plain ``.npz`` file, which
:func:`load_exported_odb_npz` then reads back with ordinary numpy -- no
proprietary code executes inside this process, only inside Abaqus's own
interpreter, which is the only thing actually licensed and guaranteed to
read its own file format correctly. This requires the caller to have
Abaqus installed and licensed; there is no way around that for ``.odb``,
by design of the format.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# .inp mesh reader (open format, no Abaqus required)
# ---------------------------------------------------------------------------

_KEYWORD_RE = re.compile(r"^\*([A-Za-z][\w\- ]*)", re.M)


def read_abaqus_inp_mesh(path: str):
    """Parse ``*NODE`` and ``*ELEMENT`` blocks out of an Abaqus ``.inp``
    keyword file.

    Returns
    -------
    dict with ``"node_ids"`` (N,), ``"coords"`` (N, 3), and
    ``"elements"`` (``{element_type: {"ids": (M,), "connectivity": (M, k)}}``,
    one entry per ``*ELEMENT, TYPE=...`` block encountered). Multi-line
    continuation (a trailing comma with more data on the next line, common
    for higher-order elements) is handled.
    """
    import numpy as np

    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    node_ids: List[int] = []
    coords: List[List[float]] = []
    elements: Dict[str, Dict[str, list]] = {}

    mode = None  # None | "node" | "element"
    current_elem_type = None
    pending: List[str] = []

    def _flush_node(row: str):
        parts = [p.strip() for p in row.split(",") if p.strip() != ""]
        if len(parts) < 2:
            return
        nid = int(float(parts[0]))
        xyz = [float(v) for v in parts[1:4]]
        while len(xyz) < 3:
            xyz.append(0.0)
        node_ids.append(nid)
        coords.append(xyz)

    def _flush_element(row: str, etype: str):
        parts = [p.strip() for p in row.split(",") if p.strip() != ""]
        if len(parts) < 2:
            return
        eid = int(float(parts[0]))
        conn = [int(float(v)) for v in parts[1:]]
        bucket = elements.setdefault(etype, {"ids": [], "connectivity": []})
        bucket["ids"].append(eid)
        bucket["connectivity"].append(conn)

    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        line = raw.strip()
        if line.startswith("**"):
            i += 1
            continue
        if line.startswith("*"):
            header = line.upper()
            if header.startswith("*NODE"):
                mode = "node"
            elif header.startswith("*ELEMENT"):
                mode = "element"
                m = re.search(r"TYPE\s*=\s*([\w\-\.]+)", line, re.I)
                current_elem_type = m.group(1) if m else "UNKNOWN"
            else:
                mode = None
            i += 1
            continue
        if mode == "node" and line:
            _flush_node(line)
        elif mode == "element" and line:
            # A line ending in a bare comma continues on the next line
            # (common for second-order/multi-node element connectivity).
            full = line
            while full.rstrip().endswith(",") and i + 1 < n:
                i += 1
                full = full.rstrip() + lines[i].strip()
            _flush_element(full, current_elem_type)
        i += 1

    return {
        "node_ids": np.array(node_ids, dtype=np.int64),
        "coords": np.array(coords, dtype=np.float64),
        "elements": {
            k: {"ids": np.array(v["ids"], dtype=np.int64),
                "connectivity": v["connectivity"]}  # ragged across element types; kept as list of lists
            for k, v in elements.items()
        },
    }


def abaqus_inp_to_upd(path: str):
    """Package an ``.inp`` mesh (geometry + connectivity, no result
    fields -- ``.inp`` is a pre-processor input deck, not a results file;
    see :func:`export_odb_fields` for actual field data) as a UPD
    ``PhysicalSample``."""
    import numpy as np
    from pinneapple_data.physical_sample import PhysicalSample
    from ..openfoam.mesh_reader import MeshGeometry

    mesh = read_abaqus_inp_mesh(path)
    coords = mesh["coords"]
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
        state={},
        geometry=geom,
        domain={"type": "mesh", "n_cells": coords.shape[0]},
        provenance={
            "version": "0.1", "physics_domain": "structural", "source": "abaqus_inp",
            "case_dir": os.path.abspath(path),
            "element_types": list(mesh["elements"].keys()),
            "validation": "open text format, parsed directly -- geometry/connectivity only, "
                           "no result fields (.inp has none; see export_odb_fields for .odb)",
        },
        schema={"units": {}},
    )


# ---------------------------------------------------------------------------
# .odb results bridge (requires a real, licensed Abaqus installation)
# ---------------------------------------------------------------------------

_ODB_EXPORT_SCRIPT = os.path.join(os.path.dirname(__file__), "_abaqus_odb_export_script.py")


def export_odb_fields(
    odb_path: str,
    out_npz_path: str,
    *,
    step_name: Optional[str] = None,
    frame_index: int = -1,
    field_outputs: Sequence[str] = ("U", "S"),
    abaqus_executable: str = "abaqus",
    timeout: int = 1800,
) -> str:
    """Export field data from an ``.odb`` to a plain ``.npz`` file by
    running Abaqus's own Python (``odbAccess``) as a subprocess.

    Requires a local, licensed Abaqus installation with ``abaqus`` (or
    ``abaqus_executable``) on ``PATH``, or an executable path pointing at
    one. This is the only correct way to read ``.odb`` data without
    Abaqus's proprietary format specification -- see the module docstring.

    Parameters
    ----------
    odb_path : path to the .odb file.
    out_npz_path : where to write the exported .npz (node coords under
        ``"coords"``, each requested field under its own name).
    step_name : step to read; last step in the odb if None.
    frame_index : frame within the step (-1 = last, the converged/final
        result -- the common case for a steady analysis).
    field_outputs : Abaqus field-output identifiers to export (e.g.
        ``"U"`` displacement, ``"S"`` stress, ``"NT11"`` temperature).
    abaqus_executable : the Abaqus command-line entry point.
    timeout : seconds to allow the subprocess before giving up.

    Returns
    -------
    ``out_npz_path``, once the export subprocess has completed
    successfully.

    Raises
    ------
    FileNotFoundError
        If ``abaqus_executable`` is not found on PATH (no local Abaqus
        installation -- there is no fallback, by design; see module
        docstring for why).
    RuntimeError
        If the Abaqus subprocess itself fails (bad odb path, licensing
        issue, unknown step/field name, ...); the subprocess's own stderr
        is included in the message.
    """
    if not os.path.exists(odb_path):
        raise FileNotFoundError(odb_path)

    args_json = json.dumps({
        "odb_path": os.path.abspath(odb_path),
        "out_npz_path": os.path.abspath(out_npz_path),
        "step_name": step_name,
        "frame_index": frame_index,
        "field_outputs": list(field_outputs),
    })

    try:
        proc = subprocess.run(
            [abaqus_executable, "python", _ODB_EXPORT_SCRIPT, args_json],
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"'{abaqus_executable}' not found on PATH -- .odb export requires a local, licensed "
            "Abaqus installation (there is no way to read .odb without one; see this module's "
            "docstring). Pass abaqus_executable= with a full path if it is installed but not on PATH."
        ) from e

    # The subprocess runs in Abaqus's own Python, which is not guaranteed to
    # have numpy -- it writes plain JSON (see _abaqus_odb_export_script.py's
    # own docstring); this process (regular Python, numpy guaranteed via
    # PINNeAPPle's own dependencies) converts that to the final .npz.
    json_path = out_npz_path + ".json"
    if proc.returncode != 0 or not os.path.exists(json_path):
        raise RuntimeError(
            f"Abaqus odb export failed (exit code {proc.returncode}).\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )

    import numpy as np
    with open(json_path, "r") as f:
        payload = json.load(f)
    save_kwargs = {"coords": np.asarray(payload["coords"], dtype=np.float64)}
    for name, values in payload["fields"].items():
        save_kwargs[name] = np.asarray(values, dtype=np.float64)
    # np.savez() auto-appends ".npz" to a bare path string, which would
    # silently break the out_npz_path contract for a caller who didn't
    # already include the extension -- pass an open file handle instead,
    # which numpy writes to as-is.
    with open(out_npz_path, "wb") as fh:
        np.savez(fh, **save_kwargs)
    os.remove(json_path)
    return out_npz_path


def load_exported_odb_npz(npz_path: str):
    """Load a ``.npz`` produced by :func:`export_odb_fields` as a UPD
    ``PhysicalSample``."""
    import numpy as np
    from pinneapple_data.physical_sample import PhysicalSample
    from ..openfoam.mesh_reader import MeshGeometry

    data = np.load(npz_path)
    coords = data["coords"]
    fields = {k: data[k] for k in data.files if k != "coords"}
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
        state=fields,
        geometry=geom,
        domain={"type": "mesh", "n_cells": coords.shape[0]},
        provenance={
            "version": "0.1", "physics_domain": "structural", "source": "abaqus_odb",
            "case_dir": os.path.abspath(npz_path),
            "validation": "read via Abaqus's own odbAccess API (subprocess bridge) -- "
                           "correctness depends on the local Abaqus installation, not on any "
                           "format guess made in this repository",
        },
        schema={"units": {}},
    )
