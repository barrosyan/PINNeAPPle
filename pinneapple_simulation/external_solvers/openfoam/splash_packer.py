"""Pack a finished OpenFOAM case directory into a ``.splash`` archive.

No writer for this format existed anywhere in PINNeAPPle before this
module -- ``binary_reader.py``/``mesh_reader.py``/``field_reader.py`` are
all read-only, built to open ``.splash`` files (or plain case
directories) that already exist. A ``.splash`` file is nothing exotic: a
zip of a standard OpenFOAM case directory (``system/``, ``constant/``,
time directories, optionally ``postProcessing/``) plus one extra
top-level ``splash-manifest.json`` recording per-file size/sha256 and a
small summary block -- this module writes exactly that, so any case this
package's own tools (or plain ``blockMesh``/... run by hand) produce can
be packed into something ``field_reader``/``mesh_reader`` can read back.

Manifest schema (reverse-engineered from a real, externally-produced
``.splash`` archive, since no spec ships anywhere in this codebase)::

    {
      "format": 1,
      "splash_version": "<free-form string>",
      "case_name": "<string>",
      "generated_by": "<string>",
      "files": {"<relative/path>": {"size": <int>, "sha256": "<hex>"}, ...},
      "summary": {<caller-supplied dict, e.g. solver/cells/physical params>}
    }
"""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from typing import Any, Callable, Dict, Iterable, Optional


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack_splash_archive(
    case_dir: str,
    out_path: str,
    *,
    case_name: Optional[str] = None,
    summary: Optional[Dict[str, Any]] = None,
    generated_by: str = "pinneapple_simulation.external_solvers.openfoam.splash_packer",
    splash_version: str = "1.0-pinneapple",
    exclude: Optional[Callable[[str], bool]] = None,
) -> Dict[str, Any]:
    """Zip ``case_dir`` (a finished OpenFOAM case: ``system/``,
    ``constant/``, time directories, ...) into a ``.splash`` archive at
    ``out_path``, with a ``splash-manifest.json`` listing every packed
    file's size and sha256 plus a caller-supplied ``summary`` block.

    Parameters
    ----------
    case_dir : path to the case directory to pack.
    out_path : where to write the ``.splash`` zip.
    case_name : recorded in the manifest; defaults to ``basename(case_dir)``.
    summary : free-form dict merged into the manifest's ``"summary"`` key
        (e.g. solver name, cell count, physical parameters) -- whatever a
        reader of this archive later would want without opening a case
        file.
    exclude : optional ``(relative_path) -> bool``; files it returns True
        for are skipped. Defaults to excluding ``processor*`` directories
        (decomposed-run artifacts that shouldn't ship in the packed case)
        and ``case.foam`` (a ParaView placeholder file with no data).

    Returns
    -------
    The manifest dict that was written (also embedded in the archive).
    """
    if exclude is None:
        def exclude(rel: str) -> bool:
            first = rel.split(os.sep, 1)[0]
            return first.startswith("processor") or rel == "case.foam"

    files = []
    for root, _dirs, filenames in os.walk(case_dir):
        for fn in filenames:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, case_dir)
            if exclude(rel):
                continue
            files.append((rel, full))

    manifest: Dict[str, Any] = {
        "format": 1,
        "splash_version": splash_version,
        "case_name": case_name or os.path.basename(os.path.normpath(case_dir)),
        "generated_by": generated_by,
        "files": {},
        "summary": dict(summary or {}),
    }
    for rel, full in files:
        manifest["files"][rel] = {"size": os.path.getsize(full), "sha256": _sha256(full)}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel, full in files:
            zf.write(full, rel)
        zf.writestr("splash-manifest.json", json.dumps(manifest, indent=2))

    return manifest


def read_splash_manifest(splash_path: str) -> Dict[str, Any]:
    """Read just the ``splash-manifest.json`` entry out of a ``.splash``
    archive, without extracting anything else."""
    with zipfile.ZipFile(splash_path, "r") as zf:
        with zf.open("splash-manifest.json") as fh:
            return json.load(fh)


def verify_splash_archive(splash_path: str) -> Iterable[str]:
    """Yield a description for every file whose recorded sha256/size in
    the archive's own manifest doesn't match its actual packed content --
    empty if the archive is internally consistent. Cheap integrity check
    before trusting a ``.splash`` file (e.g. after copying/transferring
    one) without needing the original case directory."""
    manifest = read_splash_manifest(splash_path)
    with zipfile.ZipFile(splash_path, "r") as zf:
        for rel, meta in manifest.get("files", {}).items():
            try:
                data = zf.read(rel)
            except KeyError:
                yield f"{rel}: listed in manifest but missing from archive"
                continue
            if len(data) != meta.get("size"):
                yield f"{rel}: size mismatch (manifest={meta.get('size')}, actual={len(data)})"
                continue
            actual_sha = hashlib.sha256(data).hexdigest()
            if actual_sha != meta.get("sha256"):
                yield f"{rel}: sha256 mismatch"
