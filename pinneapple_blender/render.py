"""Subprocess bridge to a real local Blender installation -- same pattern
as ``cfd_formats.abaqus_reader.export_odb_fields``: never guess at
Blender's internals from outside, shell out to Blender's own bundled
Python (``blender --background --python <script>``) running genuine
``bpy`` code (``blender_addon/import_pinneapple_sequence.py``).
"""
from __future__ import annotations

import os
import subprocess
from typing import Optional

_ADDON_SCRIPT = os.path.join(os.path.dirname(__file__), "blender_addon", "import_pinneapple_sequence.py")


def build_scene(
    ply_dir: str,
    *,
    output_blend: Optional[str] = None,
    blender_executable: str = "blender",
    timeout: int = 600,
) -> str:
    """Run Blender headless to import a ``.ply`` sequence (as written by
    ``pinneapple_blender.export``) into a scene, optionally saving it as a
    ``.blend`` file.

    Requires a local Blender installation with ``blender`` (or
    ``blender_executable``) on ``PATH``.

    Returns
    -------
    ``output_blend`` if given, else ``ply_dir`` (the scene was built in
    Blender's memory but not saved -- pass ``output_blend`` to keep it).

    Raises
    ------
    FileNotFoundError
        If ``blender_executable`` is not found on PATH.
    RuntimeError
        If the Blender subprocess itself fails.
    """
    args = [blender_executable, "--background", "--python", _ADDON_SCRIPT, "--", ply_dir]
    if output_blend:
        args.append(output_blend)

    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"'{blender_executable}' not found on PATH -- pinneapple_blender requires a local "
            "Blender installation. Pass blender_executable= with a full path if it is installed "
            "but not on PATH."
        ) from e

    if proc.returncode != 0:
        raise RuntimeError(
            f"Blender scene build failed (exit code {proc.returncode}).\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return output_blend or ply_dir
