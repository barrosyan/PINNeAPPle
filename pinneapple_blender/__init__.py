"""Blender bridge: export a PINNeAPPle field/trajectory as a ``.ply``
sequence (``export.py``, pure Python + numpy, no Blender needed), and
optionally build/save a Blender scene from it via a real local Blender
installation (``render.py``, subprocess bridge to Blender's own ``bpy``,
mirroring the ``cfd_formats.abaqus_reader`` ``.odb`` bridge pattern).
"""
from .export import export_scene, export_trajectory, write_ply

try:
    from .render import build_scene
except Exception:
    build_scene = None  # type: ignore

__all__ = ["export_scene", "export_trajectory", "write_ply", "build_scene"]
