"""Geometry I/O helpers.

Some adapters (CADQuery, OpenFOAM) are optional and may not be installed in all
environments (e.g., minimal web backend containers). We therefore guard imports
to keep the base package usable.
"""

from __future__ import annotations

# STEP → mesh (gmsh)
try:
    from .step import step_to_mesh  # noqa: F401
except Exception:
    step_to_mesh = None  # type: ignore

# CADQuery bridge (optional)
try:
    from .cadquery_bridge import build_parametric_part  # noqa: F401
except Exception:
    build_parametric_part = None  # type: ignore

# OpenFOAM adapter (optional)
try:
    from .openfoam import export_openfoam_case  # noqa: F401
except Exception:
    export_openfoam_case = None  # type: ignore

__all__ = [
    "step_to_mesh",
    "build_parametric_part",
    "export_openfoam_case",
]
