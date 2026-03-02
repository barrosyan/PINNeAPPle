
"""STEP file import utilities.

This module provides optional STEP -> mesh conversion.

Preferred backend:
  - pythonocc-core (OpenCascade) for reading STEP
  - gmsh (Python API) for meshing the CAD geometry

Both are optional dependencies. If not installed, functions raise a clear error.

Notes:
  - STEP is CAD (B-Rep). For PINNs / operators we typically convert to a triangle mesh
    (surface) or a volume mesh (tetra) depending on the downstream solver/model.
  - For training datasets, a surface triangle mesh is often enough to sample boundary
    points; volume meshes help for FEM/FVM but are heavier.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

import numpy as np

from pinneaple_geom.core.mesh import MeshData
from pinneaple_geom.io.meshio_bridge import load_meshio, _require_meshio


StepMeshKind = Literal["surface", "volume"]


@dataclass
class StepImportConfig:
    kind: StepMeshKind = "surface"
    mesh_size: float = 0.02
    # gmsh algorithm hints (best-effort)
    algorithm_2d: int = 6   # Frontal-Delaunay (often good)
    algorithm_3d: int = 1   # Delaunay
    curvature_refine: bool = True
    optimize: bool = True


def _require_gmsh():
    try:
        import gmsh  # type: ignore
    except Exception as e:
        raise ImportError(
            "gmsh is required for STEP meshing. Install with: pip install gmsh\n"
            "(On some OS you may also need system libs.)"
        ) from e
    return gmsh


def step_to_mesh(
    step_path: str | Path,
    *,
    cfg: Optional[StepImportConfig] = None,
    cache_dir: Optional[str | Path] = None,
) -> MeshData:
    """Convert a STEP file into MeshData via gmsh.

    Parameters
    ----------
    step_path:
        Path to .step/.stp file.
    cfg:
        Meshing configuration.
    cache_dir:
        If provided, writes intermediate .msh there (useful for debugging).
    """
    _require_meshio()
    gmsh = _require_gmsh()

    cfg = cfg or StepImportConfig()
    step_path = Path(step_path)
    if not step_path.exists():
        raise FileNotFoundError(str(step_path))

    cache_dir_p = Path(cache_dir) if cache_dir is not None else None
    if cache_dir_p is not None:
        cache_dir_p.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    try:
        gmsh.model.add("pinneaple_step")

        # Import STEP into gmsh model
        gmsh.model.occ.importShapes(str(step_path))
        gmsh.model.occ.synchronize()

        # Meshing options
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", float(cfg.mesh_size))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(cfg.mesh_size))

        gmsh.option.setNumber("Mesh.Algorithm", int(cfg.algorithm_2d))
        gmsh.option.setNumber("Mesh.Algorithm3D", int(cfg.algorithm_3d))

        if cfg.curvature_refine:
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
        if cfg.optimize:
            gmsh.option.setNumber("Mesh.Optimize", 1)

        # Generate mesh
        if cfg.kind == "surface":
            gmsh.model.mesh.generate(2)
        elif cfg.kind == "volume":
            gmsh.model.mesh.generate(3)
        else:
            raise ValueError(f"Unknown kind: {cfg.kind}")

        # Write to .msh (in-memory read via meshio is easiest from file)
        out_msh = (cache_dir_p / (step_path.stem + ".msh")) if cache_dir_p else (step_path.with_suffix(".msh"))
        gmsh.write(str(out_msh))

    finally:
        gmsh.finalize()

    import meshio  # type: ignore
    msh = meshio.read(str(out_msh))

    # Convert to MeshData (triangles preferred for surface PINNs)
    mesh = load_meshio(msh)

    # If we wrote a temporary .msh next to STEP and no cache_dir specified, clean up best-effort
    if cache_dir_p is None:
        try:
            out_msh.unlink()
        except Exception:
            pass

    return mesh
