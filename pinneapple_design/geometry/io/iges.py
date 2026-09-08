
"""IGES file import utilities.

This module provides optional IGES -> mesh conversion.

Preferred backend:
  - gmsh (Python API), which embeds an OpenCASCADE (OCC) kernel capable of
    reading IGES (.iges/.igs) files via the same ``model.occ.importShapes``
    entry point used for STEP in :mod:`pinneapple_design.geometry.io.step`.

This is an optional dependency. If not installed, functions raise a clear error.

Notes:
  - IGES, like STEP, is CAD (B-Rep / surface) data. For PINNs / operators we
    typically convert to a triangle mesh (surface) or a volume mesh (tetra)
    depending on the downstream solver/model.
  - IGES is an older, surface-oriented format: solids are frequently
    represented as (possibly imperfectly stitched) collections of trimmed
    surfaces rather than a single watertight B-Rep solid the way STEP AP203/
    AP214 files usually are. Volume meshing ("kind='volume'") can therefore
    fail on IGES files that STEP handles fine -- callers needing a solid mesh
    should prefer STEP when available and treat IGES as a surface-mesh source.
  - This module intentionally mirrors ``step.py``'s config/return shape so the
    two importers are interchangeable from a caller's point of view.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

from pinneapple_design.geometry.core.mesh import MeshData
from pinneapple_design.geometry.io.meshio_bridge import load_meshio, _require_meshio


IgesMeshKind = Literal["surface", "volume"]


@dataclass
class IgesImportConfig:
    kind: IgesMeshKind = "surface"
    mesh_size: float = 0.02
    # gmsh algorithm hints (best-effort)
    algorithm_2d: int = 6   # Frontal-Delaunay (often good)
    algorithm_3d: int = 1   # Delaunay
    curvature_refine: bool = True
    optimize: bool = True
    # IGES-specific: attempt to heal/stitch surfaces on import. gmsh/OCC expose
    # this as a global option (Geometry.OCCFixDegenerated / OCCSewFaces-style
    # knobs); IGES files are more prone to needing this than STEP.
    heal_shapes: bool = True


def _require_gmsh():
    try:
        import gmsh  # type: ignore
    except Exception as e:
        raise ImportError(
            "gmsh is required for IGES meshing. Install with: pip install gmsh\n"
            "(On some OS you may also need system libs.)"
        ) from e
    return gmsh


def iges_to_mesh(
    iges_path: str | Path,
    *,
    cfg: Optional[IgesImportConfig] = None,
    cache_dir: Optional[str | Path] = None,
) -> MeshData:
    """Convert an IGES file into MeshData via gmsh.

    Parameters
    ----------
    iges_path:
        Path to .iges/.igs file.
    cfg:
        Meshing configuration.
    cache_dir:
        If provided, writes intermediate .msh there (useful for debugging).
    """
    _require_meshio()
    gmsh = _require_gmsh()

    cfg = cfg or IgesImportConfig()
    iges_path = Path(iges_path)
    if not iges_path.exists():
        raise FileNotFoundError(str(iges_path))

    cache_dir_p = Path(cache_dir) if cache_dir is not None else None
    if cache_dir_p is not None:
        cache_dir_p.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    try:
        gmsh.model.add("pinneapple_iges")

        if cfg.heal_shapes:
            # Best-effort shape healing; IGES imports are more prone to gaps
            # between surfaces than STEP. Silently ignored by older gmsh
            # versions that don't expose this option.
            try:
                gmsh.option.setNumber("Geometry.OCCFixDegenerated", 1)
                gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
                gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)
                gmsh.option.setNumber("Geometry.OCCSewFaces", 1)
            except Exception:
                pass

        # Import IGES into gmsh model (same OCC-backed entry point as STEP)
        gmsh.model.occ.importShapes(str(iges_path))
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
        out_msh = (cache_dir_p / (iges_path.stem + ".msh")) if cache_dir_p else (iges_path.with_suffix(".msh"))
        gmsh.write(str(out_msh))

    finally:
        gmsh.finalize()

    # Convert to MeshData (triangles preferred for surface PINNs).
    # NB: load_meshio() takes a *path* and does its own meshio.read()
    # internally, so we pass out_msh directly rather than pre-reading it.
    mesh = load_meshio(out_msh)

    # If we wrote a temporary .msh next to the IGES file and no cache_dir
    # specified, clean up best-effort.
    if cache_dir_p is None:
        try:
            out_msh.unlink()
        except Exception:
            pass

    return mesh
