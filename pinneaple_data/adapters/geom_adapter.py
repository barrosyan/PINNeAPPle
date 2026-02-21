"""Geometry loading and attachment utilities for PhysicalSample via pinneaple_geom."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, overload


# Accepts:
#  - path to geometry file (.stl, .obj, .ply, ...)
#  - path to mesh file (.vtu, .vtk, .msh, ...)
#  - dict spec {"kind": "...", ...}
#  - already-built GeometryAsset (from pinneaple_geom)
GeometryInput = Union[str, Path, Dict[str, Any], Any]


@dataclass
class GeometryLoadOptions:
    """
    Configuration container for geometry or mesh loading behavior.

    This structure defines transformation parameters and basic processing
    flags applied during geometry asset construction.

    Attributes
    ----------
    scale : Optional[float]
        Uniform scaling factor applied to the geometry.
    translate : Optional[Tuple[float, float, float]]
        Translation vector (tx, ty, tz).
    rotate_euler_deg : Optional[Tuple[float, float, float]]
        Euler rotation angles in degrees (rx, ry, rz).
    repair : bool
        Whether to attempt mesh repair operations.
    compute_normals : bool
        Whether to compute or recompute surface normals.
    boundary_labels : Optional[Dict[str, Any]]
        Optional mapping of boundary group names to labeling metadata.
    """
    """
    Options for geometry/mesh loading and basic processing.

    This MVP focuses on:
      - fast load (trimesh/meshio)
      - basic transforms
      - optional simplification hooks (for later)

    Note: operations such as remeshing/decimation are expected to live in pinneaple_geom.ops.
    """
    # transforms
    scale: Optional[float] = None
    translate: Optional[Tuple[float, float, float]] = None
    rotate_euler_deg: Optional[Tuple[float, float, float]] = None  # (rx, ry, rz) degrees

    # optional processing flags
    repair: bool = True
    compute_normals: bool = True

    # optional labeling (boundary groups)
    # e.g., {"inlet": [...], "wall": [...]} (format depends on geom module)
    boundary_labels: Optional[Dict[str, Any]] = None


def _is_pathlike(x: Any) -> bool:
    """
    Determine whether the input behaves like a filesystem path.

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
        True if the object can be interpreted as a valid Path
        and is either a string or Path instance; otherwise False.
    """
    try:
        Path(x)
        return isinstance(x, (str, Path))
    except Exception:
        return False


def _default_spec_from_path(p: Union[str, Path]) -> Dict[str, Any]:
    """
    Infer a geometry specification dictionary from a file path.

    The file extension determines whether the input is treated
    as a surface geometry file or a volumetric mesh file.

    Parameters
    ----------
    p : Union[str, Path]
        Path to the geometry or mesh file.

    Returns
    -------
    Dict[str, Any]
        A minimal specification dictionary compatible with
        pinneaple_geom's registry builder.
    """
    p = Path(p)
    ext = p.suffix.lower().lstrip(".")
    
    if ext in ("stl", "obj", "ply", "glb", "gltf", "off"):
        return {"kind": "file", "path": str(p)}
    if ext in ("vtk", "vtu", "msh", "mesh", "xdmf", "xmf"):
        return {"kind": "mesh_file", "path": str(p)}
    # fallback
    return {"kind": "file", "path": str(p)}


def load_geometry_asset(
    geom: GeometryInput,
    *,
    options: Optional[GeometryLoadOptions] = None,
) -> Any:
    """
    Load or construct a GeometryAsset via pinneaple_geom.

    This function supports multiple input formats and delegates
    asset construction to the geometry registry system.

    Parameters
    ----------
    geom : GeometryInput
        Geometry specification, which may be:
            - Path or string (auto-inferred by extension),
            - Dictionary specification,
            - Pre-built GeometryAsset-like object.
    options : Optional[GeometryLoadOptions]
        Loading and preprocessing configuration.

    Returns
    -------
    Any
        GeometryAsset instance as defined by pinneaple_geom.

    Raises
    ------
    TypeError
        If the input type is unsupported.
    ImportError
        If pinneaple_geom is not available.
    """
    """
    Loads/creates a GeometryAsset using pinneaple_geom.

    Returns:
      GeometryAsset (opaque here; defined in pinneaple_geom)

    Supported inputs:
      - Path/str: inferred spec by file extension
      - dict: GeometrySpec-like
      - GeometryAsset: returned as-is
    """
    options = options or GeometryLoadOptions()

    # If user passes an already-built asset, return it.
    # We avoid importing pinneaple_geom types explicitly (keeps adapters light).
    if not isinstance(geom, (str, Path, dict)):
        # Heuristic: treat as GeometryAsset if it has 'mesh' or 'vertices/faces'
        if hasattr(geom, "mesh") or hasattr(geom, "vertices") or hasattr(geom, "faces"):
            return geom

    if _is_pathlike(geom):
        spec = _default_spec_from_path(Path(geom))
    elif isinstance(geom, dict):
        spec = dict(geom)
    else:
        raise TypeError(f"Unsupported geometry input type: {type(geom)}")

    try:
        from pinneaple_geom.core.registry import build_geometry_asset  # type: ignore
    except Exception as e:
        raise ImportError(
            "pinneaple_geom is required to load geometry assets. "
            "Install geometry extras or add pinneaple_geom module."
        ) from e

    asset = build_geometry_asset(spec, options=options)  # type: ignore
    return asset


def attach_geometry(sample: Any, geom_asset: Any) -> Any:
    """
    Attach a GeometryAsset to a PhysicalSample-like structure.

    This function supports both dictionary-style and object-style
    samples and ensures the domain type defaults to "mesh".

    Parameters
    ----------
    sample : Any
        PhysicalSample-like object or dictionary.
    geom_asset : Any
        GeometryAsset instance to attach.

    Returns
    -------
    Any
        The updated sample with geometry attached.
    """
    """
    Attaches a GeometryAsset to a PhysicalSample-like object.

    Expects `sample` to have `.geometry` and `.domain` (dict) or be dict-like.

    This keeps things flexible while we iterate on the PhysicalSample dataclass.
    """
    # dict-like support
    if isinstance(sample, dict):
        sample["geometry"] = geom_asset
        dom = sample.get("domain", {}) or {}
        dom.setdefault("type", "mesh")
        sample["domain"] = dom
        return sample

    # object-like support
    if hasattr(sample, "geometry"):
        setattr(sample, "geometry", geom_asset)
    if hasattr(sample, "domain"):
        dom = getattr(sample, "domain") or {}
        if not isinstance(dom, dict):
            dom = {"value": dom}
        dom.setdefault("type", "mesh")
        setattr(sample, "domain", dom)
    return sample