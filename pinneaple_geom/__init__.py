# pinneaple_geom/__init__.py

"""
pinneaple_geom

Geometry + mesh utilities for Pinneaple.

MVP-1 focus (trimesh + meshio):
- Load geometry files (STL/OBJ/PLY/GLTF/...)
- Load mesh files (VTK/VTU/MSH/...)
- Generate simple parametric primitives -> meshes
- Apply fast transforms (scale/translate/rotate)
- Provide surface sampling hooks for PINN collocation/BC points

Design principles:
- Keep `pinneaple_geom` independent from `pinneaple_pdb` (ETL) and `pinneaple_pinn` (loss/models)
- Expose a small stable API surface from `__init__`
"""

from .core.geometry import GeometrySpec, GeometryAsset
from .core.mesh import MeshData
from .core.registry import build_geometry_asset, load_geometry_asset
from .builders import STLDomainBatchBuilder, STLDomainBatchConfig

__all__ = [
    "GeometrySpec",
    "GeometryAsset",
    "MeshData",
    "build_geometry_asset",
    "load_geometry_asset",
    "STLDomainBatchBuilder",
    "STLDomainBatchConfig"
]
