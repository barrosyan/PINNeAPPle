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
"""Public API for `pinneaple_geom`.

Important: this package is used in a few contexts (library use, webapp backend,
Colab notebooks). Some optional integration pieces (like the webapp batch
builder) may depend on modules that aren't installed in every environment.

So we keep imports *soft* where needed.
"""

try:
    from .builders import STLDomainBatchBuilder, STLDomainBatchConfig
except Exception:  # optional dependency chain
    STLDomainBatchBuilder = None  # type: ignore
    STLDomainBatchConfig = None  # type: ignore

__all__ = [
    "GeometrySpec",
    "GeometryAsset",
    "MeshData",
    "build_geometry_asset",
    "load_geometry_asset",
    "STLDomainBatchBuilder",
    "STLDomainBatchConfig",
]


# IO
from .io.step import step_to_mesh, StepImportConfig
from .io.sdf_meshing import sdf2d_to_tri_mesh, sample_boundary_points_sdf2d, marching_squares_contours, SDFGrid2D

# Point clouds
from .ops.pointcloud import PointCloud, mesh_to_pointcloud, sdf2d_to_pointcloud

# Optimization
from .optimize.loop import ParamSpace, GeometryOptimizer
