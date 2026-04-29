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
    # 3D Collocation
    "MeshCollocator",
    "MeshCollocatorConfig",
    "CollocationBatch3D",
    "GeometrySpec",
    "GeometryAsset",
    "MeshData",
    "build_geometry_asset",
    "load_geometry_asset",
    "STLDomainBatchBuilder",
    "STLDomainBatchConfig",
    # SDF shapes
    "SDF",
    "sdf2d_circle", "sdf2d_rectangle", "sdf2d_ellipse", "sdf2d_annulus",
    "sdf2d_capsule", "sdf2d_triangle", "sdf2d_convex_polygon",
    "sdf3d_sphere", "sdf3d_box", "sdf3d_cylinder", "sdf3d_torus", "sdf3d_capsule",
    "sdf_union", "sdf_intersection", "sdf_difference",
    "sdf_smooth_union", "sdf_smooth_intersection", "sdf_smooth_difference",
    "sdf_translate", "sdf_scale", "sdf_rotate_2d", "sdf_onion", "sdf_repeat_2d",
    "circle", "rectangle", "ellipse", "annulus", "capsule2d",
    "sphere3d", "box3d", "cylinder3d", "torus3d",
    # Physics domains
    "PhysicsDomain2D",
    "BoundaryRegion",
    "ChannelDomain2D",
    "ChannelWithObstacleDomain2D",
    "LidDrivenCavityDomain2D",
    "LShapeDomain2D",
    "AnnularDomain2D",
    "MultiObstacleDomain2D",
    "TJunctionDomain2D",
    "SDFDomain2D",
    "get_domain",
    "list_domains",
    # 3D physics domains
    "PhysicsDomain3D",
    "LidDrivenCavityDomain3D",
    "ChannelDomain3D",
    "PipeFlowDomain3D",
    "get_domain_3d",
    "list_domains_3d",
    # 2D mesher
    "Mesh2D",
    "mesh_rectangle_structured",
    "mesh_sdf_2d",
    "mesh_polygon_2d",
    "mesh_quality_report",
    # CSG (Feature 12)
    "SDFShape",
    "CSGRectangle",
    "CSGCircle",
    "CSGEllipse",
    "CSGPolygon",
    "CSGUnion",
    "CSGIntersection",
    "CSGDifference",
    "lshape",
    "csg_annulus",
    "channel_with_hole",
    "t_junction",
]


# 3D Collocation
from .mesh_collocator import MeshCollocator, MeshCollocatorConfig, CollocationBatch3D

# IO
from .io.step import step_to_mesh, StepImportConfig
from .io.sdf_meshing import sdf2d_to_tri_mesh, sample_boundary_points_sdf2d, marching_squares_contours, SDFGrid2D

# Point clouds
from .ops.pointcloud import PointCloud, mesh_to_pointcloud, sdf2d_to_pointcloud

# Optimization
from .optimize.loop import ParamSpace, GeometryOptimizer

# New: rich SDF library
from .gen.sdf_shapes import (
    SDF,
    sdf2d_circle, sdf2d_rectangle, sdf2d_ellipse, sdf2d_annulus,
    sdf2d_capsule, sdf2d_triangle, sdf2d_convex_polygon,
    sdf3d_sphere, sdf3d_box, sdf3d_cylinder, sdf3d_torus, sdf3d_capsule,
    sdf_union, sdf_intersection, sdf_difference,
    sdf_smooth_union, sdf_smooth_intersection, sdf_smooth_difference,
    sdf_translate, sdf_scale, sdf_rotate_2d, sdf_onion, sdf_repeat_2d,
    circle, rectangle, ellipse, annulus, capsule2d,
    sphere3d, box3d, cylinder3d, torus3d,
)

# New: physics domains (2D)
from .gen.domains import (
    PhysicsDomain2D,
    BoundaryRegion,
    ChannelDomain2D,
    ChannelWithObstacleDomain2D,
    LidDrivenCavityDomain2D,
    LShapeDomain2D,
    AnnularDomain2D,
    MultiObstacleDomain2D,
    TJunctionDomain2D,
    SDFDomain2D,
    get_domain,
    list_domains,
)

# New: physics domains (3D)
from .gen.domains3d import (
    PhysicsDomain3D,
    LidDrivenCavityDomain3D,
    ChannelDomain3D,
    PipeFlowDomain3D,
    get_domain_3d,
    list_domains_3d,
)

# New: 2D mesher
from .mesh import (
    Mesh2D,
    mesh_rectangle_structured,
    mesh_sdf_2d,
    mesh_polygon_2d,
    mesh_quality_report,
)

# New: CSG (Feature 12)
from .csg import (
    SDFShape,
    CSGRectangle,
    CSGCircle,
    CSGEllipse,
    CSGPolygon,
    CSGUnion,
    CSGIntersection,
    CSGDifference,
    lshape,
    annulus as csg_annulus,
    channel_with_hole,
    t_junction,
)
