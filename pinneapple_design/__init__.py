"""pinneapple_design — Geometry construction and physics-guided design optimization.

Sub-modules
-----------
geometry         (was pinneapple_geom)
    Geometry utilities: rich SDF library (20+ primitives), CSG operations,
    physics domains (2D & 3D channel, cavity, pipe, L-shape, annular, …),
    mesh generation (structured / SDF-based), mesh collocation for 3D PINNs,
    airfoil generation (NACA), RBF interpolation, and STL/STEP import.

design_optimizer  (was pinneapple_design_opt)
    Physics-guided design optimization: shape parametrization, adjoint solver,
    surrogate model, gradient / Bayesian / evolutionary optimizers, Pareto
    fronts (multi-objective), manufacturing constraints, and PINN refinement.

Integration helpers
-------------------
``build_domain(name, ...)``
    Wraps ``geometry.get_domain`` with fallback to get_domain_3d.
``sample_domain(domain, n_interior, n_boundary)``
    Returns interior and boundary collocation points from a domain object.
``optimize(param_space, surrogate, objective, ...)``
    One-shot design optimization shortcut backed by DesignOptLoop.

Usage
-----
>>> from pinneapple_design import build_domain, sample_domain, optimize
>>> domain = build_domain("lid_driven_cavity_2d")
>>> x_int, x_bnd = sample_domain(domain, 4096, 512)
>>> result = optimize(param_space, surrogate, drag_objective, method="bayesian")
"""
from __future__ import annotations

# ── sub-modules (new descriptive names) ───────────────────────────────────────
from . import geometry
from . import design_optimizer

# backward-compat aliases
geom       = geometry
design_opt = design_optimizer

# ── geometry re-exports ───────────────────────────────────────────────────────
from .geometry import (
    GeometrySpec, GeometryAsset, MeshData,
    build_geometry_asset, load_geometry_asset,
    MeshCollocator, MeshCollocatorConfig, CollocationBatch3D,
    # SDF primitives
    SDF,
    sdf2d_circle, sdf2d_rectangle, sdf2d_ellipse, sdf2d_annulus,
    sdf2d_capsule, sdf2d_triangle, sdf2d_convex_polygon,
    sdf3d_sphere, sdf3d_box, sdf3d_cylinder, sdf3d_torus, sdf3d_capsule,
    sdf_union, sdf_intersection, sdf_difference,
    sdf_smooth_union, sdf_smooth_intersection, sdf_smooth_difference,
    sdf_translate, sdf_scale, sdf_rotate_2d, sdf_onion, sdf_repeat_2d,
    circle, rectangle, ellipse, annulus, capsule2d,
    sphere3d, box3d, cylinder3d, torus3d,
    # Physics domains 2D
    PhysicsDomain2D, BoundaryRegion,
    ChannelDomain2D, ChannelWithObstacleDomain2D, LidDrivenCavityDomain2D,
    LShapeDomain2D, AnnularDomain2D, MultiObstacleDomain2D,
    TJunctionDomain2D, SDFDomain2D, get_domain, list_domains,
    # Physics domains 3D
    PhysicsDomain3D, LidDrivenCavityDomain3D, ChannelDomain3D,
    PipeFlowDomain3D, get_domain_3d, list_domains_3d,
    # Mesh
    Mesh2D, mesh_rectangle_structured, mesh_sdf_2d,
    mesh_polygon_2d, mesh_quality_report,
    # Meshfree
    RBFInterpolator, ImplicitSurfaceRBF,
    # Airfoil
    naca_parametric,
    # CSG
    SDFShape, CSGRectangle, CSGCircle, CSGEllipse, CSGPolygon,
    CSGUnion, CSGIntersection, CSGDifference,
    lshape, csg_annulus, channel_with_hole, t_junction,
)

try:
    from .geometry import STLDomainBatchBuilder, STLDomainBatchConfig
except Exception:
    STLDomainBatchBuilder = None
    STLDomainBatchConfig = None

# ── design_optimizer re-exports ───────────────────────────────────────────────
from .design_optimizer import (
    ShapeParametrization, ContinuousAdjointSolver,
    DragAdjointObjective, naca_parametric as naca_opt,
    ParetoFront, pareto_dominates, compute_pareto_front,
    ObjectiveBase, DragObjective, ThermalEfficiencyObjective,
    StructuralObjective, WeightMinimizationObjective, CompositeObjective,
    ConstraintBase, BoxConstraint, MassConservationConstraint,
    GeometricConstraint, ManufacturabilityConstraint, ConstraintSet,
    SurrogateConfig, PhysicsSurrogate,
    ParamSpace, OptState,
    DesignOptimizerConfig, GradientDesignOptimizer,
    BayesianDesignOptimizer, EvolutionaryDesignOptimizer,
    RefinementConfig, PINNRefinement, RefinementResult,
    DesignOptConfig, DesignOptResult, DesignOptLoop,
)


# ── Integration helpers ────────────────────────────────────────────────────────

def build_domain(name: str, dim: int = 2, **kwargs):
    """Return a physics domain object by name."""
    if dim == 3:
        return get_domain_3d(name, **kwargs)
    return get_domain(name, **kwargs)


def sample_domain(domain, n_interior: int = 4096, n_boundary: int = 512):
    """Sample interior and boundary collocation points from a domain."""
    x_int = domain.sample_interior(n_interior)
    x_bnd = domain.sample_boundary(n_boundary)
    return x_int, x_bnd


def optimize(param_space: "ParamSpace", surrogate, objective, *,
             constraints=None, method: str = "bayesian",
             n_trials: int = 100, **loop_kwargs) -> "DesignOptResult":
    """Run physics-guided design optimization."""
    if not isinstance(surrogate, PhysicsSurrogate):
        surrogate = PhysicsSurrogate(surrogate)
    cfg = DesignOptConfig(method=method, n_trials=n_trials, **loop_kwargs)
    loop = DesignOptLoop(param_space, surrogate, objective,
                         constraints=constraints, config=cfg)
    return loop.run()


__all__ = [
    # Sub-modules (new names)
    "geometry", "design_optimizer",
    # Sub-modules (old aliases — backward compat)
    "geom", "design_opt",
    # Integration
    "build_domain", "sample_domain", "optimize",
    # geometry
    "GeometrySpec", "GeometryAsset", "MeshData",
    "build_geometry_asset", "load_geometry_asset",
    "MeshCollocator", "MeshCollocatorConfig", "CollocationBatch3D",
    "STLDomainBatchBuilder", "STLDomainBatchConfig",
    "SDF",
    "sdf2d_circle", "sdf2d_rectangle", "sdf2d_ellipse", "sdf2d_annulus",
    "sdf2d_capsule", "sdf2d_triangle", "sdf2d_convex_polygon",
    "sdf3d_sphere", "sdf3d_box", "sdf3d_cylinder", "sdf3d_torus", "sdf3d_capsule",
    "sdf_union", "sdf_intersection", "sdf_difference",
    "sdf_smooth_union", "sdf_smooth_intersection", "sdf_smooth_difference",
    "sdf_translate", "sdf_scale", "sdf_rotate_2d", "sdf_onion", "sdf_repeat_2d",
    "circle", "rectangle", "ellipse", "annulus", "capsule2d",
    "sphere3d", "box3d", "cylinder3d", "torus3d",
    "PhysicsDomain2D", "BoundaryRegion",
    "ChannelDomain2D", "ChannelWithObstacleDomain2D", "LidDrivenCavityDomain2D",
    "LShapeDomain2D", "AnnularDomain2D", "MultiObstacleDomain2D",
    "TJunctionDomain2D", "SDFDomain2D", "get_domain", "list_domains",
    "PhysicsDomain3D", "LidDrivenCavityDomain3D", "ChannelDomain3D",
    "PipeFlowDomain3D", "get_domain_3d", "list_domains_3d",
    "Mesh2D", "mesh_rectangle_structured", "mesh_sdf_2d",
    "mesh_polygon_2d", "mesh_quality_report",
    "RBFInterpolator", "ImplicitSurfaceRBF",
    "naca_parametric",
    "SDFShape", "CSGRectangle", "CSGCircle", "CSGEllipse", "CSGPolygon",
    "CSGUnion", "CSGIntersection", "CSGDifference",
    "lshape", "csg_annulus", "channel_with_hole", "t_junction",
    # design_optimizer
    "ShapeParametrization", "ContinuousAdjointSolver",
    "DragAdjointObjective", "naca_opt",
    "ParetoFront", "pareto_dominates", "compute_pareto_front",
    "ObjectiveBase", "DragObjective", "ThermalEfficiencyObjective",
    "StructuralObjective", "WeightMinimizationObjective", "CompositeObjective",
    "ConstraintBase", "BoxConstraint", "MassConservationConstraint",
    "GeometricConstraint", "ManufacturabilityConstraint", "ConstraintSet",
    "SurrogateConfig", "PhysicsSurrogate",
    "ParamSpace", "OptState",
    "DesignOptimizerConfig", "GradientDesignOptimizer",
    "BayesianDesignOptimizer", "EvolutionaryDesignOptimizer",
    "RefinementConfig", "PINNRefinement", "RefinementResult",
    "DesignOptConfig", "DesignOptResult", "DesignOptLoop",
]
