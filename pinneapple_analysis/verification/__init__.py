"""pinneapple_analysis.verification -- Physics verification building blocks.

Relocated from ``saas/physics_verification_engine/core/*.py`` (a single
standalone FastAPI microservice) so any use case -- ``examples/``,
templates, or any other ``saas/`` app -- can import this real,
already-tested logic directly, instead of it being locked inside one
app. ``saas/physics_verification_engine/core/*.py`` now contain thin
re-export shims pointing back here (the same convention as
``pinneapple_models/registry.py`` / ``pinneapple_solvers/fft.py``), so
every existing caller keeps working unchanged.

Sub-modules
-----------
dimensional_analysis
    Real, first-principles dimensionless-number calculations (Reynolds,
    Prandtl, Peclet, Mach, Froude, Weber, Damkohler, a Nusselt
    correlation when its validity range is satisfied) plus a
    geometry-aware flow-regime classifier.

mesh_intelligence
    Mesh-quality assessment: per-triangle aspect ratio / equiangular
    skewness (real, computed from the mesh's own geometry), an
    explicitly-labeled y+-driven required-first-cell-height ESTIMATE,
    and a structured ``MeshQualityReport`` of what failed and why.

causal_discrepancy
    Two independent tools: ``run_intervention``, a causal/intervention
    engine ("what happens if I change parameter X?") that retrains and
    re-verifies real runs and reports real percentage changes; and
    ``fit_discrepancy_model``, a Kennedy & O'Hagan (2001) model-form
    discrepancy correction (``y(x) = physics_model(x) + delta(x)``) fit
    by ordinary MSE regression on a held-out split.

knowledge_graph
    A real Physics Knowledge Graph (Phenomenon -> Equation ->
    Parameters -> Preset -> Reference -> VerificationMethod), rebuilt
    at call time from PINNeAPPle's own live preset registry and
    AST-scanned validation test files -- never a fabricated ontology or
    a stale snapshot.

convergence
    Mesh-independence / grid-convergence-study automation: Richardson
    extrapolation and Roache's (1994) Grid Convergence Index (GCI) from
    three systematically-refined solves, plus a solver-agnostic
    ``mesh_independence_study`` driver that calls an arbitrary
    ``solve_fn(resolution)`` across an ascending list of resolutions.
    Generic infrastructure only -- it never imports or knows about any
    specific PDE solver (FDM/FEM/LBM/etc.).
"""
from __future__ import annotations

from pinneapple_analysis.verification.dimensional_analysis import (
    DimensionlessNumbers,
    compute_dimensionless_numbers,
    classify_flow_regime,
)
from pinneapple_analysis.verification.mesh_intelligence import (
    TriangleQualityStats,
    OrthogonalityResult,
    YPlusEstimate,
    MeshQualityReport,
    estimate_required_first_cell_height,
    assess_mesh_quality,
)
from pinneapple_analysis.verification.causal_discrepancy import (
    InterventionRunResult,
    InterventionReport,
    run_intervention,
    DiscrepancyModel,
    fit_discrepancy_model,
)
from pinneapple_analysis.verification.knowledge_graph import (
    build_knowledge_graph,
    find_presets_for_phenomenon,
    find_verified_equations,
    explain_preset,
    graph_stats,
)
from pinneapple_analysis.verification.convergence import (
    ConvergenceResult,
    MeshIndependenceStudyResult,
    richardson_extrapolate,
    mesh_independence_study,
)

__all__ = [
    # dimensional_analysis
    "DimensionlessNumbers",
    "compute_dimensionless_numbers",
    "classify_flow_regime",
    # mesh_intelligence
    "TriangleQualityStats",
    "OrthogonalityResult",
    "YPlusEstimate",
    "MeshQualityReport",
    "estimate_required_first_cell_height",
    "assess_mesh_quality",
    # causal_discrepancy
    "InterventionRunResult",
    "InterventionReport",
    "run_intervention",
    "DiscrepancyModel",
    "fit_discrepancy_model",
    # knowledge_graph
    "build_knowledge_graph",
    "find_presets_for_phenomenon",
    "find_verified_equations",
    "explain_preset",
    "graph_stats",
    # convergence
    "ConvergenceResult",
    "MeshIndependenceStudyResult",
    "richardson_extrapolate",
    "mesh_independence_study",
]
