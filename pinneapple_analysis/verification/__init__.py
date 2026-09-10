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

physics_confidence_score
    A transparent, componentized aggregate confidence score built ONLY
    from real, already-executed checks a caller supplies (a
    ``PhysicsGuardrail`` report, a convergence-study ``ConvergenceResult``,
    real UQ calibration numbers, a benchmark comparison) -- never a
    check-running black box, never a fabricated number, and always paired
    with an explicit ``coverage`` fraction so a score built from a subset
    of the four possible checks can never be mistaken for one built from
    all of them. See that module's own docstring for the full
    anti-fabrication design rationale.

geometry_intelligence
    CAD/mesh -> semantic boundary regions -> boundary conditions. Two
    stage: real, deterministic normal-vector-similarity face segmentation
    + boundary-loop tracing + isoperimetric-quotient/axis-alignment
    heuristics (wall/opening/symmetry_plane candidates), then optional
    LLM semantic disambiguation ONLY for genuinely-ambiguous opening
    regions (a fixed, checked label/BC-type vocabulary -- hallucinated
    region ids/labels are rejected, never guessed at).

solver_orchestration
    Live introspection of which solver families (PINN, the classical
    FDM/FEM/FVM/LBM/... registry, OpenFOAM, FEniCS) are ACTUALLY runnable
    on this machine right now (never a hardcoded table), a
    documented-scope-matching recommendation heuristic that only ever
    recommends a family it just verified is available, and an honest
    numeric solver-vs-solver comparison (max/mean/RMSE) of two
    already-produced results.

uncertainty_inverse
    Wires PINNeAPPle's real UQ (``pinneapple_analysis.uncertainty``) and
    inverse-problem (``pinneapple_analysis.inverse_problems``) machinery
    end to end: ensemble/MC-dropout/decomposed/aleatoric uncertainty
    reporting (each field honestly ``None`` when that method's underlying
    class doesn't support it -- never fabricated), and inverse-parameter
    estimation via a real alternating field-training/parameter-fitting
    scheme (needed because the real solver only ever optimises a model's
    ``inverse_params``, never its field weights) with parametric-bootstrap
    parameter uncertainty.

provenance
    A structured, JSON-serializable ``ProvenanceRecord`` for one analysis
    run -- problem description, drafted spec, architecture/training
    config, environment, and verification report -- so a run can be
    replayed or audited later. Intentionally a plain dataclass +
    ``to_dict()``/``save()``/``load()``, not a database model.

tool_recommendation
    The sibling of ``solver_orchestration`` for tools OUTSIDE PINNeAPPle:
    a small, explicitly-cited starter catalog of real, named CFD/FEM/
    multiphysics/meshing/visualization/UQ software (OpenFOAM, ANSYS,
    FEniCSx, COMSOL, Siemens STAR-CCM+, SU2, MOOSE, CalculiX, Gmsh,
    ParaView/PyVista, OpenTURNS), ranked against a ``ProblemSpec`` by
    documented scope match (never a live availability/license probe,
    never an accuracy claim -- see the module's own docstring for the
    full honesty scope).

architecture_recommendation
    Which PINNeAPPle neural architecture family to reach for FIRST, from
    a problem's real characteristics (data availability, whether a
    solver can generate more, cross-parameter/cross-geometry
    generalization needs, forward vs. inverse). A recommendation-BEFORE-
    you-train tool -- ``pinneapple_arena``'s ``physics_aware_rank()`` is
    the after-you've-tried-things empirical comparison this narrows the
    search space for, not a competitor to it.

architecture_critique
    A structured, checked-menu "adversarial review" of a proposed
    architecture/pipeline design -- an LLM playing skeptical Principal
    Engineer, forced to address every category in a fixed
    ``FAILURE_MODE_CHECKLIST`` (data leakage, shortcut learning,
    dimensional inconsistency, identifiability, extrapolation failure,
    unstable training, bad benchmark design, misleading metrics,
    malformed physics constraints, deployment dependency risk) with a
    verdict/severity drawn from a fixed vocabulary -- a hallucinated
    category, verdict, or severity is rejected, never silently accepted.

evidence_graph
    A run-level Evidence Graph -- ties one run's ``ProvenanceRecord``
    together with an optional ``PhysicsConfidenceScore`` and any
    ``ComparisonReport`` objects into one small, queryable claim ->
    evidence graph (``build_evidence_graph``), plus ``evidence_summary``
    and ``explain_trust`` (a plain-text "why trust this result" render).
    Distinct from ``knowledge_graph`` -- that module builds ONE graph of
    PINNeAPPle's general physics knowledge; this one builds a DIFFERENT
    graph per run, grounded only in that run's own already-computed
    evidence, never re-deriving or fabricating a new verdict.
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
from pinneapple_analysis.verification.physics_confidence_score import (
    ConfidenceComponent,
    CalibrationSummary,
    PhysicsConfidenceScore,
    compute_physics_confidence,
)
from pinneapple_analysis.verification.geometry_intelligence import (
    BoundaryLoop,
    SurfaceRegion,
    GeometryClassification,
    classify_geometry,
)
from pinneapple_analysis.verification.solver_orchestration import (
    list_available_solver_families,
    SolverRecommendation,
    select_solver_family,
    ComparisonReport,
    compare_solvers,
)
from pinneapple_analysis.verification.uncertainty_inverse import (
    UncertaintyReport,
    InverseProblemResult,
    InverseParamAdapter,
    quantify_uncertainty,
    solve_inverse_problem,
)
from pinneapple_analysis.verification.provenance import ProvenanceRecord
from pinneapple_analysis.verification.tool_recommendation import (
    ExternalTool,
    TOOL_CATALOG,
    ToolRecommendation,
    recommend_tools,
    get_tools_by_category,
    Player,
    PLAYER_CATALOG,
    get_player_for_tool,
    BUY_VS_BUILD_GUIDANCE,
    buy_vs_build_recommendation,
    list_buy_vs_build_needs,
)
from pinneapple_analysis.verification.architecture_recommendation import (
    ArchitectureCandidate,
    ARCHITECTURE_CATALOG,
    ArchitectureRecommendation,
    recommend_architecture,
)
from pinneapple_analysis.verification.architecture_critique import (
    FAILURE_MODE_CHECKLIST,
    CritiqueFinding,
    AdversarialReviewReport,
    run_adversarial_review,
)
from pinneapple_analysis.verification.evidence_graph import (
    build_evidence_graph,
    evidence_summary,
    explain_trust,
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
    # physics_confidence_score
    "ConfidenceComponent",
    "CalibrationSummary",
    "PhysicsConfidenceScore",
    "compute_physics_confidence",
    # geometry_intelligence
    "BoundaryLoop",
    "SurfaceRegion",
    "GeometryClassification",
    "classify_geometry",
    # solver_orchestration
    "list_available_solver_families",
    "SolverRecommendation",
    "select_solver_family",
    "ComparisonReport",
    "compare_solvers",
    # uncertainty_inverse
    "UncertaintyReport",
    "InverseProblemResult",
    "InverseParamAdapter",
    "quantify_uncertainty",
    "solve_inverse_problem",
    # provenance
    "ProvenanceRecord",
    # tool_recommendation
    "ExternalTool",
    "TOOL_CATALOG",
    "ToolRecommendation",
    "recommend_tools",
    "get_tools_by_category",
    "Player",
    "PLAYER_CATALOG",
    "get_player_for_tool",
    "BUY_VS_BUILD_GUIDANCE",
    "buy_vs_build_recommendation",
    "list_buy_vs_build_needs",
    # architecture_recommendation
    "ArchitectureCandidate",
    "ARCHITECTURE_CATALOG",
    "ArchitectureRecommendation",
    "recommend_architecture",
    # architecture_critique
    "FAILURE_MODE_CHECKLIST",
    "CritiqueFinding",
    "AdversarialReviewReport",
    "run_adversarial_review",
    # evidence_graph
    "build_evidence_graph",
    "evidence_summary",
    "explain_trust",
]
