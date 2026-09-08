"""Plan builder: dispatches to FNO-first or PINN-first based on task_type."""
from __future__ import annotations

from typing import Dict, List, Optional
from ..schema import ProblemSpec, Plan, PlanStep, Gap, uses_pinn_approach, uses_cfd_approach
from ..method_selection import recommend_method_from_spec


# ---------------------------------------------------------------------------
# Optional bridge to pinneapple_worldmodel's PhysicsOrchestrator tool registry
# ---------------------------------------------------------------------------
#
# ``pinneapple_worldmodel`` is treated as an OPTIONAL dependency of
# ``pinneapple_problemdesign``: this module never assumes it is importable
# and never fails if it (or any of the heavier physics stacks its tools
# lazily import) is missing. This mirrors the try/except ImportError
# fallback pattern used elsewhere in the repo for optional deps, e.g.
# ``pinneapple_neural.trainer.adaptive_sweep`` (Optuna) and
# ``pinneapple_design.geometry.io.step`` (pythonocc-core).
#
# The dependency direction is strictly one-way: problemdesign -> (optionally)
# worldmodel. ``pinneapple_worldmodel`` never imports anything from
# ``pinneapple_problemdesign``, so no circular import is introduced.

# Categories to query in pinneapple_worldmodel.physics_tools.PhysicsToolRegistry
# for each plan flavour, reusing the exact same dispatch signals
# (uses_cfd_approach / uses_pinn_approach) that already select which static
# plan builder runs, so the live-tool lookup stays consistent with the
# textual plan it augments.
_CFD_TOOL_CATEGORIES = ("simulation", "pde_solving", "geometry")
_PINN_TOOL_CATEGORIES = ("pde_solving", "training")
_FNO_TOOL_CATEGORIES = ("training", "timeseries", "data_generation")


def available_orchestrator_tools(spec: ProblemSpec) -> List[Dict[str, object]]:
    """Read-only lookup of real, currently-registered tools from
    ``pinneapple_worldmodel.orchestrator.PhysicsOrchestrator``'s
    ``PhysicsToolRegistry`` that are relevant to *spec*.

    This never raises: if ``pinneapple_worldmodel`` is not importable, or
    the registry cannot be constructed for any reason, it returns ``[]``.
    Only tools that are actually available (``PhysicsTool.is_available()``,
    i.e. their underlying module could be imported) are returned, since the
    point of this bridge is to surface what could be executed *right now*
    via the orchestrator -- not a wishlist of tool names.

    Returns a list of plain dicts (one per matching tool) with keys
    ``name``, ``category``, ``description``, ``module_path``, ``tags`` --
    i.e. the real metadata already carried by ``PhysicsTool``, not an
    invented shape.
    """
    try:
        from pinneapple_worldmodel.physics_tools import PhysicsToolRegistry
    except ImportError:
        return []

    try:
        registry = PhysicsToolRegistry()
        registry.register_all()

        if uses_cfd_approach(spec):
            categories = _CFD_TOOL_CATEGORIES
        elif uses_pinn_approach(spec):
            categories = _PINN_TOOL_CATEGORIES
        else:
            categories = _FNO_TOOL_CATEGORIES

        seen = set()
        matches = []
        for category in categories:
            for tool in registry.list_by_category(category):
                if tool.name in seen or not tool.is_available():
                    continue
                seen.add(tool.name)
                matches.append(tool)

        return [
            {
                "name": tool.name,
                "category": tool.category,
                "description": tool.description,
                "module_path": tool.module_path,
                "tags": list(tool.tags),
            }
            for tool in matches
        ]
    except Exception:
        # Defensive: any instantiation/registration error in the optional
        # dependency must never break problemdesign's plan generation.
        return []


def _orchestrator_tools_step(spec: ProblemSpec) -> Optional[PlanStep]:
    """Build an additive ``PlanStep`` naming real orchestrator tools, or
    ``None`` if the bridge found nothing (never a fake placeholder step)."""
    tools = available_orchestrator_tools(spec)
    if not tools:
        return None
    return PlanStep(
        title="Available orchestrator tools",
        why=(
            "pinneapple_worldmodel.orchestrator.PhysicsOrchestrator's live "
            "PhysicsToolRegistry currently has these registered, executable "
            "tools relevant to this problem -- consider calling them "
            "directly (via PhysicsOrchestrator or PhysicsToolRegistry.get) "
            "instead of reimplementing equivalent functionality."
        ),
        actions=[
            f"{t['name']} [{t['category']}]: {t['description']}" for t in tools
        ],
        pinneapple_modules=sorted({t["module_path"] for t in tools}),
        exit_criteria=[],
    )


def _method_selection_step(spec: ProblemSpec) -> Optional[PlanStep]:
    """Build an additive ``PlanStep`` naming a real recommended numerical
    method + turbulence closure (from
    ``pinneapple_problemdesign.method_selection.recommend_method_from_spec``),
    or ``None`` if it found nothing to recommend (never a fake placeholder).

    Mirrors ``_orchestrator_tools_step`` above: read-only, additive-only,
    and safe to call for any ``spec`` regardless of task_type -- today's
    ``ProblemSpec`` schema carries no structured numeric physics parameters
    (see ``method_selection.recommend_method_from_spec``'s docstring), so
    this returns ``None`` for essentially every spec until that changes.
    """
    rec = recommend_method_from_spec(spec)
    if rec is None:
        return None
    return PlanStep(
        title="Recommended numerical method",
        why=(
            "pinneapple_problemdesign.method_selection.recommend_method_from_spec "
            "computed a flow regime from the elicited physical parameters and "
            "mapped it to a real, registered numerical solver (and turbulence "
            "closure where relevant) via an explicit decision table -- "
            "consider this as a starting point rather than a validated "
            "CFD-expert-system recommendation."
        ),
        actions=[
            f"flow_regime: {rec.flow_regime}",
            f"numerical_method: {rec.numerical_method}",
            f"turbulence_model: {rec.turbulence_model.value if rec.turbulence_model else 'none'}",
            f"confidence: {rec.confidence}",
            f"rationale: {rec.rationale}",
        ],
        pinneapple_modules=["pinneapple_problemdesign.method_selection"],
        exit_criteria=[],
    )


def build_plan(
    spec: ProblemSpec,
    gaps: List[Gap],
    use_orchestrator_bridge: bool = True,
) -> Plan:
    """Dispatch to the appropriate plan builder based on spec.task_type.

    When ``use_orchestrator_bridge`` is True (the default), this additionally
    -- and read-only -- consults ``pinneapple_worldmodel``'s
    ``PhysicsToolRegistry`` and appends an extra "Available orchestrator
    tools" step naming real, currently-registered tools relevant to *spec*,
    on top of (never instead of) the existing static plan. If
    ``pinneapple_worldmodel`` is unavailable or no relevant tools are found,
    the plan is identical to what it would have been with the bridge
    disabled.

    This also always (regardless of ``use_orchestrator_bridge``) -- and
    read-only -- consults
    ``pinneapple_problemdesign.method_selection.recommend_method_from_spec``
    and appends a "Recommended numerical method" step when it returns a
    recommendation, on top of (never instead of) the existing static plan.
    When it returns ``None`` (today's ``ProblemSpec`` schema has no
    structured numeric physics parameters), no such step is added and the
    plan is unchanged.
    """
    if uses_cfd_approach(spec):
        plan = build_plan_cfd_first(spec, gaps)
    elif uses_pinn_approach(spec):
        plan = build_plan_pinn_first(spec, gaps)
    else:
        plan = build_plan_fno_first(spec, gaps)

    if use_orchestrator_bridge:
        extra_step = _orchestrator_tools_step(spec)
        if extra_step is not None:
            plan.steps.append(extra_step)

    method_step = _method_selection_step(spec)
    if method_step is not None:
        plan.steps.append(method_step)

    return plan


def build_plan_pinn_first(spec: ProblemSpec, gaps: List[Gap]) -> Plan:
    """Plan for PDE-solution and inverse-problem tasks via PINNFactory."""
    recommended = (
        "PINN-first: define PDE residuals symbolically via PINNFactory, enforce BCs/ICs "
        "as condition losses, and optionally add supervised data loss."
    )
    alternatives = [
        "Neural operator (FNO/DeepONet) as surrogate if many initial conditions are needed",
        "Hybrid: supervised pre-training + PINN fine-tuning",
        "Classical solver for validation baselines (FEM/FD)",
    ]
    steps = [
        PlanStep(
            title="Define PDE residuals and boundary/initial conditions",
            why="PINN training quality depends entirely on correctly stated physics.",
            actions=[
                "Write PDE residuals as SymPy strings (e.g. 'u_t + u*u_x - nu*u_xx').",
                "List ICs and BCs with their equations and domain definitions.",
                "Identify any inverse parameters to recover.",
            ],
            pinneapple_modules=["pinneapple_pinn.factory (PINNProblemSpec, PINNFactory)"],
            exit_criteria=["PDE + BCs compile without error via PINNFactory."],
        ),
        PlanStep(
            title="Build model and loss function",
            why="The factory compiles symbolic equations into a unified torch loss.",
            actions=[
                "Instantiate VanillaPINN (or NeuralNetwork) with appropriate depth/width.",
                "Call PINNFactory(spec).generate_loss_function().",
                "Verify loss components (pde, conditions, data) are nonzero on a test batch.",
            ],
            pinneapple_modules=[
                "pinneapple_pinn.factory (VanillaPINN, PINNFactory)",
                "pinneapple_train.trainer.Trainer",
            ],
            exit_criteria=["Loss function evaluates without error; PDE residual > 0 before training."],
        ),
        PlanStep(
            title="Train with collocation + condition sampling",
            why="PINNs require careful sampling of collocation and BC points.",
            actions=[
                "Sample interior collocation points and BC/IC boundary points.",
                "Train using Trainer with Adam → L-BFGS schedule for convergence.",
                "Monitor each loss component separately.",
            ],
            pinneapple_modules=[
                "pinneapple_train.trainer.Trainer",
                "pinneapple_train.losses.CombinedLoss",
            ],
            exit_criteria=["PDE residual < 1e-3; BC loss < 1e-4."],
        ),
        PlanStep(
            title="Validate against analytic solution or reference data",
            why="PINN convergence to the wrong solution is a known failure mode.",
            actions=[
                "Compare against analytic solution (if available) or high-fidelity solver.",
                "Check error by region (interior vs boundary).",
                "For inverse problems: verify recovered parameters vs ground truth.",
            ],
            pinneapple_modules=["pinneapple_train.metrics"],
            exit_criteria=["L2 relative error < acceptance threshold defined in spec."],
        ),
    ]
    return Plan(
        recommended_approach=recommended,
        alternatives=alternatives,
        steps=steps,
        go_no_go=[
            "GO: PDE residual and BC losses converge; solution matches reference.",
            "NO-GO: PDE residuals are not physics-faithful (wrong BCs or domain).",
            "REVISE: switch to neural operator surrogate if many PDE solves needed.",
        ],
    )


def build_plan_fno_first(spec: ProblemSpec, gaps: List[Gap]) -> Plan:
    recommended = (
        "FNO-first baseline (direct multi-horizon forecast), then iterate on data quality, "
        "robustness, and optionally add physics-inspired constraints or hybrid losses."
    )

    alternatives = [
        "Autoregressive 1-step baseline (rollout) for simplicity",
        "Transformer-based time series model for long-range dependencies",
        "Hybrid supervised + constraints (bounds/monotonicity/conservation if applicable)",
        "PINN if PDE residuals + BC/IC are reliable and data is scarce",
    ]

    steps: List[PlanStep] = []

    steps.append(PlanStep(
        title="Consolidate the ProblemSpec and close critical gaps",
        why="Prevents building the wrong pipeline and ensures success is measurable.",
        actions=[
            "Confirm inputs/outputs and units.",
            "Confirm sampling frequency, input window, and forecast horizon.",
            "Confirm temporal validation policy and acceptance criteria.",
            "List key data issues (missingness, drift, outliers).",
        ],
        pinneapple_modules=[],
        exit_criteria=[
            "No 'blocker' gaps remain.",
            "Primary metrics and acceptance criteria are defined.",
        ],
    ))

    steps.append(PlanStep(
        title="Define dataset windowing and temporal splits",
        why="Time series modeling requires leakage-safe splits and consistent windowing.",
        actions=[
            "Implement windowing (input_window, horizon, stride) and scaling/normalization.",
            "Apply temporal split policy (e.g., last 20% time as validation).",
            "Check missingness and distribution shift across splits.",
        ],
        pinneapple_modules=[
            "pinneapple_timeseries (windowed datasets + datamodule)",
        ],
        exit_criteria=[
            "Dataset yields consistent (x, y) shapes for train/val.",
            "Split policy avoids future leakage.",
        ],
    ))

    steps.append(PlanStep(
        title="Train the FNO-first baseline (direct multi-horizon)",
        why="FNO is a strong baseline for operator-like dynamics and can generalize well with sufficient data.",
        actions=[
            "Choose initial FNO config (width, modes, layers) appropriate for hardware.",
            "Train with supervised loss (MSE/MAE) and save best checkpoint.",
            "Compare against naive baselines (persistence, simple AR).",
        ],
        pinneapple_modules=[
            "pinneapple_models.neural_operators (FNO)",
            "pinneapple_train.trainer.Trainer",
            "pinneapple_train.losses (CombinedLoss + SupervisedLoss)",
            "pinneapple_train.metrics.default_metrics",
        ],
        exit_criteria=[
            "Baseline beats persistence on primary metric.",
            "Error by horizon is acceptable or gaps are revisited.",
        ],
    ))

    steps.append(PlanStep(
        title="Validate robustness (stress tests)",
        why="Production failures often come from drift, missingness, and rare extremes.",
        actions=[
            "Evaluate error by horizon (short vs long).",
            "Test synthetic missingness, noise, drift, and extreme scenarios.",
            "Record failures and prioritize mitigations.",
        ],
        pinneapple_modules=[
            "pinneapple_train.metrics (custom TS metrics as needed)",
        ],
        exit_criteria=[
            "Clear list of failure modes and mitigation plan.",
            "Acceptance criteria met OR next iteration decision is justified.",
        ],
    ))

    steps.append(PlanStep(
        title="Iterate: data/features, architecture, and optional physics/hybrid constraints",
        why="Second iteration often yields the biggest gains.",
        actions=[
            "If long-horizon error dominates: consider exogenous features, multi-resolution inputs, or transformer alternative.",
            "If generalization is weak: regularization, augmentation, scaling fixes, or constraints.",
            "If reliable physics exists: add constraint losses (bounds, conservation) as hybrid training.",
        ],
        pinneapple_modules=[
            "pinneapple_train.losses.PhysicsLossHook (when applicable)",
            "pinneapple_models (transformers/recurrent as alternatives)",
        ],
        exit_criteria=[
            "Measured improvement in metric + robustness.",
            "Deployment plan updated (latency, monitoring, update policy).",
        ],
    ))

    return Plan(
        recommended_approach=recommended,
        alternatives=alternatives,
        steps=steps,
        go_no_go=[
            "GO: baseline beats naive and meets acceptance criteria.",
            "NO-GO: data is insufficient/ambiguous (critical gaps), leakage exists, or target definition is unstable.",
            "REVISE: adjust horizon/window/metrics if the real use-case demands it.",
        ],
    )


def build_plan_cfd_first(spec: ProblemSpec, gaps: List[Gap]) -> Plan:
    """Plan for external-aerodynamics/CFD/fluid-dynamics problems, using
    PINNeAPPle's Lattice-Boltzmann solver, RANS/LES turbulence-closure
    presets, vortex-identification post-processing, and (optionally) the
    OpenFOAM bridge, instead of defaulting to generic PINN/FNO advice."""
    recommended = (
        "CFD-first: resolve the flow with the Lattice-Boltzmann solver (or bridge to an "
        "external OpenFOAM case for higher-fidelity/complex geometry), apply an appropriate "
        "RANS/LES turbulence closure, and post-process with vortex-identification diagnostics."
    )
    alternatives = [
        "External OpenFOAM run (case_builder/runner) when geometry or Reynolds number is "
        "outside what the in-repo LBM solver handles well",
        "PINN surrogate trained on LBM/OpenFOAM output for fast AoA-sweep interpolation "
        "once reference solves exist",
        "Neural operator (FNO) surrogate across the AoA sweep if many geometries/conditions "
        "must be evaluated cheaply",
    ]
    steps: List[PlanStep] = [
        PlanStep(
            title="Define geometry, voxelization/CAD input, and AoA sweep",
            why="LBM/OpenFOAM meshing and boundary conditions depend on a concrete "
                "geometry representation and the set of angles of attack to evaluate.",
            actions=[
                "Confirm CAD format (e.g. STL/STEP) or voxel resolution for the body.",
                "Confirm the angle-of-attack sweep values (geometry.aoa_sweep_deg).",
                "Confirm Reynolds number / inflow velocity and domain size.",
            ],
            pinneapple_modules=[
                "pinneapple_simulation.numerical_solvers.lbm (LBMSolver / LBMSolver3D)",
                "pinneapple_simulation.external_solvers.openfoam (case_builder, mesh_reader)",
            ],
            exit_criteria=["Geometry, voxel/mesh resolution, and AoA sweep are all specified."],
        ),
        PlanStep(
            title="Select turbulence closure and numerical method",
            why="Reynolds number and flow regime (attached vs separated/high-AoA) determine "
                "whether a RANS closure, LES closure, or laminar LBM run is appropriate.",
            actions=[
                "Pick a RANS closure (k-omega-SST or Spalart-Allmaras) for attached, "
                "moderate-Re flow, or WALE LES for transient/separated flow.",
                "Confirm physics.numerical_method (e.g. 'LBM' vs an external finite-volume run).",
            ],
            pinneapple_modules=[
                "pinneapple_physics.pde_environment.turbulence_presets "
                "(KOmegaSSTResiduals, SpalartAllmarasResiduals, WALEResiduals)",
            ],
            exit_criteria=["Turbulence model and numerical method are recorded in the spec."],
        ),
        PlanStep(
            title="Run the flow solve across the AoA sweep",
            why="Each angle of attack requires its own solve; sweeping is how "
                "lift/drag-vs-AoA behavior (including high-AoA separation) is characterized.",
            actions=[
                "Run LBMSolver.from_problem_spec()/solve_from_spec() (2D) or construct "
                "LBMSolver3D directly (3D) for each AoA, or stage_case_for_scenario()+"
                "run_openfoam_case() for an external solve.",
                "Save trajectory output (rho, u) or exported OpenFOAM fields per AoA.",
            ],
            pinneapple_modules=[
                "pinneapple_simulation.numerical_solvers.lbm",
                "pinneapple_simulation.external_solvers.openfoam (runner, export_bundle)",
            ],
            exit_criteria=["A converged flow field exists for every AoA in the sweep."],
        ),
        PlanStep(
            title="Post-process: vortex identification and force coefficients",
            why="Q-criterion/lambda2/vorticity diagnostics are how leading-edge vortices "
                "and separation are visualized and validated at high AoA.",
            actions=[
                "Compute Q-criterion/lambda2/vorticity fields from the solved velocity field.",
                "Derive lift/drag coefficients from the surface pressure/stress field.",
                "Compare against a cite-able reference (e.g. a named pinneapple_pdb benchmark, "
                "or thin-airfoil/vortex-lift theory at low-to-moderate AoA) rather than an "
                "unverified number.",
            ],
            pinneapple_modules=[
                "pinneapple_tools.visualization.vortex "
                "(compute_q_criterion_2d/3d, compute_lambda2_3d, compute_vorticity_2d/3d)",
            ],
            exit_criteria=["Vortex diagnostics and force coefficients are computed for each AoA."],
        ),
    ]
    return Plan(
        recommended_approach=recommended,
        alternatives=alternatives,
        steps=steps,
        go_no_go=[
            "GO: flow solves converge across the AoA sweep and diagnostics match the "
            "expected qualitative behavior (e.g. vortex lift onset at high AoA).",
            "NO-GO: solver fails to converge, or no cite-able reference exists to sanity-check "
            "the resulting force coefficients.",
            "REVISE: switch to the external OpenFOAM bridge if the in-repo LBM solver's "
            "geometry/Reynolds-number range is insufficient.",
        ],
    )
