"""Which PINNeAPPle neural architecture to reach for FIRST, given a
problem's real characteristics -- data availability, whether an existing
solver can generate more, whether the model needs to generalize across
parameters/geometry, and whether the target is a forward or inverse
problem. The sibling of ``solver_orchestration`` (which recommends among
PINNeAPPle's own classical numerical solvers) and ``tool_recommendation``
(external software) for the neural-architecture axis specifically.

This is deliberately a RECOMMENDATION-BEFORE-YOU-TRAIN tool, not a
replacement for empirical comparison. ``pinneapple_arena``'s
``physics_aware_rank()`` (train several real architectures, rank by
accuracy while flagging PDE-residual outliers) is the AFTER-you've-tried-
things tool this module's recommendation is meant to narrow down to a
short list for, not compete with -- this module has never trained
anything and makes no accuracy claim; it only says which architectures'
own well-known, documented design intent (not this module's opinion)
covers the stated problem characteristics.

Decision structure (the "PDE type x data availability x parameter-space
generalization -> architecture" pattern): the two axes that matter most
are (1) how much labeled/simulated data is available, and if none, is
there at least an existing solver to fall back on, and (2) whether the
trained model needs to generalize across a family of parameters/
geometries (an operator-learning problem) or only needs to solve ONE
fixed instance well (a function-fitting problem):

    little/no data, no solver available  -> PINN (pure physics-residual
        training -- vanilla_pinn/modified_mlp; xpinn if the domain is
        large/geometrically complex enough to benefit from domain
        decomposition)
    little/no data, solver available     -> hybrid_ml (use the solver to
        generate the missing data on demand, or as an online physics
        check during training -- not a different architecture per se,
        a data-generation strategy layered on top of whichever
        architecture the OTHER axis below recommends)
    lots of data, no generalization need -> PINN/XPINN still applies
        (a single, fixed-domain fit) -- more data mainly helps as extra
        supervision alongside the residual, not a reason by itself to
        reach for an operator-learning architecture
    lots of data, generalization need    -> a neural OPERATOR: FNO (fixed
        structured grid, periodic/near-periodic domains -- literally what
        FNO was demonstrated on, Li et al. 2020), DeepONet (irregular/
        scattered geometry, a branch/trunk split naturally handles varying
        sensor locations), MeshGraphNet (arbitrary unstructured mesh
        topology that itself varies across the family), PINO (wants BOTH
        the operator-learning generalization AND a physics-residual loss,
        not supervision alone)

``is_inverse_problem=True`` adds ``inverse_pinn``/an ``inverse_params``-
capable base architecture as a caveat/addition regardless of the other
axes, since PINNeAPPle's inverse-problem machinery
(``pinneapple_analysis.inverse_problems``, see ``uncertainty_inverse.py``)
specifically expects that convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ArchitectureCandidate:
    """One named architecture family and when its own well-known design
    intent (cited in ``source``) actually applies -- never an accuracy
    claim, only a documented-scope-match input (same discipline as
    ``solver_orchestration.ExternalTool``/``_CLASSICAL_SOLVER_SCOPE``)."""
    name: str
    registry_key: str  # the real pinneapple_neural.architectures.registry.ModelRegistry key
    category: str  # "PINN" | "neural_operator" | "graph" | "domain_decomposition" | "inverse"
    when_to_use: str
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    source: str = ""


ARCHITECTURE_CATALOG: Dict[str, ArchitectureCandidate] = {
    "pinn": ArchitectureCandidate(
        name="Vanilla PINN", registry_key="vanilla_pinn", category="PINN",
        when_to_use="A single, fixed problem instance (fixed geometry/parameters), little or no "
                    "labeled data, physics residual as the primary training signal.",
        strengths=["No simulation data required at all.",
                   "Directly enforces the governing PDE + BCs/ICs as a loss, not just fit-to-data."],
        weaknesses=["Must be retrained from scratch for a new parameter value or geometry -- "
                    "no cross-instance generalization by construction.",
                    "Known to struggle on stiff/multi-scale/chaotic problems without extra "
                    "machinery (causal training, adaptive weighting, domain decomposition)."],
        source="Raissi, Perdikaris & Karniadakis (2019), 'Physics-informed neural networks'.",
    ),
    "xpinn": ArchitectureCandidate(
        name="XPINN (domain-decomposed PINN)", registry_key="xpinn", category="domain_decomposition",
        when_to_use="Same regime as a plain PINN, but the domain is large/geometrically complex "
                    "enough that one network struggles to fit it uniformly well.",
        strengths=["Each subdomain network is smaller/easier to train than one global network "
                    "over the whole domain.", "Naturally parallelizable across subdomains."],
        weaknesses=["Extra interface-continuity losses to tune; more moving parts than a plain PINN.",
                    "Still a single-instance fit, same as plain PINN -- no cross-parameter "
                    "generalization."],
        source="Jagtap & Karniadakis (2020), 'Extended Physics-Informed Neural Networks (XPINNs)'.",
    ),
    "fno": ArchitectureCandidate(
        name="Fourier Neural Operator", registry_key="fno3d", category="neural_operator",
        when_to_use="Many simulations available, need to generalize across parameters/initial "
                    "conditions, domain is a fixed STRUCTURED grid (periodic or near-periodic "
                    "boundary treatment fits especially well -- FFT-based spectral convolution "
                    "implicitly assumes periodic boundaries).",
        strengths=["Resolution-invariant (train/evaluate at different grid resolutions).",
                   "Very fast inference once trained -- a genuine surrogate for expensive CFD/FEM."],
        weaknesses=["Needs a real structured grid; not a natural fit for arbitrary unstructured "
                    "mesh geometry (use MeshGraphNet instead).",
                    "Purely data-driven by default (no physics residual) unless combined with a "
                    "residual loss -- see PINO below for that combination."],
        source="Li et al. (2020), 'Fourier Neural Operator for Parametric Partial Differential Equations'.",
    ),
    "deeponet": ArchitectureCandidate(
        name="DeepONet", registry_key="deeponet", category="neural_operator",
        when_to_use="Many simulations available, need to generalize across an input function "
                    "(e.g. varying boundary/initial conditions or source terms) at ARBITRARY "
                    "query/sensor locations, not tied to one fixed grid.",
        strengths=["Branch/trunk split naturally handles irregular/varying sensor locations, "
                   "unlike FNO's fixed-grid assumption.",
                   "Well-suited to operator learning where the varying input is a function, not "
                   "just a scalar parameter."],
        weaknesses=["Typically needs more training data than FNO for comparable accuracy on "
                    "grid-structured problems (no resolution-invariance built in the same way)."],
        source="Lu, Jin & Karniadakis (2021), 'Learning nonlinear operators via DeepONet'.",
    ),
    "mesh_graph_net": ArchitectureCandidate(
        name="MeshGraphNet", registry_key="mesh_graph_net", category="graph",
        when_to_use="Many simulations available, need to generalize across GEOMETRY itself "
                    "(not just parameters) -- an arbitrary, possibly-varying unstructured mesh "
                    "topology is the natural representation.",
        strengths=["Handles arbitrary mesh connectivity directly -- no fixed-grid assumption at all.",
                   "Learned local message-passing update rule generalizes across meshes with "
                   "different node counts/topology."],
        weaknesses=["Message-passing over a large mesh (100k+ nodes) is more compute/memory "
                    "-intensive per forward pass than FNO's FFT-based convolution.",
                    "Needs a real mesh for every training sample -- meshing cost is part of the "
                    "data-generation budget, not a training-time detail."],
        source="Pfaff et al. (2021), 'Learning Mesh-Based Simulation with Graph Networks' "
               "(MeshGraphNets).",
    ),
    "pino": ArchitectureCandidate(
        name="Physics-Informed Neural Operator", registry_key="pino", category="neural_operator",
        when_to_use="Wants BOTH operator-learning generalization (like FNO/DeepONet) AND a "
                    "physics-residual loss during training -- e.g. simulation data is available "
                    "but scarce/expensive, and the governing PDE should still constrain the fit "
                    "between data points.",
        strengths=["Can reduce the amount of simulation data needed versus a purely data-driven "
                   "operator, since the residual loss supplies extra, free supervision."],
        weaknesses=["Combines the training complexity/cost of both PINN residual computation and "
                    "operator-learning data pipelines -- more moving parts than either alone."],
        source="Li et al. (2021), 'Physics-Informed Neural Operator for Learning Partial "
               "Differential Equations'.",
    ),
    "inverse_pinn": ArchitectureCandidate(
        name="Inverse PINN", registry_key="inverse_pinn", category="inverse",
        when_to_use="The unknown is a PHYSICAL PARAMETER (or source term) to be calibrated "
                    "against observed data, not the field itself -- see "
                    "pinneapple_analysis.verification.uncertainty_inverse.solve_inverse_problem "
                    "for the real, already-wired PINNeAPPle machinery this maps to.",
        strengths=["Directly exposes an `inverse_params` convention PINNeAPPle's own inverse-"
                   "problem solver (InverseProblemSolver) is built to calibrate."],
        weaknesses=["Same single-instance-fit limitation as a plain PINN -- add operator-learning "
                    "machinery separately if the calibrated parameter itself needs to generalize "
                    "across many problem instances."],
        source="pinneapple_analysis.inverse_problems / this repo's own uncertainty_inverse.py "
               "docstring (see its real, documented alternating field/parameter-fitting scheme).",
    ),
}


@dataclass
class ArchitectureRecommendation:
    """Result of :func:`recommend_architecture`. ``decision_path`` makes
    the reasoning auditable step by step (matching this whole engine's
    "explain, don't just assert" norm) rather than a single opaque pick."""
    recommended: List[str]  # ARCHITECTURE_CATALOG keys, ranked
    reasoning: str
    decision_path: List[str]
    alternatives: List[Dict[str, Any]]
    caveats: List[str] = field(default_factory=list)


# Below this many real, already-run high-fidelity simulations, "little/no
# data" branch applies -- a round number informed by this session's own
# real experience (a working FNO surrogate prototype trained cleanly on
# ~100 dense snapshots; PINNeAPPle's own benchmark_suite/arena examples
# commonly cite figures in the same 50-200 range for a first useful
# operator-learning fit), not a universal theoretical threshold. Callers
# with a better-informed number for their own problem should pass
# ``has_lots_of_data`` explicitly instead of relying on this default.
_DEFAULT_DATA_THRESHOLD = 50


def recommend_architecture(
    *,
    n_high_fidelity_simulations: int = 0,
    has_lots_of_data: Optional[bool] = None,
    has_analytical_solver: bool = False,
    needs_parameter_generalization: bool = False,
    geometry_varies: bool = False,
    is_inverse_problem: bool = False,
    catalog: Optional[Dict[str, ArchitectureCandidate]] = None,
) -> ArchitectureRecommendation:
    """Recommend which PINNeAPPle neural architecture family to reach for
    first, from the problem's own real characteristics -- see the module
    docstring for the full decision structure and honesty scope (this
    never claims accuracy; only documented-design-intent match).

    Parameters
    ----------
    n_high_fidelity_simulations : how many real simulation results are
        already available (or realistically obtainable) -- used to infer
        ``has_lots_of_data`` if not given explicitly.
    has_lots_of_data : override the ``n_high_fidelity_simulations``-based
        threshold heuristic with a direct answer, when the caller has a
        better-informed judgment for their specific problem.
    has_analytical_solver : whether an existing solver (classical or
        otherwise) could generate more training data on demand -- only
        matters when data is scarce; changes the recommendation to
        "hybrid_ml" (a data-generation STRATEGY, not a different base
        architecture) rather than a pure physics-only PINN.
    needs_parameter_generalization : the trained model must generalize
        across a FAMILY of parameter values/initial-boundary conditions,
        not just solve one fixed instance.
    geometry_varies : the family also varies in GEOMETRY (not just
        parameters) -- pushes the recommendation toward MeshGraphNet
        over FNO/DeepONet when combined with ``needs_parameter_generalization``.
    is_inverse_problem : the unknown is a physical parameter to calibrate
        against observations, not the field itself -- adds
        ``inverse_pinn`` as a caveat/addition regardless of the other axes.
    catalog : override ``ARCHITECTURE_CATALOG``. The decision tree's
        branches resolve to fixed category keys (``"pinn"``, ``"fno"``,
        ``"mesh_graph_net"``, ``"xpinn"``, ``"deeponet"``, ``"pino"``,
        ``"inverse_pinn"``), so a replacement catalog must provide entries
        under those same keys (e.g. to swap in a different
        ``registry_key``/description for one category) -- this overrides
        entry CONTENT, not the category taxonomy itself. A catalog missing
        a key some branch resolves to raises ``KeyError`` rather than
        silently falling back, so an incompatible override is caught
        immediately, not at some later, harder-to-trace call site.
    """
    catalog = catalog if catalog is not None else ARCHITECTURE_CATALOG
    if has_lots_of_data is None:
        has_lots_of_data = n_high_fidelity_simulations >= _DEFAULT_DATA_THRESHOLD

    decision_path: List[str] = []
    caveats: List[str] = []

    if not has_lots_of_data:
        decision_path.append(
            f"data availability: {n_high_fidelity_simulations} simulation(s) available "
            f"(< threshold {_DEFAULT_DATA_THRESHOLD}) -> treated as data-scarce."
        )
        if has_analytical_solver:
            decision_path.append(
                "an existing solver can generate more data on demand -> recommend a hybrid "
                "strategy: use the solver to supply additional training data (or an online "
                "physics check) alongside a PINN-family base architecture, rather than a pure "
                "physics-only PINN or a purely data-driven operator."
            )
            primary_key = "pinn"
            caveats.append(
                "'hybrid_ml' here is a DATA-GENERATION STRATEGY (use the available solver to "
                "supplement scarce data), not a distinct architecture family in "
                "ARCHITECTURE_CATALOG -- the base architecture recommendation is still driven "
                "by the generalization/geometry axes below."
            )
        else:
            decision_path.append(
                "no existing solver to lean on -> recommend a pure physics-residual PINN "
                "(no simulation data required at all)."
            )
            primary_key = "pinn"
    else:
        decision_path.append(
            f"data availability: {n_high_fidelity_simulations} simulation(s) available "
            f"(>= threshold {_DEFAULT_DATA_THRESHOLD}) -> treated as data-rich."
        )
        if needs_parameter_generalization or geometry_varies:
            decision_path.append(
                "needs to generalize across parameters/geometry -> this is an OPERATOR-LEARNING "
                "problem, not a single-instance fit."
            )
            if geometry_varies:
                decision_path.append(
                    "geometry itself varies across the family -> MeshGraphNet's arbitrary-mesh "
                    "representation is the natural fit (FNO/DeepONet both assume a fixed "
                    "grid/sensor layout across the family)."
                )
                primary_key = "mesh_graph_net"
            else:
                decision_path.append(
                    "geometry is fixed, only parameters/ICs/BCs vary -> FNO if the domain is a "
                    "structured grid (periodic-friendly), DeepONet if query/sensor points are "
                    "scattered/irregular."
                )
                primary_key = "fno"
        else:
            decision_path.append(
                "no cross-instance generalization need despite having data -> a single-instance "
                "PINN fit (with the extra data as supervision alongside the residual) still "
                "applies; more data by itself is not a reason to reach for operator learning."
            )
            primary_key = "pinn"

    matches = [primary_key]
    if primary_key == "fno":
        matches.append("deeponet")
        matches.append("pino")
    elif primary_key == "mesh_graph_net":
        pass
    elif primary_key == "pinn":
        matches.append("xpinn")

    if is_inverse_problem:
        decision_path.append(
            "is_inverse_problem=True -> add inverse_pinn regardless of the other axes: the "
            "unknown is a calibrated PHYSICAL PARAMETER, which PINNeAPPle's inverse-problem "
            "machinery (uncertainty_inverse.solve_inverse_problem) specifically expects via an "
            "inverse_params convention."
        )
        if "inverse_pinn" not in matches:
            matches.append("inverse_pinn")

    primary = matches[0]
    reasoning = (
        f"{catalog[primary].name} ({catalog[primary].category}): {catalog[primary].when_to_use} "
        f"[{catalog[primary].source}]"
    )
    alternatives = [
        {
            "architecture": key,
            "name": catalog[key].name,
            "registry_key": catalog[key].registry_key,
            "why_also_listed": catalog[key].when_to_use,
        }
        for key in matches[1:]
    ]

    return ArchitectureRecommendation(
        recommended=matches, reasoning=reasoning, decision_path=decision_path,
        alternatives=alternatives, caveats=caveats,
    )
