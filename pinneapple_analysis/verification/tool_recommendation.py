"""External tool/player recommendation -- the sibling of
``solver_orchestration`` for tools OUTSIDE PINNeAPPle: given a physics
problem, which real, named CFD/FEM/multiphysics/meshing/visualization/UQ
software (open-source or commercial) is actually a documented fit, and
what are the real tradeoffs (license, typical cost tier, maturity)
against the alternatives?

This module never claims to run, license, or benchmark any of these
tools -- there is no live-availability probe here the way
``solver_orchestration.list_available_solver_families()`` has for
PINNeAPPle's own internal solvers (checking whether ANSYS is installed
and licensed on the caller's machine is out of scope; commercial tools
in particular have no reliable universal "is it on PATH" signal the way
open-source CLI solvers do). Instead, ``TOOL_CATALOG`` is a small,
explicitly-cited, hand-curated starter set of well-known, real software
-- each entry's ``pde_kinds``/``dims``/``license``/``strengths``/
``caveats`` sourced from that tool's own public documentation (cited in
``source``) -- and :func:`recommend_tools` ranks catalog entries against
a ``ProblemSpec`` the same documented-scope-matching way
``select_solver_family`` does for internal solvers, never inventing a
capability a tool doesn't publicly document.

Honest scope limits, stated plainly rather than glossed over:

- ``typical_cost_tier`` is a coarse, qualitative bucket (``"free"``,
  ``"$"``, ``"$$"``, ``"$$$"``) -- real commercial CFD/FEM licensing is
  typically quote-based, seat/core-count-dependent, and changes over
  time; this module does not claim to know current prices.
- The catalog is a STARTER set (a dozen or so well-known tools spanning
  CFD/FEM/multiphysics/meshing/viz/UQ), not an exhaustive or continuously
  -updated market database -- callers needing broader coverage should
  extend ``TOOL_CATALOG`` (or pass a custom ``catalog=`` to
  :func:`recommend_tools`), which is a first-class, documented extension
  point, not an afterthought.
- Like ``solver_orchestration``, this has no model of numerical accuracy
  or which tool would "perform better" -- only which tools' own
  documented scope covers the problem at hand.
- ``TOOL_CATALOG`` has no ``"ML/surrogate"`` entries yet (the category
  exists in :class:`ExternalTool`'s docstring but nothing is catalogued
  under it): the ~94-library third-party physics-ML landscape (PINNs,
  neural operators, differentiable simulators, ML potentials) is a
  fast-moving, install-status-dependent thing a static, hand-curated
  entry here can't honestly represent -- that comparison lives instead
  in PINNeAPPle-arena's external model catalog (a separate app,
  deliberately not imported from here: this library must stay
  installable standalone, and a live per-model install/run status isn't
  a "tool exists and documents this scope" fact this module deals in).
  See ``BUY_VS_BUILD_GUIDANCE["ml_surrogate_landscape_comparison"]``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set


@dataclass(frozen=True)
class ExternalTool:
    """One real, named external tool and its documented scope.

    ``pde_kinds`` : substrings matched against ``spec.pde.kind``, the same
        convention ``solver_orchestration._CLASSICAL_SOLVER_SCOPE`` uses.
    ``dims`` : which spatial dimensionalities the tool documents support
        for (``{2}``, ``{3}``, or ``{2, 3}``).
    ``license`` : ``"open-source"`` or ``"commercial"`` (a tool with a
        free tier alongside paid tiers, e.g. COMSOL's evaluation license,
        is still classified by its primary/production licensing model).
    ``typical_cost_tier`` : ``"free"``, ``"$"``, ``"$$"``, or ``"$$$"`` --
        a coarse qualitative bucket, not a quote (see module docstring).
    ``strengths`` / ``caveats`` : short, real, documented facts about the
        tool -- not marketing copy, not fabricated numbers.
    ``source`` : where this entry's scope claims were sourced from (the
        tool's own public documentation/website) -- exactly the citation
        discipline ``_CLASSICAL_SOLVER_SCOPE`` already uses internally.
    """
    name: str
    category: str  # "CFD" | "FEM" | "multiphysics" | "meshing" | "visualization" | "UQ" | "ML/surrogate"
    license: str  # "open-source" | "commercial"
    typical_cost_tier: str  # "free" | "$" | "$$" | "$$$"
    pde_kinds: Sequence[str] = field(default_factory=tuple)
    dims: Set[int] = field(default_factory=lambda: {2, 3})
    strengths: Sequence[str] = field(default_factory=tuple)
    caveats: Sequence[str] = field(default_factory=tuple)
    source: str = ""


# Starter catalog -- see module docstring for scope/extension notes. Every
# entry's pde_kinds/strengths/caveats are real, documented facts about that
# tool, cited in `source` (the tool's own public documentation), the same
# discipline `_CLASSICAL_SOLVER_SCOPE` in solver_orchestration.py uses.
TOOL_CATALOG: Dict[str, ExternalTool] = {
    "openfoam": ExternalTool(
        name="OpenFOAM", category="CFD", license="open-source", typical_cost_tier="free",
        pde_kinds=["navier_stokes", "channel_flow", "heat", "combustion", "multiphase"],
        dims={2, 3},
        strengths=[
            "Finite-volume CFD covering incompressible/compressible flow, heat transfer, "
            "combustion, and multiphase solvers out of the box.",
            "No license cost at any scale; source available for custom solver development.",
            "Large user/developer community and third-party training ecosystem.",
        ],
        caveats=[
            "No official GUI (ParaView is the standard companion for pre/post-processing); "
            "steeper initial learning curve than GUI-driven commercial tools.",
            "No vendor support contract by default (paid support available via third parties, "
            "e.g. the OpenFOAM Foundation's partner network).",
        ],
        source="openfoam.org / openfoam.com documentation (OpenFOAM User Guide).",
    ),
    "ansys_fluent": ExternalTool(
        name="ANSYS Fluent", category="CFD", license="commercial", typical_cost_tier="$$$",
        pde_kinds=["navier_stokes", "channel_flow", "heat", "combustion", "multiphase", "turbulence"],
        dims={2, 3},
        strengths=[
            "Industry-standard commercial CFD with a mature GUI, extensive turbulence/"
            "combustion/multiphase model library, and automated meshing pipeline.",
            "Vendor support, certification, and validation documentation suited to "
            "regulated-industry (aerospace/automotive/energy) workflows.",
        ],
        caveats=[
            "Per-seat/per-core commercial licensing; cost scales with concurrent usage.",
            "Workflow is GUI/journal-script-centric, less naturally scriptable/version-"
            "controllable than a plain-text-case-file tool like OpenFOAM.",
        ],
        source="ansys.com Fluent product documentation.",
    ),
    "ansys_mechanical": ExternalTool(
        name="ANSYS Mechanical", category="FEM", license="commercial", typical_cost_tier="$$$",
        pde_kinds=["linear_elasticity", "structural", "modal", "thermal_stress"],
        dims={2, 3},
        strengths=[
            "Mature commercial structural FEM: linear/nonlinear elasticity, modal, "
            "thermal-stress, fatigue, contact mechanics.",
            "Extensive material model library and vendor-validated solver documentation.",
        ],
        caveats=["Per-seat/per-core commercial licensing, same tier as ANSYS Fluent."],
        source="ansys.com Mechanical product documentation.",
    ),
    "fenicsx": ExternalTool(
        name="FEniCSx (dolfinx)", category="FEM", license="open-source", typical_cost_tier="free",
        pde_kinds=["heat_equation_steady", "linear_elasticity_plane_stress",
                   "linear_elasticity_plane_strain", "poisson", "laplace"],
        dims={2, 3},
        strengths=[
            "Python/C++-native finite-element library with a high-level variational-form "
            "language (UFL) -- well suited to custom/research PDE formulations.",
            "No license cost; source available.",
        ],
        caveats=[
            "No built-in GUI or automated meshing pipeline -- typically paired with Gmsh "
            "for mesh generation and ParaView/PyVista for post-processing.",
            "Native install depends on PETSc/MPI/HDF5 native builds -- reliably distributed "
            "via conda-forge rather than plain pip on most platforms (confirmed directly "
            "this session: not pip-installable into an arbitrary macOS venv, but installs "
            "cleanly via `mamba create -c conda-forge fenics-dolfinx`).",
        ],
        source="fenicsproject.org documentation; DOLFINx installation guide.",
    ),
    "comsol": ExternalTool(
        name="COMSOL Multiphysics", category="multiphysics", license="commercial", typical_cost_tier="$$$",
        pde_kinds=["heat", "navier_stokes", "linear_elasticity", "electromagnetics", "acoustics",
                   "chemical_species_transport"],
        dims={2, 3},
        strengths=[
            "Tightly-coupled multiphysics GUI: fluid-structure interaction, electro-thermal, "
            "acoustic-structural, and more, without hand-coding the coupling.",
            "Per-physics-module licensing lets a customer buy only what they need "
            "(though the whole-platform total can still be substantial).",
        ],
        caveats=[
            "Commercial, per-module licensing -- multiphysics coupling across many "
            "modules can become expensive at scale.",
            "Proprietary mesh/model file format; less naturally scriptable/CI-friendly "
            "than an open-source, plain-text-driven tool.",
        ],
        source="comsol.com product/licensing documentation.",
    ),
    "siemens_starccm": ExternalTool(
        name="Siemens Simcenter STAR-CCM+", category="CFD", license="commercial", typical_cost_tier="$$$",
        pde_kinds=["navier_stokes", "channel_flow", "heat", "multiphase", "turbulence"],
        dims={2, 3},
        strengths=[
            "Strong automated/adaptive meshing pipeline, reducing manual mesh-cleanup time "
            "for complex real-world geometry.",
            "Broad multiphysics coupling (CFD + structural + electromagnetics) within one "
            "integrated environment, commonly used in automotive/aerospace.",
        ],
        caveats=["Commercial, quote-based licensing, same general tier as ANSYS/COMSOL."],
        source="plm.sw.siemens.com Simcenter STAR-CCM+ product documentation.",
    ),
    "su2": ExternalTool(
        name="SU2", category="CFD", license="open-source", typical_cost_tier="free",
        pde_kinds=["navier_stokes", "compressible_euler", "aerodynamics"],
        dims={2, 3},
        strengths=[
            "Purpose-built for compressible aerodynamics and adjoint-based shape "
            "optimization (gradient-based design workflows are a first-class use case, "
            "not an add-on).",
            "No license cost; developed originally at Stanford's Aerospace Design Lab, "
            "widely used in academic/research aerodynamics.",
        ],
        caveats=[
            "Narrower general-purpose CFD scope than OpenFOAM (less mature multiphase/"
            "combustion tooling) -- strongest specifically for aero/compressible-flow "
            "and design-optimization workflows.",
        ],
        source="su2code.github.io documentation.",
    ),
    "moose": ExternalTool(
        name="MOOSE", category="multiphysics", license="open-source", typical_cost_tier="free",
        pde_kinds=["heat", "linear_elasticity", "reaction_diffusion", "neutron_transport"],
        dims={2, 3},
        strengths=[
            "A finite-element MULTIPHYSICS FRAMEWORK (not a single fixed solver) -- built "
            "for coupling many physics kernels (thermal/mechanical/chemical/neutronics) "
            "in one implicit solve; originated at Idaho National Laboratory for nuclear "
            "fuel-performance modeling and has since generalized to other domains.",
            "No license cost; source available; strong for problems needing tight, "
            "custom multiphysics coupling that a fixed commercial GUI doesn't expose.",
        ],
        caveats=[
            "Framework, not an out-of-the-box GUI application -- building a new physics "
            "module/app typically requires C++ development, a materially higher setup "
            "cost than a ready-made solver for a standard single-physics problem.",
        ],
        source="mooseframework.inl.gov documentation.",
    ),
    "calculix": ExternalTool(
        name="CalculiX", category="FEM", license="open-source", typical_cost_tier="free",
        pde_kinds=["linear_elasticity", "structural", "modal", "thermal_stress"],
        dims={2, 3},
        strengths=[
            "Free structural FEM solver with an Abaqus-compatible input-deck format, "
            "easing migration from/interop with Abaqus-based workflows.",
        ],
        caveats=[
            "Smaller user/support community and narrower material-model library than "
            "ANSYS Mechanical or Abaqus itself.",
        ],
        source="calculix.de documentation.",
    ),
    "gmsh": ExternalTool(
        name="Gmsh", category="meshing", license="open-source", typical_cost_tier="free",
        pde_kinds=[],  # mesh generation, not a PDE solver -- never scope-matched by pde_kind
        dims={2, 3},
        strengths=[
            "Open-source 2D/3D mesh generator with a built-in CAD kernel and a scriptable "
            "(.geo) format -- the standard free pairing for FEniCSx/OpenFOAM/SU2 pipelines "
            "needing programmatic mesh generation.",
        ],
        caveats=["Meshing only -- not a solver; always paired with a separate solver."],
        source="gmsh.info documentation.",
    ),
    "paraview": ExternalTool(
        name="ParaView / PyVista", category="visualization", license="open-source", typical_cost_tier="free",
        pde_kinds=[],
        dims={2, 3},
        strengths=[
            "Open-source, VTK-based post-processing/visualization -- ParaView's GUI for "
            "interactive exploration, PyVista's Python API for scripted/automated figure "
            "generation and CI-pipeline integration.",
        ],
        caveats=["Post-processing only -- not a solver."],
        source="paraview.org / pyvista.org documentation.",
    ),
    "openturns": ExternalTool(
        name="OpenTURNS", category="UQ", license="open-source", typical_cost_tier="free",
        pde_kinds=[],
        dims={2, 3},
        strengths=[
            "Open-source uncertainty-quantification/sensitivity-analysis library "
            "(polynomial chaos, Sobol indices, reliability analysis) usable alongside "
            "any external solver's output, not tied to one CFD/FEM vendor.",
        ],
        caveats=["A UQ/statistics library, not a physics solver."],
        source="openturns.org documentation.",
    ),
}


@dataclass
class ToolRecommendation:
    """Result of :func:`recommend_tools`. Mirrors
    ``solver_orchestration.SolverRecommendation``'s shape/honesty
    contract, for the external-tool axis instead of PINNeAPPle's
    internal solver families."""
    recommended: List[str]  # TOOL_CATALOG keys, ranked; [] if nothing in-catalog documents a scope match
    reasoning: str
    alternatives: List[Dict[str, Any]]  # [{"tool": key, "why_not_primary": str}, ...]
    caveats: List[str] = field(default_factory=list)


def recommend_tools(
    spec: Any,
    dimensionless_numbers: Optional[Any] = None,
    *,
    catalog: Optional[Dict[str, ExternalTool]] = None,
    category: Optional[str] = None,
) -> ToolRecommendation:
    """Recommend external tools whose documented scope matches *spec* (a
    ``pinneapple_physics.ProblemSpec``), ranked the same documented-scope
    -matching way ``solver_orchestration.select_solver_family`` ranks
    PINNeAPPle's own internal solvers -- see the module docstring for
    exactly what this does and does NOT claim to know (no live
    availability/license probing, no accuracy/performance ranking, a
    starter catalog rather than an exhaustive market database).

    Parameters
    ----------
    spec : ProblemSpec, used for ``spec.pde.kind``/``spec.dim`` matching
        against each catalog entry's ``pde_kinds``/``dims`` (same
        substring-match convention as ``_CLASSICAL_SOLVER_SCOPE``).
    dimensionless_numbers : optional ``DimensionlessNumbers`` -- used only
        to ADD an honest caveat (e.g. flagging a turbulent-regime Reynolds
        number when the matched tools are meshing/viz-only), never to
        change which tools are matched.
    catalog : override ``TOOL_CATALOG`` (e.g. to add organization-specific
        tools) -- the documented extension point.
    category : optionally restrict matching to one category
        (``"CFD"``/``"FEM"``/``"multiphysics"``/``"meshing"``/
        ``"visualization"``/``"UQ"``/``"ML/surrogate"``).

    Returns
    -------
    ToolRecommendation
        ``recommended`` is ranked open-source-first (a deliberate,
        stated tie-break -- see ``reasoning`` -- since this module has no
        accuracy/performance signal to break ties on) among every tool
        whose ``pde_kinds``/``dims`` match; empty if nothing in the
        catalog documents a scope match for this problem (never a forced
        pick).
    """
    catalog = catalog if catalog is not None else TOOL_CATALOG
    kind = str(getattr(spec.pde, "kind", "")).lower().replace("-", "_").replace(" ", "_")
    dim = int(getattr(spec, "dim", 2))

    matches: List[str] = []
    for key, tool in catalog.items():
        if category is not None and tool.category != category:
            continue
        if not tool.pde_kinds:
            continue  # meshing/viz/UQ tools are never pde_kind-scope-matched; see get_tools_by_category
        kind_hit = any(sub in kind for sub in tool.pde_kinds)
        dim_hit = dim in tool.dims
        if kind_hit and dim_hit:
            matches.append(key)

    # Deliberate, stated tie-break: open-source first (no license-cost barrier to
    # actually trying the recommendation), then alphabetical for determinism.
    # This is NOT a claim that open-source tools are technically superior --
    # this module has no accuracy/performance signal to make that claim with.
    matches.sort(key=lambda k: (catalog[k].license != "open-source", catalog[k].name))

    caveats: List[str] = []
    if dimensionless_numbers is not None:
        re = getattr(dimensionless_numbers, "reynolds", None)
        if re is not None and re >= 4000 and any(catalog[m].category == "CFD" for m in matches):
            caveats.append(
                f"Reynolds number {re:.3g} indicates a turbulent regime -- confirm the "
                f"recommended tool's specific turbulence-closure model (RANS/LES/DES "
                f"variant) is appropriate for this problem; this module only matches on "
                f"documented PDE-kind/dimension scope, not turbulence-model fidelity."
            )

    if not matches:
        return ToolRecommendation(
            recommended=[],
            reasoning=(
                f"spec.pde.kind={kind!r} (dim={dim}) does not match any catalog entry's "
                f"documented pde_kinds/dims. This is a starter catalog (see module "
                f"docstring), not an exhaustive database -- absence of a match here means "
                f"'not yet catalogued', not 'no real-world tool exists for this problem'."
            ),
            alternatives=[],
            caveats=caveats,
        )

    primary = matches[0]
    alternates = matches[1:]
    reasoning = (
        f"spec.pde.kind={kind!r} (dim={dim}) matches {catalog[primary].name}'s documented "
        f"scope ({catalog[primary].source}). Ranked open-source-first among every "
        f"scope-matching tool (a cost-of-entry tie-break, not an accuracy claim -- see "
        f"this function's docstring)."
    )
    alternatives = [
        {
            "tool": key,
            "name": catalog[key].name,
            "license": catalog[key].license,
            "why_not_primary": (
                f"also matches spec.pde.kind={kind!r}/dim={dim}, but ranked after "
                f"{catalog[primary].name} by this module's open-source-first tie-break "
                f"(license={catalog[key].license})."
            ),
        }
        for key in alternates
    ]

    return ToolRecommendation(
        recommended=matches, reasoning=reasoning, alternatives=alternatives, caveats=caveats,
    )


def get_tools_by_category(category: str, *, catalog: Optional[Dict[str, ExternalTool]] = None) -> List[str]:
    """List catalog tool keys in *category* -- the entry point for
    non-PDE-scoped categories (``"meshing"``, ``"visualization"``,
    ``"UQ"``) that :func:`recommend_tools` never scope-matches by design
    (a mesh generator or a UQ library doesn't "solve" a ``pde.kind``)."""
    catalog = catalog if catalog is not None else TOOL_CATALOG
    return sorted(k for k, t in catalog.items() if t.category == category)


# ═══════════════════════════════════════════════════════════════════════
# Players (companies/ecosystems) vs. Tools (specific software) -- a
# deliberate distinction: ANSYS the COMPANY makes both Fluent (CFD) and
# Mechanical (FEM); recommending "ANSYS" and recommending "ANSYS Fluent"
# are different-grained answers to different questions ("who could I buy
# a support relationship from" vs. "which specific software solves this
# PDE"). TOOL_CATALOG/recommend_tools answers the second question;
# PLAYER_CATALOG answers the first.
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Player:
    """One real company/ecosystem and what it's known for. ``tool_keys``
    links to ``TOOL_CATALOG`` entries this player makes/sells, where one
    exists in the (starter) catalog -- a player can be real and relevant
    with an empty ``tool_keys`` (e.g. a cloud/compute provider that hosts
    other vendors' software rather than making its own solver)."""
    name: str
    focus_areas: Sequence[str]  # e.g. ["CFD", "FEM", "multiphysics", "cloud/HPC", "GPU compute"]
    tool_keys: Sequence[str] = field(default_factory=tuple)  # TOOL_CATALOG keys this player makes
    notes: str = ""


PLAYER_CATALOG: Dict[str, Player] = {
    "ansys": Player(
        name="ANSYS, Inc.", focus_areas=["CFD", "FEM", "multiphysics", "electromagnetics"],
        tool_keys=["ansys_fluent", "ansys_mechanical"],
        notes="Broad commercial engineering-simulation suite; one of the most widely deployed "
              "in regulated industries (aerospace/automotive/energy).",
    ),
    "siemens_dis": Player(
        name="Siemens Digital Industries Software", focus_areas=["CFD", "PLM", "multiphysics"],
        tool_keys=["siemens_starccm"],
        notes="Simcenter product family plus broader PLM/CAD (NX, Teamcenter) -- simulation is "
              "one part of a larger industrial-software portfolio.",
    ),
    "comsol_inc": Player(
        name="COMSOL, Inc.", focus_areas=["multiphysics"], tool_keys=["comsol"],
        notes="Multiphysics-first commercial vendor; per-module licensing lets a customer buy "
              "only the physics they need.",
    ),
    "dassault": Player(
        name="Dassault Systèmes", focus_areas=["CAD", "FEM", "multiphysics", "PLM"],
        tool_keys=[],
        notes="SolidWorks/CATIA/Abaqus/SIMULIA product family -- CAD-centric engineering suite; "
              "no entry in TOOL_CATALOG yet (a real extension candidate, see module docstring).",
    ),
    "altair": Player(
        name="Altair Engineering", focus_areas=["FEM", "CFD", "optimization", "HPC"], tool_keys=[],
        notes="Strong in structural optimization/topology optimization (OptiStruct) and HPC "
              "workload management (PBS) alongside its simulation suite.",
    ),
    "nvidia": Player(
        name="NVIDIA", focus_areas=["GPU compute", "ML/surrogate", "accelerated CFD"],
        tool_keys=[],
        notes="GPU hardware plus PhysicsNeMo (physics-ML framework) and Modulus/Warp -- the "
              "dominant compute-hardware layer under most GPU-accelerated PINN/neural-operator "
              "training, regardless of which higher-level framework is used.",
    ),
    "cadence": Player(
        name="Cadence Design Systems", focus_areas=["electromagnetics", "electronics thermal", "CFD"],
        tool_keys=[],
        notes="Electronic design automation (EDA) heritage; Cadence's simulation portfolio "
              "(via the Fidelity/Celsius product lines) is strongest for electronics-adjacent "
              "thermal/electromagnetic problems specifically.",
    ),
    "mathworks": Player(
        name="MathWorks", focus_areas=["scripting/prototyping", "control systems", "signal processing"],
        tool_keys=[],
        notes="MATLAB/Simulink -- not a CFD/FEM solver vendor itself, but a common "
              "prototyping/control-systems layer many engineering teams already standardize on.",
    ),
    "aws": Player(
        name="Amazon Web Services", focus_areas=["cloud/HPC", "GPU compute"], tool_keys=[],
        notes="Compute infrastructure provider, not a physics-tool vendor -- relevant as a "
              "deployment target for any of the above (self-hosted OpenFOAM/FEniCSx clusters, "
              "managed HPC, GPU instances for training).",
    ),
    "azure": Player(
        name="Microsoft Azure", focus_areas=["cloud/HPC", "GPU compute"], tool_keys=[],
        notes="Same role as AWS -- compute infrastructure, not a physics-tool vendor.",
    ),
    "gcp": Player(
        name="Google Cloud Platform", focus_areas=["cloud/HPC", "GPU compute"], tool_keys=[],
        notes="Same role as AWS/Azure -- compute infrastructure, not a physics-tool vendor.",
    ),
    "kitware": Player(
        name="Kitware, Inc.", focus_areas=["visualization"], tool_keys=["paraview"],
        notes="The commercial-support company behind the open-source VTK/ParaView stack -- "
              "ParaView itself stays free/open-source; Kitware sells support/custom development.",
    ),
}


def get_player_for_tool(tool_key: str, *, player_catalog: Optional[Dict[str, Player]] = None) -> Optional[str]:
    """Reverse lookup: which ``PLAYER_CATALOG`` key makes the given
    ``TOOL_CATALOG`` tool, if any (``None`` for a community/consortium
    project with no single corporate player behind it, e.g. FEniCSx/Gmsh/
    OpenFOAM's Foundation side/SU2/MOOSE/CalculiX/OpenTURNS)."""
    player_catalog = player_catalog if player_catalog is not None else PLAYER_CATALOG
    for key, player in player_catalog.items():
        if tool_key in player.tool_keys:
            return key
    return None


# ═══════════════════════════════════════════════════════════════════════
# Buy vs. Build guidance -- a small, honestly-scoped need -> recommendation
# lookup. Deliberately qualitative (category-level guidance, not a
# specific-vendor ranking with fabricated scores) -- see module docstring.
# ═══════════════════════════════════════════════════════════════════════

BUY_VS_BUILD_GUIDANCE: Dict[str, str] = {
    "CFD_high_fidelity": (
        "Buy or use open-source, don't build: OpenFOAM (open-source) or ANSYS Fluent/Siemens "
        "STAR-CCM+ (commercial, regulated-industry validation/support) -- decades of validated "
        "numerics behind each; re-deriving a general-purpose CFD solver from scratch is rarely "
        "justified outside of research into the numerics themselves."
    ),
    "FEM_structural": (
        "Buy or use open-source: FEniCSx/CalculiX (open-source) or ANSYS Mechanical (commercial). "
        "Same reasoning as CFD -- mature, validated numerics already exist."
    ),
    "multiphysics_coupling": (
        "Buy for tightly-integrated GUI coupling (COMSOL, Siemens), or use MOOSE (open-source "
        "framework) if the coupling is unusual enough that a fixed commercial GUI doesn't expose "
        "the needed kernels -- MOOSE trades GUI convenience for C++ extensibility."
    ),
    "GPU_native_physics_ai": (
        "Build on top of an existing framework (PyTorch + PINNeAPPle's own architectures, or "
        "NVIDIA PhysicsNeMo) rather than writing custom CUDA kernels -- the differentiated value "
        "is in the physics/architecture/verification layer, not in re-implementing GPU primitives "
        "that PyTorch/cuDNN/cuFFT already provide."
    ),
    "mesh_generation": "Use open-source: Gmsh -- mature, scriptable, free; rarely worth building.",
    "uncertainty_quantification": (
        "Use open-source: OpenTURNS -- mature UQ/sensitivity-analysis library usable alongside "
        "any solver's output; building UQ machinery from scratch is rarely the differentiated part "
        "of a physics-AI product."
    ),
    "visualization": "Use open-source: ParaView/PyVista -- building a VTK-equivalent from scratch is not a good use of engineering time.",
    "digital_twin_platform": (
        "Build the domain-specific parts (sensor mapping, calibration loop, decision logic) on "
        "top of an existing sensor/assimilation substrate rather than a bespoke stack -- see "
        "pinneapple_systems.digital_twin for PINNeAPPle's own real, tested building blocks "
        "(SensorRegistry, MQTT/Kafka/HTTP-poll streams, Kalman assimilation, anomaly detection) "
        "before building new IoT-ingestion/calibration code."
    ),
    "hpc_infrastructure": (
        "Buy/rent: AWS/Azure/GCP for elastic capacity, or on-prem HPC only once utilization is "
        "high/steady enough to justify capital cost -- infrastructure is rarely a product "
        "differentiator this early."
    ),
    "commercial_engineering_workflow_integration": (
        "Buy: ANSYS/Siemens/Dassault -- when the customer's own downstream workflow (CAD "
        "provenance, PLM, certification paperwork) is already built around one of these "
        "ecosystems, integrating with it beats replacing it."
    ),
    "fast_surrogate_inference": (
        "Build (this is genuinely the differentiated layer): FNO/GNN/DeepONet/PINO trained on "
        "your own domain's simulation data -- see pinneapple_analysis.verification."
        "architecture_recommendation for which family fits which data/generalization regime."
    ),
    "ml_surrogate_landscape_comparison": (
        "Don't hand-pick from a static list: PINNeAPPle-arena's external model catalog runs "
        "~94 third-party physics-ML libraries (PINNs/neural operators/differentiable "
        "simulators/ML potentials) side by side against PINNeAPPle's own architectures, with a "
        "real per-model install/run status (never a fabricated 'it works') -- consult it before "
        "assuming PINNeAPPle's own architecture zoo is the only option, or before picking a "
        "third-party library on reputation alone."
    ),
}


def buy_vs_build_recommendation(need: str, *, guidance: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Look up qualitative buy-vs-build guidance for a named *need* (a key
    into :data:`BUY_VS_BUILD_GUIDANCE`, e.g. ``"CFD_high_fidelity"``).
    Returns ``None`` (not a guess) for a need this starter table doesn't
    cover yet -- callers should extend ``BUY_VS_BUILD_GUIDANCE`` or pass
    a custom ``guidance=`` dict rather than get a fabricated answer.
    Call :func:`list_buy_vs_build_needs` to see every covered key."""
    guidance = guidance if guidance is not None else BUY_VS_BUILD_GUIDANCE
    return guidance.get(need)


def list_buy_vs_build_needs(*, guidance: Optional[Dict[str, str]] = None) -> List[str]:
    guidance = guidance if guidance is not None else BUY_VS_BUILD_GUIDANCE
    return sorted(guidance.keys())
