"""Recommend a numerical method + turbulence closure from a computed flow regime.

Three real pieces already exist in this repo but nothing connects them:

  * :func:`pinneapple_analysis.verification.dimensional_analysis.classify_flow_regime`
    computes Reynolds number from physical parameters and classifies the flow
    as laminar/transitional/turbulent (geometry-aware: internal pipe flow vs.
    external bluff-body flow have different transition thresholds).
  * :class:`pinneapple_simulation.numerical_solvers.registry.SolverRegistry`
    holds the real, buildable numerical PDE solvers (``lbm``, ``fem``,
    ``fvm``, ``fdm``, ``spectral``, ``sph``, ...).
  * :func:`pinneapple_physics.pde_environment.turbulence_selector.get_turbulence_closure`
    builds/validates a real turbulence closure once a
    :class:`~pinneapple_physics.pde_environment.turbulence_selector.TurbulenceModel`
    has already been chosen.

This module is the missing "given these problem characteristics, here's the
recommended numerical approach" step: it calls the real ``classify_flow_regime``
to get a regime, applies a small explicit decision table to pick a real,
registered solver name, and (where turbulence closure is relevant) calls the
real ``get_turbulence_closure`` to validate the recommended
``TurbulenceModel`` is actually constructible for that solver family.

HONESTY / SCOPE CAVEAT (read before trusting this for a real design decision)
------------------------------------------------------------------------
The decision table below is a small, explicit, inspectable heuristic
grounded in real dimensional analysis and this repo's *actual* registered
solver descriptions -- it is NOT a validated CFD-expert-system and does not
replace engineering judgement. In particular:

* Inspecting the registered solvers' own descriptions/tags
  (``SolverRegistry.spec(name)``) shows that, in this repo, ``fem`` is a
  problem-agnostic Poisson/Helmholtz/elasticity solver and ``fvm`` is a
  problem-agnostic scalar diffusion/convection solver -- *neither actually
  solves the Navier-Stokes equations here*. ``lbm``/``lbm_3d`` (tags
  ``fluids``, ``navier_stokes``, ``cfd``) is the only registered solver in
  this repo that is genuinely a fluid-flow solver. So for essentially every
  incompressible-flow regime this table recommends ``lbm`` -- not because
  every other solver is unsuitable in general CFD practice, but because it
  is the only one of *this repo's* registered solvers actually built for
  the physics ``classify_flow_regime`` is reasoning about.
* No solver registered in this repo implements genuinely compressible
  (shock-capturing / high-Mach) Navier-Stokes. For compressible flow this
  table falls back to ``fvm`` (the conventional method family for
  compressible CFD in general practice) while explicitly flagging
  ``confidence="low"`` and saying so in the rationale, rather than silently
  implying a validated compressible solver exists here.
* Turbulence-closure selection is fully delegated to the real
  ``get_turbulence_closure``: this repo's LBM solvers only support
  Smagorinsky LES natively (no RANS), so a turbulent regime recommends
  ``LES_SMAGORINSKY`` when the chosen solver is ``lbm``, and
  ``RANS_K_OMEGA_SST`` (a PINN-residual closure, solver-agnostic in the
  sense that it names the appropriate physical closure family) otherwise.
  Wiring a PINN-residual closure into a specific classical solver's own
  solve loop is not implemented in this repo and is left to the caller.
* Thresholds and regime labels are exactly whatever ``classify_flow_regime``
  already documents (including its own "order-of-magnitude estimate"
  caveat for external bluff-body turbulent wakes) -- this module adds no
  new physics, only a mapping from regime label -> (solver, closure).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from pinneapple_analysis.verification.dimensional_analysis import (
    classify_flow_regime,
    compute_dimensionless_numbers,
)
from pinneapple_physics.pde_environment.turbulence_selector import (
    TurbulenceModel,
    get_turbulence_closure,
)
from pinneapple_simulation.numerical_solvers.registry import SolverRegistry, register_all
from .schema import ProblemSpec

# Standard gas-dynamics threshold for "flow can no longer be treated as
# incompressible" (White, *Fluid Mechanics*): Ma >= 0.3.
_COMPRESSIBLE_MACH_THRESHOLD = 0.3

# A small, honest, clearly-justified capability note per candidate solver --
# NOT asserted as authoritative SolverRegistry metadata (the registry only
# carries free-text `description`/`tags`, inspected below and quoted in the
# rationale where relevant). See the module docstring for the reasoning.
SOLVER_CAPABILITY_NOTES = {
    "lbm": (
        "Lattice-Boltzmann (D2Q9 BGK): the only solver in this repo's "
        "registry actually tagged fluids/navier_stokes/cfd. Good for "
        "transient, low-to-moderate-Re, complex-geometry, near-incompressible "
        "flow; natively supports only Smagorinsky LES as a turbulence "
        "closure (no RANS); not suited to genuinely compressible/high-Mach "
        "flow (BGK-LBM is a weakly-compressible scheme valid near the "
        "incompressible limit)."
    ),
    "fvm": (
        "Finite Volume: in general CFD practice this is the conventional "
        "family for compressible, shock-capturing flow. This repo's "
        "registered `fvm` solver, however, is a problem-agnostic scalar "
        "diffusion/convection MVP, not a compressible Navier-Stokes solver "
        "-- recommended here only as the closest registered analogue, with "
        "low confidence."
    ),
}


@dataclass
class MethodRecommendation:
    """A recommended (numerical_method, turbulence_model) pair for a flow.

    Attributes
    ----------
    numerical_method : str
        A real, registered name from ``SolverRegistry.list()``.
    turbulence_model : Optional[TurbulenceModel]
        A real ``TurbulenceModel`` member, or ``None`` when no turbulence
        closure is recommended (laminar/Stokes regimes, or when the regime
        could not be determined).
    flow_regime : str
        The exact string returned by ``classify_flow_regime``.
    rationale : str
        Short, human-readable explanation citing the actual numbers that
        drove the decision.
    confidence : str
        ``"high"`` or ``"low"`` -- qualitative only, low whenever inputs are
        incomplete/ambiguous (unknown regime, transitional regime, an
        external-bluff-body "order-of-magnitude" turbulent estimate, or a
        compressible flow for which no dedicated solver is registered).
    """
    numerical_method: str
    turbulence_model: Optional[TurbulenceModel]
    flow_regime: str
    rationale: str
    confidence: str


def _validate_registered(name: str) -> None:
    """Raise loudly if a name this module's decision table produced isn't
    actually a registered solver -- that would mean a bug in the table
    below, not a user-input problem."""
    register_all()  # idempotent: re-importing already-registered modules is a no-op
    available = SolverRegistry.list()
    if name not in available:
        raise RuntimeError(
            f"method_selection internal bug: decision table recommended "
            f"solver '{name}', which is not registered in SolverRegistry "
            f"(available: {available}). This indicates a bug in "
            f"pinneapple_problemdesign.method_selection, not a user error."
        )


def _regime_category(regime: str) -> str:
    """Bucket a ``classify_flow_regime`` label into one of a small set of
    categories this module's decision table dispatches on."""
    if regime.startswith("unknown"):
        return "unknown"
    if regime.startswith("transitional"):
        return "transitional"
    if regime.startswith("turbulent"):
        return "turbulent"
    # "laminar (...)", "Stokes / creeping flow (...)", "laminar separated wake (...)"
    return "laminar"


def recommend_method(
    velocity: Optional[float],
    length_scale: Optional[float],
    kinematic_viscosity: Optional[float],
    *,
    geometry: str = "internal_pipe",
    compressible: bool = False,
    dim: int = 2,
    **kwargs: Any,
) -> MethodRecommendation:
    """Recommend a numerical solver + turbulence closure for a flow.

    Calls the real ``classify_flow_regime`` (via ``compute_dimensionless_numbers``
    to get Reynolds/Mach) and applies the small, explicit decision table
    documented in this module's docstring. ``geometry`` must be one of the
    values ``classify_flow_regime`` itself accepts (``"internal_pipe"`` or
    ``"external_bluff_body"``); an invalid value raises exactly as that
    function does. ``**kwargs`` are forwarded to
    ``compute_dimensionless_numbers`` (e.g. ``speed_of_sound`` to compute
    Mach, ``density``/``dynamic_viscosity`` as an alternative to
    ``kinematic_viscosity``).

    This never raises for "insufficient information" -- it degrades to
    ``confidence="low"`` and an explanatory rationale instead, since a
    recommendation object (not ``None``) is always returned here. (The
    ``ProblemSpec``-based wrapper below, consistent with this repo's
    non-invention policy, returns ``None`` instead when it cannot find the
    physical inputs needed to call this function at all.)
    """
    numbers = compute_dimensionless_numbers(
        velocity=velocity,
        length=length_scale,
        kinematic_viscosity=kinematic_viscosity,
        **kwargs,
    )
    regime = classify_flow_regime(numbers.reynolds, geometry=geometry)
    category = _regime_category(regime)

    effective_compressible = compressible or (
        numbers.mach is not None and numbers.mach >= _COMPRESSIBLE_MACH_THRESHOLD
    )

    confidence = "high"
    rationale_bits = [f"flow_regime = {regime!r}."]

    if category == "unknown":
        numerical_method = "lbm"
        turbulence_model = None
        confidence = "low"
        rationale_bits.append(
            "Reynolds number could not be computed from the given "
            "parameters (velocity/length_scale/kinematic_viscosity "
            "incomplete or kinematic_viscosity<=0) -- defaulting to 'lbm' "
            "as this repo's only general-purpose fluid-flow solver, but "
            "this recommendation is unconfirmed without a Reynolds number."
        )
    elif effective_compressible:
        numerical_method = "fvm"
        turbulence_model = None
        confidence = "low"
        mach_txt = f"Ma={numbers.mach:.3g}" if numbers.mach is not None else "compressible=True"
        rationale_bits.append(
            f"{mach_txt} -> flow treated as compressible; no solver in this "
            "repo's registry implements compressible/shock-capturing "
            "Navier-Stokes, so 'fvm' (the conventional family for "
            "compressible CFD in general practice) is recommended with low "
            "confidence rather than presenting an unvalidated solver as a "
            "confident match. No turbulence closure is recommended here "
            "for the same reason."
        )
    else:
        numerical_method = "lbm"
        if category == "laminar":
            turbulence_model = None
            rationale_bits.append(
                "laminar/Stokes regime -> no turbulence closure recommended; "
                "'lbm' handles transient, complex-geometry incompressible "
                "laminar flow well."
            )
        elif category == "transitional":
            turbulence_model = None
            confidence = "low"
            rationale_bits.append(
                "transitional regime is inherently ambiguous -> no "
                "turbulence closure recommended by default; consider "
                "enabling Smagorinsky LES (Cs>0) on the 'lbm' solver if Re "
                "trends toward the turbulent end of this range, or RANS if "
                "Re increases further."
            )
        else:  # "turbulent"
            turbulence_model = TurbulenceModel.LES_SMAGORINSKY
            # Validate constructibility against the REAL closure dispatch --
            # this repo's LBM solvers only support Smagorinsky LES natively.
            get_turbulence_closure(turbulence_model, dim=dim, solver_family="lbm")
            rationale_bits.append(
                "turbulent regime -> LES_SMAGORINSKY recommended (this "
                "repo's LBM solvers support Smagorinsky LES natively; RANS "
                "closures here are PINN-residual-only, not implemented for "
                "LBM's own solve loop)."
            )
            if "order-of-magnitude" in regime:
                confidence = "low"
                rationale_bits.append(
                    "classify_flow_regime itself flags this external "
                    "bluff-body turbulent-wake estimate as order-of-magnitude, "
                    "not a sharp transition -- confidence lowered accordingly."
                )

    _validate_registered(numerical_method)

    return MethodRecommendation(
        numerical_method=numerical_method,
        turbulence_model=turbulence_model,
        flow_regime=regime,
        rationale=" ".join(rationale_bits),
        confidence=confidence,
    )


def recommend_method_from_spec(spec: ProblemSpec) -> Optional[MethodRecommendation]:
    """Convenience wrapper: pull physical parameters out of a ``ProblemSpec``
    and call :func:`recommend_method`, or return ``None`` if they aren't
    present.

    Consistent with this repo's non-invention policy (see e.g.
    ``codegen.py``'s handling of ``PhysicsSpec.parameters_known``, which
    elicits only parameter *names*, never numeric values): today's
    ``ProblemSpec``/``PhysicsSpec``/``GeometrySpec`` schema has no
    structured numeric fields for velocity, a characteristic length, or
    kinematic viscosity -- ``parameters_known`` is a list of bare names and
    ``units`` maps names to unit strings, neither of which is a number this
    function may safely guess. Parsing such values out of free-text fields
    (``domain_context``, ``goal``, etc.) would be exactly the kind of
    guessed-input this repo's policy forbids, so this function does not do
    that.

    Instead, it looks (via ``getattr`` with no assumed default) for
    optional, currently-nonexistent-but-forward-compatible numeric
    attributes -- ``spec.physics.velocity``, ``spec.physics.length_scale``,
    ``spec.physics.kinematic_viscosity`` -- so that if a future elicitation
    pipeline adds structured numeric physics fields to ``PhysicsSpec``, this
    wrapper activates automatically with no further changes here. Until
    then, it returns ``None`` for every spec, which is the honest answer:
    the information needed to recommend a numerical method is not yet
    captured anywhere in ``ProblemSpec``.
    """
    physics = spec.physics
    velocity = getattr(physics, "velocity", None)
    length_scale = getattr(physics, "length_scale", None) or getattr(physics, "characteristic_length", None)
    kinematic_viscosity = getattr(physics, "kinematic_viscosity", None)

    if velocity is None or length_scale is None or kinematic_viscosity is None:
        return None

    geometry = "internal_pipe"
    domain_text = f"{spec.geometry.domain} {spec.domain_context}".lower()
    if any(kw in domain_text for kw in ("airfoil", "sphere", "cylinder", "bluff", "external")):
        geometry = "external_bluff_body"

    compressible = bool(getattr(physics, "compressible", False))
    speed_of_sound = getattr(physics, "speed_of_sound", None)

    kwargs = {}
    if speed_of_sound is not None:
        kwargs["speed_of_sound"] = speed_of_sound

    return recommend_method(
        velocity,
        length_scale,
        kinematic_viscosity,
        geometry=geometry,
        compressible=compressible,
        **kwargs,
    )


__all__ = [
    "MethodRecommendation",
    "SOLVER_CAPABILITY_NOTES",
    "recommend_method",
    "recommend_method_from_spec",
]
