"""``PhysicsGuardrail``: the actual anti-hallucination mechanism.

State this precisely, because overclaiming it is exactly the failure mode
it exists to prevent: **this does not guarantee a result is correct, and
nothing can.** What it does is compute a small set of independent,
numeric, re-checkable signals -- a re-evaluated PDE residual on fresh
points, dimensional/units sanity of the declared parameters, a
conservation-law balance (for the pde_kind families where one has a known
closed form), and (when a reference is supplied) an error metric against
real data -- and refuse to label a result "trustworthy" unless every
signal it can compute passes. A raw LLM asked to solve a physics problem,
by contrast, produces an answer with no residual behind it at all; there
is nothing to check. The differentiation this module gives PINNeAPPle
over "just ask a language model" is not that its answers are smarter,
it's that every claim is required to pass through this gate, or be
reported as *not* having passed it -- explicitly, with the failing check
named, not silently.

Independent of ``pinneapple_llm``'s drafting module (``draft.py``): this
is useful on *any* ``solve_pde``/``pipeline`` result, whether or not an
LLM was involved in producing the ``ProblemSpec`` at all.

Coverage is explicit, not implied
----------------------------------
Two of ``check()``'s signals -- the dimensional-analysis parameter check
and the conservation check -- are only *real, structural* verification
for a documented subset of ``pde_kind`` families. Outside that subset,
``check()`` never silently claims the same rigor: the dimensional check
falls back to a legacy positivity-only heuristic (still run, just weaker
-- see ``_check_parameter_sanity_legacy``), and the conservation check is
simply *absent* from the report (see ``GuardrailReport.trustworthy``'s
docstring for what "absent" means: neither a pass nor a fail). See
``PhysicsGuardrail``'s own docstring below for the exact covered lists.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class GuardrailReport:
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def checked_names(self) -> List[str]:
        """Names of every check that actually ran and produced a
        ``CheckResult``, in the order they ran. This is the ground truth
        for "what was and wasn't evaluated" for a given ``check()`` call
        -- a name absent from this list was not applicable (e.g. no
        reference data supplied, or this ``pde_kind`` has no known
        conservation law implemented here) and therefore counts as
        neither a pass nor a fail; see ``trustworthy``'s docstring."""
        return [c.name for c in self.checks]

    @property
    def skipped(self) -> List[str]:
        """Which of ``PhysicsGuardrail.ALL_CHECK_NAMES`` did NOT appear in
        this report. Read this with one caveat: a few of those names are
        mutually-exclusive alternatives for the *same* slot rather than
        independent checks -- exactly one of ``dimensional_analysis``
        (real units check) or ``parameter_sanity`` (legacy positivity-only
        fallback) always runs, never both, and at most one of
        ``conservation_mass_continuity``/``conservation_heat_conduction``
        can ever apply to a single ``pde_kind``. Seeing one of those
        names here does not by itself mean less rigor was applied --
        check ``checked_names`` for which one actually ran, and
        ``PhysicsGuardrail``'s class docstring for exactly which
        ``pde_kind`` families get which."""
        from pinneapple_llm.guardrail import PhysicsGuardrail  # local import: avoids a module-level forward reference
        return [n for n in PhysicsGuardrail.ALL_CHECK_NAMES if n not in self.checked_names]

    @property
    def trustworthy(self) -> bool:
        """True only if every check that actually ran, passed. A check
        that could not run (e.g. no reference data supplied) is neither a
        pass nor a fail -- it is absent, and does not count either way;
        see ``checked_names``/``skipped`` to see exactly what was and
        was not evaluated, rather than inferring it from a single bool."""
        return all(c.passed for c in self.checks)

    def as_error(self) -> RuntimeError:
        failing = [c for c in self.checks if not c.passed]
        lines = [f"PhysicsGuardrail: {len(failing)} check(s) failed:"]
        for c in failing:
            lines.append(f"  - {c.name}: {c.detail}")
        return RuntimeError("\n".join(lines))

    def summary(self) -> str:
        lines = ["PhysicsGuardrail report:"]
        for c in self.checks:
            mark = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{mark}] {c.name}: {c.detail}")
        lines.append(f"  Overall: {'TRUSTWORTHY' if self.trustworthy else 'NOT TRUSTWORTHY'}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dimensional-analysis registry.
#
# Base-unit ordering fixed throughout this module: (length, time, mass,
# temperature) SI dimension exponents -- e.g. kinematic viscosity/thermal
# diffusivity (m^2/s) is (2, -1, 0, 0).
#
# Parameter names are looked up PER FAMILY, not globally, because the same
# name means different physical quantities in different pde_kinds -- e.g.
# 'nu' is kinematic viscosity ([L^2/T], must be > 0) in the Navier-Stokes
# and diffusion families below, but Poisson's ratio (dimensionless, valid
# range (-1, 0.5), can be legitimately NEGATIVE for an auxetic material)
# in the elasticity family. Confirmed by grepping this repo's actual
# preset parameters (pinneapple_physics/pde_environment/presets/): e.g.
# linear_elasticity_3d's nu=0.3 vs. burgers_1d's nu=0.01, aircraft_wing_
# aerodynamics' nu=1.5e-05 -- genuinely different quantities, same name.
# A global "nu must be positive" rule (the previous heuristic) would
# already incorrectly reject a physically valid nu=-0.2 auxetic material
# spec; this table-driven, per-family version does not make that mistake.
# ---------------------------------------------------------------------------
_DIMENSIONLESS = (0, 0, 0, 0)
_DIFFUSIVITY = (2, -1, 0, 0)     # m^2/s: kinematic viscosity nu, thermal/mass diffusivity alpha/kappa
_CONDUCTIVITY = (1, -3, 1, -1)   # W/(m*K) = kg*m/(s^3*K): thermal conductivity k
_DENSITY = (-3, 0, 1, 0)         # kg/m^3
_SPECIFIC_HEAT = (2, -2, 0, -1)  # J/(kg*K) = m^2/(s^2*K)
_STRESS = (-1, -2, 1, 0)         # Pa = kg/(m*s^2): Young's modulus E, Lame parameters lambda/mu
_VELOCITY = (1, -1, 0, 0)        # m/s

_DIM_NAMES = ("L", "T", "M", "Theta")


def _dim_str(d: Tuple[int, int, int, int]) -> str:
    parts = [f"{n}^{e}" for n, e in zip(_DIM_NAMES, d) if e != 0]
    return "dimensionless" if not parts else "*".join(parts)


class PhysicsGuardrail:
    """Runs a fixed sequence of independent checks against a trained
    model + the ``ProblemSpec`` it was trained on.

    Coverage -- read this before trusting a report's silence on a check
    ----------------------------------------------------------------------
    ``dimensional_analysis`` (real units-consistency check, replacing the
    old positivity-only heuristic) is implemented for exactly these
    ``pde_kind`` families (see ``_DIFFUSION_KINDS``/``_CONDUCTION_KINDS``/
    ``_TRANSIENT_HEAT_KINDS``/``_NS_KINDS``/``_ELASTICITY_KINDS`` below for
    the literal membership):
      - diffusion/advection-diffusion/Burgers (``heat_equation``,
        ``advection_diffusion``, ``burgers``): checks the diffusion/
        viscosity coefficient (``alpha``/``kappa``/``nu``, whichever this
        pde_kind's compiled residual actually reads) is a positive
        real number -- its required dimension is [L^2/T].
      - steady conduction (``heat_equation_steady``,
        ``heat_equation_steady_multilayer``,
        ``heat_equation_steady_anisotropic``): checks the thermal
        conductivity parameter(s) (``k``/``k_eff``/``k_x``/``k_y``/``k_z``)
        are positive -- required dimension [M*L/(T^3*Theta)].
      - transient conduction (``heat_equation_transient``): checks
        ``alpha`` is a positive diffusivity, AND -- when ``k``, ``rho``,
        ``cp`` are *also* declared (as ``car_brake_thermal`` does,
        alongside the ``alpha`` the compiled residual actually reads) --
        verifies the NUMERIC relation ``alpha == k/(rho*cp)`` holds
        (the dimensions force this: [k]/([rho]*[cp]) = [L^2/T] exactly).
      - incompressible Navier-Stokes (``navier_stokes_incompressible``,
        ``incompressible_navier_stokes_2d``): checks ``Re``/``inv_Re`` are
        positive and dimensionless, that ``Re*inv_Re == 1`` when both are
        given, and that ``nu`` (if given) is a positive [L^2/T] viscosity.
      - linear elasticity (``linear_elasticity``,
        ``linear_elasticity_plane_strain``, ``linear_elasticity_plane_
        stress``): checks ``E`` and ``mu`` are positive stresses
        [M/(L*T^2)] (Young's modulus and shear modulus must always be
        positive for a stable isotropic material), that ``lambda``/
        ``lam`` (the first Lame parameter) is at least a finite stress-
        dimensioned number -- NOT required to be positive on its own,
        since an auxetic material (``nu`` < 0) genuinely makes it
        negative -- with the real stability constraint checked instead
        on the implied bulk modulus ``lambda + 2*mu/3 > 0`` when both are
        given; that Poisson's ratio ``nu`` is in the valid isotropic
        range (-1, 0.5) (NOT simply "> 0" -- see the module-level note on
        the ``nu`` naming collision above); and -- when ``E``, ``nu``,
        ``lambda``, ``mu`` are all given -- that the standard isotropic
        relations ``lambda == E*nu/((1+nu)(1-2nu))`` and
        ``mu == E/(2*(1+nu))`` numerically hold.
    Every other ``pde_kind`` (laplace, poisson, wave_equation, darcy,
    stokes, hyperelasticity_neo_hookean, ... — everything not listed
    above) falls back to the ORIGINAL positivity-only heuristic
    (``_check_parameter_sanity_legacy``) -- still run, so nothing
    regresses, but NOT the same rigor; the report's ``dimensional_
    analysis`` vs. ``parameter_sanity`` check name tells you which one
    actually ran for a given spec.

    ``conservation_*`` (spatial-integral balance over the domain
    boundary) is implemented for exactly two families, and is ABSENT
    (not appended to the report at all -- never faked as a pass) for
    every other ``pde_kind``:
      - incompressible continuity (``navier_stokes_incompressible``,
        ``incompressible_navier_stokes_2d``): the net outward volume flux
        of the model's velocity field through ``self.spec.domain_bounds``'
        box boundary must be ~0 for a divergence-free field (divergence
        theorem). Estimated via Monte Carlo sampling on each of the box's
        2*d faces (``conservation_n_points_per_face`` samples/face); for a
        time-dependent spec, evaluated at a single fixed time slice (the
        midpoint of the 't' domain bound) since continuity holds at every
        instant independently, not just in aggregate over time.
      - steady heat conduction with NO source term (``heat_equation_
        steady``, ``heat_equation_steady_multilayer``,
        ``heat_equation_steady_anisotropic``): the net ``-k*grad(T)``
        (Fourier's law; ``-k_i*dT/dx_i`` per-axis for the anisotropic
        kind) flux through the boundary must be ~0. This explicitly
        ASSUMES the steady solution has no volumetric source term (q=0)
        -- if the spec's problem was solved with a nonzero ``ctx
        ['source_fn']``, this check has no way to know that (that
        function lives in a training-time ``ctx`` dict, not in
        ``ProblemSpec`` itself) and will report a nonzero imbalance even
        for an otherwise-correct solution. This is the explicit, honest
        scope limit for this check, not a bug.

    Both conservation checks report a normalised imbalance ratio
    (``|net flux| / sum(|per-face flux contribution|)``), not a raw flux
    value, specifically so ONE threshold works across wildly different
    domain sizes/field magnitudes without being re-tuned per preset. The
    default threshold (0.15) and default sample count (2048 points/face)
    were picked from an empirical noise-floor characterization done
    before choosing them (see ``tests/test_physics_guardrail.py``'s
    ``test_conservation_navier_stokes_noise_floor_is_below_threshold``-
    style tests and this class's own change history): on a NUMERICALLY
    EXACT divergence-free 3D velocity field (built from ``curl`` of a
    smooth vector potential, so its divergence is exactly zero by
    construction, not merely small), the observed imbalance ratio at
    2048 points/face had mean ~0.007-0.010 and a max of ~0.027-0.05 over
    50-trial resampling on domains matching this repo's own
    ``channel_flow_3d``/``lid_driven_cavity_3d`` presets. The steady-
    conduction analogue is far quieter still (~1e-4 at the same sample
    count, on a harmonic ``T``). A deliberately non-conservative field
    (nonzero-divergence velocity; non-harmonic ``T`` implying a hidden
    source) gave a ratio of exactly 1.0 in both cases (no cancellation at
    all across faces). 0.15 sits roughly 3-20x above the observed
    noise ceiling and ~6.7x below the "clearly broken" ratio of 1.0 --
    comfortable margin on both sides for a Monte-Carlo estimator, not a
    guessed number.

    Parameters
    ----------
    spec : the ``ProblemSpec`` (from ``pinneapple_physics.pde_environment``)
        the model was trained against.
    residual_threshold : max acceptable mean-squared PDE residual
        (re-evaluated on FRESH collocation points, never the ones used
        during training -- catches memorisation as well as an under-
        converged fit).
    n_check_points : how many fresh collocation points to re-evaluate the
        residual on.
    conservation_n_points_per_face : Monte-Carlo sample count per
        boundary face for the conservation check (see class docstring
        above for the empirical noise-floor justification behind the
        default).
    conservation_flux_ratio_threshold : max acceptable normalised
        boundary-flux imbalance ratio (see class docstring above).
    """

    # Every CheckResult.name this class can ever produce -- used by
    # GuardrailReport.skipped. Note some of these are mutually-exclusive
    # alternatives for the same slot (see GuardrailReport.skipped's own
    # docstring), not independent checks that all run every time.
    ALL_CHECK_NAMES = (
        "parameter_sanity",
        "dimensional_analysis",
        "pde_residual",
        "reference_data_match",
        "conservation_mass_continuity",
        "conservation_heat_conduction",
    )

    # pde_kind family membership for the dimensional-analysis check.
    _DIFFUSION_KINDS = {"heat_equation", "advection_diffusion", "burgers"}
    _CONDUCTION_KINDS = {
        "heat_equation_steady",
        "heat_equation_steady_multilayer",
        "heat_equation_steady_anisotropic",
    }
    _TRANSIENT_HEAT_KINDS = {"heat_equation_transient"}
    _NS_KINDS = {"navier_stokes_incompressible", "incompressible_navier_stokes_2d"}
    _ELASTICITY_KINDS = {
        "linear_elasticity",
        "linear_elasticity_plane_strain",
        "linear_elasticity_plane_stress",
    }

    # pde_kind family membership for the conservation check.
    _NS_CONSERVATION_KINDS = _NS_KINDS
    _STEADY_HEAT_CONSERVATION_KINDS = _CONDUCTION_KINDS

    def __init__(
        self,
        spec: Any,
        *,
        residual_threshold: float = 1e-2,
        n_check_points: int = 4096,
        conservation_n_points_per_face: int = 2048,
        conservation_flux_ratio_threshold: float = 0.15,
    ):
        self.spec = spec
        self.residual_threshold = residual_threshold
        self.n_check_points = n_check_points
        self.conservation_n_points_per_face = conservation_n_points_per_face
        self.conservation_flux_ratio_threshold = conservation_flux_ratio_threshold

    # ------------------------------------------------------------------
    # Dimensional analysis / parameter sanity
    # ------------------------------------------------------------------
    def _check_parameter_sanity(self) -> CheckResult:
        """Dispatches to a real dimensional-analysis check for the
        covered ``pde_kind`` families documented in this class's
        docstring, or falls back to the legacy positivity-only heuristic
        for everything else. Always produces exactly one CheckResult
        (never absent) -- the two possible ``name`` values
        (``dimensional_analysis`` vs. ``parameter_sanity``) are what
        distinguishes which rigor level actually ran; see
        ``GuardrailReport.skipped``'s docstring."""
        real_check = self._check_dimensional_analysis()
        if real_check is not None:
            return real_check
        params = dict(getattr(self.spec.pde, "params", {}) or {})
        return self._check_parameter_sanity_legacy(params)

    def _check_parameter_sanity_legacy(self, params: Dict[str, Any]) -> CheckResult:
        """The ORIGINAL heuristic this module shipped with: a fixed set of
        parameter names must be positive numbers if present. Kept as the
        fallback for every ``pde_kind`` not yet covered by
        ``_check_dimensional_analysis`` -- weaker than a real units check
        (e.g. it would incorrectly flag a valid negative Poisson's ratio
        as an error, which is exactly why elasticity kinds are routed to
        the real check instead of this one), but still catches an
        obviously unphysical value (negative viscosity/diffusivity/
        Reynolds number) for pde_kinds this module hasn't derived a real
        equation-structure check for yet."""
        problems = []
        # Parameters that are physically required to be positive for any
        # of this module's supported pde_kinds (diffusivity/viscosity/
        # Reynolds-number-like quantities). Deliberately conservative: an
        # unrecognised parameter name is not flagged (this checks known
        # physical quantities, not the shape of an arbitrary dict).
        positive_only = {"nu", "Re", "alpha", "D", "k", "inv_Re", "diffusivity"}
        for name in positive_only & params.keys():
            val = params[name]
            if isinstance(val, (int, float)) and val <= 0:
                problems.append(f"{name}={val} must be > 0")
        passed = not problems
        detail = (
            "all recognised physical parameters are positive (legacy positivity-only "
            "heuristic -- this pde_kind is not yet in PhysicsGuardrail's real "
            "dimensional-analysis table, see its class docstring for the covered list)"
            if passed else "; ".join(problems)
        )
        return CheckResult(name="parameter_sanity", passed=passed, detail=detail)

    def _check_dimensional_analysis(self) -> Optional[CheckResult]:
        """Real units/dimensional-consistency check for the pde_kind
        families listed in this class's docstring. Returns ``None`` (not
        a passing or failing CheckResult) if ``self.spec.pde.kind`` isn't
        one of them -- the caller (``_check_parameter_sanity``) falls
        back to the legacy heuristic in that case."""
        kind = self.spec.pde.kind
        p = dict(getattr(self.spec.pde, "params", {}) or {})
        problems: List[str] = []
        checked: List[str] = []

        if kind in self._DIFFUSION_KINDS:
            # heat_equation reads params["alpha"] (falling back to
            # "kappa"); advection_diffusion reads params["kappa"];
            # burgers reads params["nu"] -- see compile.py's own
            # p.get(...) fallback chains for each. All three play the
            # SAME structural role (the coefficient multiplying the
            # diffusive/2nd-derivative term), so all three require the
            # same dimension: [L^2/T].
            for name in ("alpha", "kappa", "nu"):
                if name in p:
                    checked.append(name)
                    val = p[name]
                    if not isinstance(val, (int, float)) or val <= 0:
                        problems.append(
                            f"{name}={val}: a diffusion/viscosity coefficient "
                            f"([{_dim_str(_DIFFUSIVITY)}]) must be a positive real number"
                        )
            for name in ("u0", "v0", "w0"):
                if name in p:
                    checked.append(name)
                    val = p[name]
                    if not isinstance(val, (int, float)) or not math.isfinite(val):
                        problems.append(
                            f"{name}={val}: advection velocity ([{_dim_str(_VELOCITY)}]) "
                            "must be a finite real number (may be negative -- flow direction)"
                        )

        elif kind in self._CONDUCTION_KINDS:
            # k / k_eff (isotropic steady conduction, possibly a
            # series-resistance effective conductivity) / k_x,k_y,k_z
            # (anisotropic) -- all play the role of Fourier's-law thermal
            # conductivity, dimension [M*L/(T^3*Theta)].
            for name in ("k", "k_eff", "k_x", "k_y", "k_z"):
                if name in p:
                    checked.append(name)
                    val = p[name]
                    if not isinstance(val, (int, float)) or val <= 0:
                        problems.append(
                            f"{name}={val}: thermal conductivity "
                            f"([{_dim_str(_CONDUCTIVITY)}]) must be > 0"
                        )

        elif kind in self._TRANSIENT_HEAT_KINDS:
            alpha = p.get("alpha")
            if alpha is not None:
                checked.append("alpha")
                if not isinstance(alpha, (int, float)) or alpha <= 0:
                    problems.append(
                        f"alpha={alpha}: thermal diffusivity ([{_dim_str(_DIFFUSIVITY)}]) must be > 0"
                    )
            # Structural cross-check: heat_equation_transient's own
            # governing equation is rho*cp*dT/dt = k*laplacian(T) + q;
            # dividing through by rho*cp gives dT/dt = alpha*laplacian(T)
            # + ..., i.e. alpha IS DEFINED as k/(rho*cp) -- and the
            # dimensions force this exactly: [k]/([rho]*[cp]) =
            # ([M*L/(T^3*Theta)]) / (([M/L^3])*([L^2/(T^2*Theta)])) =
            # [L^2/T], matching alpha's own required dimension. When a
            # preset supplies k, rho, cp ALONGSIDE the alpha the compiled
            # residual actually reads (as car_brake_thermal's does),
            # verify the NUMERIC relation holds -- a spec whose declared
            # constitutive parameters (k, rho, cp) don't agree with its
            # own declared alpha has an internally inconsistent physical
            # description, even though the compiled residual (which only
            # reads alpha) would never itself detect it.
            k, rho, cp = p.get("k"), p.get("rho"), p.get("cp")
            if alpha is not None and None not in (k, rho, cp):
                checked.append("alpha==k/(rho*cp)")
                try:
                    implied_alpha = float(k) / (float(rho) * float(cp))
                    rel_err = abs(implied_alpha - float(alpha)) / (abs(float(alpha)) + 1e-30)
                    if rel_err > 1e-2:
                        problems.append(
                            f"alpha={alpha} but k/(rho*cp)={implied_alpha:.6g} "
                            f"(relative mismatch {rel_err:.2%}) -- alpha, k, rho and cp are "
                            "supposed to be the same physical diffusivity expressed two "
                            "ways; the compiled residual only reads 'alpha' directly, so "
                            "this mismatch would silently have no effect on training, but "
                            "it means the spec's own declared parameters disagree with "
                            "each other"
                        )
                except (TypeError, ValueError, ZeroDivisionError) as exc:
                    problems.append(f"k={k}, rho={rho}, cp={cp}: could not evaluate k/(rho*cp) ({exc})")

        elif kind in self._NS_KINDS:
            Re, inv_Re, nu = p.get("Re"), p.get("inv_Re"), p.get("nu")
            if Re is not None:
                checked.append("Re")
                if not isinstance(Re, (int, float)) or Re <= 0:
                    problems.append(f"Re={Re}: Reynolds number (dimensionless) must be > 0")
            if inv_Re is not None:
                checked.append("inv_Re")
                if not isinstance(inv_Re, (int, float)) or inv_Re <= 0:
                    problems.append(f"inv_Re={inv_Re}: 1/Re (dimensionless) must be > 0")
            if Re is not None and inv_Re is not None:
                checked.append("Re*inv_Re==1")
                prod = float(Re) * float(inv_Re)
                if abs(prod - 1.0) > 1e-2:
                    problems.append(
                        f"Re={Re} and inv_Re={inv_Re} are not reciprocal "
                        f"(Re*inv_Re={prod:.6g}, expected 1) -- the compiled residual uses "
                        "inv_Re directly, so a Re/inv_Re pair that doesn't actually match "
                        "means whichever one the residual reads is not the Reynolds number "
                        "the rest of the spec claims"
                    )
            if nu is not None:
                checked.append("nu")
                if not isinstance(nu, (int, float)) or nu <= 0:
                    problems.append(
                        f"nu={nu}: kinematic viscosity ([{_dim_str(_DIFFUSIVITY)}]) must be > 0"
                    )

        elif kind in self._ELASTICITY_KINDS:
            E = p.get("E")
            nu = p.get("nu")
            lam = p.get("lambda", p.get("lam"))
            mu = p.get("mu")
            if p.get("lambda") is None and p.get("lam") is not None:
                # compile.py's linear_elasticity*/thermoelasticity_2d
                # residual reads params["lambda"] ONLY (falling back to a
                # hardcoded default of 1.0 if that key is absent) -- it
                # never looks at "lam". A spec that declares "lam" but not
                # "lambda" (found this way while validating this check
                # against this repo's own presets) has its real material
                # Lame parameter silently ignored in favour of that
                # default, a genuine spec/compiled-physics mismatch, not
                # just a spelling variant.
                checked.append("lam-vs-lambda-key")
                problems.append(
                    f"params has 'lam'={p['lam']} but no 'lambda' key -- the compiled "
                    "residual reads params['lambda'] specifically (default 1.0 if absent) "
                    "and never reads 'lam', so this preset's declared Lame parameter has "
                    "no effect on the physics actually being solved"
                )
            if E is not None:
                checked.append("E")
                if not isinstance(E, (int, float)) or E <= 0:
                    problems.append(f"E={E}: Young's modulus ([{_dim_str(_STRESS)}]) must be > 0")
            if nu is not None:
                checked.append("nu")
                # Poisson's ratio is DIMENSIONLESS here, NOT kinematic
                # viscosity -- see the module-level note on this naming
                # collision. Valid isotropic-elastic range is (-1, 0.5),
                # not simply "> 0": an auxetic material genuinely has
                # nu < 0, and the old global positivity heuristic would
                # have wrongly rejected it.
                if not isinstance(nu, (int, float)) or not (-1.0 < nu < 0.5):
                    problems.append(
                        f"nu={nu}: Poisson's ratio (dimensionless) must be in (-1, 0.5) "
                        "for an isotropic elastic material"
                    )
            if mu is not None:
                checked.append("mu")
                # Shear modulus: must be > 0 for a stable isotropic
                # material (unlike lambda below, there is no valid
                # negative regime for mu).
                if not isinstance(mu, (int, float)) or mu <= 0:
                    problems.append(f"mu={mu}: shear modulus ([{_dim_str(_STRESS)}]) must be > 0")
            for name, val in (("lambda", p.get("lambda")), ("lam", p.get("lam"))):
                if val is not None:
                    checked.append(name)
                    # Lambda (the first Lame parameter) is NOT required to
                    # be positive on its own -- for an auxetic material
                    # (nu<0) the standard isotropic relation
                    # lambda=E*nu/((1+nu)(1-2nu)) makes lambda negative
                    # while the material is still perfectly stable/valid.
                    # The real stability constraint is on the BULK
                    # modulus K=lambda+2*mu/3 (must be > 0), checked
                    # below once mu is known, not on lambda in isolation.
                    if not isinstance(val, (int, float)) or not math.isfinite(val):
                        problems.append(
                            f"{name}={val}: Lame parameter lambda ([{_dim_str(_STRESS)}]) must be a finite real number"
                        )
            if lam is not None and mu is not None:
                checked.append("bulk_modulus(lambda,mu)")
                if isinstance(lam, (int, float)) and isinstance(mu, (int, float)):
                    bulk_modulus = float(lam) + (2.0 / 3.0) * float(mu)
                    if bulk_modulus <= 0:
                        problems.append(
                            f"lambda={lam}, mu={mu}: implied bulk modulus lambda+2*mu/3={bulk_modulus:.6g} "
                            "must be > 0 for a stable isotropic material"
                        )
            if E is not None and nu is not None and lam is not None:
                checked.append("lambda==E*nu/((1+nu)(1-2nu))")
                try:
                    implied_lam = float(E) * float(nu) / ((1.0 + float(nu)) * (1.0 - 2.0 * float(nu)))
                    rel_err = abs(implied_lam - float(lam)) / (abs(float(lam)) + 1e-30)
                    if rel_err > 1e-2:
                        problems.append(
                            f"lambda/lam={lam} but the standard isotropic relation "
                            f"E*nu/((1+nu)(1-2nu))={implied_lam:.6g} (mismatch {rel_err:.2%})"
                        )
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
            if E is not None and nu is not None and mu is not None:
                checked.append("mu==E/(2*(1+nu))")
                try:
                    implied_mu = float(E) / (2.0 * (1.0 + float(nu)))
                    rel_err = abs(implied_mu - float(mu)) / (abs(float(mu)) + 1e-30)
                    if rel_err > 1e-2:
                        problems.append(
                            f"mu={mu} but the standard isotropic relation "
                            f"E/(2*(1+nu))={implied_mu:.6g} (mismatch {rel_err:.2%})"
                        )
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
        else:
            return None

        passed = not problems
        if checked:
            detail = (
                f"dimensional/units consistency verified for: {', '.join(checked)}"
                if passed else "; ".join(problems)
            )
        else:
            detail = (
                "pde_kind is in the dimensional-analysis coverage table, but none of its "
                "recognised parameter names were present in params (nothing to check)"
            )
        return CheckResult(name="dimensional_analysis", passed=passed, detail=detail)

    # ------------------------------------------------------------------
    # PDE residual
    # ------------------------------------------------------------------
    def _check_residual(self, model) -> CheckResult:
        import numpy as np
        import torch
        from pinneapple_physics.pinn_solver.compiler.compile import compile_problem

        loss_fn = compile_problem(self.spec)
        coords = list(self.spec.coords)
        bounds = self.spec.domain_bounds
        cols = [
            torch.as_tensor(np.random.uniform(*bounds[c], size=self.n_check_points), dtype=torch.float32).reshape(-1, 1)
            for c in coords
        ]
        x = torch.cat(cols, dim=1).requires_grad_(True)
        n_fields = len(self.spec.fields)
        batch = {
            "x_col": x, "ctx": {},
            "x_bc": torch.zeros((0, len(coords))), "y_bc": torch.zeros((0, n_fields)),
            "x_ic": torch.zeros((0, len(coords))), "y_ic": torch.zeros((0, n_fields)),
            "x_data": torch.zeros((0, len(coords))), "y_data": torch.zeros((0, n_fields)),
        }
        y_hat = model(x)
        if hasattr(y_hat, "y"):
            y_hat = y_hat.y
        out = loss_fn(model, y_hat, batch)
        residual = float(out["pde"].item()) if isinstance(out, dict) and "pde" in out else float(out["total"].item())
        passed = residual <= self.residual_threshold
        return CheckResult(
            name="pde_residual", passed=passed,
            detail=f"mean-squared PDE residual on {self.n_check_points} fresh points = {residual:.4g} "
                   f"({'<=' if passed else '>'} threshold {self.residual_threshold:.4g})",
            value=residual, threshold=self.residual_threshold,
        )

    # ------------------------------------------------------------------
    # Reference-data match
    # ------------------------------------------------------------------
    def _check_reference(self, model, reference_x, reference_y, rmse_threshold: float) -> CheckResult:
        import numpy as np
        import torch

        model.eval()
        with torch.no_grad():
            x_t = torch.as_tensor(np.asarray(reference_x, dtype="float32"))
            y_hat = model(x_t)
            if hasattr(y_hat, "y"):
                y_hat = y_hat.y
            y_hat = y_hat.numpy()
        y_true = np.asarray(reference_y, dtype="float32")
        rmse = float(np.sqrt(np.mean((y_hat - y_true) ** 2)))
        passed = rmse <= rmse_threshold
        return CheckResult(
            name="reference_data_match", passed=passed,
            detail=f"RMSE against supplied reference data = {rmse:.4g} "
                   f"({'<=' if passed else '>'} threshold {rmse_threshold:.4g})",
            value=rmse, threshold=rmse_threshold,
        )

    # ------------------------------------------------------------------
    # Conservation checks
    # ------------------------------------------------------------------
    def _boundary_flux_imbalance(
        self,
        flux_component_fn,
        spatial_coord_names: Sequence[str],
        domain_bounds: Dict[str, Tuple[float, float]],
        n_points_per_face: int,
        rng,
    ) -> Tuple[float, float, List[Tuple[str, str, float]]]:
        """Monte-Carlo estimate of the net outward flux of some vector
        field through the boundary of an axis-aligned box domain
        (``domain_bounds``), one face per (axis, min/max) pair.

        ``flux_component_fn(axis_index, fixed_value, n_points) -> np.ndarray``
        must return an (n_points,) array of the physical flux component
        along the +``axis_index`` direction, evaluated at ``n_points``
        points sampled uniformly over that face (with coordinate
        ``spatial_coord_names[axis_index]`` held fixed at ``fixed_value``
        and every other spatial coordinate drawn uniformly at random from
        its own domain bound -- the caller does that sampling internally,
        this function only tells it which axis/value to fix).

        Returns ``(net_flux, total_abs_contribution, per_face_detail)``;
        callers report ``|net_flux| / (total_abs_contribution + eps)`` as
        a normalised, dimensionless, self-scaling imbalance ratio -- see
        ``PhysicsGuardrail``'s class docstring for why a normalised ratio
        is used instead of a raw flux value.
        """
        net_flux = 0.0
        abs_sum = 0.0
        per_face: List[Tuple[str, str, float]] = []
        for axis_idx, axis_name in enumerate(spatial_coord_names):
            lo, hi = domain_bounds[axis_name]
            area = 1.0
            for other in spatial_coord_names:
                if other == axis_name:
                    continue
                olo, ohi = domain_bounds[other]
                area *= (ohi - olo)
            for sign, bound_val, face_label in ((-1.0, lo, "min"), (1.0, hi, "max")):
                comp = flux_component_fn(axis_idx, bound_val, n_points_per_face)
                mean_comp = float(comp.mean())
                contrib = sign * mean_comp * area
                net_flux += contrib
                abs_sum += abs(contrib)
                per_face.append((axis_name, face_label, contrib))
        return net_flux, abs_sum, per_face

    def _check_conservation(self, model) -> Optional[CheckResult]:
        """Dispatches to the applicable conservation check, or returns
        ``None`` (absent from the report, not a pass) if ``self.spec.pde
        .kind`` has no known closed-form conservation law implemented
        here -- see this class's docstring for the exact covered list."""
        kind = self.spec.pde.kind
        if kind in self._NS_CONSERVATION_KINDS:
            return self._check_conservation_mass_continuity(model)
        if kind in self._STEADY_HEAT_CONSERVATION_KINDS:
            return self._check_conservation_heat_flux(model)
        return None

    def _check_conservation_mass_continuity(self, model) -> CheckResult:
        """Incompressible continuity: net outward volume flux of the
        model's velocity field through the domain boundary must be ~0
        for a divergence-free field (divergence theorem). See class
        docstring for the empirical noise-floor justification of the
        default threshold/sample count."""
        import numpy as np
        import torch

        coords = list(self.spec.coords)
        field_names = list(self.spec.fields)
        spatial_coord_names = [c for c in coords if c != "t"]
        spatial_dim = len(spatial_coord_names)
        vel_names = ["u", "v", "w"][:spatial_dim]
        vel_cols = [field_names.index(n) for n in vel_names]
        bounds = self.spec.domain_bounds
        has_t = "t" in coords
        t_mid = 0.5 * (bounds["t"][0] + bounds["t"][1]) if has_t else None

        rng = np.random.default_rng(0)  # fixed seed: a report shouldn't flicker pass/fail run-to-run on MC noise alone
        model.eval()

        def flux_component(axis_idx, bound_val, n):
            pts = np.zeros((n, len(coords)), dtype=np.float64)
            for ci, cname in enumerate(coords):
                if cname == "t":
                    pts[:, ci] = t_mid
                elif cname == spatial_coord_names[axis_idx]:
                    pts[:, ci] = bound_val
                else:
                    lo, hi = bounds[cname]
                    pts[:, ci] = rng.uniform(lo, hi, size=n)
            x_t = torch.as_tensor(pts, dtype=torch.float32)
            with torch.no_grad():
                y = model(x_t)
                if hasattr(y, "y"):
                    y = y.y
                y = y.numpy()
            return y[:, vel_cols[axis_idx]]

        net_flux, abs_sum, _ = self._boundary_flux_imbalance(
            flux_component, spatial_coord_names, bounds, self.conservation_n_points_per_face, rng,
        )
        ratio = abs(net_flux) / (abs_sum + 1e-12)
        passed = ratio <= self.conservation_flux_ratio_threshold
        detail = (
            f"|net outward volume flux| / (sum of |per-face flux|) = {ratio:.4g} "
            f"(net_flux={net_flux:.4g}, {self.conservation_n_points_per_face} MC samples/face"
            f"{', at fixed t=' + f'{t_mid:.4g}' if has_t else ''}) "
            f"({'<=' if passed else '>'} threshold {self.conservation_flux_ratio_threshold:.4g})"
        )
        return CheckResult(
            name="conservation_mass_continuity", passed=passed, detail=detail,
            value=ratio, threshold=self.conservation_flux_ratio_threshold,
        )

    def _check_conservation_heat_flux(self, model) -> CheckResult:
        """Steady conduction, no source: net Fourier-law heat flux
        (``-k*grad(T)``, or ``-k_i*dT/dx_i`` per-axis for the anisotropic
        kind) through the domain boundary must be ~0. Assumes q=0 -- see
        class docstring for why this is an explicit, honest scope limit,
        not a bug."""
        import numpy as np
        import torch

        coords = list(self.spec.coords)
        field_names = list(self.spec.fields)
        if len(field_names) != 1:
            return CheckResult(
                name="conservation_heat_conduction", passed=False,
                detail=f"expected exactly one scalar field (T), got {field_names}; cannot evaluate heat flux",
            )
        T_col = 0
        spatial_coord_names = list(coords)  # these pde_kinds have no 't'
        bounds = self.spec.domain_bounds
        kind = self.spec.pde.kind
        p = dict(getattr(self.spec.pde, "params", {}) or {})

        def k_for_axis(axis_name: str) -> float:
            if kind == "heat_equation_steady_anisotropic":
                return float(p.get(f"k_{axis_name}", p.get("k", 1.0)))
            return float(p.get("k_eff", p.get("k", 1.0)))

        rng = np.random.default_rng(0)
        model.eval()

        def flux_component(axis_idx, bound_val, n):
            axis_name = spatial_coord_names[axis_idx]
            axis_col = coords.index(axis_name)
            pts = np.zeros((n, len(coords)), dtype=np.float64)
            for ci, cname in enumerate(coords):
                if cname == axis_name:
                    pts[:, ci] = bound_val
                else:
                    lo, hi = bounds[cname]
                    pts[:, ci] = rng.uniform(lo, hi, size=n)
            x_t = torch.as_tensor(pts, dtype=torch.float32).requires_grad_(True)
            y = model(x_t)
            if hasattr(y, "y"):
                y = y.y
            T = y[:, T_col:T_col + 1]
            gT = torch.autograd.grad(T, x_t, grad_outputs=torch.ones_like(T), create_graph=False)[0]
            dT_dxi = gT[:, axis_col:axis_col + 1]
            k_i = k_for_axis(axis_name)
            flux = (-k_i * dT_dxi).detach().numpy().reshape(-1)
            return flux

        net_flux, abs_sum, _ = self._boundary_flux_imbalance(
            flux_component, spatial_coord_names, bounds, self.conservation_n_points_per_face, rng,
        )
        ratio = abs(net_flux) / (abs_sum + 1e-12)
        passed = ratio <= self.conservation_flux_ratio_threshold
        detail = (
            f"|net boundary heat flux| / (sum of |per-face flux|) = {ratio:.4g} "
            f"(net_flux={net_flux:.4g}, {self.conservation_n_points_per_face} MC samples/face; "
            "assumes q=0, see class docstring) "
            f"({'<=' if passed else '>'} threshold {self.conservation_flux_ratio_threshold:.4g})"
        )
        return CheckResult(
            name="conservation_heat_conduction", passed=passed, detail=detail,
            value=ratio, threshold=self.conservation_flux_ratio_threshold,
        )

    # ------------------------------------------------------------------
    def check(
        self,
        model,
        *,
        reference_x=None,
        reference_y=None,
        reference_rmse_threshold: Optional[float] = None,
    ) -> GuardrailReport:
        """Run every applicable check and return a :class:`GuardrailReport`.

        Parameters
        ----------
        model : the trained model (an ``nn.Module`` mapping
            ``(N, len(coords)) -> (N, len(fields))``).
        reference_x, reference_y : optional real reference data
            (e.g. a DNS/experimental dataset) to check the model against;
            skipped entirely if not given -- see ``GuardrailReport
            .trustworthy``'s docstring for what a skipped check means.
        reference_rmse_threshold : required if ``reference_x``/
            ``reference_y`` are given (no default -- what counts as an
            acceptable RMSE is problem-specific and must not be silently
            assumed).

        The dimensional-analysis and conservation checks are only real,
        structural verification for the ``pde_kind`` families documented
        in this class's own docstring -- outside those, the former falls
        back to a legacy positivity heuristic (still runs, weaker) and
        the latter is simply absent from ``report.checks`` (see
        ``GuardrailReport.checked_names``/``skipped``).
        """
        checks = [self._check_parameter_sanity(), self._check_residual(model)]
        conservation = self._check_conservation(model)
        if conservation is not None:
            checks.append(conservation)
        if reference_x is not None or reference_y is not None:
            if reference_x is None or reference_y is None:
                raise ValueError("reference_x and reference_y must both be given, or neither")
            if reference_rmse_threshold is None:
                raise ValueError(
                    "reference_rmse_threshold is required when reference_x/reference_y are given "
                    "-- there is no problem-agnostic default for 'acceptable error'"
                )
            checks.append(self._check_reference(model, reference_x, reference_y, reference_rmse_threshold))
        return GuardrailReport(checks=checks)
