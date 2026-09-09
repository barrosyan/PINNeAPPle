"""A small, curated, NAME-based catalog of real benchmark datasets.

This is the thing ``ROADMAP_PHYSICS_AI_HUB.md`` section P3.2 and
``pinneapple_llm.guardrail.PhysicsGuardrail._load_reference_from_upd_zarr``'s
own docstring both flagged as missing: until this module, ``pinneapple_pdb``
had exactly one name->dict lookup (``templates.schema_templates()``, which
returns physical-*schema* metadata -- governing equations, units policy --
never x/y data arrays) and no way to resolve a friendly string like
``"lane_emden_n1.5"`` to an actual reference dataset. ``PhysicalDatasetBuilder``
itself still only ever *fetches from external Earth-data hubs and writes to
disk* -- it has no reader of its own and no registry of pre-known datasets.

This module is deliberately NOT that: it does not touch
``PhysicalDatasetBuilder``, NASA CMR, or earthaccess at all. It is a small,
self-contained, in-repo catalog of reference (x, y) arrays for a couple of
benchmarks this codebase can independently verify without any external
service or file -- following the same
``@register_x`` / ``get_x(name)`` / ``list_x()`` convention as
``pinneapple_physics.pde_environment.presets.registry`` (see that module's
docstring for the pattern this mirrors).

Honesty about scope, matching this repo's established ethic: this is FIVE
entries, not a comprehensive benchmark suite. Two are the astrophysically
standard Lane-Emden polytropic indices (n=1.5: non-relativistic degenerate
star / white-dwarf core; n=3: Eddington standard model / relativistic
degenerate limit) that ``pinneapple_physics.pde_environment.presets
.astrophysics.lane_emden_polytrope`` already models but has no closed-form
solution for (see that preset's own docstring). Each entry is built by
independently integrating the Lane-Emden ODE with `scipy.integrate.solve_ivp`
(the SAME method -- not the same code -- as
``tests/test_lane_emden_numerical_validation.py``, which is not imported
here; this module reimplements the integration so it has no import-time
dependency on the test suite) and is cross-checked against the published
surface radius xi_1 (Hansen, Kawaler & Trimble, "Stellar Interiors", 2nd ed.,
Table 4.1) every time it is built -- if the integrated xi_1 ever drifted from
the published value beyond the same <0.1% tolerance the test suite uses,
``get_benchmark``/``benchmark_catalog`` would raise rather than silently
hand back an unverified profile.

The third, ``concorde_high_aoa``, is deliberately NOT a wind-tunnel or
flight-test aerodynamic dataset for the real Concorde -- no such number is
available to this codebase with a citable public source, and fabricating
one would violate the same honesty ethic as the paragraph above. Instead it
evaluates a real, cite-able CLOSED-FORM theory (the Polhamus 1966
leading-edge-suction vortex-lift analogy for slender delta wings, in its
low-aspect-ratio asymptotic limit) at Concorde's approximate wing aspect
ratio, and cross-checks its own small-angle slope against the closed-form
constant that theory predicts. See that entry's docstring for the exact
scope and limitations.

A real OpenFOAM LES channel-flow dataset (Re_tau=180, Moser-Kim-Mansour
setup) also exists, in the sibling ``splash-pinneapple`` project on this
machine -- but it was deliberately NOT added as a catalog entry here: it
lives at an absolute filesystem path outside this repository (not portable
to another checkout, CI, or contributor's machine), is a 200+MB zipped
OpenFOAM case rather than a small in-repo array, and requires bespoke
OpenFOAM-binary-format parsing code (``openfoam_binary.py``,
``splash_mesh.py`` in that other project) that does not exist anywhere in
PINNeAPPle. Wiring it in properly is real future work, not something to
fake with a hardcoded absolute path that would only work on one machine.

The fourth entry, ``blasius_flat_plate``, is the classical laminar
flat-plate boundary-layer similarity solution: the Blasius ODE
``f'''(eta) + 0.5*f(eta)*f''(eta) = 0``, ``f(0)=f'(0)=0``, ``f'(inf)=1``,
independently integrated here via a shooting method
(``scipy.integrate.solve_ivp`` + ``scipy.optimize.brentq``, not imported
from anywhere else in this codebase) and cross-checked against the
well-known tabulated wall-shear parameter ``f''(0) = 0.332057`` (Howarth,
1938 high-precision integration; tabulated in e.g. Schlichting & Gersten,
"Boundary-Layer Theory", 8th/9th ed.) -- see that entry's docstring for an
explicit note on why this is NOT the same number as the sometimes-quoted
``0.4696``, which belongs to a different similarity-variable normalization
(``eta`` scaled by an extra factor of ``sqrt(2)``) and does not apply to
the ODE form used here.

The fifth entry, ``planewall_transient_conduction_bi1``, is the classical
1D transient-conduction-in-a-plane-wall problem with convective boundary
conditions on both faces (Biot number Bi=1.0): the one-term-series
centerline (midplane) dimensionless-temperature history
``theta*_0(Fo) = C1*exp(-zeta_1^2*Fo)``, valid for Fourier number
``Fo > 0.2`` (Incropera & DeWitt, "Fundamentals of Heat and Mass
Transfer"). The eigenvalue ``zeta_1`` (root of ``zeta*tan(zeta) = Bi``)
and coefficient ``C1`` are independently solved for here
(``scipy.optimize.brentq``) and cross-checked against Incropera & DeWitt's
own published one-term-approximation table (Table 5.1, Bi=1.0:
``zeta_1=0.8603``, ``C1=1.1191``).

Entries six through eight integrate PDEBench (Takamoto, M. et al.,
"PDEBench: An Extensive Benchmark for Scientific Machine Learning", NeurIPS
2022 Datasets and Benchmarks Track, arXiv:2210.07182) -- a real, well-known
open-source PDE-ML benchmark suite -- into this catalog. PDEBench's actual
datasets are large (multi-GB) HDF5 files hosted on DaRUS (the University of
Stuttgart's Dataverse-based data repository), not something this repo
fetches or bundles. Instead, each entry below implements PDEBench's own
*governing equation*, as written in the paper's Appendix D, for a
simplified-but-legitimate parameter choice, independently solves it with a
method not borrowed from PDEBench's own code, and cross-checks the result
against a real, citable closed-form solution from the classical PDE
literature -- exactly the same "never fabricate, always re-verify" pattern
used by every other entry in this module. Where the paper's own example
figures name specific parameter values (e.g. "1D Advection (beta=0.4)"),
those same values are reused here so the entries visibly correspond to
PDEBench's own illustrations, not to arbitrarily chosen numbers.

``pdebench_advection_1d`` targets PDEBench's 1D linear advection equation
(paper Eq. 6-7: ``u_t + beta*u_x = 0``, periodic on ``x in (0,1)``, beta=0.4
matching the paper's own Figure 6 caption). This equation has an EXACT
closed-form solution by the method of characteristics,
``u(t,x) = u0(x - beta*t)`` (periodic wraparound) -- a standard result (see
e.g. Strauss, "Partial Differential Equations: An Introduction"). The
initial condition ``u0(x) = sin(2*pi*x)`` is independently evolved to
``t=1.3`` via a Fourier pseudo-spectral method-of-lines
(``numpy.fft`` for the spatial derivative + ``scipy.integrate.solve_ivp``,
DOP853) and cross-checked pointwise against the exact shifted-sinusoid
solution.

``pdebench_diffusion_reaction_1d`` targets PDEBench's 1D diffusion-reaction
equation (paper Eq. 9-10: ``u_t - nu*u_xx - rho*u*(1-u) = 0``, nu=0.5,
rho=1.0 matching the paper's own Figure 6 caption). This is exactly the
Fisher-KPP equation (Fisher, R.A., "The Wave of Advance of Advantageous
Genes", Annals of Eugenics 7(4), 1937; Kolmogorov, Petrovsky & Piskunov,
1937), which has a real, citable EXACT closed-form traveling-wave solution
for a special wave speed (Ablowitz, M.J. & Zeppetella, A., "Explicit
Solutions of Fisher's Equation for a Special Wave Speed", Bulletin of
Mathematical Biology 41(6), 1979, 835-840):
``u(x,t) = [1 + exp(sqrt(rho/(6*nu))*(x - c*t))]^(-2)``, ``c = 5*sqrt(rho*nu/6)``.
This closed form is used as the initial condition on a wide truncated
domain and independently re-evolved forward in time via a second-order
central-difference-in-space, ``scipy.integrate.solve_ivp``-in-time
method-of-lines solve of the ACTUAL PDE (diffusion + logistic reaction
terms, not the closed form itself), then cross-checked against the exact
traveling-wave solution at the later time.

``pdebench_darcy_2d`` targets PDEBench's 2D Darcy flow equation (paper
Eq. 13-14: ``-div(a(x)*grad(u(x))) = f(x)`` on the unit square, ``u=0`` on
the boundary, ``f=beta`` constant, beta=1.0 matching the paper's own
Figure 7 caption). PDEBench's own ``a(x)`` is in general a random/binary
permeability field; this entry uses the simplest legitimate member of that
family, a homogeneous (spatially constant) permeability ``a(x)=a0=1.0``,
which reduces the equation to the classical Poisson problem
``-a0*laplacian(u) = beta`` with zero Dirichlet boundary conditions. This
has a standard textbook closed-form solution as a double Fourier sine
series (see e.g. Strauss, "Partial Differential Equations: An
Introduction", section on Poisson's equation on a rectangle) -- and its
center-point value (~0.0736713, a widely cited constant for the unit-square
Poisson benchmark, e.g. as used to validate finite-element/finite-difference
Poisson solvers) is reproduced here to confirm the series is not
miscoded. The series solution (39x39 odd modes) is independently
cross-checked against a second, unrelated method: a direct 5-point-stencil
finite-difference discretization of the SAME PDE, assembled as a sparse
linear system and solved via ``scipy.sparse.linalg.spsolve`` -- not
imported from, or structurally related to, the Fourier-series code path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

__all__ = [
    "BenchmarkEntry",
    "register_benchmark",
    "get_benchmark",
    "list_benchmarks",
    "benchmark_catalog",
]


@dataclass
class BenchmarkEntry:
    """A single named, real reference dataset: enough to plug directly into
    ``pinneapple_llm.guardrail.PhysicsGuardrail._check_reference`` as
    ``(reference_x, reference_y)``.

    ``reference_x``/``reference_y`` are plain ``float32`` numpy arrays,
    already shaped ``(N, len(x_vars))`` / ``(N, len(y_vars))`` -- exactly
    what ``_check_reference`` expects, so no further extraction step is
    needed (unlike ``_load_reference_from_upd_zarr``, which has to pick
    named variables out of an xarray store; here the columns are already
    in the right order at construction time).
    """

    name: str
    description: str
    reference_source: str
    x_vars: Tuple[str, ...]
    y_vars: Tuple[str, ...]
    reference_x: np.ndarray
    reference_y: np.ndarray
    verification_note: str = ""


_REGISTRY: Dict[str, Callable[[], BenchmarkEntry]] = {}


def register_benchmark(name: str):
    """Decorator to register a zero-argument ``BenchmarkEntry`` factory
    under ``name`` -- mirrors ``pinneapple_physics.pde_environment.presets
    .registry.register_preset``'s decorator pattern. The factory is called
    fresh on every ``get_benchmark``/``benchmark_catalog`` lookup (not
    memoised), so the independent numerical-verification step below runs,
    and fails loudly, every time the entry is resolved."""

    def deco(fn: Callable[[], BenchmarkEntry]) -> Callable[[], BenchmarkEntry]:
        key = str(name).lower().strip()
        _REGISTRY[key] = fn
        return fn

    return deco


def get_benchmark(name: str) -> BenchmarkEntry:
    """Resolve a benchmark dataset by name.

    Parameters
    ----------
    name : benchmark identifier (case-insensitive), e.g. ``"lane_emden_n1.5"``.

    Raises
    ------
    KeyError if the name is not registered.
    """
    key = str(name).lower().strip()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown benchmark dataset '{name}'. Available: {list_benchmarks()}")
    return _REGISTRY[key]()


def list_benchmarks() -> List[str]:
    """Return the sorted list of all registered benchmark dataset names."""
    return sorted(_REGISTRY.keys())


def benchmark_catalog() -> Dict[str, BenchmarkEntry]:
    """Return name -> ``BenchmarkEntry`` for every registered benchmark,
    built (and independently re-verified) fresh."""
    return {key: fn() for key, fn in _REGISTRY.items()}


# ---------------------------------------------------------------------------
# Lane-Emden polytrope profiles (n=1.5, n=3): real, independently-integrated
# reference data for `pinneapple_physics...presets.astrophysics
# .lane_emden_polytrope` -- see module docstring for method and citation.
# ---------------------------------------------------------------------------

def _integrate_lane_emden_profile(
    n: float, xi0: float = 1e-3, n_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Independently integrate theta''(xi) + (2/xi)theta'(xi) + theta^n = 0,
    theta(0)=1, theta'(0)=0, from ``xi0`` to the first zero crossing xi_1
    (the star's dimensionless surface), then densely resample the interior
    solution on ``n_points`` points spanning [xi0, xi1].

    Returns ``(reference_x, reference_y, xi1)`` where ``reference_x`` is
    ``(n_points, 1)`` (the radial coordinate xi -- what
    ``lane_emden_polytrope``'s single coord, named "t" in that preset for
    compiler-convention reasons, actually represents physically) and
    ``reference_y`` is ``(n_points, 2)`` (columns ``theta``, ``phi=dtheta/dxi``,
    matching ``lane_emden_polytrope``'s ``fields=("theta", "phi")`` order
    exactly). Both float32, matching the dtype convention
    ``_load_reference_from_upd_zarr`` already uses.
    """
    from scipy.integrate import solve_ivp

    theta0 = 1.0 - xi0 ** 2 / 6.0  # near-origin series expansion, valid to O(xi^2) for any n
    phi0 = -xi0 / 3.0

    def rhs(xi, s):
        theta, phi = s
        theta_safe = max(theta, 0.0)  # avoid a negative base for non-integer n past the surface
        return [phi, -theta_safe ** n - (2.0 / xi) * phi]

    def event_zero(xi, s):
        return s[0]

    event_zero.terminal = True
    event_zero.direction = -1

    sol = solve_ivp(
        rhs, [xi0, 10.0], [theta0, phi0], events=event_zero,
        rtol=1e-12, atol=1e-12, method="DOP853", max_step=0.01, dense_output=True,
    )
    if sol.t_events[0].size != 1:
        raise RuntimeError(f"Lane-Emden n={n}: expected exactly one zero crossing in [0, 10], got {sol.t_events[0]}")
    xi1 = float(sol.t_events[0][0])

    xi_grid = np.linspace(xi0, xi1, n_points)
    theta_phi = sol.sol(xi_grid)  # shape (2, n_points)
    reference_x = xi_grid.reshape(-1, 1).astype("float32")
    reference_y = np.stack([theta_phi[0], theta_phi[1]], axis=1).astype("float32")
    return reference_x, reference_y, xi1


def _lane_emden_entry(n: float, published_xi1: float, name: str, description: str) -> BenchmarkEntry:
    reference_x, reference_y, xi1 = _integrate_lane_emden_profile(n)
    rel_err = abs(xi1 - published_xi1) / published_xi1
    if rel_err > 1e-3:
        # Same <0.1% tolerance tests/test_lane_emden_numerical_validation.py
        # asserts -- if this ever fails it means the integration itself
        # regressed, and this catalog entry refuses to hand back an
        # unverified profile rather than silently doing so.
        raise RuntimeError(
            f"Lane-Emden benchmark '{name}': independently-integrated surface xi_1={xi1:.6f} "
            f"does not match the published table value {published_xi1} (rel_err={100 * rel_err:.4f}%, "
            "expected <0.1%) -- refusing to hand back an unverified reference dataset"
        )
    return BenchmarkEntry(
        name=name,
        description=description,
        reference_source=(
            "Hansen, Kawaler & Trimble, 'Stellar Interiors', 2nd ed., Table 4.1 "
            f"(xi_1={published_xi1}); see also Chandrasekhar (1939)"
        ),
        x_vars=("xi",),
        y_vars=("theta", "phi"),
        reference_x=reference_x,
        reference_y=reference_y,
        verification_note=(
            f"independently re-integrated (scipy.integrate.solve_ivp, DOP853, not imported from "
            f"the preset's compiled residual) surface xi_1={xi1:.5f} vs. published {published_xi1} "
            f"(rel_err={100 * rel_err:.4f}%)"
        ),
    )


@register_benchmark("lane_emden_n1.5")
def _lane_emden_n1_5() -> BenchmarkEntry:
    return _lane_emden_entry(
        n=1.5,
        published_xi1=3.65375,
        name="lane_emden_n1.5",
        description=(
            "Lane-Emden polytrope, index n=1.5 (non-relativistic degenerate star / white-dwarf "
            "core), theta(xi) and phi(xi)=dtheta/dxi from the center (xi~0) to the surface "
            "(xi_1, theta=0). Matches pinneapple_physics.pde_environment.presets.astrophysics"
            ".lane_emden_polytrope(n=1.5)'s fields=('theta','phi')."
        ),
    )


@register_benchmark("lane_emden_n3")
def _lane_emden_n3() -> BenchmarkEntry:
    return _lane_emden_entry(
        n=3.0,
        published_xi1=6.89685,
        name="lane_emden_n3",
        description=(
            "Lane-Emden polytrope, index n=3 (Eddington standard model / relativistic degenerate "
            "limit), theta(xi) and phi(xi)=dtheta/dxi from the center (xi~0) to the surface "
            "(xi_1, theta=0). Matches pinneapple_physics.pde_environment.presets.astrophysics"
            ".lane_emden_polytrope(n=3.0)'s fields=('theta','phi')."
        ),
    )


# ---------------------------------------------------------------------------
# Concorde high-angle-of-attack lift coefficient: a CLOSED-FORM THEORETICAL
# cross-check, not real Concorde data. See module docstring for why no
# wind-tunnel/flight-test number is used here.
# ---------------------------------------------------------------------------

# Widely published approximate aspect ratio of Concorde's low-aspect-ratio
# ogival-delta wing (AR = b^2/S ~ 1.7 for span ~25.6 m, wing area ~358 m^2).
# This is a rounded, commonly-cited planform figure, not a precise
# manufacturer specification -- appropriate for a THEORETICAL cross-check,
# not for anything claiming flight-test-grade precision.
_CONCORDE_ASPECT_RATIO = 1.7


def _polhamus_slender_delta_cl(alpha_deg: np.ndarray, aspect_ratio: float) -> np.ndarray:
    """Evaluate the Polhamus (1966) leading-edge-suction vortex-lift analogy
    for a sharp-leading-edge delta wing, in its low-aspect-ratio (slender
    wing) asymptotic limit:

        CL(alpha) = Kp * sin(alpha) * cos(alpha)^2 + Kv * cos(alpha) * sin(alpha)^2

    with the potential-flow term's constant Kp = pi*AR/2 (the classic
    slender-wing potential lift-curve slope; R. T. Jones, "Properties of
    Low-Aspect-Ratio Pointed Wings at Speeds Below and Above the Speed of
    Sound", NACA Report 835, 1946) and the vortex-lift term's constant
    Kv = 2 (Polhamus's own slender-wing asymptote as AR -> 0).

    This is deliberately the ASYMPTOTIC closed form (fixed Kp/Kv formulas),
    not Polhamus's full empirical Kp(AR)/Kv(AR) design charts for an
    arbitrary aspect ratio -- see the calling entry's ``verification_note``
    for exactly what that means for fidelity at Concorde's actual AR.
    """
    alpha_rad = np.deg2rad(alpha_deg)
    kp = np.pi * aspect_ratio / 2.0
    kv = 2.0
    return kp * np.sin(alpha_rad) * np.cos(alpha_rad) ** 2 + kv * np.cos(alpha_rad) * np.sin(alpha_rad) ** 2


@register_benchmark("concorde_high_aoa")
def _concorde_high_aoa() -> BenchmarkEntry:
    alpha_deg = np.linspace(0.0, 20.0, 11)  # 0..20 deg in 2 deg steps; includes AoA=10 deg
    cl = _polhamus_slender_delta_cl(alpha_deg, _CONCORDE_ASPECT_RATIO)

    # Independent re-evaluation, mirroring the Lane-Emden entries' pattern of
    # re-deriving and cross-checking against a citable closed-form number
    # (there is only a THEORY to check this formula against here, not a
    # published experimental number -- see module/entry docstrings): the
    # formula's own small-angle slope (finite difference at a tiny AoA, not
    # the analytic derivative used to build `cl` above) must recover the
    # cited closed-form slender-wing lift-curve-slope constant
    # Kp = pi*AR/2 (R. T. Jones, NACA Report 835) to <0.1%.
    eps_deg = 1e-4
    slope_numeric = float(_polhamus_slender_delta_cl(np.array([eps_deg]), _CONCORDE_ASPECT_RATIO)[0]) / np.deg2rad(eps_deg)
    kp_expected = np.pi * _CONCORDE_ASPECT_RATIO / 2.0
    rel_err = abs(slope_numeric - kp_expected) / kp_expected
    if rel_err > 1e-3:
        raise RuntimeError(
            f"concorde_high_aoa: independently re-evaluated small-angle CL slope "
            f"{slope_numeric:.6f}/rad does not match the closed-form slender-wing constant "
            f"Kp=pi*AR/2={kp_expected:.6f}/rad (rel_err={100 * rel_err:.4f}%, expected <0.1%) "
            "-- refusing to hand back an unverified profile"
        )

    reference_x = alpha_deg.reshape(-1, 1).astype("float32")
    reference_y = cl.reshape(-1, 1).astype("float32")

    return BenchmarkEntry(
        name="concorde_high_aoa",
        description=(
            "THEORETICAL cross-check, not real Concorde data: the Polhamus (1966) "
            "leading-edge-suction vortex-lift analogy for a sharp-edged slender delta wing, "
            "CL(alpha) = Kp*sin(alpha)*cos(alpha)^2 + Kv*cos(alpha)*sin(alpha)^2, evaluated in "
            "its low-aspect-ratio asymptotic limit (Kp=pi*AR/2, Kv=2) at Concorde's widely "
            "published approximate wing aspect ratio (AR~1.7), across an angle-of-attack sweep "
            "from 0 to 20 degrees (including AoA=10 deg). This captures the qualitative "
            "vortex-lift behavior that makes a slender delta wing's CL keep rising well past "
            "the angle at which a conventional wing would stall -- it does NOT reproduce a "
            "real Concorde's actual measured CL at any angle."
        ),
        reference_source=(
            "Polhamus, E.C., 'A Concept of the Vortex Lift of Sharp-Edge Delta Wings Based on "
            "a Leading-Edge-Suction Analogy', NASA TN D-3767 (1966); slender-wing potential-flow "
            "lift-curve-slope constant Kp=pi*AR/2 per R. T. Jones, 'Properties of Low-Aspect-Ratio "
            "Pointed Wings at Speeds Below and Above the Speed of Sound', NACA Report 835 (1946)."
        ),
        x_vars=("alpha_deg",),
        y_vars=("cl",),
        reference_x=reference_x,
        reference_y=reference_y,
        verification_note=(
            "LOW/MEDIUM-fidelity THEORETICAL cross-check only -- explicitly NOT a substitute "
            "for real Concorde wind-tunnel or flight-test data, which is not available to this "
            "codebase with a citable public source (see module docstring). Kp=pi*AR/2 and Kv=2 "
            "are the Polhamus formula's own low-aspect-ratio ASYMPTOTIC limit, not chart-read "
            "values for Concorde's exact aspect ratio, so applying them at AR~1.7 (not "
            "vanishingly small) is itself an approximation on top of the theory. What IS "
            "verified: an independent finite-difference re-evaluation of this implementation's "
            "small-angle CL slope reproduces the cited closed-form constant Kp=pi*AR/2 "
            "(R. T. Jones, NACA Report 835) to <0.1% -- i.e. the code correctly implements the "
            "cited formula, not that the formula correctly predicts a real Concorde's lift."
        ),
    )


# ---------------------------------------------------------------------------
# Blasius flat-plate laminar boundary layer: real, independently-integrated
# similarity solution. See module docstring for method and citation.
# ---------------------------------------------------------------------------

# Wall-shear (second-derivative) parameter for the Blasius similarity
# solution, in the ETA = y*sqrt(U/(nu*x)) normalization that makes the ODE
# f'''+0.5*f*f''=0 (the form integrated below): Howarth's (1938) high-
# precision shooting-method integration, tabulated in e.g. Schlichting &
# Gersten, "Boundary-Layer Theory", 8th/9th ed. NOTE: this is explicitly
# NOT the sometimes-quoted value 0.4696, which is f''(0) under a DIFFERENT
# similarity-variable scaling (eta multiplied by an extra sqrt(2), giving
# the alternative ODE form f'''+f*f''=0) -- the two normalizations are not
# interchangeable, and mixing them up is a common source of error when
# citing this benchmark. 0.332057 is the correct value for the ODE form
# used here (verified by the independent re-derivation below, which
# reproduces it to <0.001%).
_BLASIUS_FPP0_PUBLISHED = 0.332057
_BLASIUS_ETA_MAX = 10.0  # far enough that f'(eta_max)=1 (freestream) to within integration tolerance


def _blasius_rhs(eta: float, state: np.ndarray) -> list:
    """The Blasius ODE f'''+0.5*f*f''=0, written as a first-order system
    ``state=[f, f', f'']`` -> ``[f', f'', f''']= [f', f'', -0.5*f*f'']``."""
    f, fp, fpp = state
    return [fp, fpp, -0.5 * f * fpp]


def _integrate_blasius_profile(n_points: int = 200) -> Tuple[np.ndarray, np.ndarray, float]:
    """Independently solve the Blasius similarity ODE via a shooting
    method: integrate ``f'''+0.5*f*f''=0`` from ``eta=0`` (``f=f'=0``,
    unknown ``f''(0)=s``) out to ``eta=_BLASIUS_ETA_MAX``, and use
    ``scipy.optimize.brentq`` to find the ``s`` for which the far-field
    boundary condition ``f'(eta_max)=1`` (the freestream velocity match)
    is satisfied.

    Returns ``(reference_x, reference_y, fpp0)`` where ``reference_x`` is
    ``(n_points, 1)`` (the similarity coordinate eta) and ``reference_y``
    is ``(n_points, 3)`` (columns ``f`` (scaled stream function), ``f'``
    (``u/U_inf``, the velocity profile), ``f''`` (proportional to the
    local shear stress)). Both float32, matching the dtype convention
    the other entries in this module use.
    """
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    def _fprime_at_eta_max(s: float) -> float:
        sol = solve_ivp(
            _blasius_rhs, [0.0, _BLASIUS_ETA_MAX], [0.0, 0.0, s],
            rtol=1e-12, atol=1e-13, method="DOP853", dense_output=True,
        )
        return float(sol.y[1, -1]) - 1.0

    # f''(0) is known (from the physics of a monotonically-accelerating boundary
    # layer profile) to lie well within (0.1, 1.0); the residual is monotonic
    # increasing in s over that bracket.
    fpp0 = brentq(_fprime_at_eta_max, 0.1, 1.0, xtol=1e-13, rtol=1e-13)

    sol = solve_ivp(
        _blasius_rhs, [0.0, _BLASIUS_ETA_MAX], [0.0, 0.0, fpp0],
        rtol=1e-12, atol=1e-13, method="DOP853", dense_output=True,
    )
    eta_grid = np.linspace(0.0, _BLASIUS_ETA_MAX, n_points)
    profile = sol.sol(eta_grid)  # shape (3, n_points): f, f', f''
    reference_x = eta_grid.reshape(-1, 1).astype("float32")
    reference_y = profile.T.astype("float32")  # (n_points, 3)
    return reference_x, reference_y, float(fpp0)


@register_benchmark("blasius_flat_plate")
def _blasius_flat_plate() -> BenchmarkEntry:
    reference_x, reference_y, fpp0 = _integrate_blasius_profile()
    rel_err = abs(fpp0 - _BLASIUS_FPP0_PUBLISHED) / _BLASIUS_FPP0_PUBLISHED
    if rel_err > 1e-3:
        raise RuntimeError(
            f"blasius_flat_plate: independently-integrated wall-shear parameter f''(0)={fpp0:.6f} "
            f"does not match the published value {_BLASIUS_FPP0_PUBLISHED} (rel_err={100 * rel_err:.4f}%, "
            "expected <0.1%) -- refusing to hand back an unverified profile"
        )
    return BenchmarkEntry(
        name="blasius_flat_plate",
        description=(
            "Blasius laminar flat-plate boundary-layer similarity solution: f(eta), f'(eta)=u/U_inf "
            "(the velocity profile), and f''(eta) (proportional to local shear stress), for "
            "eta=y*sqrt(U_inf/(nu*x)) from the wall (eta=0) out to the freestream "
            f"(eta={_BLASIUS_ETA_MAX:.0f}, where f'->1 to within integration tolerance). The classical "
            "zero-pressure-gradient laminar boundary-layer benchmark."
        ),
        reference_source=(
            "Blasius, H. (1908); high-precision integration by Howarth, L. (1938); tabulated in "
            "Schlichting, H. & Gersten, K., 'Boundary-Layer Theory', 8th/9th ed. "
            f"(wall-shear parameter f''(0)={_BLASIUS_FPP0_PUBLISHED})."
        ),
        x_vars=("eta",),
        y_vars=("f", "f_prime", "f_pprime"),
        reference_x=reference_x,
        reference_y=reference_y,
        verification_note=(
            f"independently re-integrated (shooting method: scipy.integrate.solve_ivp DOP853 + "
            f"scipy.optimize.brentq, not imported from any other solver in this codebase) "
            f"wall-shear parameter f''(0)={fpp0:.6f} vs. published {_BLASIUS_FPP0_PUBLISHED} "
            f"(rel_err={100 * rel_err:.4f}%). NOTE: this is the f''(0) value for the "
            "eta=y*sqrt(U/(nu*x)) normalization (ODE f'''+0.5*f*f''=0) -- NOT the same as the "
            "sometimes-quoted 0.4696, which belongs to a different similarity-variable scaling "
            "(see module docstring)."
        ),
    )


# ---------------------------------------------------------------------------
# 1D transient conduction in a plane wall (convective BCs both faces),
# one-term-series centerline temperature history. See module docstring.
# ---------------------------------------------------------------------------

# Biot number for this entry: Bi = h*L/k = 1.0 (a standard, round, widely
# tabulated case -- e.g. Incropera & DeWitt's own worked examples use it).
_PLANEWALL_BIOT_NUMBER = 1.0
# Incropera & DeWitt, "Fundamentals of Heat and Mass Transfer", Table 5.1
# ("One-term approximation coefficients"), plane wall, Bi=1.0.
_PLANEWALL_ZETA1_PUBLISHED = 0.8603
_PLANEWALL_C1_PUBLISHED = 1.1191
# The one-term approximation is only valid for Fo > 0.2 (Incropera & DeWitt's
# own stated criterion, <2% error above this) -- the reference curve below is
# restricted to that range so it is never used outside its documented validity.
_PLANEWALL_FO_MIN = 0.2
_PLANEWALL_FO_MAX = 2.0


def _plane_wall_one_term_zeta1_c1(bi: float) -> Tuple[float, float]:
    """Independently solve for the first eigenvalue zeta_1 -- the smallest
    positive root of the transcendental eigen-condition
    ``zeta*tan(zeta) = Bi`` that governs 1D transient conduction in a
    plane wall with convective boundary conditions on both faces -- via
    ``scipy.optimize.brentq`` bisection on ``(0, pi/2)`` (the first branch
    of ``tan``, where the smallest positive root always lies for any
    finite Bi > 0), and the corresponding one-term series coefficient
    ``C1 = 4*sin(zeta_1) / (2*zeta_1 + sin(2*zeta_1))``.
    """
    from scipy.optimize import brentq

    def _eigencondition(zeta: float) -> float:
        return zeta * np.tan(zeta) - bi

    eps = 1e-9
    zeta1 = brentq(_eigencondition, eps, np.pi / 2.0 - eps, xtol=1e-14, rtol=1e-14)
    c1 = 4.0 * np.sin(zeta1) / (2.0 * zeta1 + np.sin(2.0 * zeta1))
    return float(zeta1), float(c1)


@register_benchmark("planewall_transient_conduction_bi1")
def _planewall_transient_conduction_bi1() -> BenchmarkEntry:
    zeta1, c1 = _plane_wall_one_term_zeta1_c1(_PLANEWALL_BIOT_NUMBER)

    zeta1_rel_err = abs(zeta1 - _PLANEWALL_ZETA1_PUBLISHED) / _PLANEWALL_ZETA1_PUBLISHED
    c1_rel_err = abs(c1 - _PLANEWALL_C1_PUBLISHED) / _PLANEWALL_C1_PUBLISHED
    if zeta1_rel_err > 1e-3 or c1_rel_err > 1e-3:
        raise RuntimeError(
            f"planewall_transient_conduction_bi1: independently-solved (zeta_1={zeta1:.6f}, "
            f"C1={c1:.6f}) does not match Incropera & DeWitt Table 5.1's published Bi=1.0 values "
            f"(zeta_1={_PLANEWALL_ZETA1_PUBLISHED}, C1={_PLANEWALL_C1_PUBLISHED}) to <0.1% "
            f"(rel_err zeta_1={100 * zeta1_rel_err:.4f}%, C1={100 * c1_rel_err:.4f}%) -- refusing "
            "to hand back an unverified reference curve"
        )

    fourier_number = np.linspace(_PLANEWALL_FO_MIN, _PLANEWALL_FO_MAX, 50)
    # Centerline (x*=0, so cos(zeta_1*x*)=1): theta*_0(Fo) = C1*exp(-zeta_1^2*Fo).
    theta_star_centerline = c1 * np.exp(-(zeta1 ** 2) * fourier_number)

    reference_x = fourier_number.reshape(-1, 1).astype("float32")
    reference_y = theta_star_centerline.reshape(-1, 1).astype("float32")

    return BenchmarkEntry(
        name="planewall_transient_conduction_bi1",
        description=(
            "1D transient conduction in a plane wall of half-thickness L, initially at a uniform "
            "temperature, suddenly exposed on both faces (at t=0) to convection with a fluid at a "
            "different temperature (Biot number Bi=h*L/k=1.0). Centerline (midplane, x*=0) "
            "dimensionless temperature history theta*_0(Fo) = C1*exp(-zeta_1^2*Fo), the one-term "
            f"series approximation, tabulated over its documented validity range "
            f"Fo=alpha*t/L^2 in [{_PLANEWALL_FO_MIN}, {_PLANEWALL_FO_MAX}] (Fo>0.2, <2% error per "
            "Incropera & DeWitt's own stated criterion)."
        ),
        reference_source=(
            "Incropera, F.P. & DeWitt, D.P., 'Fundamentals of Heat and Mass Transfer' -- one-term "
            f"approximation, plane wall, Table 5.1 (Bi={_PLANEWALL_BIOT_NUMBER}: "
            f"zeta_1={_PLANEWALL_ZETA1_PUBLISHED}, C1={_PLANEWALL_C1_PUBLISHED})."
        ),
        x_vars=("fourier_number",),
        y_vars=("theta_star_centerline",),
        reference_x=reference_x,
        reference_y=reference_y,
        verification_note=(
            f"independently re-solved (scipy.optimize.brentq on the zeta*tan(zeta)=Bi eigen-condition) "
            f"zeta_1={zeta1:.6f} vs. published {_PLANEWALL_ZETA1_PUBLISHED} "
            f"(rel_err={100 * zeta1_rel_err:.4f}%), C1={c1:.6f} vs. published "
            f"{_PLANEWALL_C1_PUBLISHED} (rel_err={100 * c1_rel_err:.4f}%). Reference curve restricted "
            f"to Fo>={_PLANEWALL_FO_MIN} -- the one-term approximation's own documented validity range -- "
            "so this benchmark is never used outside the regime its citation actually supports."
        ),
    )


# ---------------------------------------------------------------------------
# PDEBench (Takamoto et al., NeurIPS 2022 Datasets and Benchmarks,
# arXiv:2210.07182) integration: three of PDEBench's own governing PDEs
# (as written in the paper's Appendix D), each independently solved via a
# method not borrowed from PDEBench's own code, and cross-checked against a
# real, citable closed-form solution. See module docstring for full method
# and citation for each entry -- PDEBench's actual (multi-GB, DaRUS-hosted)
# HDF5 datasets are never fetched here.
# ---------------------------------------------------------------------------

_PDEBENCH_CITATION = (
    "Takamoto, M., Praditia, T., Leiteritz, R., MacKinlay, D., Alesiani, F., Pflueger, D. "
    "& Niepert, M., 'PDEBench: An Extensive Benchmark for Scientific Machine Learning', "
    "NeurIPS 2022 Datasets and Benchmarks Track, arXiv:2210.07182"
)

# --- 1D linear advection (PDEBench paper Eq. 6-7) --------------------------

_ADVECTION_BETA = 0.4  # matches the paper's own Figure 6 caption "1D Advection (beta=0.4)"
_ADVECTION_T_FINAL = 1.3
_ADVECTION_N_X = 200


def _integrate_advection_1d(
    beta: float = _ADVECTION_BETA, t_final: float = _ADVECTION_T_FINAL, n_x: int = _ADVECTION_N_X,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Independently evolve u_t + beta*u_x = 0 (periodic, x in (0,1)) from
    u0(x)=sin(2*pi*x) to ``t_final`` via a Fourier pseudo-spectral
    method-of-lines (exact spatial derivative via ``numpy.fft``, time
    integration via ``scipy.integrate.solve_ivp``, DOP853 -- not imported
    from anywhere else in this codebase), and compare pointwise against the
    PDE's exact method-of-characteristics solution
    ``u(t,x) = u0(x - beta*t)``.

    Returns ``(reference_x, reference_y, max_abs_err)`` where
    ``reference_x`` is ``(n_x, 1)`` (the spatial grid) and ``reference_y``
    is ``(n_x, 1)`` (the exact solution u(t_final, x); the independently
    -integrated numeric solution is used only for the cross-check, not as
    the stored reference, since here the exact closed form IS the ground
    truth and the numeric PDE solve is what is being validated against it).
    """
    from scipy.integrate import solve_ivp

    x = np.linspace(0.0, 1.0, n_x, endpoint=False)
    u0 = np.sin(2.0 * np.pi * x)
    k = 2.0 * np.pi * np.fft.rfftfreq(n_x, d=1.0 / n_x)

    def rhs(t, u):
        du_dx = np.fft.irfft(1j * k * np.fft.rfft(u), n=n_x)
        return -beta * du_dx

    sol = solve_ivp(
        rhs, [0.0, t_final], u0, t_eval=[t_final], rtol=1e-12, atol=1e-13, method="DOP853",
    )
    u_numeric = sol.y[:, -1]
    u_exact = np.sin(2.0 * np.pi * (x - beta * t_final))
    max_abs_err = float(np.max(np.abs(u_numeric - u_exact)))

    reference_x = x.reshape(-1, 1).astype("float32")
    reference_y = u_exact.reshape(-1, 1).astype("float32")
    return reference_x, reference_y, max_abs_err


@register_benchmark("pdebench_advection_1d")
def _pdebench_advection_1d() -> BenchmarkEntry:
    reference_x, reference_y, max_abs_err = _integrate_advection_1d()
    if max_abs_err > 1e-6:
        raise RuntimeError(
            f"pdebench_advection_1d: independent Fourier-spectral PDE solve disagrees with the "
            f"exact method-of-characteristics solution by max_abs_err={max_abs_err:.3e} "
            "(expected <1e-6) -- refusing to hand back an unverified profile"
        )
    return BenchmarkEntry(
        name="pdebench_advection_1d",
        description=(
            "PDEBench's 1D linear advection equation, u_t + beta*u_x = 0, x in (0,1), periodic "
            f"boundary conditions, advection speed beta={_ADVECTION_BETA} (matching PDEBench's own "
            f"Figure 6 example), initial condition u0(x)=sin(2*pi*x), evaluated at t={_ADVECTION_T_FINAL}. "
            "This entry reproduces PDEBench's own governing equation and example parameter, NOT any "
            "of PDEBench's actual (multi-GB, DaRUS-hosted) sample trajectories -- the reference curve "
            "is the equation's own exact closed-form solution."
        ),
        reference_source=(
            f"{_PDEBENCH_CITATION} (Eq. 6-7: 1D advection equation, beta=0.4 example per Fig. 6); "
            "exact solution u(t,x)=u0(x-beta*t) by the method of characteristics is a standard result, "
            "e.g. Strauss, W.A., 'Partial Differential Equations: An Introduction'."
        ),
        x_vars=("x",),
        y_vars=("u",),
        reference_x=reference_x,
        reference_y=reference_y,
        verification_note=(
            f"the stored reference IS the PDE's exact closed-form solution u(t,x)=u0(x-beta*t); "
            f"independently cross-checked by evolving u0 forward via a Fourier pseudo-spectral "
            f"method-of-lines (numpy.fft spatial derivative + scipy.integrate.solve_ivp DOP853 time "
            f"integration, not imported from PDEBench or any other solver in this codebase) to "
            f"t={_ADVECTION_T_FINAL}, giving max_abs_err={max_abs_err:.3e} against the exact solution "
            "(expected <1e-6)."
        ),
    )


# --- 1D diffusion-reaction / Fisher-KPP (PDEBench paper Eq. 9-10) ----------

_FISHER_NU = 0.5  # matches the paper's own Figure 6 caption "1D Reaction-Diffusion (nu=0.5, rho=1)"
_FISHER_RHO = 1.0
_FISHER_T_FINAL = 2.0
_FISHER_DOMAIN_L = 20.0
_FISHER_N_X = 201


def _fisher_kpp_exact(x: np.ndarray, t: float, nu: float, rho: float) -> np.ndarray:
    """Ablowitz & Zeppetella (1979) exact traveling-wave solution of the
    Fisher-KPP equation ``u_t = nu*u_xx + rho*u*(1-u)`` (PDEBench's own
    diffusion-reaction equation, ``u_t - nu*u_xx - rho*u*(1-u) = 0``,
    rearranged), for the special wave speed ``c = 5*sqrt(rho*nu/6)``."""
    c = 5.0 * np.sqrt(rho * nu / 6.0)
    kk = np.sqrt(rho / (6.0 * nu))
    s = kk * (x - c * t)
    return 1.0 / (1.0 + np.exp(s)) ** 2


def _integrate_diffusion_reaction_1d(
    nu: float = _FISHER_NU, rho: float = _FISHER_RHO, t_final: float = _FISHER_T_FINAL,
    domain_l: float = _FISHER_DOMAIN_L, n_x: int = _FISHER_N_X,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Independently re-evolve PDEBench's actual diffusion-reaction PDE
    (``u_t = nu*u_xx + rho*u*(1-u)``, NOT the closed-form formula itself) on
    a wide truncated domain ``[-domain_l, domain_l]``, initialized from the
    Ablowitz-Zeppetella exact traveling wave at ``t=0``, via a second-order
    central-difference spatial Laplacian + ``scipy.integrate.solve_ivp``
    time integration (boundary points held fixed at their far-field ``t=0``
    values, which are within ~1e-5 of the true asymptotic 0/1 states given
    how far the domain edges are from the wavefront over ``t_final``).
    Returns ``(reference_x, reference_y, max_abs_err)`` -- ``reference_y``
    is the exact closed-form profile at ``t_final``; ``max_abs_err`` is the
    independent numeric PDE solve's disagreement with it.
    """
    from scipy.integrate import solve_ivp

    x = np.linspace(-domain_l, domain_l, n_x)
    dx = x[1] - x[0]
    u0 = _fisher_kpp_exact(x, 0.0, nu, rho)
    u_left, u_right = float(u0[0]), float(u0[-1])

    def rhs(t, u_interior):
        u = np.empty(n_x)
        u[0] = u_left
        u[-1] = u_right
        u[1:-1] = u_interior
        lap = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx ** 2
        reaction = rho * u[1:-1] * (1.0 - u[1:-1])
        return nu * lap + reaction

    sol = solve_ivp(
        rhs, [0.0, t_final], u0[1:-1], method="RK45", rtol=1e-10, atol=1e-12, t_eval=[t_final],
    )
    u_numeric = np.empty(n_x)
    u_numeric[0], u_numeric[-1] = u_left, u_right
    u_numeric[1:-1] = sol.y[:, -1]

    u_exact_final = _fisher_kpp_exact(x, t_final, nu, rho)
    max_abs_err = float(np.max(np.abs(u_numeric - u_exact_final)))

    reference_x = x.reshape(-1, 1).astype("float32")
    reference_y = u_exact_final.reshape(-1, 1).astype("float32")
    return reference_x, reference_y, max_abs_err


@register_benchmark("pdebench_diffusion_reaction_1d")
def _pdebench_diffusion_reaction_1d() -> BenchmarkEntry:
    reference_x, reference_y, max_abs_err = _integrate_diffusion_reaction_1d()
    if max_abs_err > 1e-3:
        raise RuntimeError(
            f"pdebench_diffusion_reaction_1d: independent finite-difference PDE solve disagrees "
            f"with the Ablowitz-Zeppetella exact traveling-wave solution by max_abs_err="
            f"{max_abs_err:.3e} (expected <1e-3) -- refusing to hand back an unverified profile"
        )
    return BenchmarkEntry(
        name="pdebench_diffusion_reaction_1d",
        description=(
            "PDEBench's 1D diffusion-reaction equation, u_t - nu*u_xx - rho*u*(1-u) = 0, "
            f"nu={_FISHER_NU}, rho={_FISHER_RHO} (matching PDEBench's own Figure 6 example) -- "
            "exactly the Fisher-KPP equation. Reference profile is the Ablowitz-Zeppetella (1979) "
            f"exact traveling-wave solution evaluated at t={_FISHER_T_FINAL} on x in "
            f"[-{_FISHER_DOMAIN_L:.0f}, {_FISHER_DOMAIN_L:.0f}]. Reproduces PDEBench's own governing "
            "equation and example parameters, NOT any of PDEBench's actual sample trajectories."
        ),
        reference_source=(
            f"{_PDEBENCH_CITATION} (Eq. 9-10: 1D diffusion-reaction equation, nu=0.5/rho=1 example "
            "per Fig. 6); equation is the Fisher-KPP equation (Fisher, R.A., 'The Wave of Advance of "
            "Advantageous Genes', Annals of Eugenics 7(4), 1937); exact traveling-wave solution per "
            "Ablowitz, M.J. & Zeppetella, A., 'Explicit Solutions of Fisher's Equation for a Special "
            "Wave Speed', Bulletin of Mathematical Biology 41(6), 1979, 835-840."
        ),
        x_vars=("x",),
        y_vars=("u",),
        reference_x=reference_x,
        reference_y=reference_y,
        verification_note=(
            f"the stored reference IS the Ablowitz-Zeppetella (1979) exact closed-form traveling-wave "
            f"solution; independently cross-checked by evolving the ACTUAL diffusion-reaction PDE "
            f"forward from that same exact solution's t=0 profile (central-difference-in-space + "
            f"scipy.integrate.solve_ivp-in-time, not the closed form itself) to t={_FISHER_T_FINAL}, "
            f"giving max_abs_err={max_abs_err:.3e} against the exact solution (expected <1e-3)."
        ),
    )


# --- 2D Darcy flow, homogeneous permeability (PDEBench paper Eq. 13-14) ----

_DARCY_A0 = 1.0  # homogeneous/constant specialization of PDEBench's general permeability field a(x)
_DARCY_BETA = 1.0  # matches the paper's own Figure 7 caption "2D Darcy Flow (beta=1.0)"
_DARCY_N_MODES = 39  # odd Fourier modes per axis for the closed-form series
_DARCY_N_REF = 15  # reference grid points per axis (kept small/curated, per module docstring)
_DARCY_FD_N = 61  # finite-difference cross-check grid points per axis (interior)
_DARCY_CENTER_PUBLISHED = 0.0736713  # widely cited unit-square Poisson center-value constant


def _darcy_fourier_series(x: np.ndarray, y: np.ndarray, c: float, n_modes: int = _DARCY_N_MODES) -> np.ndarray:
    """Closed-form double Fourier sine series solution of
    ``-laplacian(u) = c`` on the unit square with ``u=0`` on the boundary
    (standard textbook result -- see module docstring):
    ``u(x,y) = sum_{m,n odd} [16*c/(pi^4*m*n*(m^2+n^2))] * sin(m*pi*x)*sin(n*pi*y)``.
    """
    u = np.zeros_like(x)
    for m in range(1, n_modes + 1, 2):
        for n in range(1, n_modes + 1, 2):
            coeff = 16.0 * c / (np.pi ** 4 * m * n * (m ** 2 + n ** 2))
            u = u + coeff * np.sin(m * np.pi * x) * np.sin(n * np.pi * y)
    return u


def _darcy_finite_difference_center(c: float, n_grid: int = _DARCY_FD_N) -> Tuple[np.ndarray, float]:
    """Independently solve the SAME Poisson problem (``-laplacian(u)=c``,
    ``u=0`` on the unit-square boundary) via a completely different method:
    a direct 5-point-stencil finite-difference discretization assembled as
    a sparse linear system and solved with
    ``scipy.sparse.linalg.spsolve`` -- no relation to the Fourier-series
    code path above. Returns the relative L2 disagreement against the
    series solution (evaluated on the same fine grid) and that grid's
    center value (for the extra published-constant cross-check).
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    h = 1.0 / (n_grid + 1)
    xs = np.linspace(h, 1.0 - h, n_grid)
    xg, yg = np.meshgrid(xs, xs, indexing="ij")

    main = np.ones(n_grid)
    tri = sp.diags([main[:-1], -2.0 * main, main[:-1]], [-1, 0, 1]) / h ** 2
    ident = sp.identity(n_grid)
    laplacian = sp.kron(tri, ident) + sp.kron(ident, tri)
    u_fd = spla.spsolve(laplacian.tocsr(), -c * np.ones(n_grid * n_grid)).reshape(n_grid, n_grid)

    u_series_fine = _darcy_fourier_series(xg, yg, c)
    rel_l2_err = float(np.linalg.norm(u_fd - u_series_fine) / np.linalg.norm(u_series_fine))
    center_value = float(u_series_fine[n_grid // 2, n_grid // 2])
    return rel_l2_err, center_value


@register_benchmark("pdebench_darcy_2d")
def _pdebench_darcy_2d() -> BenchmarkEntry:
    c = _DARCY_BETA / _DARCY_A0
    rel_l2_err, center_value = _darcy_finite_difference_center(c)
    center_rel_err = abs(center_value - _DARCY_CENTER_PUBLISHED) / _DARCY_CENTER_PUBLISHED

    if rel_l2_err > 1e-3:
        raise RuntimeError(
            f"pdebench_darcy_2d: independent finite-difference Poisson solve disagrees with the "
            f"Fourier-series closed-form solution by rel_l2_err={rel_l2_err:.3e} (expected <1e-3) "
            "-- refusing to hand back an unverified field"
        )
    if center_rel_err > 1e-2:
        raise RuntimeError(
            f"pdebench_darcy_2d: center-point value {center_value:.6f} does not match the widely "
            f"cited unit-square Poisson benchmark constant {_DARCY_CENTER_PUBLISHED} "
            f"(rel_err={100 * center_rel_err:.4f}%, expected <1%) -- refusing to hand back an "
            "unverified field"
        )

    xs_ref = np.linspace(0.0, 1.0, _DARCY_N_REF + 2)[1:-1]
    x_ref, y_ref = np.meshgrid(xs_ref, xs_ref, indexing="ij")
    u_ref = _darcy_fourier_series(x_ref, y_ref, c)

    reference_x = np.stack([x_ref.ravel(), y_ref.ravel()], axis=1).astype("float32")
    reference_y = u_ref.ravel().reshape(-1, 1).astype("float32")

    return BenchmarkEntry(
        name="pdebench_darcy_2d",
        description=(
            "PDEBench's 2D Darcy flow equation, -div(a(x)*grad(u(x))) = f(x) on the unit square, "
            f"u=0 on the boundary, f=beta={_DARCY_BETA} constant (matching PDEBench's own Figure 7 "
            f"example), specialized to a homogeneous permeability field a(x)=a0={_DARCY_A0} -- the "
            "simplest legitimate member of PDEBench's general (in general random/binary) a(x) "
            "family -- which reduces the equation to the classical Poisson problem "
            "-a0*laplacian(u)=beta. Reproduces PDEBench's own governing equation and example "
            "parameter, NOT any of PDEBench's actual sample permeability fields/solutions."
        ),
        reference_source=(
            f"{_PDEBENCH_CITATION} (Eq. 13-14: 2D Darcy flow, beta=1.0 example per Fig. 7); "
            "closed-form double Fourier sine series solution of the unit-square Poisson problem "
            "with zero Dirichlet boundary conditions is a standard textbook result, e.g. Strauss, "
            "W.A., 'Partial Differential Equations: An Introduction'; center-point value "
            f"{_DARCY_CENTER_PUBLISHED} is a widely cited constant for this exact benchmark problem."
        ),
        x_vars=("x", "y"),
        y_vars=("u",),
        reference_x=reference_x,
        reference_y=reference_y,
        verification_note=(
            f"the stored reference is the closed-form Fourier sine series solution ({_DARCY_N_MODES}"
            f"x{_DARCY_N_MODES} odd modes); independently cross-checked two ways: (1) a completely "
            f"separate 5-point-stencil finite-difference discretization solved via "
            f"scipy.sparse.linalg.spsolve gives rel_l2_err={rel_l2_err:.3e} against the series "
            f"(expected <1e-3); (2) the series' own center-point value {center_value:.6f} matches "
            f"the widely cited unit-square Poisson benchmark constant {_DARCY_CENTER_PUBLISHED} "
            f"(rel_err={100 * center_rel_err:.4f}%, expected <1%)."
        ),
    )
