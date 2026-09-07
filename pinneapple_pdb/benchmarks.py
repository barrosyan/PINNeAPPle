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

Honesty about scope, matching this repo's established ethic: this is TWO
entries, not a comprehensive benchmark suite. Both are the astrophysically
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

A real OpenFOAM LES channel-flow dataset (Re_tau=180, Moser-Kim-Mansour
setup) also exists, in the sibling ``splash-pinneapple`` project on this
machine -- but it was deliberately NOT added as a third catalog entry here:
it lives at an absolute filesystem path outside this repository (not
portable to another checkout, CI, or contributor's machine), is a 200+MB
zipped OpenFOAM case rather than a small in-repo array, and requires
bespoke OpenFOAM-binary-format parsing code (``openfoam_binary.py``,
``splash_mesh.py`` in that other project) that does not exist anywhere in
PINNeAPPle. Wiring it in properly is real future work, not something to
fake with a hardcoded absolute path that would only work on one machine.
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
