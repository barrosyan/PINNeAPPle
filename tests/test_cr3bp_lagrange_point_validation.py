"""Independent validation of `cr3bp_planar_synodic`'s Lagrange-point
equilibria and L4 stability claim -- explicitly tracked as a genuine,
newly-added astrophysics preset in ROADMAP_PHYSICS_AI_HUB.md's
astrophysics section (N-body dynamics, N=3, the circular restricted
three-body problem).

Method: reimplement the SAME CR3BP acceleration equations independently
with `scipy.integrate.solve_ivp` and `scipy.optimize.brentq` (NOT
imported from `compile.py` or from `presets/astrophysics.py`'s own
helper functions -- a genuinely separate implementation, matching the
style of `test_j2_secular_validation.py` and
`test_lane_emden_numerical_validation.py`), and check two independent
things:

1. The 5 Lagrange-point equilibria (L1-L3 solved here via `brentq` on the
   collinear force-balance equation; L4/L5 the well-known closed-form
   equilateral-triangle points, true for ANY mass ratio mu) give an
   EXACT force balance (dOmega/dx = dOmega/dy = 0) to numerical
   root-finding precision, and the collinear points' distances from
   Earth match the well-known tabulated Earth-Moon Lagrange-point
   distances (~326,400 km for L1, ~448,900 km for L2, ~-381,600 km for
   L3, e.g. as tabulated on Wikipedia's "Lagrangian point" article) to
   well under 1%.
2. A small perturbation from L4 stays BOUNDED (a librating "tadpole"
   orbit, not a runaway) when integrated for 10 synodic periods with
   `solve_ivp` -- confirming L4's well-known linear stability for the
   Earth-Moon mass ratio (which sits well below Routh's critical mass
   ratio ~0.0385, above which the triangular points become unstable;
   Szebehely, "Theory of Orbits", 1967, Ch. 5).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy.integrate import solve_ivp  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

MU_EARTH_MOON = 0.012150585609624
EARTH_MOON_DISTANCE_KM = 384400.0


def _domega(x, y, mu):
    """Independent (from `presets/astrophysics.py`'s `cr3bp_omega_gradient`
    and from `compile.py`'s torch residual) plain-Python reimplementation
    of the CR3BP effective-potential gradient."""
    r1 = math.sqrt((x + mu) ** 2 + y ** 2)
    r2 = math.sqrt((x - 1.0 + mu) ** 2 + y ** 2)
    dOx = x - (1.0 - mu) * (x + mu) / r1 ** 3 - mu * (x - 1.0 + mu) / r2 ** 3
    dOy = y - (1.0 - mu) * y / r1 ** 3 - mu * y / r2 ** 3
    return dOx, dOy


def _cr3bp_rhs(mu):
    def rhs(t, s):
        x, y, vx, vy = s
        dOx, dOy = _domega(x, y, mu)
        ax = 2.0 * vy + dOx
        ay = -2.0 * vx + dOy
        return [vx, vy, ax, ay]
    return rhs


def test_l4_l5_equilateral_points_are_exact_equilibria_for_any_mu():
    """L4=(1/2-mu, sqrt(3)/2), L5=(1/2-mu, -sqrt(3)/2): force balance
    should be zero to root-finding/floating-point precision for several
    different mass ratios, not just the Earth-Moon default."""
    for mu in (0.001, 0.0121505856, 0.1, 0.3):
        for sign in (1.0, -1.0):
            x, y = 0.5 - mu, sign * math.sqrt(3.0) / 2.0
            dOx, dOy = _domega(x, y, mu)
            assert abs(dOx) < 1e-12 and abs(dOy) < 1e-12, (
                f"mu={mu}: L4/L5 force balance should be ~0, got dOx={dOx}, dOy={dOy}"
            )


def test_collinear_lagrange_points_match_known_earth_moon_distances():
    mu = MU_EARTH_MOON

    def f(x):
        return _domega(x, 0.0, mu)[0]

    xL1 = brentq(f, -mu + 0.5, 1 - mu - 1e-6, xtol=1e-14)
    xL2 = brentq(f, 1 - mu + 1e-6, 1.5, xtol=1e-14)
    xL3 = brentq(f, -1.2, -mu - 1e-6, xtol=1e-14)

    # Force balance at the roots themselves, to root-finder precision.
    for name, x in [("L1", xL1), ("L2", xL2), ("L3", xL3)]:
        dOx, dOy = _domega(x, 0.0, mu)
        assert abs(dOx) < 1e-10, f"{name}: force balance should be ~0 at the root, got {dOx}"

    # Earth-centered distances (barycentric x + mu, since primary 1/Earth
    # sits at x=-mu), compared against the well-known tabulated values.
    d_L1 = (xL1 + mu) * EARTH_MOON_DISTANCE_KM
    d_L2 = (xL2 + mu) * EARTH_MOON_DISTANCE_KM
    d_L3 = (xL3 + mu) * EARTH_MOON_DISTANCE_KM
    literature = {"L1": 326400.0, "L2": 448900.0, "L3": -381600.0}
    computed = {"L1": d_L1, "L2": d_L2, "L3": d_L3}
    for name in ("L1", "L2", "L3"):
        rel_err = abs(computed[name] - literature[name]) / abs(literature[name])
        print(f"{name}: computed={computed[name]:.1f} km, literature~{literature[name]:.1f} km, "
              f"rel_err={100 * rel_err:.3f}%")
        assert rel_err < 0.01, (
            f"{name} distance from Earth should match the well-known tabulated value to <1%, "
            f"got {100 * rel_err:.3f}%"
        )


def test_l4_perturbation_stays_bounded_confirming_linear_stability():
    """Earth-Moon mu is well below Routh's critical mass ratio (~0.0385),
    so L4 should be linearly stable: a small perturbation should librate
    (stay bounded), not run away, over many synodic periods."""
    mu = MU_EARTH_MOON
    xL4, yL4 = 0.5 - mu, math.sqrt(3.0) / 2.0
    perturbation = 1e-4
    s0 = [xL4 + perturbation, yL4, 0.0, 0.0]

    n_periods = 10
    period_synodic = 2.0 * math.pi
    t_eval = np.linspace(0, n_periods * period_synodic, n_periods * 50)
    sol = solve_ivp(_cr3bp_rhs(mu), [0, t_eval[-1]], s0, t_eval=t_eval,
                     rtol=1e-12, atol=1e-13, method="DOP853", max_step=0.01)
    assert sol.success

    dist_from_l4 = np.sqrt((sol.y[0] - xL4) ** 2 + (sol.y[1] - yL4) ** 2)
    max_excursion = float(dist_from_l4.max())
    print(f"L4 perturbation stability: initial={perturbation:.2e}, "
          f"max excursion over {n_periods} synodic periods={max_excursion:.6e} "
          f"(ratio={max_excursion / perturbation:.2f}x)")
    # Bounded libration, not runaway: stays within a modest, O(10x) multiple
    # of the initial perturbation, not growing by orders of magnitude.
    assert max_excursion < 50.0 * perturbation, (
        f"L4 perturbation should stay bounded (linearly stable) over {n_periods} synodic periods, "
        f"got max excursion {max_excursion:.4e} vs initial {perturbation:.4e}"
    )


def test_l4_perturbation_matches_compiled_preset_lagrange_points():
    """Cross-check: the Lagrange points this independent scipy-based
    validation computes should match the ones `cr3bp_planar_synodic`
    stores in its own `ProblemSpec.pde.meta["lagrange_points"]` (computed
    by `presets/astrophysics.py`'s own Newton-Raphson, a THIRD independent
    implementation) -- agreement here is evidence the preset's numbers are
    not just self-consistent with its own code, but with an entirely
    separate root-finder too."""
    from pinneapple_physics.pde_environment.presets.astrophysics import cr3bp_planar_synodic

    mu = MU_EARTH_MOON
    spec = cr3bp_planar_synodic(mu=mu)
    lpoints = spec.pde.meta["lagrange_points"]

    def f(x):
        return _domega(x, 0.0, mu)[0]

    xL1 = brentq(f, -mu + 0.5, 1 - mu - 1e-6, xtol=1e-14)
    xL2 = brentq(f, 1 - mu + 1e-6, 1.5, xtol=1e-14)
    xL3 = brentq(f, -1.2, -mu - 1e-6, xtol=1e-14)

    for name, x_indep in [("L1", xL1), ("L2", xL2), ("L3", xL3)]:
        x_preset, _ = lpoints[name]
        rel_err = abs(x_preset - x_indep) / abs(x_indep)
        assert rel_err < 1e-6, (
            f"{name}: preset's Newton-Raphson value {x_preset} should match this file's "
            f"independent brentq value {x_indep} to <1e-6, got rel_err={rel_err}"
        )
