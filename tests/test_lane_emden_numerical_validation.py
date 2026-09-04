"""Numerical validation of the Lane-Emden equation for the astrophysically
standard polytropic indices n=1.5 (non-relativistic degenerate star, e.g.
a white dwarf core) and n=3 (Eddington standard model / relativistic
degenerate limit) -- explicitly tracked as NOT done in
ROADMAP_PHYSICS_AI_HUB.md's astrophysics section, since
`lane_emden_polytrope`'s closed-form Tier B checks
(tests/test_astrophysics_validation.py) only cover n in {0, 1, 5}, the
only indices with an exact analytic solution.

Method: independently integrate the Lane-Emden equation
    theta''(xi) + (2/xi) theta'(xi) + theta(xi)^n = 0,  theta(0)=1, theta'(0)=0
with `scipy.integrate.solve_ivp` (a fresh, independent implementation, NOT
imported from compile.py's `lane_emden_polytrope` branch -- this checks
the underlying physics/equation, not the compiler code against itself),
find the first zero crossing xi_1 (the star's dimensionless surface
radius), and compare against the well-known published values (tabulated
in essentially every stellar-structure textbook, e.g. Chandrasekhar 1939;
Hansen, Kawaler & Trimble, "Stellar Interiors", 2nd ed., Table 4.1):
    n=1.5: xi_1 = 3.65375
    n=3.0: xi_1 = 6.89685

As a sanity check on this integrator itself (before trusting it for the
n=1.5/3 cases with no closed form), it is also run for n=0 and n=1 --
where the exact answer is known in closed form (xi_1 = sqrt(6) and pi
respectively, both already covered by `test_astrophysics_validation.py`'s
residual-based checks) -- and matches to within 1e-6 relative error,
confirming the integration method itself is trustworthy.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy.integrate import solve_ivp  # noqa: E402


def _lane_emden_rhs(n: float):
    def rhs(xi, s):
        theta, phi = s
        theta_safe = max(theta, 0.0)  # avoid a negative base for non-integer n past the surface
        return [phi, -theta_safe ** n - (2.0 / xi) * phi]
    return rhs


def _first_zero_crossing(n: float, xi_max: float = 10.0) -> float:
    xi0 = 1e-4
    theta0 = 1.0 - xi0 ** 2 / 6.0  # near-origin series expansion, valid to O(xi^2) for any n
    phi0 = -xi0 / 3.0

    def event_zero(xi, s):
        return s[0]
    event_zero.terminal = True
    event_zero.direction = -1

    sol = solve_ivp(_lane_emden_rhs(n), [xi0, xi_max], [theta0, phi0],
                     events=event_zero, rtol=1e-12, atol=1e-12, method="DOP853", max_step=0.01)
    assert sol.t_events[0].size == 1, f"n={n}: expected exactly one zero crossing in [0, {xi_max}]"
    return float(sol.t_events[0][0])


@pytest.mark.parametrize("n,exact_xi1", [(0.0, math.sqrt(6)), (1.0, math.pi)])
def test_lane_emden_integrator_matches_closed_form(n, exact_xi1):
    """Sanity check on the integration method itself, using the two cases
    with a known exact answer, before trusting it for n=1.5/3 below."""
    xi1 = _first_zero_crossing(n)
    rel_err = abs(xi1 - exact_xi1) / exact_xi1
    assert rel_err < 1e-6, f"n={n}: integrator should match the closed form xi_1={exact_xi1}, got {xi1}"


@pytest.mark.parametrize("n,published_xi1", [(1.5, 3.65375), (3.0, 6.89685)])
def test_lane_emden_astrophysically_standard_n_matches_published_tables(n, published_xi1):
    """n=1.5 (non-relativistic degenerate/white dwarf) and n=3 (Eddington
    standard model) have no closed form -- this is the actual physics-
    correctness check for the astrophysically realistic cases, cross-
    referenced against published tabulated values."""
    xi1 = _first_zero_crossing(n)
    rel_err = abs(xi1 - published_xi1) / published_xi1
    print(f"n={n}: integrated xi_1={xi1:.6f}, published={published_xi1}, rel_err={100*rel_err:.5f}%")
    assert rel_err < 1e-3, f"n={n}: xi_1 should match the published table value to <0.1%, got {100*rel_err:.4f}%"
