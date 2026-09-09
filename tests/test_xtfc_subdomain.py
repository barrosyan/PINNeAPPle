"""Validates pinneapple_simulation.numerical_solvers.xtfc_subdomain against
two independent references on a genuinely stiff two-species linear kinetics
system:

    dy1/dt = -k1*y1
    dy2/dt =  k1*y1 - k2*y2,   y1(0)=1, y2(0)=0

with k1=2.0, k2=3e12 (stiffness ratio k2/k1 = 1.5e12, within the 1e10-1e14
target range): (1) the closed-form solution y1(t)=exp(-k1*t), y2(t) =
k1/(k2-k1)*(exp(-k1*t)-exp(-k2*t)), and (2) an independent
`solve_stiff_reaction_network` (scipy Radau) integration from
`stiff_kinetics_ode.py`.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_simulation.numerical_solvers.xtfc_subdomain import (
    XTFCSubdomainSolver,
    log_subdomain_boundaries,
    solve_xtfc_subdomains,
)
from pinneapple_simulation.numerical_solvers.stiff_kinetics_ode import solve_stiff_reaction_network
from pinneapple_simulation.numerical_solvers.registry import SolverRegistry, register_all

K1 = 2.0
K2 = 3e12
Y0 = [1.0, 0.0]
T_SPAN = (0.0, 5.0)


def _rate_fn(t, y):
    y1, y2 = y[:, 0], y[:, 1]
    return np.stack([-K1 * y1, K1 * y1 - K2 * y2], axis=-1)


def _rate_fn_scalar(t, y):
    y1, y2 = y
    return np.array([-K1 * y1, K1 * y1 - K2 * y2])


def _jac_fn(t, y):
    n = y.shape[0]
    J = np.zeros((n, 2, 2))
    J[:, 0, 0] = -K1
    J[:, 1, 0] = K1
    J[:, 1, 1] = -K2
    return J


def _jac_fn_scalar(t, y):
    return np.array([[-K1, 0.0], [K1, -K2]])


def _y_exact(t):
    y1 = np.exp(-K1 * t)
    y2 = K1 / (K2 - K1) * (np.exp(-K1 * t) - np.exp(-K2 * t))
    return y1, y2


def test_log_subdomain_boundaries_span_many_decades():
    boundaries = log_subdomain_boundaries(0.0, 5.0, n_subdomains=16, first_width=1e-11)
    assert boundaries[0] == 0.0
    assert boundaries[-1] == 5.0
    assert len(boundaries) == 17
    widths = np.diff(boundaries)
    assert np.all(widths > 0)
    # Widths should grow roughly geometrically (many orders of magnitude
    # between the first and last subdomain) -- the entire point of log
    # spacing for a stiff initial transient.
    assert widths[-1] / widths[0] > 1e8


def test_stiff_chain_matches_closed_form_and_scipy():
    t_query = np.concatenate([np.geomspace(1e-12, 1e-1, 30), np.linspace(0.2, 5.0, 30)])

    sol = solve_xtfc_subdomains(
        _rate_fn, Y0, T_SPAN, n_subdomains=40, first_width=1e-11,
        n_basis=30, n_collocation=40, seed=0, linear=True, jac_fn=_jac_fn,
    )
    y_pred = sol["predict"](t_query)

    y1_exact, y2_exact = _y_exact(t_query)
    err1_closed = np.max(np.abs(y_pred[:, 0] - y1_exact))
    err2_closed = np.max(np.abs(y_pred[:, 1] - y2_exact))
    assert err1_closed < 2e-3, f"y1 abs err {err1_closed} vs closed form too large"
    assert err2_closed < 1e-8, f"y2 abs err {err2_closed} vs closed form too large"

    ref = solve_stiff_reaction_network(
        _rate_fn_scalar, Y0, T_SPAN, t_eval=t_query, jac_fn=_jac_fn_scalar,
        method="Radau", rtol=1e-10, atol=1e-14,
    )
    assert ref["success"]
    err1_scipy = np.max(np.abs(y_pred[:, 0] - ref["y"][0]))
    err2_scipy = np.max(np.abs(y_pred[:, 1] - ref["y"][1]))
    assert err1_scipy < 2e-3, f"y1 abs err {err1_scipy} vs scipy Radau too large"
    assert err2_scipy < 1e-8, f"y2 abs err {err2_scipy} vs scipy Radau too large"


def test_continuity_is_exact_at_subdomain_boundaries():
    sol = solve_xtfc_subdomains(
        _rate_fn, Y0, T_SPAN, n_subdomains=10, first_width=1e-10,
        n_basis=20, n_collocation=25, seed=1, linear=True, jac_fn=_jac_fn,
    )
    boundaries = sol["boundaries"]
    for k in range(len(sol["subdomains"]) - 1):
        y_end_k = sol["subdomains"][k]["y_final"]
        # subdomain k+1's IC (its own y0) must equal subdomain k's final state
        # EXACTLY (chaining is a plain assignment, not a fitted match).
        y0_kp1 = sol["subdomains"][k + 1]["y0"]
        assert np.array_equal(y_end_k, y0_kp1)


def test_uniform_spacing_is_far_worse_than_log_for_this_stiff_system():
    # Demonstrates *why* log spacing matters: with the SAME total subdomain
    # budget, uniform spacing wastes almost all of it on the (already-smooth)
    # slow part of the trajectory and cannot resolve the sub-domain-1e-12
    # initial transient at all.
    t_query = np.array([1e-11, 1e-9, 1e-6])
    y1_exact, _ = _y_exact(t_query)

    sol_log = solve_xtfc_subdomains(
        _rate_fn, Y0, T_SPAN, n_subdomains=20, first_width=1e-11,
        n_basis=20, n_collocation=25, seed=0, linear=True, jac_fn=_jac_fn, spacing="log",
    )
    sol_uniform = solve_xtfc_subdomains(
        _rate_fn, Y0, T_SPAN, n_subdomains=20,
        n_basis=20, n_collocation=25, seed=0, linear=True, jac_fn=_jac_fn, spacing="uniform",
    )
    err_log = np.max(np.abs(sol_log["predict"](t_query)[:, 0] - y1_exact))
    err_uniform = np.max(np.abs(sol_uniform["predict"](t_query)[:, 0] - y1_exact))
    assert err_log < 1e-6
    assert err_uniform > err_log * 10


def test_extra_param_reuses_features_and_shifts_solution():
    # Same seed -> same (w, w_param, b); only the effective bias (and hence
    # the fit) should change with extra_param. Uses a resolution adequate for
    # tight accuracy (per test_stiff_chain_matches_closed_form_and_scipy's
    # own tuning) so this test isolates extra_param's effect from ordinary
    # under-resolution -- NOT n_subdomains=4, which by itself already gives
    # only ~0.2 accuracy for this stiffness ratio regardless of extra_param.
    common = dict(n_subdomains=30, first_width=1e-11, n_basis=25, n_collocation=30,
                   seed=2, linear=True, jac_fn=_jac_fn, param_range=(-1.0, 1.0))
    sol_a = solve_xtfc_subdomains(_rate_fn, Y0, T_SPAN, extra_param=0.0, **common)
    sol_b = solve_xtfc_subdomains(_rate_fn, Y0, T_SPAN, extra_param=0.8, **common)

    assert np.array_equal(sol_a["w"], sol_b["w"])
    assert np.array_equal(sol_a["w_param"], sol_b["w_param"])
    assert not np.array_equal(sol_a["b_eff"], sol_b["b_eff"])
    # Both should still solve the SAME ODE accurately (extra_param doesn't
    # change the physics here, just the internal feature bias) -- a sanity
    # check that folding extra_param into b_eff didn't break correctness.
    t_query = np.linspace(0.0, sol_a["boundaries"][-1], 10)
    y1_exact, _ = _y_exact(t_query)
    for sol in (sol_a, sol_b):
        err = np.max(np.abs(sol["predict"](t_query)[:, 0] - y1_exact))
        assert err < 5e-3


def test_invalid_spacing_raises():
    with pytest.raises(ValueError):
        solve_xtfc_subdomains(_rate_fn, Y0, T_SPAN, spacing="quadratic")


def test_registered_in_solver_registry_and_forward_runs():
    register_all()
    assert "xtfc_subdomain" in SolverRegistry.list()
    spec = SolverRegistry.spec("xtfc_subdomain")
    assert spec.family == "ode"

    solver = XTFCSubdomainSolver(n_subdomains=6, first_width=1e-9, n_basis=15, n_collocation=20,
                                  linear=True)
    out = solver.forward(_rate_fn, Y0, T_SPAN, jac_fn=_jac_fn, n_query=20)
    assert out.result.shape == (20, 2)
    assert "residual" in out.losses
