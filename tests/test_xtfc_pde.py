"""Validates pinneapple_simulation.numerical_solvers.xtfc_pde against
independent references:
  - heat_1d: the closed-form solution u(x,t) = exp(-alpha*pi^2*t)*sin(pi*x)
    (exact for the module's default IC/BC on x in [0,1]).
  - burgers_1d: a method-of-lines + RK4 reference integrated independently
    in this test (same style as
    pinneapple_tools.benchmark_suite.tasks.burgers_1d.Burgers1DTask's own
    reference, but at a milder viscosity -- see note below).

Note on burgers_1d's viscosity: the named benchmark's standard nu=0.01/pi
produces a sharp interior layer that a modest fixed-ELM-basis Gauss-Newton
fit (this module's whole design point -- no gradient descent, no adaptive
collocation) cannot resolve tightly without a much larger n_basis/
n_collocation than is practical for a fast test; this is a known, documented
limitation of the exact-least-squares X-TFC approach for shock-like layers,
not unique to this implementation. This test instead validates the SAME
nonlinear Gauss-Newton code path at nu=0.1 (still genuinely nonlinear, no
sharp layer), where convergence and accuracy are both easy to verify tightly.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_simulation.numerical_solvers.xtfc_pde import (
    XTFCPDESolver,
    solve_xtfc_pde,
)
from pinneapple_simulation.numerical_solvers.registry import SolverRegistry, register_all


def test_heat_1d_matches_closed_form_solution():
    alpha = 1.0
    sol = solve_xtfc_pde("heat_1d", n_basis=40, n_collocation_x=20, n_collocation_t=20,
                          seed=1, param=alpha)
    assert sol["converged"]

    x_query = np.linspace(0.0, 1.0, 60)
    for t_val in (0.0, 0.25, 0.5, 1.0):
        t_query = np.full_like(x_query, t_val)
        u_pred = sol["predict"](x_query, t_query)
        u_exact = np.exp(-alpha * np.pi ** 2 * t_val) * np.sin(np.pi * x_query)
        max_err = np.max(np.abs(u_pred - u_exact))
        assert max_err < 2e-3, f"t={t_val}: max abs err {max_err} too large vs closed-form heat solution"


def test_heat_1d_exact_at_ic_and_boundaries_for_any_seed():
    # The IC/BC identities are supposed to hold IDENTICALLY (by construction,
    # not by fitting) regardless of the random ELM feature draw.
    for seed in (0, 7, 99):
        sol = solve_xtfc_pde("heat_1d", n_basis=20, n_collocation_x=15, n_collocation_t=15, seed=seed)
        x_query = np.linspace(0.0, 1.0, 25)
        u_ic = sol["predict"](x_query, np.zeros_like(x_query))
        assert np.allclose(u_ic, np.sin(np.pi * x_query), atol=1e-9)

        t_query = np.linspace(0.0, 1.0, 25)
        u_left = sol["predict"](np.full_like(t_query, 0.0), t_query)
        u_right = sol["predict"](np.full_like(t_query, 1.0), t_query)
        assert np.allclose(u_left, 0.0, atol=1e-9)
        assert np.allclose(u_right, 0.0, atol=1e-9)


def _burgers_mol_rk4_reference(nu: float, x: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    dx = x[1] - x[0]
    u = -np.sin(np.pi * x)
    u[0] = 0.0
    u[-1] = 0.0
    # Explicit-RK4 diffusion stability bound (dt <~ 0.5*dx^2/nu); nu here can
    # be much larger than the Burgers1DTask benchmark's own nu=0.01/pi, so a
    # fixed dt=5e-4 (fine for that much smaller nu) is NOT safe in general --
    # pick dt from the actual nu/dx used by this reference instead.
    dt = min(5e-4, 0.4 * dx ** 2 / nu)
    t_current = 0.0
    out = {0.0: u.copy()}

    def rhs(u_):
        du = np.zeros_like(u_)
        u_x = (u_[2:] - u_[:-2]) / (2.0 * dx)
        u_xx = (u_[2:] - 2.0 * u_[1:-1] + u_[:-2]) / dx ** 2
        du[1:-1] = -u_[1:-1] * u_x + nu * u_xx
        return du

    n_steps = int(np.max(t_eval) / dt) + 2
    next_targets = sorted(t_eval)
    idx = 0
    for _ in range(n_steps):
        if idx >= len(next_targets):
            break
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)
        u = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        u[0] = 0.0
        u[-1] = 0.0
        t_current += dt
        while idx < len(next_targets) and t_current >= next_targets[idx] - 1e-9:
            out[next_targets[idx]] = u.copy()
            idx += 1
    return out


def test_burgers_1d_nonlinear_branch_matches_mol_rk4_reference():
    nu = 0.1
    sol = solve_xtfc_pde("burgers_1d", param=nu, n_basis=60, n_collocation_x=25,
                          n_collocation_t=25, seed=3, max_iter=80, tol=1e-10)
    assert sol["n_iter"] > 1, "Gauss-Newton should take more than one iteration for a nonlinear PDE"

    x = np.linspace(-1.0, 1.0, 300)
    t_eval = [0.3, 0.6, 1.0]
    ref = _burgers_mol_rk4_reference(nu, x, t_eval)

    for t_val in t_eval:
        u_ref = ref[t_val]
        u_pred = sol["predict"](x, np.full_like(x, t_val))
        max_err = np.max(np.abs(u_pred - u_ref))
        assert max_err < 0.05, f"t={t_val}: max abs err {max_err} too large vs MOL/RK4 Burgers reference"


def test_burgers_1d_ic_and_bc_exact_by_construction():
    sol = solve_xtfc_pde("burgers_1d", n_basis=30, n_collocation_x=15, n_collocation_t=15, seed=5)
    x_query = np.linspace(-1.0, 1.0, 30)
    u_ic = sol["predict"](x_query, np.zeros_like(x_query))
    assert np.allclose(u_ic, -np.sin(np.pi * x_query), atol=1e-9)

    t_query = np.linspace(0.0, 1.0, 30)
    u_left = sol["predict"](np.full_like(t_query, -1.0), t_query)
    u_right = sol["predict"](np.full_like(t_query, 1.0), t_query)
    assert np.allclose(u_left, 0.0, atol=1e-9)
    assert np.allclose(u_right, 0.0, atol=1e-9)


def test_invalid_pde_name_raises():
    with pytest.raises(ValueError):
        solve_xtfc_pde("wave_1d")


def test_registered_in_solver_registry_and_forward_runs():
    register_all()
    assert "xtfc_pde" in SolverRegistry.list()
    spec = SolverRegistry.spec("xtfc_pde")
    assert spec.family == "pde"

    solver = XTFCPDESolver(pde="heat_1d", n_basis=20, n_collocation_x=12, n_collocation_t=12)
    out = solver.forward(n_query=15)
    assert out.result.shape == (15, 15)
    assert "residual" in out.losses
    assert out.extras["pde"] == "heat_1d"
