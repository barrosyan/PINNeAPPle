"""Validates pinneapple_simulation.numerical_solvers.discrete_time_pinn:

  1. `stage_derivatives`' forward-mode-AD (torch.func.jvp) stage derivatives
     against ordinary reverse-mode autograd, on an untrained (random-weight)
     network -- a fast, non-flaky check that doesn't depend on training
     convergence at all.
  2. End-to-end training accuracy for both supported presets (`burgers_1d`,
     `allen_cahn_1d`) against the SAME method-of-lines/RK4 reference build
     already used by
     `pinneapple_tools.benchmark_suite.tasks.{burgers_1d,allen_cahn_1d}`
     (reused here directly, not re-derived).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pinneapple_simulation.numerical_solvers.discrete_time_pinn import (
    DiscreteTimePINNSolver,
    DiscreteTimeRKNet,
    discrete_time_rk_residual,
    stage_derivatives,
    train_discrete_time,
)
from pinneapple_simulation.numerical_solvers.irk_gauss_legendre import get_irk_tableau
from pinneapple_simulation.numerical_solvers.registry import SolverRegistry, register_all
from pinneapple_tools.benchmark_suite.tasks.allen_cahn_1d import AllenCahn1DTask
from pinneapple_tools.benchmark_suite.tasks.burgers_1d import Burgers1DTask


def test_stage_derivatives_match_autograd():
    torch.manual_seed(0)
    model = DiscreteTimeRKNet(q=5, hidden_layers=(10, 10))
    x = torch.randn(20, 1)

    U, U_x, U_xx = stage_derivatives(model, x)

    x_ref = x.clone().requires_grad_(True)
    U_ref = model(x_ref)
    grads = [torch.autograd.grad(U_ref[:, k].sum(), x_ref, create_graph=True)[0]
             for k in range(U_ref.shape[1])]
    U_x_ref = torch.cat(grads, dim=1)
    grads2 = [torch.autograd.grad(U_x_ref[:, k].sum(), x_ref, retain_graph=True)[0]
              for k in range(U_x_ref.shape[1])]
    U_xx_ref = torch.cat(grads2, dim=1)

    assert torch.allclose(U, U_ref.detach(), atol=1e-6)
    assert torch.allclose(U_x, U_x_ref.detach(), atol=1e-5)
    assert torch.allclose(U_xx, U_xx_ref.detach(), atol=1e-4)


def test_discrete_time_rk_residual_shapes_and_zero_for_matching_data():
    q = 4
    A_np, b_np, c_np = get_irk_tableau(q)
    A = torch.tensor(A_np, dtype=torch.float32)
    b = torch.tensor(b_np, dtype=torch.float32)

    torch.manual_seed(0)
    model = DiscreteTimeRKNet(q=q, hidden_layers=(8, 8))
    x = torch.linspace(-1.0, 1.0, 15).reshape(-1, 1)

    # If u_n_data is defined AS the network's own (untrained) prediction run
    # backward through the tableau with dt=0, the residual must be exactly
    # U_i - u_n_data with N_j irrelevant (dt=0 kills the RK term) -- a shape
    # and wiring sanity check independent of any PDE.
    nonlinear_op = lambda u, ux, uxx: torch.zeros_like(u)
    U, _, _ = stage_derivatives(model, x)
    u_n_data = U[:, -1:].detach()  # arbitrary reference values, shape (N,1)

    res_stage, res_final = discrete_time_rk_residual(model, x, u_n_data, dt=0.0, A=A, b=b,
                                                       nonlinear_op=nonlinear_op)
    assert res_stage.shape == (15, q)
    assert res_final.shape == (15, 1)
    # dt=0 -> res_stage[:,i] == U_i - u_n_data, res_final == U_final - u_n_data
    assert torch.allclose(res_final, U[:, -1:] - u_n_data, atol=1e-6)


def test_burgers_1d_discrete_step_matches_mol_rk4_reference():
    task = Burgers1DTask()
    x_n = task._ref_x[::3]
    u_n = -np.sin(np.pi * x_n)
    dt = 0.2

    result = train_discrete_time(
        "burgers_1d", x_n, u_n, dt, q=25, hidden_layers=(30, 30, 30),
        max_iter=250, bc_weight=20.0, seed=0,
    )

    t_idx = int(np.argmin(np.abs(task._ref_t - dt)))
    u_ref = task._ref_u[t_idx]
    u_pred = result.predict_final(task._ref_x)
    err = np.abs(u_pred - u_ref)

    assert np.max(err) < 0.1, f"burgers_1d max abs err {np.max(err)} too large"
    assert np.sqrt(np.mean(err ** 2)) < 0.05, f"burgers_1d rms err {np.sqrt(np.mean(err**2))} too large"


def test_allen_cahn_1d_discrete_step_matches_mol_rk4_reference():
    task = AllenCahn1DTask()
    x_n = task._ref_x[::3]
    u_n = x_n ** 2 * np.cos(np.pi * x_n)
    dt = 0.1

    result = train_discrete_time(
        "allen_cahn_1d", x_n, u_n, dt, q=25, hidden_layers=(30, 30, 30),
        max_iter=250, bc_weight=20.0, seed=0,
    )

    t_idx = int(np.argmin(np.abs(task._ref_t - dt)))
    u_ref = task._ref_u[t_idx]
    u_pred = result.predict_final(task._ref_x)
    err = np.abs(u_pred - u_ref)

    assert np.max(err) < 0.1, f"allen_cahn_1d max abs err {np.max(err)} too large"
    assert np.sqrt(np.mean(err ** 2)) < 0.05, f"allen_cahn_1d rms err {np.sqrt(np.mean(err**2))} too large"


def test_boundary_condition_enforced_at_all_stages():
    task = Burgers1DTask()
    x_n = task._ref_x[::4]
    u_n = -np.sin(np.pi * x_n)
    result = train_discrete_time(
        "burgers_1d", x_n, u_n, dt=0.2, q=10, hidden_layers=(20, 20),
        max_iter=150, bc_weight=50.0, seed=0,
    )
    x_bc = torch.tensor([[-1.0], [1.0]])
    with torch.no_grad():
        u_bc = result.model(x_bc)  # (2, q+1) -- all stages, not just the final one
    assert torch.allclose(u_bc, torch.zeros_like(u_bc), atol=0.05)


def test_invalid_pde_name_raises():
    with pytest.raises(ValueError):
        train_discrete_time("wave_1d", np.array([0.0]), np.array([0.0]), dt=0.1)


def test_registered_in_solver_registry_and_forward_runs():
    register_all()
    assert "discrete_time_pinn" in SolverRegistry.list()
    spec = SolverRegistry.spec("discrete_time_pinn")
    assert spec.family == "pde"

    task = Burgers1DTask()
    x_n = task._ref_x[::6]
    u_n = -np.sin(np.pi * x_n)
    solver = DiscreteTimePINNSolver(pde="burgers_1d", q=8, hidden_layers=(15, 15),
                                     max_iter=60, n_boundary=5)
    out = solver.forward(x_n, u_n, dt=0.1, n_query=20)
    assert out.result.shape == (20,)
    assert "loss" in out.losses
