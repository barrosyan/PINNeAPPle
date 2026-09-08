"""Tests for the D3Q19 LBMSolver3D bounce-back obstacles and Smagorinsky LES
closure added in pinneapple_simulation/numerical_solvers/lbm.py.

LBMSolver3D previously only supported periodic BCs. These tests validate the
new features generalised from the already-tested D2Q9 LBMSolver:
  - full-way bounce-back at solid nodes (D3Q19)
  - local-omega Smagorinsky LES adjustment (D3Q19)
  - stability warnings (Ma > 0.3, tau < 0.6)
"""
from __future__ import annotations

import warnings

import torch
import pytest

from pinneapple_simulation.numerical_solvers.lbm import (
    LBMSolver3D,
    airfoil_naca_mask_3d,
    _bounce_back_3d,
    _d3q19_tensors,
    _equilibrium_3d,
    _lbm_step_3d,
    _smagorinsky_omega_3d,
)


def _uniform_flow_f(nx, ny, nz, u_in, device=torch.device("cpu")):
    c, w, _ = _d3q19_tensors(device)
    cx, cy, cz = c[:, 0], c[:, 1], c[:, 2]
    rho = torch.ones(nx, ny, nz, device=device)
    ux = torch.full_like(rho, u_in)
    uy = torch.zeros_like(rho)
    uz = torch.zeros_like(rho)
    return _equilibrium_3d(rho, ux, uy, uz, cx, cy, cz, w), (cx, cy, cz, w)


# ---------------------------------------------------------------------------
# Bounce-back
# ---------------------------------------------------------------------------

def test_bounce_back_3d_zero_velocity_at_rest_solid_node():
    """
    A solid node whose own pre-collision state is the rest equilibrium
    (rho=1, u=0) collides to a direction-symmetric f_post (opposite D3Q19
    directions share equal weights, so feq_i(u=0) == feq_opp[i](u=0)).
    Bounce-back (f_new[i] = f_post[opp[i]]) leaves such a symmetric state
    unchanged, so the macroscopic velocity at that node is exactly zero --
    the defining property of full-way bounce-back no-slip walls.
    """
    nx = ny = nz = 8
    device = torch.device("cpu")
    # Uniform freestream everywhere...
    f, (cx, cy, cz, w) = _uniform_flow_f(nx, ny, nz, u_in=0.05, device=device)
    # ...except the solid node itself starts at rest equilibrium.
    rest_feq, _ = _uniform_flow_f(1, 1, 1, u_in=0.0, device=device)
    f[:, 4, 4, 4] = rest_feq[:, 0, 0, 0]

    solid = torch.zeros(nx, ny, nz, dtype=torch.bool)
    solid[4, 4, 4] = True
    _, _, opp = _d3q19_tensors(device)

    f_new = _lbm_step_3d(f, omega=1.0, solid=solid, cx=cx, cy=cy, cz=cz, w=w, opp=opp)

    Q = f_new.shape[0]
    for i in range(Q):
        assert torch.allclose(
            f_new[i, 4, 4, 4], f_new[opp[i], 4, 4, 4], atol=1e-6
        ), f"direction {i} not bounced back symmetrically at solid node"

    rho = f_new[:, 4, 4, 4].sum()
    ux = (f_new[:, 4, 4, 4] * cx).sum() / rho
    uy = (f_new[:, 4, 4, 4] * cy).sum() / rho
    uz = (f_new[:, 4, 4, 4] * cz).sum() / rho
    assert abs(float(ux)) < 1e-6
    assert abs(float(uy)) < 1e-6
    assert abs(float(uz)) < 1e-6


def test_bounce_back_3d_reverses_velocity_sign_for_moving_solid_node():
    """
    Algebraic identity of full-way bounce-back: since opp() pairs a
    direction with its negation (c_opp = -c), overwriting f_new[i] with
    f_post[opp[i]] at a solid node exactly negates that node's own
    post-collision velocity (rho is preserved: opp is a permutation of the
    19 directions, so sum_i f_post[opp[i]] == sum_i f_post[i]).
    """
    nx = ny = nz = 4
    device = torch.device("cpu")
    f, (cx, cy, cz, w) = _uniform_flow_f(nx, ny, nz, u_in=0.05, device=device)
    solid = torch.zeros(nx, ny, nz, dtype=torch.bool)
    solid[1, 1, 1] = True
    _, _, opp = _d3q19_tensors(device)

    rho_pre = f[:, 1, 1, 1].sum()
    ux_pre = (f[:, 1, 1, 1] * cx).sum() / rho_pre

    f_new = _lbm_step_3d(f, omega=1.0, solid=solid, cx=cx, cy=cy, cz=cz, w=w, opp=opp)

    rho_post = f_new[:, 1, 1, 1].sum()
    ux_post = (f_new[:, 1, 1, 1] * cx).sum() / rho_post
    assert torch.allclose(rho_post, rho_pre, atol=1e-5)
    assert torch.allclose(ux_post, -ux_pre, atol=1e-5)


def test_bounce_back_3d_helper_matches_reversed_populations():
    """_bounce_back_3d directly: streamed values at solid cells must be
    overwritten with the pre-streaming opposite-direction populations."""
    nx = ny = nz = 4
    device = torch.device("cpu")
    _, w, opp = _d3q19_tensors(device)
    Q = w.shape[0]
    f_post = torch.arange(Q * nx * ny * nz, dtype=torch.float32).reshape(Q, nx, ny, nz)
    f_streamed = torch.zeros_like(f_post)  # distinguishable from f_post

    solid = torch.zeros(nx, ny, nz, dtype=torch.bool)
    solid[1, 1, 1] = True

    out = _bounce_back_3d(f_post, f_streamed.clone(), solid, opp)
    for i in range(Q):
        assert torch.equal(out[i, 1, 1, 1], f_post[opp[i], 1, 1, 1])
    # Non-solid cells must be untouched (still the zeroed f_streamed values)
    assert torch.equal(out[:, 0, 0, 0], f_streamed[:, 0, 0, 0])


def test_lbm_solver_3d_forward_with_obstacle_runs_and_stays_finite():
    """End-to-end smoke test: LBMSolver3D with a solid block obstacle should
    run without NaNs/Infs and should damp velocity inside the obstacle."""
    nx = ny = nz = 10
    solid = torch.zeros(nx, ny, nz, dtype=torch.bool)
    solid[4:6, 4:6, 4:6] = True

    solver = LBMSolver3D(nx=nx, ny=ny, nz=nz, Re=50.0, u_in=0.03, obstacle_mask=solid)
    out = solver.forward(steps=20, save_every=10)

    ux, uy, uz = out.extras["ux"], out.extras["uy"], out.extras["uz"]
    assert torch.isfinite(ux).all()
    assert torch.isfinite(uy).all()
    assert torch.isfinite(uz).all()
    # Velocity magnitude strictly inside the solid block stays small (no-slip)
    vel_mag_obstacle = out.extras["vel_mag"][4:6, 4:6, 4:6]
    assert vel_mag_obstacle.max() < 0.03


# ---------------------------------------------------------------------------
# Smagorinsky LES
# ---------------------------------------------------------------------------

def test_smagorinsky_omega_3d_responds_to_strain_rate():
    """
    Higher local strain (non-equilibrium stress) should push tau_eff (and
    hence omega) away from the base BGK value -- omega field must differ
    from the uniform omega0 once Cs>0 and there's non-zero shear, and must
    stay within the [0, 2) stability band.
    """
    nx = ny = nz = 6
    device = torch.device("cpu")
    c, w, _ = _d3q19_tensors(device)
    cx, cy, cz = c[:, 0], c[:, 1], c[:, 2]

    # Build a sheared velocity field: ux depends on y -> non-zero strain
    rho = torch.ones(nx, ny, nz)
    y = torch.arange(ny, dtype=torch.float32).view(1, ny, 1)
    ux = 0.02 * y.expand(nx, ny, nz)
    uy = torch.zeros(nx, ny, nz)
    uz = torch.zeros(nx, ny, nz)
    feq = _equilibrium_3d(rho, ux, uy, uz, cx, cy, cz, w)

    # Non-equilibrium perturbation on top of feq so f != feq (else Pi_neq = 0)
    f = feq + 0.01 * torch.randn_like(feq)

    omega0 = 1.2
    omega_field = _smagorinsky_omega_3d(f, feq, rho, omega0, Cs=0.15, cx=cx, cy=cy, cz=cz)

    assert omega_field.shape == (nx, ny, nz)
    assert torch.isfinite(omega_field).all()
    assert (omega_field > 0).all() and (omega_field < 2.0).all()
    # LES adjustment must actually move omega away from the bare BGK value
    assert not torch.allclose(omega_field, torch.full_like(omega_field, omega0), atol=1e-4)


def test_lbm_step_3d_with_les_stays_finite():
    """A full step with Cs>0 should run and stay numerically finite."""
    nx = ny = nz = 6
    device = torch.device("cpu")
    f, (cx, cy, cz, w) = _uniform_flow_f(nx, ny, nz, u_in=0.05, device=device)
    _, _, opp = _d3q19_tensors(device)

    f_new = _lbm_step_3d(f, omega=1.0, solid=None, cx=cx, cy=cy, cz=cz, w=w, opp=opp, Cs=0.15)
    assert torch.isfinite(f_new).all()


def test_lbm_solver_3d_with_les_runs():
    solver = LBMSolver3D(nx=8, ny=8, nz=8, Re=100.0, u_in=0.04, Cs=0.15)
    out = solver.forward(steps=10, save_every=5)
    assert torch.isfinite(out.extras["ux"]).all()


# ---------------------------------------------------------------------------
# Stability warnings
# ---------------------------------------------------------------------------

def test_lbm_solver_3d_warns_on_high_mach_number():
    with pytest.warns(UserWarning, match="Ma="):
        LBMSolver3D(nx=16, ny=16, nz=16, Re=1000.0, u_in=0.3)


def test_lbm_solver_3d_warns_near_stability_limit():
    # Small nx/Re combination that drives tau close to 0.5
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        LBMSolver3D(nx=8, ny=8, nz=8, Re=1000.0, u_in=0.02)
    messages = [str(w.message) for w in caught]
    assert any("stability limit" in m for m in messages)


# ---------------------------------------------------------------------------
# Obstacle helper
# ---------------------------------------------------------------------------

def test_airfoil_naca_mask_3d_is_extrusion_of_2d_mask():
    nx, ny, nz, chord = 40, 20, 6, 16
    mask3d = airfoil_naca_mask_3d(nx, ny, nz, chord=chord)
    assert mask3d.shape == (nx, ny, nz)
    assert mask3d.dtype == torch.bool
    # Every z-slice must be identical (uniform extrusion along span)
    for z in range(1, nz):
        assert torch.equal(mask3d[:, :, 0], mask3d[:, :, z])
    assert mask3d.any()  # non-trivial mask


def test_airfoil_naca_mask_3d_respects_span_bounds():
    nx, ny, nz, chord = 40, 20, 8, 16
    mask3d = airfoil_naca_mask_3d(nx, ny, nz, chord=chord, z0=2, z1=5)
    assert not mask3d[:, :, 0].any()
    assert not mask3d[:, :, 6].any()
    assert mask3d[:, :, 3].any()
