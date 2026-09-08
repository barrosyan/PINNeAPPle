"""Tests for the 3-D vorticity, enstrophy and dissipation-rate functions
added to pinneapple_tools/visualization/vortex.py.

Validated against analytically-known flow fields where the exact answer is
computable by hand:
  - solid-body rotation u=(-Omega*y, Omega*x, 0): pure rotation, zero strain,
    vorticity_z = 2*Omega everywhere, dissipation = 0.
  - simple shear u=(gamma*y, 0, 0): pure strain (S12=gamma/2), zero net
    rotation about x/y, vorticity_z = -gamma, dissipation = nu*gamma**2.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_tools.visualization.vortex import (
    compute_dissipation_rate,
    compute_enstrophy,
    compute_q_criterion_3d,
    compute_strain_rate_tensor_3d,
    compute_vorticity_2d,
    compute_vorticity_3d,
)

N = 16


def _grid():
    x = np.arange(N, dtype=float)
    return np.meshgrid(x, x, x, indexing="ij")


def _interior_slice():
    """np.gradient is only 2nd-order accurate away from the domain edges;
    compare in the interior to avoid one-sided boundary stencils."""
    return (slice(2, -2), slice(2, -2), slice(2, -2))


# ---------------------------------------------------------------------------
# compute_vorticity_3d
# ---------------------------------------------------------------------------

def test_vorticity_3d_solid_body_rotation():
    X, Y, Z = _grid()
    Omega = 0.4
    u = -Omega * Y
    v = Omega * X
    w = np.zeros_like(X)

    omega = compute_vorticity_3d(u, v, w)
    assert omega.shape == (3, N, N, N)

    interior = _interior_slice()
    omega_x, omega_y, omega_z = omega[0][interior], omega[1][interior], omega[2][interior]
    assert np.allclose(omega_x, 0.0, atol=1e-8)
    assert np.allclose(omega_y, 0.0, atol=1e-8)
    assert np.allclose(omega_z, 2.0 * Omega, atol=1e-8)


def test_vorticity_3d_simple_shear():
    X, Y, Z = _grid()
    gamma = 0.7
    u = gamma * Y
    v = np.zeros_like(X)
    w = np.zeros_like(X)

    omega = compute_vorticity_3d(u, v, w)
    interior = _interior_slice()
    assert np.allclose(omega[0][interior], 0.0, atol=1e-8)
    assert np.allclose(omega[1][interior], 0.0, atol=1e-8)
    assert np.allclose(omega[2][interior], -gamma, atol=1e-8)


# ---------------------------------------------------------------------------
# compute_strain_rate_tensor_3d / compute_q_criterion_3d consistency
# ---------------------------------------------------------------------------

def test_strain_rate_tensor_3d_zero_for_rigid_rotation():
    """Pure rotation carries zero strain by definition."""
    X, Y, Z = _grid()
    Omega = 0.4
    u = -Omega * Y
    v = Omega * X
    w = np.zeros_like(X)

    S11, S22, S33, S12, S13, S23 = compute_strain_rate_tensor_3d(u, v, w)
    interior = _interior_slice()
    for S in (S11, S22, S33, S12, S13, S23):
        assert np.allclose(S[interior], 0.0, atol=1e-8)


def test_strain_rate_tensor_3d_simple_shear():
    X, Y, Z = _grid()
    gamma = 0.7
    u = gamma * Y
    v = np.zeros_like(X)
    w = np.zeros_like(X)

    S11, S22, S33, S12, S13, S23 = compute_strain_rate_tensor_3d(u, v, w)
    interior = _interior_slice()
    assert np.allclose(S11[interior], 0.0, atol=1e-8)
    assert np.allclose(S22[interior], 0.0, atol=1e-8)
    assert np.allclose(S33[interior], 0.0, atol=1e-8)
    assert np.allclose(S12[interior], 0.5 * gamma, atol=1e-8)
    assert np.allclose(S13[interior], 0.0, atol=1e-8)
    assert np.allclose(S23[interior], 0.0, atol=1e-8)


def test_q_criterion_3d_positive_for_pure_rotation_negative_for_pure_shear():
    X, Y, Z = _grid()
    Omega = 0.4
    u_rot, v_rot, w_rot = -Omega * Y, Omega * X, np.zeros_like(X)
    Q_rot = compute_q_criterion_3d(u_rot, v_rot, w_rot)
    interior = _interior_slice()
    assert np.all(Q_rot[interior] > 0)  # rotation dominates (zero strain)

    gamma = 0.7
    u_sh, v_sh, w_sh = gamma * Y, np.zeros_like(X), np.zeros_like(X)
    Q_sh = compute_q_criterion_3d(u_sh, v_sh, w_sh)
    # Simple shear splits evenly into equal rotation + strain -> Q ~ 0
    assert np.allclose(Q_sh[interior], 0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# compute_enstrophy
# ---------------------------------------------------------------------------

def test_enstrophy_2d_scalar_field():
    omega_z = np.full((5, 5), 2.0)
    # 0.5 * sum(omega^2) = 0.5 * 25 * 4 = 50
    assert compute_enstrophy(omega_z) == pytest.approx(50.0)


def test_enstrophy_3d_vector_field_matches_manual_sum():
    rng = np.random.default_rng(0)
    vort = rng.normal(size=(3, 4, 4, 4))
    expected = 0.5 * np.sum(vort ** 2)
    assert compute_enstrophy(vort) == pytest.approx(expected)


def test_enstrophy_solid_body_rotation_matches_hand_calc():
    X, Y, Z = _grid()
    Omega = 0.5
    u, v, w = -Omega * Y, Omega * X, np.zeros_like(X)
    omega = compute_vorticity_3d(u, v, w)
    # Exact everywhere for this field: omega_z = 2*Omega, omega_x=omega_y=0
    expected = 0.5 * np.sum((2.0 * Omega) ** 2 * np.ones_like(X))
    assert compute_enstrophy(omega) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# compute_dissipation_rate
# ---------------------------------------------------------------------------

def test_dissipation_rate_zero_for_rigid_rotation():
    X, Y, Z = _grid()
    Omega = 0.4
    u, v, w = -Omega * Y, Omega * X, np.zeros_like(X)
    S = compute_strain_rate_tensor_3d(u, v, w)
    eps = compute_dissipation_rate(S, nu=0.05)
    interior = _interior_slice()
    assert np.allclose(eps[interior], 0.0, atol=1e-8)


def test_dissipation_rate_simple_shear_matches_hand_calc():
    """eps = 2*nu*S_ij*S_ij = 2*nu*2*S12^2 = 4*nu*(gamma/2)^2 = nu*gamma^2."""
    X, Y, Z = _grid()
    gamma = 0.6
    nu = 0.02
    u, v, w = gamma * Y, np.zeros_like(X), np.zeros_like(X)
    S = compute_strain_rate_tensor_3d(u, v, w)
    eps = compute_dissipation_rate(S, nu=nu)
    interior = _interior_slice()
    expected = nu * gamma ** 2
    assert np.allclose(eps[interior], expected, atol=1e-8)
