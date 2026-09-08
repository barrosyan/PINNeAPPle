"""Tests for ``pinneapple_analysis.state_estimation.kalman`` (EKF / EnKF).

This module was relocated from
``pinneapple_systems/digital_twin/assimilation/kalman.py`` so it can be
used standalone for sequential state estimation / data assimilation
outside a full ``DigitalTwin`` (see that package's docstring). These
tests exercise the promoted, standalone module directly with simple,
analytically-tractable linear systems -- a static state observed with
noise, and a 1D constant-velocity kinematic model -- checking that both
filters converge toward the true state and that estimated covariance
shrinks as more observations are assimilated.

``tests/test_breadth_six_packages.py`` separately covers the
backward-compatible shim at the old import path
(``pinneapple_systems.digital_twin.assimilation.kalman``); this module
does not duplicate that, beyond a short re-export identity check.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.state_estimation.kalman import (
    ExtendedKalmanFilter,
    EnsembleKalmanFilter,
)


def _identity(x: np.ndarray) -> np.ndarray:
    return x.copy()


# ---------------------------------------------------------------------------
# Static (constant) state, observed with noise
# ---------------------------------------------------------------------------

def test_ekf_converges_to_true_static_state():
    """EKF fusing repeated noisy observations of a fixed 2D state should
    converge close to the true value, well inside raw observation noise."""
    rng = np.random.default_rng(42)
    true_state = np.array([3.0, -2.0])
    obs_std = 0.5

    Q = np.eye(2) * 1e-6  # state is truly constant -> ~no process noise
    R = np.eye(2) * obs_std ** 2

    ekf = ExtendedKalmanFilter(n_state=2, n_obs=2, f=_identity, h=_identity, Q=Q, R=R)
    ekf.initialize(np.zeros(2), P0=np.eye(2) * 10.0)

    for _ in range(200):
        y = true_state + rng.normal(scale=obs_std, size=2)
        ekf.step(y)

    err = np.linalg.norm(ekf.x - true_state)
    assert err < 0.15, f"EKF estimate {ekf.x} did not converge to {true_state} (err={err})"
    assert np.all(np.isfinite(ekf.P))


def test_ekf_covariance_shrinks_with_more_observations():
    """Posterior covariance trace should shrink monotonically-ish as more
    observations are assimilated (uncertainty reduction)."""
    rng = np.random.default_rng(0)
    true_state = np.array([1.0, 1.0])
    Q = np.eye(2) * 1e-6
    R = np.eye(2) * 0.25

    ekf = ExtendedKalmanFilter(n_state=2, n_obs=2, f=_identity, h=_identity, Q=Q, R=R)
    ekf.initialize(np.zeros(2), P0=np.eye(2) * 10.0)

    trace0 = np.trace(ekf.P)
    for _ in range(50):
        y = true_state + rng.normal(scale=0.5, size=2)
        ekf.step(y)
    trace_after = np.trace(ekf.P)

    assert trace_after < trace0, "covariance should shrink after many observations"
    assert trace_after < 0.1 * trace0


# ---------------------------------------------------------------------------
# 1D constant-velocity kinematic model (linear, analytically tractable)
# ---------------------------------------------------------------------------

def _make_constant_velocity_model(dt: float = 1.0):
    """x = [position, velocity]; observe position only."""

    def f(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] + dt * x[1], x[1]])

    def h(x: np.ndarray) -> np.ndarray:
        return np.array([x[0]])

    return f, h


def test_ekf_tracks_constant_velocity_trajectory():
    rng = np.random.default_rng(7)
    dt = 1.0
    true_velocity = 2.0
    n_steps = 60
    obs_std = 1.0

    f, h = _make_constant_velocity_model(dt)
    Q = np.diag([1e-4, 1e-4])
    R = np.array([[obs_std ** 2]])

    ekf = ExtendedKalmanFilter(n_state=2, n_obs=1, f=f, h=h, Q=Q, R=R)
    ekf.initialize(np.array([0.0, 0.0]), P0=np.diag([1.0, 4.0]))

    true_pos = 0.0
    pos_errors = []
    for _ in range(n_steps):
        true_pos += dt * true_velocity
        y = np.array([true_pos + rng.normal(scale=obs_std)])
        out = ekf.step(y)
        pos_errors.append(abs(out["x"][0] - true_pos))

    # Velocity should be identified even though only position is observed.
    assert abs(ekf.x[1] - true_velocity) < 0.3, ekf.x
    # Late-trajectory position tracking error should be well below raw
    # observation noise (the filter is using the velocity state to help).
    assert np.mean(pos_errors[-10:]) < obs_std


# ---------------------------------------------------------------------------
# EnKF
# ---------------------------------------------------------------------------

def test_enkf_converges_to_true_static_state():
    rng_seed = 3
    true_state = np.array([3.0, -2.0])
    obs_std = 0.5

    Q = np.eye(2) * 1e-6
    R = np.eye(2) * obs_std ** 2

    enkf = EnsembleKalmanFilter(
        n_state=2, n_obs=2, f=_identity, h=_identity, Q=Q, R=R,
        n_ens=100, inflation=1.0, seed=rng_seed,
    )
    enkf.initialize(np.zeros(2), P0=np.eye(2) * 10.0)

    rng = np.random.default_rng(123)
    for _ in range(200):
        y = true_state + rng.normal(scale=obs_std, size=2)
        enkf.step(y)

    err = np.linalg.norm(enkf.mean - true_state)
    assert err < 0.3, f"EnKF mean {enkf.mean} did not converge to {true_state} (err={err})"
    assert np.all(np.isfinite(enkf.covariance))


def test_enkf_covariance_shrinks_with_more_observations():
    rng = np.random.default_rng(11)
    true_state = np.array([1.0, 1.0])
    Q = np.eye(2) * 1e-6
    R = np.eye(2) * 0.25

    enkf = EnsembleKalmanFilter(
        n_state=2, n_obs=2, f=_identity, h=_identity, Q=Q, R=R,
        n_ens=100, inflation=1.0, seed=1,
    )
    enkf.initialize(np.zeros(2), P0=np.eye(2) * 10.0)

    trace0 = np.trace(enkf.covariance)
    for _ in range(50):
        y = true_state + rng.normal(scale=0.5, size=2)
        enkf.step(y)
    trace_after = np.trace(enkf.covariance)

    assert trace_after < trace0
    assert trace_after < 0.2 * trace0


# ---------------------------------------------------------------------------
# Backward-compatible shim
# ---------------------------------------------------------------------------

def test_digital_twin_shim_reexports_same_classes():
    """The old import path must keep working and resolve to the exact
    same classes now living in pinneapple_analysis.state_estimation."""
    from pinneapple_systems.digital_twin.assimilation.kalman import (
        ExtendedKalmanFilter as ShimEKF,
        EnsembleKalmanFilter as ShimEnKF,
    )

    assert ShimEKF is ExtendedKalmanFilter
    assert ShimEnKF is EnsembleKalmanFilter
