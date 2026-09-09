"""
Validation tests for the REBOUND N-body solver (`nbody_rebound.py`).

REBOUND (Rein & Liu 2012, A&A 537, A128) is a real, widely-cited
open-source N-body integration library. These tests exercise the *real*
`rebound.Simulation` machinery via `NBodySolver`/`SolverRegistry`, not a
mock:

  1. test_kepler_two_body_period_closure / test_kepler_two_body_energy_conservation
     A real Earth-Sun-like two-body Kepler problem (mass ratio
     m_earth/m_sun = 3.003e-6, semi-major axis a=1 in G=1 dimensionless
     units), integrated with IAS15 (REBOUND's adaptive, high-precision
     Gauss-Radau15 integrator) for one full orbital period. Checks the
     standard N-body correctness criteria: the body returns close to its
     starting position after one period, and total energy is conserved to
     within IAS15's expected near-machine-precision tolerance.

  2. test_registry_build
     Confirms `SolverRegistry.build("nbody_rebound", ...)` constructs and
     runs a working solver through the real registry (not direct
     construction), i.e. that `nbody_rebound` is wired into
     `_ALL_SOLVER_MODULES` / `register_all()`.

  3. test_cr3bp_l4_stays_fixed_in_rotating_frame
     Cross-validates the new numerical N-body solver against this repo's
     existing *analytical* CR3BP content in
     `pinneapple_physics/pde_environment/presets/astrophysics.py`
     (`cr3bp_lagrange_points`). Seeds a REBOUND simulation with two
     massive primaries (Earth-Moon mass ratio mu, in circular corotation)
     plus a massless test particle placed exactly at the analytically
     computed equilateral L4 point, moving with the same corotation
     velocity. A full (non-restricted) N-body integration of this exact
     configuration is an exact solution of the CR3BP, so after
     transforming the test particle's position back into the rotating
     (synodic) frame, it should remain essentially exactly at L4 for the
     entire synodic period -- this is checked to a tight tolerance.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

rebound = pytest.importorskip("rebound", reason="rebound is an optional dependency for NBodySolver")

from pinneapple_simulation.numerical_solvers.nbody_rebound import NBodySolver
from pinneapple_simulation.numerical_solvers.registry import SolverRegistry, register_all
from pinneapple_physics.pde_environment.presets.astrophysics import cr3bp_lagrange_points


# ===========================================================================
# Two-body Kepler problem (Earth-Sun-like), IAS15
# ===========================================================================

M_SUN = 1.0
M_EARTH = 3.003e-6   # real Earth/Sun mass ratio
A_AU = 1.0           # 1 AU, dimensionless (G=1) units


def _earth_sun_bodies():
    return [
        {"m": M_SUN, "x": 0.0, "y": 0.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
        {"m": M_EARTH, "a": A_AU, "e": 0.0},
    ]


def _integrate_one_period():
    solver = NBodySolver(bodies=_earth_sun_bodies(), integrator="ias15", G=1.0)
    sim0 = solver._build_simulation()
    period = sim0.particles[1].P  # REBOUND's own Kepler-element period for this mass pair
    # Sanity: for m_earth << m_sun and a=1, G=1, period should be close to 2*pi.
    assert period == pytest.approx(2.0 * math.pi, rel=1e-5)
    out = solver.forward(t_end=period, n_steps=2000)
    return out, period


def test_kepler_two_body_period_closure():
    """After exactly one orbital period, the planet returns close to its
    starting position (standard N-body correctness check)."""
    out, period = _integrate_one_period()
    pos = out.result.numpy()  # (n_t, n_bodies, 3)
    r0 = pos[0, 1]
    rT = pos[-1, 1]
    closure_err = np.linalg.norm(rT - r0)
    # IAS15 is a high-precision adaptive integrator: closure should be tight,
    # not just "roughly the same place". Require < 1e-9 * a (a=1 AU here).
    assert closure_err < 1e-9 * A_AU, f"position did not close: |r(T)-r(0)| = {closure_err:.3e}"


def test_kepler_two_body_energy_conservation():
    """Total energy is conserved to within IAS15's near-machine-precision
    tolerance over one full orbital period."""
    out, period = _integrate_one_period()
    energy = out.extras["energy"].numpy()
    E0 = energy[0]
    rel_drift = np.max(np.abs((energy - E0) / E0))
    # IAS15 conserves energy to ~1e-12 or better for a well-resolved orbit
    # like this one; require a tight but safely-above-machine-noise bound.
    assert rel_drift < 1e-10, f"relative energy drift too large: {rel_drift:.3e}"


# ===========================================================================
# SolverRegistry wiring
# ===========================================================================

def test_registry_build():
    """`nbody_rebound` is reachable through SolverRegistry.build(...), not
    just via direct NBodySolver construction."""
    register_all()
    assert "nbody_rebound" in SolverRegistry.list()
    solver = SolverRegistry.build("nbody_rebound", bodies=_earth_sun_bodies(), integrator="ias15", G=1.0)
    assert isinstance(solver, NBodySolver)
    out = solver.forward(t_end=1.0, n_steps=5)
    assert out.result.shape == (6, 2, 3)
    assert "energy" in out.extras and out.extras["energy"].shape == (6,)


# ===========================================================================
# CR3BP L4 cross-check against astrophysics.py's analytical Lagrange points
# ===========================================================================

def _corotation_velocity(x: float, y: float, n: float = 1.0):
    """Velocity of a point at (x, y) undergoing rigid corotation with
    angular velocity n about the origin (CCW): v = n * z_hat x r."""
    return (-y * n, x * n, 0.0)


def test_cr3bp_l4_stays_fixed_in_rotating_frame():
    """Seed a real full N-body REBOUND simulation (two massive primaries +
    a massless test particle) at the analytically-derived L4 equilateral
    Lagrange point of `cr3bp_lagrange_points` (astrophysics.py), with all
    three bodies in circular corotation. This exact configuration is an
    exact solution of the (restricted) three-body problem, so the massless
    particle should remain essentially exactly at L4 in the rotating frame
    over a full synodic period -- an independent, numerical cross-check of
    the analytical L4 equilibrium against a real N-body integrator."""
    mu = 0.012150585609624  # Earth-Moon mass ratio (matches cr3bp_planar_synodic default)
    lpoints = cr3bp_lagrange_points(mu)
    xL4, yL4 = lpoints["L4"]

    n = 1.0  # synodic angular velocity, dimensionless CR3BP normalization
    x1, y1 = -mu, 0.0
    x2, y2 = 1.0 - mu, 0.0
    v1x, v1y, _ = _corotation_velocity(x1, y1, n)
    v2x, v2y, _ = _corotation_velocity(x2, y2, n)
    v3x, v3y, _ = _corotation_velocity(xL4, yL4, n)

    bodies = [
        {"m": 1.0 - mu, "x": x1, "y": y1, "z": 0.0, "vx": v1x, "vy": v1y, "vz": 0.0},
        {"m": mu, "x": x2, "y": y2, "z": 0.0, "vx": v2x, "vy": v2y, "vz": 0.0},
        {"m": 0.0, "x": xL4, "y": yL4, "z": 0.0, "vx": v3x, "vy": v3y, "vz": 0.0},
    ]
    solver = NBodySolver(bodies=bodies, integrator="ias15", G=1.0, move_to_com=False)

    period_synodic = 2.0 * math.pi / n
    out = solver.forward(t_end=period_synodic, n_steps=2000)
    t = out.extras["t"].numpy()
    pos = out.result.numpy()  # (n_t, 3, 3)

    theta = n * t
    x_in, y_in = pos[:, 2, 0], pos[:, 2, 1]
    x_rot = x_in * np.cos(theta) + y_in * np.sin(theta)
    y_rot = -x_in * np.sin(theta) + y_in * np.cos(theta)

    deviation = np.sqrt((x_rot - xL4) ** 2 + (y_rot - yL4) ** 2)
    # Exact solution of the restricted 3-body problem -> deviation should be
    # essentially at IAS15's numerical-precision floor, not merely "small".
    assert deviation.max() < 1e-8, f"L4 test particle drifted in rotating frame: max |dev| = {deviation.max():.3e}"
