"""Independent, many-orbit validation of `satellite_j2_perturbation`'s
cited secular drift-rate formulas (nodal regression, apsidal precession)
-- explicitly tracked as NOT done in ROADMAP_PHYSICS_AI_HUB.md's
astrophysics section when the preset/compiler kind were added, because
tests/test_astrophysics_validation.py only checks the INSTANTANEOUS
residual (no closed-form trajectory exists to check against directly).

Method: numerically integrate the SAME J2 acceleration formula used in
`compile_problem`'s "satellite_j2_perturbation" branch (reproduced here
independently with `scipy.integrate.solve_ivp`, not imported from
compile.py, so this is a genuine independent check rather than testing
code against itself) over several hundred orbits, extract the osculating
right ascension of ascending node (RAAN) and argument of perigee from the
angular-momentum and eccentricity vectors at each output time, fit their
linear secular trend, and compare against the literature-cited formulas
(Vallado, 2013, Ch. 9):
    Omega_dot = -1.5 * n * J2 * (Re/p)^2 * cos(i)
    omega_dot =  0.75 * n * J2 * (Re/p)^2 * (5*cos(i)^2 - 1)

Both were found to agree with the numerically integrated secular trend to
better than 1% (see the printed values in each test) -- strong evidence
the derived acceleration formula (compile.py's satellite_j2_perturbation
branch) and the cited literature secular-rate formulas are mutually
consistent.

Caveat, stated honestly: apsidal precession's formula has a zero-crossing
at the critical inclination (~63.435 deg, where 5cos^2(i)-1=0) -- a naive
relative-error comparison blows up near that inclination (confirmed while
building this test: ~45% "error" at i=63.4 deg, vs. ~0.5% at i=30 deg,
purely a near-zero-denominator artifact, not a real discrepancy). The
inclination used below is chosen well away from the critical value for
exactly this reason.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy.integrate import solve_ivp  # noqa: E402


def _j2_rhs(mu, J2, Re):
    def rhs(t, s):
        x, y, z, vx, vy, vz = s
        r2 = x * x + y * y + z * z
        r = math.sqrt(r2)
        r5 = r2 * r2 * r
        j2c = 1.5 * J2 * mu * Re * Re / r5
        z2r2 = z * z / r2
        ax = -mu * x / r ** 3 + j2c * x * (5 * z2r2 - 1)
        ay = -mu * y / r ** 3 + j2c * y * (5 * z2r2 - 1)
        az = -mu * z / r ** 3 + j2c * z * (5 * z2r2 - 3)
        return [vx, vy, vz, ax, ay, az]
    return rhs


def _initial_state(mu, a, e, inc):
    p_orb = a * (1 - e ** 2)
    r_p = a * (1 - e)
    v_p = math.sqrt(mu / p_orb) * (1 + e)
    return np.array([r_p, 0.0, 0.0, 0.0, v_p * math.cos(inc), v_p * math.sin(inc)])


def test_j2_nodal_regression_matches_literature_secular_rate():
    mu, J2, Re = 398600.4418, 1.08262668e-3, 6378.137
    a, e, inc_deg = 7000.0, 0.001, 98.7  # Sun-synchronous-like, same as the preset's defaults
    inc = math.radians(inc_deg)
    p_orb = a * (1 - e ** 2)
    n_mean = math.sqrt(mu / a ** 3)
    period = 2 * math.pi / n_mean

    n_orbits = 800
    t_eval = np.linspace(0, n_orbits * period, n_orbits * 4)
    sol = solve_ivp(_j2_rhs(mu, J2, Re), [0, t_eval[-1]], _initial_state(mu, a, e, inc),
                     t_eval=t_eval, rtol=1e-11, atol=1e-8, method="DOP853")
    assert sol.success

    x, y, z, vx, vy, vz = sol.y
    hx = y * vz - z * vy
    hy = z * vx - x * vz
    Omega = np.unwrap(np.arctan2(hx, -hy))
    slope, _ = np.linalg.lstsq(np.vstack([sol.t, np.ones_like(sol.t)]).T, Omega, rcond=None)[0]

    literature = -1.5 * n_mean * J2 * (Re / p_orb) ** 2 * math.cos(inc)
    rel_err = abs(slope - literature) / abs(literature)
    print(f"Nodal regression: integrated={slope:.6e} rad/s, literature={literature:.6e} rad/s, "
          f"rel_err={100 * rel_err:.3f}%")
    assert rel_err < 0.01, f"Nodal regression rate should match Vallado's secular formula to <1%, got {100*rel_err:.2f}%"


def test_j2_apsidal_precession_matches_literature_secular_rate():
    mu, J2, Re = 398600.4418, 1.08262668e-3, 6378.137
    # e=0.1 (well-defined argument of perigee) and i=30 deg (well away from
    # the critical inclination ~63.435 deg where the literature formula's
    # 5cos^2(i)-1 factor crosses zero -- see module docstring).
    a, e, inc_deg = 7000.0, 0.1, 30.0
    inc = math.radians(inc_deg)
    p_orb = a * (1 - e ** 2)
    n_mean = math.sqrt(mu / a ** 3)
    period = 2 * math.pi / n_mean

    n_orbits = 400
    t_eval = np.linspace(0, n_orbits * period, n_orbits * 6)
    sol = solve_ivp(_j2_rhs(mu, J2, Re), [0, t_eval[-1]], _initial_state(mu, a, e, inc),
                     t_eval=t_eval, rtol=1e-12, atol=1e-9, method="DOP853")
    assert sol.success

    x, y, z, vx, vy, vz = sol.y
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    hx = y * vz - z * vy
    hy = z * vx - x * vz
    hz = x * vy - y * vx

    vxh_x = vy * hz - vz * hy
    vxh_y = vz * hx - vx * hz
    vxh_z = vx * hy - vy * hx
    ex = vxh_x / mu - x / r
    ey = vxh_y / mu - y / r
    ez = vxh_z / mu - z / r

    nx, ny = -hy, hx
    nmag = np.sqrt(nx ** 2 + ny ** 2)
    emag = np.sqrt(ex ** 2 + ey ** 2 + ez ** 2)
    cos_w = np.clip((nx * ex + ny * ey) / (nmag * emag), -1.0, 1.0)
    omega = np.arccos(cos_w)
    omega = np.where(ez < 0, 2 * np.pi - omega, omega)
    omega_unwrapped = np.unwrap(omega)

    slope, _ = np.linalg.lstsq(np.vstack([sol.t, np.ones_like(sol.t)]).T, omega_unwrapped, rcond=None)[0]

    literature = 0.75 * n_mean * J2 * (Re / p_orb) ** 2 * (5 * math.cos(inc) ** 2 - 1)
    rel_err = abs(slope - literature) / abs(literature)
    print(f"Apsidal precession: integrated={slope:.6e} rad/s, literature={literature:.6e} rad/s, "
          f"rel_err={100 * rel_err:.3f}%")
    assert rel_err < 0.01, f"Apsidal precession rate should match Vallado's secular formula to <1%, got {100*rel_err:.2f}%"
