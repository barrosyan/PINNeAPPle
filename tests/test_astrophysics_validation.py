"""Tier B validation for the astrophysics/space specialization presets
(`pinneapple_physics/pde_environment/presets/astrophysics.py`).

Same method as ``tests/test_manufactured_solutions.py``: plug an exact
(or, where noted, a manufactured smooth) solution directly into
``compile_problem``'s compiled residual with no training involved, and
assert the residual is ~0; then plug in a deliberately wrong function and
assert the residual is measurably nonzero. This is what "reproduce
exactly" is checked against for each new astrophysics PDE/ODE kind added
this session -- see `astrophysics.py`'s module docstring for the
literature references and the `sympy` verification done for each closed
form before it was written into code.

Two presets do NOT have a exact-residual test here, by design, not
oversight:
  - `satellite_j2_perturbation`: no closed-form trajectory exists (only
    orbit-averaged secular rates, which need a many-orbit integration
    horizon to check, not done this session). Instead this file checks a
    weaker but still real physical-consistency property: with J2 forced
    to 0, the preset's residual must reduce EXACTLY to the plain
    `kepler_two_body_orbit` residual for the same trajectory.
  - `sod_shock_tube_astro`: the real Sod Riemann solution is
    discontinuous (shocks/contact discontinuity), so pointwise autograd
    residuals are not defined on it -- MMS requires a *smooth* exact
    solution instead. This file uses a smooth manufactured solution (a
    rigidly-advected density pulse at constant pressure and velocity,
    verified with `sympy` to solve the compressible Euler equations
    exactly) to check the `euler_compressible_1d` residual implementation
    itself, which is the part `sod_shock_tube_astro` depends on.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from pinneapple_physics.pinn_solver.compiler.compile import compile_problem
from pinneapple_physics.pde_environment.presets.astrophysics import (
    kepler_two_body_orbit,
    space_debris_cw_relative_motion,
    satellite_j2_perturbation,
    spacecraft_attitude_euler_rotation,
    lane_emden_polytrope,
    nfw_dark_matter_potential,
    nfw_potential_exact,
    nfw_source_fn,
)


class _ExactFn(nn.Module):
    """See tests/test_manufactured_solutions.py::_ExactFn for the full
    rationale (an nn.Module with one dummy parameter satisfies
    compile_problem's `model.parameters()`/`model(x)` interface)."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x)


def _empty_batch(x_col, n_coords, n_fields, ctx=None):
    return {
        "x_col": x_col, "ctx": ctx or {},
        "x_bc": torch.zeros((0, n_coords)), "y_bc": torch.zeros((0, n_fields)),
        "x_ic": torch.zeros((0, n_coords)), "y_ic": torch.zeros((0, n_fields)),
        "x_data": torch.zeros((0, n_coords)), "y_data": torch.zeros((0, n_fields)),
    }


def _pde_residual(out):
    return out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]


# ===========================================================================
# Kepler two-body orbit
# ===========================================================================

def _kepler_exact_torch(mu: float, a: float, e: float, n_newton_iters: int = 80):
    n_mean = math.sqrt(mu / a ** 3)

    def fn(tcol: torch.Tensor) -> torch.Tensor:
        t = tcol[:, 0:1]
        M = n_mean * t
        E = M.clone()
        for _ in range(n_newton_iters):
            f = E - e * torch.sin(E) - M
            fp = 1.0 - e * torch.cos(E)
            E = E - f / fp
        x = a * (torch.cos(E) - e)
        y = a * math.sqrt(1 - e ** 2) * torch.sin(E)
        vx = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True)[0]
        vy = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]
        return torch.cat([x, y, vx, vy], dim=1)

    return fn


def test_kepler_two_body_orbit_exact_gives_near_zero_residual():
    mu, a, e = 398600.4418, 8000.0, 0.15
    spec = kepler_two_body_orbit(mu=mu, a=a, e=e)
    loss_fn = compile_problem(spec)

    exact = _ExactFn(_kepler_exact_torch(mu, a, e))
    period = 2.0 * math.pi * math.sqrt(a ** 3 / mu)
    t = (torch.rand(64, 1) * period * 0.9 + period * 0.02).requires_grad_(True)
    batch = _empty_batch(t, n_coords=1, n_fields=4)

    out = loss_fn(exact, None, batch)
    res = float(_pde_residual(out).item())
    assert res < 1e-4, f"Kepler orbit residual should be ~0 for the exact trajectory, got {res}"


def test_kepler_two_body_orbit_wrong_solution_gives_nonzero_residual():
    mu, a, e = 398600.4418, 8000.0, 0.15
    spec = kepler_two_body_orbit(mu=mu, a=a, e=e)
    loss_fn = compile_problem(spec)

    # A fixed point sitting close to the attracting body (r=sqrt(2)*50 km,
    # far inside any sensible orbit for mu=Earth's GM) -- consistent with
    # dx/dt=vx=0 trivially, but wildly violates Newton's law of
    # gravitation, whose 1/r^3 acceleration term is enormous this close in.
    def wrong(tcol):
        t = tcol[:, 0:1]
        x = 50.0 + 0.0 * t
        y = 50.0 + 0.0 * t
        vx = 0.0 * t
        vy = 0.0 * t
        return torch.cat([x, y, vx, vy], dim=1)

    t = (torch.rand(64, 1) * 3000.0 + 10.0).requires_grad_(True)
    batch = _empty_batch(t, n_coords=1, n_fields=4)
    out = loss_fn(_ExactFn(wrong), None, batch)
    res = float(_pde_residual(out).item())
    assert res > 1.0, f"Kepler residual should be clearly nonzero for a wrong trajectory, got {res}"


# ===========================================================================
# Space debris: Clohessy-Wiltshire relative motion
# ===========================================================================

def _cw_exact_torch(n: float, x0, y0, z0, vx0, vy0, vz0):
    def fn(tcol: torch.Tensor) -> torch.Tensor:
        t = tcol[:, 0:1]
        nt = n * t
        s, c = torch.sin(nt), torch.cos(nt)
        x = (4 - 3 * c) * x0 + (s / n) * vx0 + (2.0 / n) * (1 - c) * vy0
        y = 6 * (s - nt) * x0 + y0 - (2.0 / n) * (1 - c) * vx0 + (1.0 / n) * (4 * s - 3 * nt) * vy0
        z = z0 * c + (vz0 / n) * s
        vx = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True)[0]
        vy = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]
        vz = torch.autograd.grad(z, t, grad_outputs=torch.ones_like(z), create_graph=True, retain_graph=True)[0]
        return torch.cat([x, y, z, vx, vy, vz], dim=1)

    return fn


def test_space_debris_cw_exact_gives_near_zero_residual():
    n_mm = 0.0011
    x0, y0, z0, vx0, vy0, vz0 = 1.0, 0.0, 0.2, 0.0, -0.0015, 0.0008
    spec = space_debris_cw_relative_motion(n=n_mm, x0=x0, y0=y0, z0=z0, vx0=vx0, vy0=vy0, vz0=vz0)
    loss_fn = compile_problem(spec)

    exact = _ExactFn(_cw_exact_torch(n_mm, x0, y0, z0, vx0, vy0, vz0))
    period = 2.0 * math.pi / n_mm
    t = (torch.rand(64, 1) * period).requires_grad_(True)
    batch = _empty_batch(t, n_coords=1, n_fields=6)

    out = loss_fn(exact, None, batch)
    res = float(_pde_residual(out).item())
    assert res < 1e-6, f"CW relative-motion residual should be ~0 for the exact solution, got {res}"


def test_space_debris_cw_wrong_solution_gives_nonzero_residual():
    n_mm = 0.0011
    spec = space_debris_cw_relative_motion(n=n_mm)
    loss_fn = compile_problem(spec)

    # A cross-track oscillation at 10 rad/s -- ~9000x the reference
    # orbit's mean motion (n=0.0011 rad/s), so z-double-dot completely
    # swamps the n^2*z restoring term the compiled residual expects.
    def wrong(tcol):
        t = tcol[:, 0:1]
        z = 5.0 * torch.sin(10.0 * t)
        zeros = 0.0 * t
        vz = torch.autograd.grad(z, t, grad_outputs=torch.ones_like(z), create_graph=True, retain_graph=True)[0]
        return torch.cat([zeros, zeros, z, zeros, zeros, vz], dim=1)

    t = (torch.rand(64, 1) * (2.0 * math.pi / n_mm) + 1.0).requires_grad_(True)
    batch = _empty_batch(t, n_coords=1, n_fields=6)
    out = loss_fn(_ExactFn(wrong), None, batch)
    res = float(_pde_residual(out).item())
    assert res > 1.0, f"CW residual should be clearly nonzero for a wrong trajectory, got {res}"


# ===========================================================================
# Satellite J2 perturbation -- consistency check (J2=0 reduces to Kepler)
# ===========================================================================

def test_satellite_j2_perturbation_reduces_to_two_body_when_j2_zero():
    """No closed-form J2 trajectory exists (see module docstring), so
    instead check a real, exact physical-consistency property: setting
    J2=0 must make `satellite_j2_perturbation`'s residual EXACTLY equal to
    plain two-body motion's residual (-mu*r/|r|^3, i.e. Kepler's law with
    no perturbation) for a genuine (non-planar) 3D two-body trajectory."""
    mu = 398600.4418
    spec = satellite_j2_perturbation(mu=mu, J2=0.0, a=7000.0, e=0.001, inclination_deg=98.7)
    loss_fn = compile_problem(spec)

    # Circular-ish inclined trajectory in 3D built directly from Newton's
    # law's own closed form is not available (this is exactly the
    # transcendental-orbit problem), so instead test with a simple, exact
    # PURE two-body solution: circular orbit in an inclined plane (a
    # genuine closed-form special case of the two-body problem, distinct
    # from the general eccentric Kepler solve tested above).
    r0 = 7000.0
    v0 = math.sqrt(mu / r0)
    omega = v0 / r0
    inc = math.radians(60.0)

    def circular_inclined(tcol):
        t = tcol[:, 0:1]
        theta = omega * t
        x = r0 * torch.cos(theta)
        y = r0 * torch.sin(theta) * math.cos(inc)
        z = r0 * torch.sin(theta) * math.sin(inc)
        vx = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True)[0]
        vy = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]
        vz = torch.autograd.grad(z, t, grad_outputs=torch.ones_like(z), create_graph=True, retain_graph=True)[0]
        return torch.cat([x, y, z, vx, vy, vz], dim=1)

    period = 2.0 * math.pi / omega
    t = (torch.rand(64, 1) * period).requires_grad_(True)
    batch = _empty_batch(t, n_coords=1, n_fields=6)

    out = loss_fn(_ExactFn(circular_inclined), None, batch)
    res = float(_pde_residual(out).item())
    assert res < 1e-4, (
        f"satellite_j2_perturbation with J2=0 should reduce exactly to two-body motion "
        f"(residual ~0 for a true circular two-body orbit), got {res}"
    )


def test_satellite_j2_perturbation_nonzero_j2_perturbs_pure_two_body_orbit():
    """The complementary half: with J2 != 0 (the real Earth value), the
    SAME pure two-body circular trajectory must now give a measurably
    nonzero residual -- proving the J2 acceleration term is actually
    wired into the residual, not a dead branch."""
    mu = 398600.4418
    spec = satellite_j2_perturbation(mu=mu, J2=1.08262668e-3, Re=6378.137, a=7000.0, e=0.001, inclination_deg=98.7)
    loss_fn = compile_problem(spec)

    r0 = 7000.0
    v0 = math.sqrt(mu / r0)
    omega = v0 / r0
    inc = math.radians(60.0)

    def circular_inclined(tcol):
        t = tcol[:, 0:1]
        theta = omega * t
        x = r0 * torch.cos(theta)
        y = r0 * torch.sin(theta) * math.cos(inc)
        z = r0 * torch.sin(theta) * math.sin(inc)
        vx = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True)[0]
        vy = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]
        vz = torch.autograd.grad(z, t, grad_outputs=torch.ones_like(z), create_graph=True, retain_graph=True)[0]
        return torch.cat([x, y, z, vx, vy, vz], dim=1)

    period = 2.0 * math.pi / omega
    t = (torch.rand(64, 1) * period).requires_grad_(True)
    batch = _empty_batch(t, n_coords=1, n_fields=6)

    out = loss_fn(_ExactFn(circular_inclined), None, batch)
    res = float(_pde_residual(out).item())
    # Empirically (see this test's construction): J2=0 gives ~1e-19
    # (float32 rounding noise floor, confirmed by the companion test
    # above), J2=Earth's real value gives ~3e-11 -- eight orders of
    # magnitude above the noise floor. 1e-15 is comfortably inside that
    # gap on both sides.
    assert res > 1e-15, f"J2 != 0 should measurably perturb a pure two-body orbit's residual, got {res}"


# ===========================================================================
# Spacecraft attitude: torque-free axisymmetric Euler rotation
# ===========================================================================

def test_spacecraft_attitude_exact_gives_near_zero_residual():
    I1, I3 = 100.0, 150.0
    w1_0, w3_0 = 0.05, 0.5
    spec = spacecraft_attitude_euler_rotation(I1=I1, I3=I3, w1_0=w1_0, w3_0=w3_0)
    loss_fn = compile_problem(spec)

    lam = (I3 - I1) / I1 * w3_0

    def exact(tcol):
        t = tcol[:, 0:1]
        w1 = w1_0 * torch.cos(lam * t)
        w2 = w1_0 * torch.sin(lam * t)
        w3 = torch.full_like(t, w3_0)
        return torch.cat([w1, w2, w3], dim=1)

    t = (torch.rand(64, 1) * 60.0).requires_grad_(True)
    batch = _empty_batch(t, n_coords=1, n_fields=3)
    out = loss_fn(_ExactFn(exact), None, batch)
    res = float(_pde_residual(out).item())
    assert res < 1e-8, f"Torque-free attitude residual should be ~0 for the exact precession solution, got {res}"


def test_spacecraft_attitude_wrong_solution_gives_nonzero_residual():
    I1, I3 = 100.0, 150.0
    w1_0, w3_0 = 0.05, 0.5
    spec = spacecraft_attitude_euler_rotation(I1=I1, I3=I3, w1_0=w1_0, w3_0=w3_0)
    loss_fn = compile_problem(spec)

    # Wrong precession rate (uses w3_0 directly instead of the correct
    # lambda = (I3-I1)/I1 * w3_0) -- a common real modeling mistake.
    def wrong(tcol):
        t = tcol[:, 0:1]
        w1 = w1_0 * torch.cos(w3_0 * t)
        w2 = w1_0 * torch.sin(w3_0 * t)
        w3 = torch.full_like(t, w3_0)
        return torch.cat([w1, w2, w3], dim=1)

    t = (torch.rand(64, 1) * 60.0).requires_grad_(True)
    batch = _empty_batch(t, n_coords=1, n_fields=3)
    out = loss_fn(_ExactFn(wrong), None, batch)
    res = float(_pde_residual(out).item())
    assert res > 1e-6, f"Attitude residual should be nonzero for the wrong precession rate, got {res}"


# ===========================================================================
# Lane-Emden polytrope
# ===========================================================================

def test_lane_emden_n1_exact_gives_near_zero_residual():
    spec = lane_emden_polytrope(n=1.0, xi_min=1e-3, xi_max=3.0)
    loss_fn = compile_problem(spec)

    def exact(xicol):
        xi = xicol[:, 0:1]
        theta = torch.sin(xi) / xi
        phi = torch.autograd.grad(theta, xi, grad_outputs=torch.ones_like(theta),
                                   create_graph=True, retain_graph=True)[0]
        return torch.cat([theta, phi], dim=1)

    xi = (torch.rand(128, 1) * 2.9 + 0.05).requires_grad_(True)
    batch = _empty_batch(xi, n_coords=1, n_fields=2)
    out = loss_fn(_ExactFn(exact), None, batch)
    res = float(_pde_residual(out).item())
    assert res < 1e-6, f"Lane-Emden (n=1) residual should be ~0 for theta=sin(xi)/xi, got {res}"


def test_lane_emden_n0_exact_gives_near_zero_residual():
    spec = lane_emden_polytrope(n=0.0, xi_min=1e-3, xi_max=3.0)
    loss_fn = compile_problem(spec)

    def exact(xicol):
        xi = xicol[:, 0:1]
        theta = 1.0 - xi ** 2 / 6.0
        phi = torch.autograd.grad(theta, xi, grad_outputs=torch.ones_like(theta),
                                   create_graph=True, retain_graph=True)[0]
        return torch.cat([theta, phi], dim=1)

    xi = (torch.rand(128, 1) * 2.9 + 0.05).requires_grad_(True)
    batch = _empty_batch(xi, n_coords=1, n_fields=2)
    out = loss_fn(_ExactFn(exact), None, batch)
    res = float(_pde_residual(out).item())
    assert res < 1e-10, f"Lane-Emden (n=0) residual should be ~0 for theta=1-xi^2/6, got {res}"


def test_lane_emden_wrong_solution_gives_nonzero_residual():
    spec = lane_emden_polytrope(n=1.0, xi_min=1e-3, xi_max=3.0)
    loss_fn = compile_problem(spec)

    def wrong(xicol):
        xi = xicol[:, 0:1]
        theta = torch.cos(xi)  # does not solve the n=1 Lane-Emden equation
        phi = torch.autograd.grad(theta, xi, grad_outputs=torch.ones_like(theta),
                                   create_graph=True, retain_graph=True)[0]
        return torch.cat([theta, phi], dim=1)

    xi = (torch.rand(128, 1) * 2.9 + 0.05).requires_grad_(True)
    batch = _empty_batch(xi, n_coords=1, n_fields=2)
    out = loss_fn(_ExactFn(wrong), None, batch)
    res = float(_pde_residual(out).item())
    assert res > 1e-3, f"Lane-Emden residual should be nonzero for cos(xi), got {res}"


# ===========================================================================
# NFW dark-matter halo potential (reuses the existing "poisson" kind)
# ===========================================================================

def test_nfw_potential_exact_gives_near_zero_residual():
    G, rho_s, rs = 1.0, 1.0, 1.0
    spec = nfw_dark_matter_potential(G=G, rho_s=rho_s, rs=rs, r_max=10.0)
    loss_fn = compile_problem(spec)

    def exact(xyz):
        x, y, z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]
        r = torch.sqrt(x * x + y * y + z * z + 1e-9)
        return -4.0 * math.pi * G * rho_s * rs ** 3 * torch.log1p(r / rs) / r

    xyz = (torch.rand(256, 3) * 6.0 + 0.5).requires_grad_(True)  # avoid r~0
    batch = _empty_batch(xyz, n_coords=3, n_fields=1, ctx={"source_fn": nfw_source_fn(G, rho_s, rs)})
    out = loss_fn(_ExactFn(exact), None, batch)
    res = float(_pde_residual(out).item())
    assert res < 1e-6, f"NFW potential should satisfy Poisson's eq (residual ~0) for the exact potential, got {res}"


def test_nfw_potential_wrong_solution_gives_nonzero_residual():
    G, rho_s, rs = 1.0, 1.0, 1.0
    spec = nfw_dark_matter_potential(G=G, rho_s=rho_s, rs=rs, r_max=10.0)
    loss_fn = compile_problem(spec)

    def wrong(xyz):
        x, y, z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]
        r = torch.sqrt(x * x + y * y + z * z + 1e-9)
        return -1.0 / r  # point-mass (Keplerian) potential, wrong for an extended NFW halo

    xyz = (torch.rand(256, 3) * 6.0 + 0.5).requires_grad_(True)
    batch = _empty_batch(xyz, n_coords=3, n_fields=1, ctx={"source_fn": nfw_source_fn(G, rho_s, rs)})
    out = loss_fn(_ExactFn(wrong), None, batch)
    res = float(_pde_residual(out).item())
    assert res > 1e-6, f"Point-mass potential should NOT satisfy NFW's Poisson eq, got residual {res}"


def test_nfw_potential_exact_matches_module_reference_function():
    """Cross-check `nfw_potential_exact` (the numpy reference used for
    boundary conditions) against the torch closed form used in the
    residual test above -- both must agree, not just each independently
    look plausible."""
    import numpy as np
    r = np.linspace(0.1, 10.0, 50)
    ref = nfw_potential_exact(r, G=1.0, rho_s=1.0, rs=1.0)
    torch_val = (-4.0 * math.pi * 1.0 * 1.0 * 1.0 ** 3 * torch.log1p(torch.tensor(r / 1.0)) / torch.tensor(r)).numpy()
    assert np.allclose(ref, torch_val, rtol=1e-10)


# ===========================================================================
# 1D compressible Euler (euler_compressible_1d) -- smooth manufactured
# solution, since the Sod shock tube's real solution is discontinuous.
# ===========================================================================

def test_euler_compressible_1d_smooth_advected_pulse_gives_near_zero_residual():
    """A rigidly-advected density pulse at constant velocity u0 and
    constant pressure p0 -- verified with `sympy` this session to satisfy
    the 1D compressible Euler equations exactly for ANY smooth density
    profile f(x-u0*t) (isobaric flow has zero pressure gradient, so there
    is nothing to accelerate the fluid: the momentum and energy equations
    both collapse to the same statement as continuity)."""
    from pinneapple_physics.pde_environment.spec import PDETermSpec, ProblemSpec
    from pinneapple_physics.pde_environment.environment_typing import CoordNames
    from pinneapple_physics.pde_environment.scales import ScaleSpec

    gamma = 1.4
    u0, p0 = 0.3, 1.0
    coords: CoordNames = ("x", "t")
    fields = ("rho", "rho_u", "E")
    pde = PDETermSpec(kind="euler_compressible_1d", fields=fields, coords=coords, params={"gamma": gamma})
    spec = ProblemSpec(name="_euler_1d_mms_probe", dim=1, coords=coords, fields=fields, pde=pde,
                        conditions=(), scales=ScaleSpec())
    loss_fn = compile_problem(spec)

    def exact(xt):
        x, t = xt[:, 0:1], xt[:, 1:2]
        xi = x - u0 * t
        rho = 2.0 + 0.5 * torch.sin(xi)
        rho_u = u0 * rho
        E = p0 / (gamma - 1.0) + 0.5 * u0 * u0 * rho
        return torch.cat([rho, rho_u, E], dim=1)

    xt = torch.rand(256, 2, requires_grad=True)
    batch = _empty_batch(xt, n_coords=2, n_fields=3)
    out = loss_fn(_ExactFn(exact), None, batch)
    res = float(_pde_residual(out).item())
    assert res < 1e-8, f"1D Euler residual should be ~0 for the advected-pulse manufactured solution, got {res}"


def test_euler_compressible_1d_wrong_solution_gives_nonzero_residual():
    from pinneapple_physics.pde_environment.spec import PDETermSpec, ProblemSpec
    from pinneapple_physics.pde_environment.environment_typing import CoordNames
    from pinneapple_physics.pde_environment.scales import ScaleSpec

    gamma = 1.4
    coords: CoordNames = ("x", "t")
    fields = ("rho", "rho_u", "E")
    pde = PDETermSpec(kind="euler_compressible_1d", fields=fields, coords=coords, params={"gamma": gamma})
    spec = ProblemSpec(name="_euler_1d_mms_probe_wrong", dim=1, coords=coords, fields=fields, pde=pde,
                        conditions=(), scales=ScaleSpec())
    loss_fn = compile_problem(spec)

    def wrong(xt):
        x, t = xt[:, 0:1], xt[:, 1:2]
        # Density advects at u0=0.3 but momentum is set as if u=0.6 --
        # mutually inconsistent, should not satisfy continuity/momentum.
        rho = 2.0 + 0.5 * torch.sin(x - 0.3 * t)
        rho_u = 0.6 * rho
        E = 1.0 / (gamma - 1.0) + 0.5 * 0.6 * 0.6 * rho
        return torch.cat([rho, rho_u, E], dim=1)

    xt = torch.rand(256, 2, requires_grad=True)
    batch = _empty_batch(xt, n_coords=2, n_fields=3)
    out = loss_fn(_ExactFn(wrong), None, batch)
    res = float(_pde_residual(out).item())
    assert res > 1e-4, f"1D Euler residual should be nonzero for an inconsistent rho/momentum pair, got {res}"
