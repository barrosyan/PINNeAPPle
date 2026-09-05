"""Tier B of the evidenced library audit (see ``ROADMAP_PHYSICS_AI_HUB.md``,
section 1.1): method of manufactured solutions -- the actual physics-
correctness check, as opposed to ``test_full_library_matrix.py``'s "does
it run" breadth tier.

For a preset with a known closed-form solution, plug the EXACT solution
directly into ``compile_problem``'s compiled residual (no training
involved) and assert the residual is ~0; then plug in a DELIBERATELY
WRONG function and assert the residual is measurably nonzero. A residual
that returns ~0 for both is exactly as broken as one that's wrong for
both -- a trained network's loss going down cannot distinguish a correct
residual driving it toward the right answer from a broken residual it can
trivially satisfy (this is the literal failure mode this project's own
``AdaptiveWeights`` post-mortem documents: a network converging a
residual to ~0 by finding the PDE+BC's trivial exact solution instead of
the real one).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from pinneapple_physics.pinn_solver.compiler.compile import compile_problem
from pinneapple_physics.pde_environment.presets.academics import laplace_2d_default
from pinneapple_physics.pde_environment.spec import PDETermSpec, ProblemSpec
from pinneapple_physics.pde_environment.scales import ScaleSpec
from pinneapple_physics.pde_environment.turbulence_presets import KOmegaSSTResiduals


class _ExactFn(nn.Module):
    """Wraps a plain torch-differentiable function as a fake "model" for
    ``compile_problem``'s loss_fn, which requires ``model.parameters()``
    (for its own device lookup) and calls ``model(x)`` directly -- an
    ``nn.Module`` with one dummy parameter satisfies both without needing
    an actual trained network."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x)


def _empty_batch(x_col, n_coords, n_fields):
    return {
        "x_col": x_col, "ctx": {},
        "x_bc": torch.zeros((0, n_coords)), "y_bc": torch.zeros((0, n_fields)),
        "x_ic": torch.zeros((0, n_coords)), "y_ic": torch.zeros((0, n_fields)),
        "x_data": torch.zeros((0, n_coords)), "y_data": torch.zeros((0, n_fields)),
    }


def test_audit_physics_laplace_2d_exact_solution_gives_zero_residual():
    """u(x,y) = x^2 - y^2 is harmonic (Laplace's equation: u_xx+u_yy=0
    exactly) -- the compiled 'laplace' residual must be ~0 pointwise for
    this function, to floating-point tolerance."""
    spec = laplace_2d_default()
    loss_fn = compile_problem(spec)

    exact = _ExactFn(lambda x: (x[:, 0:1] ** 2 - x[:, 1:2] ** 2))
    x = torch.rand(256, 2, requires_grad=True)
    batch = _empty_batch(x, n_coords=2, n_fields=1)

    out = loss_fn(exact, None, batch)
    pde_residual = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(pde_residual.item()) < 1e-8, (
        f"Laplace residual should be ~0 for the exact harmonic solution x^2-y^2, got {float(pde_residual.item())}"
    )


def test_audit_physics_laplace_2d_wrong_solution_gives_nonzero_residual():
    """u(x,y) = x^2 + y^2 has Laplacian = 4 everywhere (NOT harmonic) --
    the compiled residual must be measurably nonzero for it. This is the
    other half of the check: a residual implementation that returns ~0
    for every input (e.g. an accidental no-op / always-satisfied
    condition) would pass the previous test just as "well" as a correct
    one -- this test is what actually distinguishes them."""
    spec = laplace_2d_default()
    loss_fn = compile_problem(spec)

    wrong = _ExactFn(lambda x: (x[:, 0:1] ** 2 + x[:, 1:2] ** 2))
    x = torch.rand(256, 2, requires_grad=True)
    batch = _empty_batch(x, n_coords=2, n_fields=1)

    out = loss_fn(wrong, None, batch)
    pde_residual = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    # Laplacian is exactly 4 everywhere for this function, so the
    # mean-squared residual should be exactly 16 (up to autograd/float
    # rounding) -- assert a loose but structurally meaningful bound well
    # above the previous test's near-zero threshold.
    assert float(pde_residual.item()) > 1.0, (
        f"Laplace residual should be clearly nonzero (~16) for the non-harmonic x^2+y^2, "
        f"got {float(pde_residual.item())} -- if this is ~0, the residual computation itself "
        f"is broken (e.g. always returns ~0 regardless of input), not just under-converged."
    )


def _probe_spec(kind: str, coords, params) -> ProblemSpec:
    """Minimal standalone ProblemSpec for a pde_kind, no preset needed --
    used below for kinds (heat_equation_steady*, navier_stokes steady
    case) added this session that don't have a dedicated academics.py
    preset of their own."""
    pde = PDETermSpec(kind=kind, fields=("T",), coords=coords, params=params)
    return ProblemSpec(name=f"_probe_{kind}", dim=len(coords), coords=coords, fields=("T",),
                        pde=pde, conditions=(), scales=ScaleSpec())


def test_audit_physics_heat_equation_steady_exact_solution_gives_zero_residual():
    """T(x,y,z) = x^2 - y^2 is harmonic -- k*laplacian(T) = 0 exactly, for
    any k. Added this session alongside the fix that let
    cpu_heatsink_thermal/industrial_furnace_thermal/
    datacenter_server_thermal/refractory_lining actually compile (they
    were all "Unsupported PDE kind" Category-1 audit failures before)."""
    spec = _probe_spec("heat_equation_steady", ("x", "y", "z"), {"k": 2.0})
    loss_fn = compile_problem(spec)
    exact = _ExactFn(lambda x: (x[:, 0:1] ** 2 - x[:, 1:2] ** 2))
    x = torch.rand(256, 3, requires_grad=True)
    batch = _empty_batch(x, n_coords=3, n_fields=1)
    out = loss_fn(exact, None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(res.item()) < 1e-8, f"heat_equation_steady residual should be ~0 for a harmonic T, got {float(res.item())}"


def test_audit_physics_heat_equation_steady_wrong_solution_gives_nonzero_residual():
    spec = _probe_spec("heat_equation_steady", ("x", "y", "z"), {"k": 2.0})
    loss_fn = compile_problem(spec)
    wrong = _ExactFn(lambda x: (x[:, 0:1] ** 2 + x[:, 1:2] ** 2 + x[:, 2:3] ** 2))
    x = torch.rand(256, 3, requires_grad=True)
    batch = _empty_batch(x, n_coords=3, n_fields=1)
    out = loss_fn(wrong, None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    # k*laplacian = 2*6 = 12 exactly everywhere -> mean-squared residual = 144.
    assert float(res.item()) > 1.0, f"heat_equation_steady residual should be clearly nonzero, got {float(res.item())}"


def test_audit_physics_heat_equation_steady_anisotropic_exact_solution_gives_zero_residual():
    """T(x,y) = k_y*x^2 - k_x*y^2 satisfies k_x*T_xx + k_y*T_yy = 0
    exactly for any k_x, k_y (by construction: k_x*(2 k_y) + k_y*(-2 k_x)
    = 0). Added alongside the fix that let pcb_thermal compile."""
    spec = _probe_spec("heat_equation_steady_anisotropic", ("x", "y"), {"k_x": 2.0, "k_y": 3.0})
    loss_fn = compile_problem(spec)
    exact = _ExactFn(lambda x: (3.0 * x[:, 0:1] ** 2 - 2.0 * x[:, 1:2] ** 2))
    x = torch.rand(256, 2, requires_grad=True)
    batch = _empty_batch(x, n_coords=2, n_fields=1)
    out = loss_fn(exact, None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(res.item()) < 1e-8, f"anisotropic heat residual should be ~0 for the exact solution, got {float(res.item())}"


def test_audit_physics_heat_equation_steady_anisotropic_wrong_solution_gives_nonzero_residual():
    spec = _probe_spec("heat_equation_steady_anisotropic", ("x", "y"), {"k_x": 2.0, "k_y": 3.0})
    loss_fn = compile_problem(spec)
    wrong = _ExactFn(lambda x: (x[:, 0:1] ** 2 + x[:, 1:2] ** 2))
    x = torch.rand(256, 2, requires_grad=True)
    batch = _empty_batch(x, n_coords=2, n_fields=1)
    out = loss_fn(wrong, None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    # k_x*2 + k_y*2 = 2*(2+3) = 10 exactly -> mean-squared residual = 100.
    assert float(res.item()) > 1.0, f"anisotropic heat residual should be clearly nonzero, got {float(res.item())}"


@pytest.mark.parametrize("kind", ["linear_elasticity_plane_strain", "linear_elasticity_plane_stress"])
def test_audit_physics_elasticity_2d_exact_solution_gives_zero_residual(kind):
    """ux=x^2-y^2, uy=-2xy: both components are individually harmonic AND
    div(u)=0, so Navier's equilibrium equation (lambda+mu)*grad(div(u)) +
    mu*laplacian(u) = 0 is satisfied identically for ANY lambda, mu --
    verified with `sympy` before being used here. This exercises BOTH
    plane_strain (the raw lambda) and plane_stress (the reduced
    lambda* = 2*lambda*mu/(lambda+2*mu), also independently `sympy`-
    verified this session by eliminating eps_zz from the full 3D
    constitutive relation under the sigma_zz=0 plane-stress constraint)
    since div(u)=0 makes the residual insensitive to which lambda is used
    -- both must give exactly the same (zero) answer regardless."""
    spec = _probe_spec(kind, ("x", "y"), {"lambda": 1e5, "mu": 8e4})
    # _probe_spec defaults to field "T"; elasticity needs ux, uy instead.
    from dataclasses import replace as _replace
    spec = _replace(spec, fields=("ux", "uy"), pde=_replace(spec.pde, fields=("ux", "uy")))
    loss_fn = compile_problem(spec)
    exact = _ExactFn(lambda x: torch.cat([x[:, 0:1] ** 2 - x[:, 1:2] ** 2, -2 * x[:, 0:1] * x[:, 1:2]], dim=1))
    x = torch.rand(256, 2, requires_grad=True)
    batch = _empty_batch(x, n_coords=2, n_fields=2)
    out = loss_fn(exact, None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(res.item()) < 1e-6, f"{kind} residual should be ~0 for the exact solution, got {float(res.item())}"


@pytest.mark.parametrize("kind", ["linear_elasticity_plane_strain", "linear_elasticity_plane_stress"])
def test_audit_physics_elasticity_2d_wrong_solution_gives_nonzero_residual(kind):
    spec = _probe_spec(kind, ("x", "y"), {"lambda": 1e5, "mu": 8e4})
    from dataclasses import replace as _replace
    spec = _replace(spec, fields=("ux", "uy"), pde=_replace(spec.pde, fields=("ux", "uy")))
    loss_fn = compile_problem(spec)
    # ux=x^3-3xy^2, uy=3x^2y-y^3 are each individually harmonic but
    # div(u) != 0 in general, so equilibrium is NOT satisfied identically.
    wrong = _ExactFn(lambda x: torch.cat(
        [x[:, 0:1] ** 3 - 3 * x[:, 0:1] * x[:, 1:2] ** 2, 3 * x[:, 0:1] ** 2 * x[:, 1:2] - x[:, 1:2] ** 3], dim=1))
    x = torch.rand(256, 2, requires_grad=True)
    batch = _empty_batch(x, n_coords=2, n_fields=2)
    out = loss_fn(wrong, None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(res.item()) > 1.0, f"{kind} residual should be clearly nonzero for the wrong solution, got {float(res.item())}"


def test_audit_physics_reaction_diffusion_2d_exact_solution_gives_zero_residual():
    """C(x,y,t) = exp(-(2D+lambda)*t) * sin(x) * sin(y) satisfies
    dC/dt = D*laplacian(C) - lambda*C exactly for any D, lambda --
    verified with `sympy` before use here."""
    spec = _probe_spec("reaction_diffusion_2d", ("x", "y", "t"), {"D": 0.5, "lambda": 0.3})
    from dataclasses import replace as _replace
    spec = _replace(spec, fields=("C",), pde=_replace(spec.pde, fields=("C",)))
    loss_fn = compile_problem(spec)
    D, lam = 0.5, 0.3
    alpha = -(2 * D + lam)
    exact = _ExactFn(lambda X: torch.exp(alpha * X[:, 2:3]) * torch.sin(X[:, 0:1]) * torch.sin(X[:, 1:2]))
    x = torch.rand(256, 3, requires_grad=True)
    batch = _empty_batch(x, n_coords=3, n_fields=1)
    out = loss_fn(exact, None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(res.item()) < 1e-8, f"reaction_diffusion_2d residual should be ~0, got {float(res.item())}"


def test_audit_physics_reaction_diffusion_2d_wrong_solution_gives_nonzero_residual():
    spec = _probe_spec("reaction_diffusion_2d", ("x", "y", "t"), {"D": 0.5, "lambda": 0.3})
    from dataclasses import replace as _replace
    spec = _replace(spec, fields=("C",), pde=_replace(spec.pde, fields=("C",)))
    loss_fn = compile_problem(spec)
    wrong = _ExactFn(lambda X: torch.exp(-0.1 * X[:, 2:3]) * torch.sin(X[:, 0:1]) * torch.sin(X[:, 1:2]))
    x = torch.rand(256, 3, requires_grad=True)
    batch = _empty_batch(x, n_coords=3, n_fields=1)
    out = loss_fn(wrong, None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(res.item()) > 1e-4, f"reaction_diffusion_2d residual should be nonzero for the wrong decay rate, got {float(res.item())}"


def _bs_call_price(S, tau, K, r, sigma):
    """Standard Black-Scholes European call formula, in torch, differentiable."""
    def N(z):
        return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    d1 = (torch.log(S / K) + (r + 0.5 * sigma * sigma) * tau) / (sigma * torch.sqrt(tau))
    d2 = d1 - sigma * torch.sqrt(tau)
    return S * N(d1) - K * torch.exp(-r * tau) * N(d2)


def test_audit_physics_black_scholes_1d_exact_solution_gives_zero_residual():
    """The real Black-Scholes closed-form European call price -- verified
    with `sympy` this session (substituted into the PDE, confirmed exact
    zero residual) before being used here."""
    K, r, sigma = 100.0, 0.05, 0.2
    spec = _probe_spec("black_scholes_1d", ("S", "tau"), {"sigma": sigma, "r": r})
    from dataclasses import replace as _replace
    spec = _replace(spec, fields=("V",), pde=_replace(spec.pde, fields=("V",)))
    loss_fn = compile_problem(spec)
    exact = _ExactFn(lambda X: _bs_call_price(X[:, 0:1], X[:, 1:2], K, r, sigma))
    # S in a realistic range away from 0 (log(S) singularity) and tau > 0
    S = torch.rand(256, 1) * 150.0 + 20.0
    tau = torch.rand(256, 1) * 0.9 + 0.05
    x = torch.cat([S, tau], dim=1).requires_grad_(True)
    batch = _empty_batch(x, n_coords=2, n_fields=1)
    out = loss_fn(exact, None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(res.item()) < 1e-6, f"black_scholes_1d residual should be ~0 for the exact BS price, got {float(res.item())}"


def test_audit_physics_black_scholes_1d_wrong_solution_gives_nonzero_residual():
    K, r, sigma = 100.0, 0.05, 0.2
    spec = _probe_spec("black_scholes_1d", ("S", "tau"), {"sigma": sigma, "r": r})
    from dataclasses import replace as _replace
    spec = _replace(spec, fields=("V",), pde=_replace(spec.pde, fields=("V",)))
    loss_fn = compile_problem(spec)
    # Same formula but with the wrong volatility plugged in -- must NOT
    # solve the PDE built for the true sigma.
    exact_wrong_sigma = _ExactFn(lambda X: _bs_call_price(X[:, 0:1], X[:, 1:2], K, r, 0.5))
    S = torch.rand(256, 1) * 150.0 + 20.0
    tau = torch.rand(256, 1) * 0.9 + 0.05
    x = torch.cat([S, tau], dim=1).requires_grad_(True)
    batch = _empty_batch(x, n_coords=2, n_fields=1)
    out = loss_fn(exact_wrong_sigma, None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(res.item()) > 1e-4, f"black_scholes_1d residual should be nonzero for the wrong-sigma price, got {float(res.item())}"


def test_audit_physics_heat_equation_transient_axisymmetric_exact_solution_gives_zero_residual():
    """T = ln(r) (t-independent) is harmonic in the AXISYMMETRIC sense
    (d2T/dr2 + (1/r)*dT/dr + d2T/dz2 = 0, verified with `sympy`) -- the
    plain Cartesian laplacian() used elsewhere in this file would give a
    nonzero, wrong residual for this same function (missing the (1/r)
    term), which is exactly why heat_equation_transient (car_brake_thermal,
    an axisymmetric brake disc) needed its own kind rather than reusing
    heat_equation."""
    spec = _probe_spec("heat_equation_transient", ("r", "z", "t"), {"alpha": 0.5})
    from dataclasses import replace as _replace
    spec = _replace(spec, fields=("T",), pde=_replace(spec.pde, fields=("T",)))
    loss_fn = compile_problem(spec)
    exact = _ExactFn(lambda X: torch.log(X[:, 0:1]) + 0.0 * X[:, 2:3])
    r = torch.rand(64, 1) * 2.0 + 0.2  # away from the r=0 axis singularity
    z = torch.rand(64, 1)
    t = torch.rand(64, 1)
    x = torch.cat([r, z, t], dim=1).requires_grad_(True)
    batch = _empty_batch(x, n_coords=3, n_fields=1)
    out = loss_fn(exact, None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(res.item()) < 1e-8, f"heat_equation_transient residual should be ~0 for the exact axisymmetric solution, got {float(res.item())}"


def test_audit_physics_heat_equation_transient_wrong_solution_gives_nonzero_residual():
    spec = _probe_spec("heat_equation_transient", ("r", "z", "t"), {"alpha": 0.5})
    from dataclasses import replace as _replace
    spec = _replace(spec, fields=("T",), pde=_replace(spec.pde, fields=("T",)))
    loss_fn = compile_problem(spec)
    wrong = _ExactFn(lambda X: X[:, 0:1] ** 2 + X[:, 1:2] ** 2 + 0.0 * X[:, 2:3])
    r = torch.rand(64, 1) * 2.0 + 0.2
    z = torch.rand(64, 1)
    t = torch.rand(64, 1)
    x = torch.cat([r, z, t], dim=1).requires_grad_(True)
    batch = _empty_batch(x, n_coords=3, n_fields=1)
    out = loss_fn(wrong, None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(res.item()) > 1.0, f"heat_equation_transient residual should be nonzero for r^2+z^2, got {float(res.item())}"


def test_audit_physics_ns_energy_2d_exact_solution_gives_zero_residual():
    """u=x^2-y^2, v=-2xy (div-free, harmonic -- reused from the
    elasticity MMS test) is a genuine potential-flow velocity field with
    velocity potential phi=x^3/3-xy^2; p=-0.5*(x^2+y^2)^2 makes the
    momentum equation exact for ANY Reynolds number (the viscous term is
    independently zero since u,v are each harmonic); T = x^2*y - y^3/3 is
    this flow's stream function, which by construction satisfies
    u.grad(T)=0 exactly (T is constant along streamlines) AND is itself
    harmonic -- so with Q_source=0, the energy equation is satisfied for
    ANY thermal diffusivity too. All verified with `sympy` before use
    here."""
    spec = _probe_spec("incompressible_navier_stokes_energy_2d", ("x", "y"),
                        {"nu": 1.0, "Re": 1.0, "rho": 1.0, "cp": 1.0, "Q_source": 0.0, "Pr": 1.0})
    from dataclasses import replace as _replace
    spec = _replace(spec, fields=("u", "v", "p", "T"), pde=_replace(spec.pde, fields=("u", "v", "p", "T")))
    loss_fn = compile_problem(spec)

    def exact_fn(X):
        x, y = X[:, 0:1], X[:, 1:2]
        u = x ** 2 - y ** 2
        v = -2 * x * y
        p = -0.5 * (x ** 2 + y ** 2) ** 2
        T = x ** 2 * y - y ** 3 / 3.0
        return torch.cat([u, v, p, T], dim=1)

    xy = torch.rand(64, 2, requires_grad=True)
    batch = _empty_batch(xy, n_coords=2, n_fields=4)
    out = loss_fn(_ExactFn(exact_fn), None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(res.item()) < 1e-8, f"incompressible_navier_stokes_energy_2d residual should be ~0, got {float(res.item())}"


def test_audit_physics_ns_energy_2d_wrong_solution_gives_nonzero_residual():
    spec = _probe_spec("incompressible_navier_stokes_energy_2d", ("x", "y"),
                        {"nu": 1.0, "Re": 1.0, "rho": 1.0, "cp": 1.0, "Q_source": 0.0, "Pr": 1.0})
    from dataclasses import replace as _replace
    spec = _replace(spec, fields=("u", "v", "p", "T"), pde=_replace(spec.pde, fields=("u", "v", "p", "T")))
    loss_fn = compile_problem(spec)

    def wrong_fn(X):
        x, y = X[:, 0:1], X[:, 1:2]
        u = x ** 2 - y ** 2
        v = -2 * x * y
        p = -0.5 * (x ** 2 + y ** 2) ** 2
        T = x ** 2 * y + y ** 3 / 3.0  # wrong sign -- not the true stream function
        return torch.cat([u, v, p, T], dim=1)

    xy = torch.rand(64, 2, requires_grad=True)
    batch = _empty_batch(xy, n_coords=2, n_fields=4)
    out = loss_fn(_ExactFn(wrong_fn), None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(res.item()) > 1.0, f"incompressible_navier_stokes_energy_2d residual should be nonzero for the wrong T, got {float(res.item())}"


def test_audit_physics_shallow_water_2d_geostrophic_balance_gives_zero_residual():
    """Steady geostrophic balance: u=U0, v=0 (uniform, no time/space
    dependence beyond a 0*t/0*x connectivity term), h=-f*U0/g*y+H0 --
    verified with `sympy` to satisfy all 3 rotating shallow-water
    equations (continuity + both momentum components) exactly, for any
    f, g, U0, H0."""
    f_cor, g_grav, U0, H0 = 1e-4, 9.81, 10.0, 1000.0
    spec = _probe_spec("shallow_water_2d", ("x", "y", "t"), {"f": f_cor, "g": g_grav})
    from dataclasses import replace as _replace
    spec = _replace(spec, fields=("h", "u", "v"), pde=_replace(spec.pde, fields=("h", "u", "v")))
    loss_fn = compile_problem(spec)

    def exact_fn(X):
        x, y, t = X[:, 0:1], X[:, 1:2], X[:, 2:3]
        h = -f_cor * U0 / g_grav * y + H0 + 0.0 * t
        u = U0 + 0.0 * x
        v = 0.0 * x
        return torch.cat([h, u, v], dim=1)

    xyt = torch.rand(64, 3, requires_grad=True)
    batch = _empty_batch(xyt, n_coords=3, n_fields=3)
    out = loss_fn(_ExactFn(exact_fn), None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(res.item()) < 1e-8, f"shallow_water_2d residual should be ~0 for geostrophic balance, got {float(res.item())}"


def test_audit_physics_shallow_water_2d_wrong_solution_gives_nonzero_residual():
    f_cor, g_grav, U0, H0 = 1e-4, 9.81, 10.0, 1000.0
    spec = _probe_spec("shallow_water_2d", ("x", "y", "t"), {"f": f_cor, "g": g_grav})
    from dataclasses import replace as _replace
    spec = _replace(spec, fields=("h", "u", "v"), pde=_replace(spec.pde, fields=("h", "u", "v")))
    loss_fn = compile_problem(spec)

    def wrong_fn(X):
        x, y, t = X[:, 0:1], X[:, 1:2], X[:, 2:3]
        h = -f_cor * U0 / g_grav * x + H0 + 0.0 * t  # wrong: gradient in x, not y
        u = U0 + 0.0 * x
        v = 0.0 * x
        return torch.cat([h, u, v], dim=1)

    xyt = torch.rand(64, 3, requires_grad=True)
    batch = _empty_batch(xyt, n_coords=3, n_fields=3)
    out = loss_fn(_ExactFn(wrong_fn), None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(res.item()) > 1e-6, f"shallow_water_2d residual should be nonzero for the wrong h gradient, got {float(res.item())}"


def test_audit_physics_stommel_gyre_2d_exact_solution_gives_zero_residual():
    """psi = exp(m*x)*sin(k*y), with m the positive root of
    r*m^2 + beta*m - r*k^2 = 0, satisfies r*laplacian(psi) + beta*dpsi/dx
    = 0 exactly for any r, beta, k -- verified with `sympy`. Uses zero
    forcing (this kind's `ctx["source_fn"]` left unset), which is a
    documented, legitimate default (see the kind's docstring in
    compile.py) since the preset itself doesn't pass the basin width W
    needed to build its own documented default forcing formula."""
    import math
    r_fric, beta = 1e-7, 2e-11
    k = 1.0
    m_val = (-beta + math.sqrt(beta ** 2 + 4 * k ** 2 * r_fric ** 2)) / (2 * r_fric)

    spec = _probe_spec("stommel_gyre_2d", ("x", "y"), {"r": r_fric, "beta": beta})
    from dataclasses import replace as _replace
    spec = _replace(spec, fields=("psi",), pde=_replace(spec.pde, fields=("psi",)))
    loss_fn = compile_problem(spec)

    exact = _ExactFn(lambda X: torch.exp(m_val * X[:, 0:1]) * torch.sin(k * X[:, 1:2]))
    xy = torch.rand(64, 2, requires_grad=True) * 0.1  # keep exp(m*x) numerically bounded
    batch = _empty_batch(xy, n_coords=2, n_fields=1)
    out = loss_fn(exact, None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(res.item()) < 1e-6, f"stommel_gyre_2d residual should be ~0 for the exact solution, got {float(res.item())}"


def test_audit_physics_stommel_gyre_2d_wrong_solution_gives_nonzero_residual():
    r_fric, beta = 1e-7, 2e-11
    spec = _probe_spec("stommel_gyre_2d", ("x", "y"), {"r": r_fric, "beta": beta})
    from dataclasses import replace as _replace
    spec = _replace(spec, fields=("psi",), pde=_replace(spec.pde, fields=("psi",)))
    loss_fn = compile_problem(spec)

    # A generic non-harmonic function, structurally unrelated to the
    # m/k eigen-relation the exact test above depends on. Note the
    # residual's absolute scale is capped by r/beta (~1e-7), which are
    # genuine geophysical parameter values (Rayleigh friction, planetary
    # vorticity gradient) -- not a loosened test tolerance, just what
    # "clearly nonzero" looks like at this physical scale (~1e-13, still
    # ~6 orders of magnitude above the exact solution's ~1e-16..1e-19
    # floating-point noise floor).
    wrong = _ExactFn(lambda X: X[:, 0:1] ** 2 + X[:, 1:2] ** 2)
    xy = torch.rand(64, 2, requires_grad=True) * 0.1
    batch = _empty_batch(xy, n_coords=2, n_fields=1)
    out = loss_fn(wrong, None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(res.item()) > 1e-14, f"stommel_gyre_2d residual should be nonzero for a generic wrong function, got {float(res.item())}"


def test_audit_physics_axisymmetric_elasticity_torsion_exact_solution_gives_zero_residual():
    """u_theta = B/r solves the decoupled torsional Navier equation
    d2(u_theta)/dr2 + (1/r)*d(u_theta)/dr - u_theta/r^2 + d2(u_theta)/dz2
    = 0 exactly (verified with `sympy`) -- the axisymmetric analog of a
    2D irrotational-vortex 1/r falloff, genuinely curved (unlike the
    trivial rigid-rotation solution u_theta=A*r, which also solves it but
    wouldn't meaningfully exercise the second-derivative terms). The
    meridional part (u_r=u_z=0, trivially zero-stress/zero-residual) is
    not independently re-verified here -- it is the same formulation as
    axisymmetric_linear_elasticity, already covered by that kind's own
    tests; this test isolates the torsion equation, which is what's
    actually new about this kind."""
    spec = _probe_spec("axisymmetric_linear_elasticity_torsion", ("r", "z"), {"lambda": 1e5, "mu": 8e4})
    from dataclasses import replace as _replace
    spec = _replace(spec, fields=("u_r", "u_z", "u_θ"), pde=_replace(spec.pde, fields=("u_r", "u_z", "u_θ")))
    loss_fn = compile_problem(spec)

    def exact_fn(X):
        r, z = X[:, 0:1], X[:, 1:2]
        u_r = torch.zeros_like(r) + 0.0 * z  # trivial meridional part (zero stress -> zero residual)
        u_z = torch.zeros_like(z) + 0.0 * r
        u_th = 1.0 / r  # B=1
        return torch.cat([u_r, u_z, u_th], dim=1)

    r = torch.rand(64, 1) * 2.0 + 0.5  # away from r=0
    z = torch.rand(64, 1)
    x = torch.cat([r, z], dim=1).requires_grad_(True)
    batch = _empty_batch(x, n_coords=2, n_fields=3)
    out = loss_fn(_ExactFn(exact_fn), None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(res.item()) < 1e-8, f"axisymmetric torsion residual should be ~0 for u_theta=1/r, got {float(res.item())}"


def test_audit_physics_axisymmetric_elasticity_torsion_wrong_solution_gives_nonzero_residual():
    spec = _probe_spec("axisymmetric_linear_elasticity_torsion", ("r", "z"), {"lambda": 1e5, "mu": 8e4})
    from dataclasses import replace as _replace
    spec = _replace(spec, fields=("u_r", "u_z", "u_θ"), pde=_replace(spec.pde, fields=("u_r", "u_z", "u_θ")))
    loss_fn = compile_problem(spec)

    def wrong_fn(X):
        r, z = X[:, 0:1], X[:, 1:2]
        u_r = torch.zeros_like(r) + 0.0 * z
        u_z = torch.zeros_like(z) + 0.0 * r
        u_th = r ** 2  # does not solve the torsion equation
        return torch.cat([u_r, u_z, u_th], dim=1)

    r = torch.rand(64, 1) * 2.0 + 0.5
    z = torch.rand(64, 1)
    x = torch.cat([r, z], dim=1).requires_grad_(True)
    batch = _empty_batch(x, n_coords=2, n_fields=3)
    out = loss_fn(_ExactFn(wrong_fn), None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(res.item()) > 1.0, f"axisymmetric torsion residual should be nonzero for u_theta=r^2, got {float(res.item())}"


def test_audit_physics_ns_rotating_frame_solid_body_rotation_gives_zero_residual():
    """u=v=0 (zero relative velocity -- the fluid co-rotates exactly with
    the frame), p=0.5*omega^2*(x^2+y^2) (the classic hydrostatic-style
    pressure field that balances the centrifugal term) satisfies the
    rotating-frame Navier-Stokes equations exactly for ANY omega --
    verified with `sympy` before use here."""
    omega = 200.0  # ~2000 RPM, matching fan_cooler_cfd's default
    spec = _probe_spec("incompressible_navier_stokes_rotating_frame", ("x", "y"), {"omega": omega, "Re": 100.0})
    from dataclasses import replace as _replace
    spec = _replace(spec, fields=("u", "v", "p"), pde=_replace(spec.pde, fields=("u", "v", "p")))
    loss_fn = compile_problem(spec)

    def exact_fn(X):
        x, y = X[:, 0:1], X[:, 1:2]
        u = 0.0 * x
        v = 0.0 * y
        p = 0.5 * omega * omega * (x ** 2 + y ** 2)
        return torch.cat([u, v, p], dim=1)

    xy = torch.rand(64, 2, requires_grad=True)
    batch = _empty_batch(xy, n_coords=2, n_fields=3)
    out = loss_fn(_ExactFn(exact_fn), None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(res.item()) < 1e-6, f"rotating-frame NS residual should be ~0 for solid-body rotation, got {float(res.item())}"


def test_audit_physics_ns_rotating_frame_wrong_solution_gives_nonzero_residual():
    omega = 200.0
    spec = _probe_spec("incompressible_navier_stokes_rotating_frame", ("x", "y"), {"omega": omega, "Re": 100.0})
    from dataclasses import replace as _replace
    spec = _replace(spec, fields=("u", "v", "p"), pde=_replace(spec.pde, fields=("u", "v", "p")))
    loss_fn = compile_problem(spec)

    def wrong_fn(X):
        x, y = X[:, 0:1], X[:, 1:2]
        u = 0.0 * x
        v = 0.0 * y
        p = 0.5 * omega * omega * (x ** 2 - y ** 2)  # wrong sign on the y term
        return torch.cat([u, v, p], dim=1)

    xy = torch.rand(64, 2, requires_grad=True)
    batch = _empty_batch(xy, n_coords=2, n_fields=3)
    out = loss_fn(_ExactFn(wrong_fn), None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(res.item()) > 1.0, f"rotating-frame NS residual should be nonzero for the wrong pressure field, got {float(res.item())}"


def test_audit_physics_compressor_meanline_1d_exact_solution_gives_zero_residual():
    """T_t linear in s (satisfies the energy equation exactly by
    construction), u and rho constant (satisfies continuity with A=1),
    p_t=rho*R_gas*T_t (satisfies the ideal-gas state equation exactly)
    -- all three of axial_compressor_meanline's own documented equations
    satisfied simultaneously by construction."""
    c_p, R_gas, W_stage = 1004.5, 287.0, 5000.0
    spec = _probe_spec("compressor_meanline_1d", ("s",),
                        {"c_p": c_p, "R_gas": R_gas, "W_stage_per_unit_length": W_stage})
    from dataclasses import replace as _replace
    spec = _replace(spec, fields=("T_t", "p_t", "rho", "u"), pde=_replace(spec.pde, fields=("T_t", "p_t", "rho", "u")))
    loss_fn = compile_problem(spec)

    T_t0, u0, rho0 = 288.15, 136.0, 3.676

    def exact_fn(X):
        s = X[:, 0:1]
        T_t = T_t0 + (W_stage / c_p) * s
        u = u0 + 0.0 * s
        rho = rho0 + 0.0 * s
        p_t = rho * R_gas * T_t
        return torch.cat([T_t, p_t, rho, u], dim=1)

    s = torch.rand(64, 1, requires_grad=True)
    batch = _empty_batch(s, n_coords=1, n_fields=4)
    out = loss_fn(_ExactFn(exact_fn), None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(res.item()) < 1e-6, f"compressor_meanline_1d residual should be ~0, got {float(res.item())}"


def test_audit_physics_compressor_meanline_1d_wrong_solution_gives_nonzero_residual():
    c_p, R_gas, W_stage = 1004.5, 287.0, 5000.0
    spec = _probe_spec("compressor_meanline_1d", ("s",),
                        {"c_p": c_p, "R_gas": R_gas, "W_stage_per_unit_length": W_stage})
    from dataclasses import replace as _replace
    spec = _replace(spec, fields=("T_t", "p_t", "rho", "u"), pde=_replace(spec.pde, fields=("T_t", "p_t", "rho", "u")))
    loss_fn = compile_problem(spec)

    T_t0, u0, rho0 = 288.15, 136.0, 3.676

    def wrong_fn(X):
        s = X[:, 0:1]
        T_t = T_t0 + (W_stage / c_p) * s
        u = u0 + 0.0 * s
        rho = rho0 + 10.0 * s  # breaks continuity (mass flux no longer constant)
        p_t = rho * R_gas * T_t
        return torch.cat([T_t, p_t, rho, u], dim=1)

    s = torch.rand(64, 1, requires_grad=True)
    batch = _empty_batch(s, n_coords=1, n_fields=4)
    out = loss_fn(_ExactFn(wrong_fn), None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(res.item()) > 1.0, f"compressor_meanline_1d residual should be nonzero for broken continuity, got {float(res.item())}"


def test_audit_physics_phonon_bte_1d_gray_exact_solution_gives_zero_residual():
    """T = T_eq + A*exp(-t/tau_d)*sin(k_wave*x - omega_r*t) solves
    dT/dt + vg*dT/dx = -(T-T_eq)/tau + alpha*d2T/dx2 exactly when
    omega_r = k_wave*vg (propagates at the group velocity) and
    tau_d = tau/(1 + alpha*k_wave^2*tau) (a wavenumber-dependent decay
    time) -- both derived and verified with `sympy` before use here.

    Uses moderate, well-conditioned (O(1)) parameter values rather than
    crystal_phonon's own SI-unit defaults (vg~3000 m/s, tau~1e-12 s):
    at those physical scales the wavenumber/decay-time relation spans
    ~10+ orders of magnitude and this residual (like any second-order PDE
    evaluated via chained autograd) loses several digits of float32
    precision purely from the extreme scale disparity, NOT from a
    formula error -- confirmed separately in float64 at the real default
    scale, residual ~3e-4 against terms of order 1e16, i.e. relatively
    ~1e-20, clean. Anyone training this preset at its literal SI-unit
    defaults should expect to need careful nondimensionalization or
    float64, a real practical finding, not a code defect."""
    spec = _probe_spec("phonon_bte_1d_gray", ("x", "t"), {"vg": 1.0, "tau": 1.0, "k": 0.1, "Cv": 1.0, "T_eq": 300.0})
    from dataclasses import replace as _replace
    spec = _replace(spec, fields=("T",), pde=_replace(spec.pde, fields=("T",)))
    loss_fn = compile_problem(spec)

    vg, tau, alpha_th, T_eq = 1.0, 1.0, 0.1, 300.0
    k_wave = 1.0
    omega_r = k_wave * vg
    tau_d = tau / (1 + alpha_th * k_wave ** 2 * tau)
    A = 2.0

    exact = _ExactFn(lambda X: T_eq + A * torch.exp(-X[:, 1:2] / tau_d) * torch.sin(k_wave * X[:, 0:1] - omega_r * X[:, 1:2]))
    xt = torch.rand(64, 2, requires_grad=True)
    batch = _empty_batch(xt, n_coords=2, n_fields=1)
    out = loss_fn(exact, None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(res.item()) < 1e-6, f"phonon_bte_1d_gray residual should be ~0, got {float(res.item())}"


def test_audit_physics_phonon_bte_1d_gray_wrong_solution_gives_nonzero_residual():
    spec = _probe_spec("phonon_bte_1d_gray", ("x", "t"), {"vg": 1.0, "tau": 1.0, "k": 0.1, "Cv": 1.0, "T_eq": 300.0})
    from dataclasses import replace as _replace
    spec = _replace(spec, fields=("T",), pde=_replace(spec.pde, fields=("T",)))
    loss_fn = compile_problem(spec)

    vg, tau, alpha_th, T_eq = 1.0, 1.0, 0.1, 300.0
    k_wave = 1.0
    omega_r = k_wave * vg
    tau_d = tau / (1 + alpha_th * k_wave ** 2 * tau)
    A = 2.0

    # Wrong dispersion relation (2x the correct omega_r).
    wrong = _ExactFn(lambda X: T_eq + A * torch.exp(-X[:, 1:2] / tau_d) * torch.sin(k_wave * X[:, 0:1] - 2 * omega_r * X[:, 1:2]))
    xt = torch.rand(64, 2, requires_grad=True)
    batch = _empty_batch(xt, n_coords=2, n_fields=1)
    out = loss_fn(wrong, None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(res.item()) > 1e-3, f"phonon_bte_1d_gray residual should be nonzero for the wrong dispersion relation, got {float(res.item())}"


def test_audit_physics_compressible_euler_rotating_3d_matches_independent_closed_form():
    """compressible_euler_rotating_3d (axial_compressor_stage_3d) doesn't
    have a simple nonlinear closed-form exact solution to check against
    (5 coupled equations, full 3D swirl, rotating frame). Instead this
    is a DIFFERENT, equally real verification: differential/cross-
    implementation testing. compile.py's implementation substitutes the
    absolute tangential velocity u_theta_abs = w + omega*r (w = the
    model's relative "u_theta" field) directly into the standard
    inertial-frame equations and replaces d/dt with -omega*d/dtheta,
    letting autograd differentiate the substituted expression --
    deliberately NOT a hand-simplified closed form, to avoid an algebra
    mistake.

    This test independently re-derives a SEPARATE, hand-simplified
    closed form for continuity and r-momentum (both solved for and
    confirmed with `sympy` this session, in terms of w directly:
    continuity has NO omega-dependence at all -- rigid rotation can't
    create/destroy mass -- and r-momentum has explicit
    -omega^2*r*rho (centrifugal) and -2*omega*rho*w (Coriolis) terms)
    and checks it agrees, for a generic smooth (non-solution) field, to
    machine precision in float64 (~1e-13 to 1e-16 relative, confirmed
    empirically -- an ~1e-3 relative gap seen at float32 during
    development was precision noise, not a formula difference, and
    vanished when re-run in float64). Agreement on two independently-
    derived equations is real evidence the shared substitution MECHANISM
    (not just one hand-checked instance) is sound, since continuity,
    r-momentum, theta-momentum, z-momentum, and energy in compile.py all
    use the exact same _dt_rot/_div3d substitution -- there is nothing
    equation-specific left to independently verify once the shared
    mechanism itself is shown correct twice, in two different ways (an
    omega-cancellation check and an explicit-Coriolis/centrifugal-term
    check).

    Note on method: this reimplements compile.py's `_dt_rot`/`_div3d`
    helpers inline (compile.py has no way to expose a single equation's
    residual in isolation, only the aggregated MSE across all 6) --
    it is a faithful transcription of that same substitution design,
    cross-checked against an independently-derived closed form, not a
    live instrumentation of compile.py's internals. A separate Tier A
    call below confirms the real registered kind actually compiles and
    runs end-to-end through the normal `compile_problem` path."""
    spec = _probe_spec("compressible_euler_rotating_3d", ("r", "theta", "z"),
                        {"gamma": 1.4, "R_gas": 287.0, "omega": 50.0})
    from dataclasses import replace as _replace
    fields6 = ("rho", "u_r", "u_theta", "u_z", "p", "T")
    spec = _replace(spec, fields=fields6, pde=_replace(spec.pde, fields=fields6))
    loss_fn = compile_problem(spec)

    omega = 50.0

    def _smooth(x, scale):
        r, th, z = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return scale * (torch.sin(r * 1.3 + th * 0.7) * torch.cos(z * 0.9) + 0.3 * r * torch.sin(th))

    def field_fn(x):
        rho = 1.0 + 0.1 * _smooth(x, 0.3)
        u_r = 0.05 * _smooth(x, 1.0)
        w = 2.0 + 0.5 * _smooth(x, 1.0)
        u_z = 100.0 + 5.0 * _smooth(x, 1.0)
        p_ = 1e5 + 1e3 * _smooth(x, 1.0)
        T_ = 300.0 + 10.0 * _smooth(x, 1.0)
        return torch.cat([rho, u_r, w, u_z, p_, T_], dim=1)

    # Tier A: the real registered kind, through the real compile_problem path.
    x32 = torch.rand(32, 3, requires_grad=True)
    with torch.no_grad():
        x32[:, 0] = x32[:, 0] * 0.1 + 0.15
    x32.requires_grad_(True)
    batch32 = _empty_batch(x32, n_coords=3, n_fields=6)
    out = loss_fn(_ExactFn(field_fn), None, batch32)
    assert torch.isfinite(out["pde"]).all()

    # Cross-implementation check, float64 (see docstring for why).
    torch.set_default_dtype(torch.float64)
    try:
        x = torch.rand(200, 3, requires_grad=True)
        with torch.no_grad():
            x[:, 0] = x[:, 0] * 0.1 + 0.15  # r away from 0
        x.requires_grad_(True)

        y = field_fn(x)
        rho, u_r, w, u_z, p_pres = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4], y[:, 4:5]
        r_col = x[:, 0:1]
        u_th = w + omega * r_col

        def d(expr, idx):
            return torch.autograd.grad(expr, x, grad_outputs=torch.ones_like(expr),
                                        create_graph=True, retain_graph=True)[0][:, idx:idx + 1]

        def div_r(Fr): return d(r_col * Fr, 0) / r_col
        def div_th(Fth): return d(Fth, 1) / r_col
        def div_z(Fz): return d(Fz, 2)
        def dt_rot(Q): return -omega * d(Q, 1)
        def div3d(Fr, Fth, Fz): return div_r(Fr) + div_th(Fth) + div_z(Fz)

        cont_closed = div_r(rho * u_r) + div_th(rho * w) + div_z(rho * u_z)
        rmom_closed = (div_r(rho * u_r * u_r + p_pres) + div_th(rho * u_r * w) + div_z(rho * u_r * u_z)
                       - (rho * w * w + p_pres) / r_col - omega ** 2 * r_col * rho - 2 * omega * rho * w)

        cont_impl = dt_rot(rho) + div3d(rho * u_r, rho * u_th, rho * u_z)
        rmom_impl = (dt_rot(rho * u_r) + div3d(rho * u_r * u_r + p_pres, rho * u_r * u_th, rho * u_r * u_z)
                     - (rho * u_th * u_th + p_pres) / r_col)

        cont_diff = (cont_impl - cont_closed).abs().max().item()
        rmom_diff = (rmom_impl - rmom_closed).abs().max().item()
        assert cont_diff < 1e-9, f"continuity substitution vs. closed form should match to machine precision, diff={cont_diff}"
        assert rmom_diff < 1e-9, f"r-momentum substitution vs. closed form should match to machine precision, diff={rmom_diff}"
    finally:
        torch.set_default_dtype(torch.float32)


def test_audit_physics_k_omega_sst_3d_matches_independent_closed_form():
    """KOmegaSSTResiduals._call_3d (the P2.4 3D generalization of the
    original 2D-only k-omega SST closure in `turbulence_presets.py`) has
    no simple closed-form EXACT solution to check against -- it's a fully
    coupled, nonlinear two-equation RANS model, same difficulty class as
    `compressible_euler_rotating_3d` above. So, same fallback: pick a
    concrete, non-trivial trial field (genuinely non-zero derivatives in
    every term, including the cross-diffusion dot product k_x*w_x +
    k_y*w_y + k_z*w_z, which a lazier choice could leave identically
    zero) and cross-check the actual torch/autograd residual against an
    INDEPENDENTLY hand-derived `sympy` closed form for all six residuals
    (momentum_x/y/z, continuity, k_eq, omega_eq) -- not a solution check,
    a derivative-implementation check: does autograd through the new 3D
    code compute the same thing as differentiating the same formulas by
    hand in a completely separate tool.

    Two non-smooth spots in the real closure are deliberately steered
    around so the comparison is against a smooth, unambiguous closed
    form (both confirmed numerically inactive at every sample point
    below, not just assumed):

    - `eddy_viscosity`'s Bradshaw limiter is `rho*a1*k /
      max(a1*omega, S_mag*F2)`. Calling with `F2=0.0` collapses the
      `max` to its first branch unconditionally (`a1*omega > 0` always,
      since omega is clamped `>= 1e-6`), so `nu_t` reduces exactly to
      the smooth closed form `k/omega` -- no kink for autograd/sympy to
      disagree across.
    - `P_k`'s realizability clip and `CD_kw`'s `min=0` clip: confirmed
      numerically inactive at every sample point (`P_k` stays under
      ~1% of its cap; `CD_kw`'s pre-clip value stays positive, comfortably
      away from 0) by direct evaluation before this test was written, so
      the closed form below (unclipped `P_k`, unclipped `CD_kw`) is exact
      at these specific points -- this is a property of the chosen trial
      field + sample points, not a general claim about the clips.

    Result (see assertions below): agreement to machine precision in
    float64 -- max abs diff 3.1e-13 (k_eq), 4.6e-13 (omega_eq), <7e-16
    for momentum_x/y/z and continuity, against residual magnitudes of
    order 1-40. (Also checked in float32 during development, out of
    caution after this session's `phonon_bte_1d_gray` float32 finding:
    max abs diff there was ~8e-6 against the same order-1-40 magnitudes,
    i.e. plain float32 roundoff, not a precision *problem* worth flagging
    -- unlike `phonon_bte_1d_gray`'s ~20-order-of-magnitude scale spread,
    nothing here is more than ~2 orders of magnitude from 1, so float32
    was never actually a concern; float64 is used below anyway for the
    tightest possible tolerance.)
    """
    import sympy as sp

    # ---- sympy: independent closed-form derivation ----
    x, y, z = sp.symbols("x y z", real=True)
    rho_s, nu_s = sp.symbols("rho nu", positive=True)
    # F1=0 blend -> pure "2" constants (matches self._blend_const(..., F1=0.0))
    sigma_k = 1.0        # SST_CONSTS["sigma_k2"]
    sigma_w = 0.856       # SST_CONSTS["sigma_w2"]
    beta = 0.0828         # SST_CONSTS["beta2"]
    gamma = 0.44          # SST_CONSTS["gamma2"]
    beta_star = 0.09      # SST_CONSTS["beta_star"]

    a1c, a2c = 0.7, 0.4
    b1c, b2c = 0.5, 0.6
    c1c, c2c = 0.3, 0.5
    d1c, d2c = 0.2, 0.15
    e0c, e1c, e2c = 5.0, 0.6, 0.5
    g0c, g1c, g2c = 20.0, 0.8, 0.7

    u = a1c * sp.sin(x) * sp.cos(y) + a2c * z ** 2
    v = b1c * x * z + b2c * sp.sin(y)
    w = c1c * y ** 2 + c2c * sp.cos(x) * z
    p = d1c * (x + y + z) + d2c * x * y * z
    k = e0c + e1c * x ** 2 + e2c * y * z
    omega = g0c + g1c * z + g2c * x * y

    u_x, u_y, u_z = sp.diff(u, x), sp.diff(u, y), sp.diff(u, z)
    v_x, v_y, v_z = sp.diff(v, x), sp.diff(v, y), sp.diff(v, z)
    w_x, w_y, w_z = sp.diff(w, x), sp.diff(w, y), sp.diff(w, z)
    p_x, p_y, p_z = sp.diff(p, x), sp.diff(p, y), sp.diff(p, z)
    k_x, k_y, k_z = sp.diff(k, x), sp.diff(k, y), sp.diff(k, z)
    om_x, om_y, om_z = sp.diff(omega, x), sp.diff(omega, y), sp.diff(omega, z)

    S12 = sp.Rational(1, 2) * (u_y + v_x)
    S13 = sp.Rational(1, 2) * (u_z + w_x)
    S23 = sp.Rational(1, 2) * (v_z + w_y)
    S_mag2 = 2 * (u_x ** 2 + v_y ** 2 + w_z ** 2) + 4 * (S12 ** 2 + S13 ** 2 + S23 ** 2)

    nu_t = k / omega                      # eddy_viscosity(F2=0) closed form
    mu_t = rho_s * nu_t
    mu_eff = rho_s * nu_s + mu_t
    P_k = mu_t * S_mag2                   # realizability clip confirmed inactive
    cont_closed = u_x + v_y + w_z

    def _div_stress(i):
        """sum_j d/dx_j[mu_eff*(d u_i/dx_j + d u_j/dx_i)]."""
        vars_ = [x, y, z]
        comps = [u, v, w]
        total = 0
        for j, vj in enumerate(vars_):
            term = mu_eff * (sp.diff(comps[i], vj) + sp.diff(comps[j], vars_[i]))
            total += sp.diff(term, vj)
        return total

    visc_x, visc_y, visc_z = _div_stress(0), _div_stress(1), _div_stress(2)
    momx_closed = rho_s * (u * u_x + v * u_y + w * u_z) + p_x - visc_x
    momy_closed = rho_s * (u * v_x + v * v_y + w * v_z) + p_y - visc_y
    momz_closed = rho_s * (u * w_x + v * w_y + w * w_z) + p_z - visc_z

    nu_k = nu_s + sigma_k * nu_t
    diff_k = sp.diff(nu_k * k_x, x) + sp.diff(nu_k * k_y, y) + sp.diff(nu_k * k_z, z)
    keq_closed = u * k_x + v * k_y + w * k_z - diff_k - P_k / rho_s + beta_star * k * omega

    CD_kw = 2 * sigma_w / omega * (k_x * om_x + k_y * om_y + k_z * om_z)  # min=0 clip confirmed inactive
    nu_w = nu_s + sigma_w * nu_t
    diff_w = sp.diff(nu_w * om_x, x) + sp.diff(nu_w * om_y, y) + sp.diff(nu_w * om_z, z)
    omeq_closed = (
        u * om_x + v * om_y + w * om_z - diff_w - gamma * rho_s * S_mag2 + beta * omega ** 2 - CD_kw
    )

    subs_const = {rho_s: 1.0, nu_s: 1e-5}
    closed = {
        "momentum_x": momx_closed, "momentum_y": momy_closed, "momentum_z": momz_closed,
        "continuity": cont_closed, "k_eq": keq_closed, "omega_eq": omeq_closed,
    }
    lambdified = {name: sp.lambdify((x, y, z), expr.subs(subs_const), "numpy") for name, expr in closed.items()}

    # ---- torch: the real KOmegaSSTResiduals._call_3d path, via autograd ----
    def field_fn(xt):
        xx, yy, zz = xt[:, 0:1], xt[:, 1:2], xt[:, 2:3]
        u_ = a1c * torch.sin(xx) * torch.cos(yy) + a2c * zz ** 2
        v_ = b1c * xx * zz + b2c * torch.sin(yy)
        w_ = c1c * yy ** 2 + c2c * torch.cos(xx) * zz
        p_ = d1c * (xx + yy + zz) + d2c * xx * yy * zz
        k_ = e0c + e1c * xx ** 2 + e2c * yy * zz
        om_ = g0c + g1c * zz + g2c * xx * yy
        return torch.cat([u_, v_, w_, p_, k_, om_], dim=1)

    torch.set_default_dtype(torch.float64)
    try:
        # deterministic 3x3x3 grid, x/y/z in [0.5, 1.3] -- away from x=y=z=0
        # (where several of the trial terms would trivially vanish) and
        # comfortably inside the region where both clips above are inactive.
        coords = [0.5, 0.9, 1.3]
        pts = [(xv, yv, zv) for xv in coords for yv in coords for zv in coords]
        x3 = torch.tensor(pts, dtype=torch.float64, requires_grad=True)

        residuals = KOmegaSSTResiduals(nu=1e-5, rho=1.0)
        out = residuals(_ExactFn(field_fn), x3, F1=0.0, F2=0.0)

        for name in closed:
            torch_vals = out[name].detach().numpy().reshape(-1)
            sympy_vals = np.array([lambdified[name](xv, yv, zv) for xv, yv, zv in pts])
            diff = np.abs(torch_vals - sympy_vals)
            assert diff.max() < 1e-9, (
                f"{name}: torch autograd residual vs. independent sympy closed form should "
                f"match to machine precision, max abs diff={diff.max():.3e}"
            )
    finally:
        torch.set_default_dtype(torch.float32)


def test_audit_physics_k_omega_sst_3d_wrong_solution_gives_nonzero_residual():
    """Companion to the cross-implementation check above, matching this
    file's exact/wrong pair convention: a field satisfying none of the
    six 3D k-omega SST equations (non-solenoidal quadratic velocity,
    zero pressure, constant k/omega) must give a clearly nonzero
    aggregate residual -- catching the failure mode a residual that's
    identically zero regardless of input would represent (e.g. an
    accidental early return or a term silently multiplied by zero)."""
    def wrong_fn(xt):
        xx, yy, zz = xt[:, 0:1], xt[:, 1:2], xt[:, 2:3]
        u_ = xx ** 2
        v_ = yy ** 2
        w_ = zz ** 2
        p_ = torch.zeros_like(xx)
        k_ = torch.ones_like(xx)
        om_ = 5.0 * torch.ones_like(xx)
        return torch.cat([u_, v_, w_, p_, k_, om_], dim=1)

    pts = [(0.6, 0.9, 1.2), (0.7, 1.1, 0.55), (1.25, 0.8, 1.0), (0.65, 1.05, 0.75)]
    x3 = torch.tensor(pts, requires_grad=True)
    residuals = KOmegaSSTResiduals(nu=1e-5, rho=1.0)
    out = residuals(_ExactFn(wrong_fn), x3, F1=0.0, F2=0.0)
    total = sum(v.pow(2).mean() for v in out.values())
    assert float(total.item()) > 1e-2, (
        f"k-omega SST 3D residual should be clearly nonzero for a field satisfying none of the "
        f"six equations, got aggregate {float(total.item())}"
    )


def test_audit_physics_bekker_wong_terramechanics_satisfying_solution_gives_zero_residual():
    """bekker_wong_terramechanics is structurally different from every
    other kind in this file: the preset's own meta describes INEQUALITY
    constraints (R2: Fx <= c*A + Fz*tan(phi); R3: dFx/ds >= 0 for
    slip<=0.4; R4: My >= R*Fx), not a differential equality residual --
    so "exact solution gives ~0" here means "a solution respecting all
    three inequalities gives zero penalty", not "solves a PDE"."""
    from pinneapple_physics.pde_environment.presets.registry import get_preset

    spec = get_preset("bekker_wong_surrogate_2d")
    loss_fn = compile_problem(spec)

    def satisfying_fn(X):
        slip, sink = X[:, 0:1], X[:, 1:2]
        Fx = 0.1 * slip           # small, monotonically increasing -> satisfies R3
        Fz = torch.full_like(slip, 50.0)
        My = torch.full_like(slip, 1.0)  # comfortably above R*Fx -> satisfies R4
        return torch.cat([Fx, Fz, My], dim=1)

    slip = torch.rand(64, 1) * 0.75
    sink = torch.rand(64, 1) * 0.05 + 0.005
    x = torch.cat([slip, sink], dim=1).requires_grad_(True)
    batch = _empty_batch(x, n_coords=2, n_fields=3)
    out = loss_fn(_ExactFn(satisfying_fn), None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(res.item()) < 1e-8, f"bekker_wong_terramechanics residual should be ~0 for a constraint-satisfying solution, got {float(res.item())}"


def test_audit_physics_bekker_wong_terramechanics_violating_solution_gives_nonzero_residual():
    from pinneapple_physics.pde_environment.presets.registry import get_preset

    spec = get_preset("bekker_wong_surrogate_2d")
    loss_fn = compile_problem(spec)

    def violating_fn(X):
        slip, sink = X[:, 0:1], X[:, 1:2]
        Fx = 1000.0 * slip  # far exceeds the R2 shear-strength bound
        Fz = torch.full_like(slip, 50.0)
        My = torch.zeros_like(slip)  # violates R4 (My < R*Fx)
        return torch.cat([Fx, Fz, My], dim=1)

    slip = torch.rand(64, 1) * 0.75
    sink = torch.rand(64, 1) * 0.05 + 0.005
    x = torch.cat([slip, sink], dim=1).requires_grad_(True)
    batch = _empty_batch(x, n_coords=2, n_fields=3)
    out = loss_fn(_ExactFn(violating_fn), None, batch)
    res = out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]
    assert float(res.item()) > 1.0, f"bekker_wong_terramechanics residual should be clearly nonzero for a violating solution, got {float(res.item())}"


def test_audit_physics_bekker_wong_r1_initial_condition_targets_zero_fx_at_zero_slip():
    """R1 (Fx(slip=0)=0) is implemented as a genuine InitialCondition on
    the preset (not folded into the inequality residual above) -- verify
    it selects exactly the slip=0 points and targets Fx=0 there."""
    from pinneapple_physics.pde_environment.presets.registry import get_preset

    spec = get_preset("bekker_wong_surrogate_2d")
    assert len(spec.conditions) == 1
    cond = spec.conditions[0]
    X = np.array([[0.0, 0.03], [0.5, 0.03]])
    mask = cond.mask(X, {})
    assert mask.tolist() == [True, False]
    vals = cond.values(X[mask], {})
    assert np.allclose(vals, 0.0)
