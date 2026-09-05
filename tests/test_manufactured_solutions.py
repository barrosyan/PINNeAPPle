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

import pytest
import torch
import torch.nn as nn

from pinneapple_physics.pinn_solver.compiler.compile import compile_problem
from pinneapple_physics.pde_environment.presets.academics import laplace_2d_default
from pinneapple_physics.pde_environment.spec import PDETermSpec, ProblemSpec
from pinneapple_physics.pde_environment.scales import ScaleSpec


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
