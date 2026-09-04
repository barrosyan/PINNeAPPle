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

import torch
import torch.nn as nn

from pinneapple_physics.pinn_solver.compiler.compile import compile_problem
from pinneapple_physics.pde_environment.presets.academics import laplace_2d_default


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
