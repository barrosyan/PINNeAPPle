"""pinneapple_systems.component_modeling.physics_residuals — generic,
closed-form PDE residuals for a handful of standard physical constraints,
computed via autograd against any differentiable model's coordinate ->
field(s) map.

Each function has the signature ``residual(model, coords, **params) ->
Tensor`` and returns a residual tensor that should be driven to zero (e.g.
via ``(residual(...) ** 2).mean()`` as a training loss term). They make no
assumption about what `model` represents beyond the shape of its output
columns — no dependency on any particular component, registry, or problem
catalog.

To use with ``pinneapple_neural.architectures.pinns.base.PINNBase``'s
``physics_loss(physics_fn=..., physics_data=...)`` contract, wrap one of
these as::

    def physics_fn(model, data):
        r = heat_conduction_residual(model, data["coords"], k=50.0, Q=0.0)
        return (r ** 2).mean()

    loss_dict = pinn_model.physics_loss(physics_fn=physics_fn,
                                         physics_data={"coords": coords})

A missing/unsatisfiable residual raises a loud ``NotImplementedError`` —
never a silent zero (a silent zero would make the physics loss term
vanish and defeat its purpose).
"""
from __future__ import annotations

from typing import Any

import torch


def _unwrap(out: Any) -> torch.Tensor:
    return out.y if hasattr(out, "y") else out


def _grad(f: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Full gradient of scalar-per-row f (N,1) w.r.t. every column of coords
    (N,dim) -> (N,dim)."""
    return torch.autograd.grad(f, coords, torch.ones_like(f), create_graph=True, retain_graph=True)[0]


def _second_derivative(gf: torch.Tensor, idx: int, coords: torch.Tensor) -> torch.Tensor:
    """d/dx_idx of an already-computed gradient column -> d2f/dx_idx^2."""
    col = gf[:, idx:idx + 1]
    return torch.autograd.grad(col, coords, torch.ones_like(col), create_graph=True, retain_graph=True)[0][:, idx:idx + 1]


def incompressible_continuity_residual(model: Any, coords: torch.Tensor) -> torch.Tensor:
    """Mass-conservation (div(u) = 0) residual for a velocity-field output.
    Assumes the model's first `dim` output columns are velocity components
    aligned with the first `dim` coordinate axes (u, v, [w])."""
    coords = coords.requires_grad_(True)
    out = _unwrap(model(coords))
    dim = coords.shape[1]
    if out.shape[1] < dim:
        raise NotImplementedError(
            f"Incompressible-flow residual needs >= {dim} output channels (one velocity "
            f"component per coordinate axis), got {out.shape[1]}."
        )
    div = torch.zeros(coords.shape[0], 1, dtype=out.dtype, device=out.device)
    for i in range(dim):
        gi = _grad(out[:, i:i + 1], coords)
        div = div + gi[:, i:i + 1]
    return div


def heat_conduction_residual(model: Any, coords: torch.Tensor, *, k: float = 50.0, Q: float = 0.0) -> torch.Tensor:
    """Steady-state heat diffusion residual: k*laplacian(T) + Q, T = model's
    first output column, laplacian summed over every axis of `coords`
    (works for any spatial dimension)."""
    coords = coords.requires_grad_(True)
    out = _unwrap(model(coords))
    if out.shape[1] < 1:
        raise NotImplementedError("Heat-conduction residual needs >= 1 output channel (temperature).")
    T = out[:, 0:1]
    gT = _grad(T, coords)
    laplacian = sum(_second_derivative(gT, i, coords) for i in range(coords.shape[1]))
    return k * laplacian + Q


def linear_elasticity_residual(model: Any, coords: torch.Tensor, *, E: float = 2.1e11, nu: float = 0.3) -> torch.Tensor:
    """Navier-Lame equilibrium (isotropic linear elasticity, no body force):
    (lambda+mu)*grad(div(u)) + mu*laplacian(u) = 0. Assumes the model's first
    `dim` output columns are displacement components [u, v, (w)]. Works in
    2D or 3D."""
    coords = coords.requires_grad_(True)
    out = _unwrap(model(coords))
    dim = coords.shape[1]
    if out.shape[1] < dim:
        raise NotImplementedError(
            f"Linear-elasticity residual needs >= {dim} output channels (one displacement "
            f"component per axis), got {out.shape[1]}."
        )
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    u = [out[:, i:i + 1] for i in range(dim)]
    gu = [_grad(u_i, coords) for u_i in u]
    div_u = sum(gu[i][:, i:i + 1] for i in range(dim))
    grad_div_u = _grad(div_u, coords)

    residuals = []
    for i in range(dim):
        laplacian_u_i = sum(_second_derivative(gu[i], j, coords) for j in range(dim))
        residuals.append((lam + mu) * grad_div_u[:, i:i + 1] + mu * laplacian_u_i)
    return torch.cat(residuals, dim=1)


def species_diffusion_residual(model: Any, coords: torch.Tensor, *, D: float = 1e-5, k: float = 0.0) -> torch.Tensor:
    """Simplified steady-state diffusion-reaction (Fick's second law at
    steady state, first-order decay): D*laplacian(C) - k*C = 0. Assumes the
    model's first output column is concentration. A generic Fickian
    diffusion-reaction constraint — not tied to any specific transport
    system."""
    coords = coords.requires_grad_(True)
    out = _unwrap(model(coords))
    if out.shape[1] < 1:
        raise NotImplementedError("Species-diffusion residual needs >= 1 output channel (concentration).")
    C = out[:, 0:1]
    gC = _grad(C, coords)
    laplacian = sum(_second_derivative(gC, i, coords) for i in range(coords.shape[1]))
    return D * laplacian - k * C
