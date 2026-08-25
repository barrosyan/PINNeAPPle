"""Continuous adjoint-based shape optimization for aerodynamic and thermal design.

Computes sensitivity dJ/d(shape) using the continuous adjoint method via
PyTorch autograd.  Supports objectives: drag, lift, heat flux, pressure drop.

Theory
------
Given primal PDE residual R(u, s) = 0 and objective J(u, s):

    dJ/ds = dJ/ds|_u  -  lambda^T  *  dR/ds

where lambda solves the *adjoint equation*:

    (dR/du)^T  lambda  =  (dJ/du)^T

In the discrete-PDE / PINN setting we use automatic differentiation to
evaluate all Jacobian-vector products without explicitly forming the Jacobians.

Quick start::

    from pinneapple_design.design_optimizer.adjoint import (
        ShapeParametrization, ContinuousAdjointSolver, naca_parametric
    )

    shape = ShapeParametrization(naca_parametric(t_c=0.12))
    solver = ContinuousAdjointSolver(pinn_model, pde_residual, objective_fn)
    result = solver.optimize(shape, x_col, n_steps=200, lr=1e-2)
    print(result["best_objective"])
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

# naca_parametric canonical location is pinneapple_geom.gen.airfoil;
# re-exported here for backwards compatibility.
from pinneapple_design.geometry.gen.airfoil import naca_parametric  # noqa: F401


# ---------------------------------------------------------------------------
# Shape parametrisation
# ---------------------------------------------------------------------------


class ShapeParametrization:
    """Parametric shape representation for gradient-based optimisation.

    Supports FFD (Free-Form Deformation) control points, Bezier curves, and
    NACA 4-digit family parameter variations.

    Parameters
    ----------
    control_points:
        Initial control-point coordinates, shape ``(n_ctrl, 2)`` (2-D) or
        ``(n_ctrl, 3)`` (3-D).
    device:
        Torch device to place the parameter on.
    """

    def __init__(self, control_points: torch.Tensor,
                 device: str = "cpu") -> None:
        self.control_points: nn.Parameter = nn.Parameter(
            control_points.clone().float().to(device)
        )
        self._device = device

    # ------------------------------------------------------------------
    # Deformation / coordinate helpers
    # ------------------------------------------------------------------

    def deform_mesh(self, mesh_points: torch.Tensor) -> torch.Tensor:
        """Apply a simple FFD-style deformation to *mesh_points*.

        The deformation is a weighted sum of the displacements of the nearest
        control points (inverse-distance weighting).  This is a lightweight
        approximation; a production FFD would use B-spline basis functions.

        Parameters
        ----------
        mesh_points:
            ``(N, d)`` tensor of mesh node coordinates.

        Returns
        -------
        torch.Tensor
            Deformed mesh points, same shape as *mesh_points*.
        """
        # Compute inverse-distance weights: shape (N, n_ctrl)
        diff = mesh_points.unsqueeze(1) - self.control_points.unsqueeze(0)  # (N, K, d)
        dist2 = (diff ** 2).sum(-1).clamp(min=1e-12)                        # (N, K)
        w = 1.0 / dist2                                                       # (N, K)
        w = w / w.sum(dim=1, keepdim=True)                                   # normalise

        # Displacement = weighted shift from initial control positions to current
        # (self.control_points already IS the current positions, so we compute
        #  the perturbation relative to the centroid as a proxy displacement)
        ctrl_mean = self.control_points.mean(0, keepdim=True)               # (1, d)
        delta = self.control_points - ctrl_mean                              # (K, d)
        displacement = w @ delta                                             # (N, d)
        return mesh_points + displacement

    def to_boundary_coordinates(self) -> torch.Tensor:
        """Return control points re-normalised to [-1, 1] chord coordinates.

        Assumes the first spatial dimension is the chord-wise direction
        ``x ∈ [x_min, x_max]``.

        Returns
        -------
        torch.Tensor
            Shape ``(n_ctrl, d)`` in normalised coordinates.
        """
        cp = self.control_points
        x_min = cp[:, 0].min()
        x_max = cp[:, 0].max()
        scale = (x_max - x_min).clamp(min=1e-12)
        cp_norm = cp.clone()
        cp_norm[:, 0] = (cp[:, 0] - x_min) / scale * 2.0 - 1.0
        return cp_norm

    def parameters(self) -> List[nn.Parameter]:
        """Return a list of trainable parameters for use with an optimiser."""
        return [self.control_points]


# ---------------------------------------------------------------------------
# Matrix-free Krylov solver (used by ``compute_adjoint``)
# ---------------------------------------------------------------------------


def _gmres(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> torch.Tensor:
    """Unrestarted, matrix-free GMRES solving ``A x = b``.

    ``matvec(v)`` must return ``A @ v`` for a tensor *v* shaped like *b*;
    ``A`` (e.g. ``(dR/du)^T``) is never assembled explicitly, only applied
    via Jacobian/vector products.

    Returns
    -------
    torch.Tensor
        Approximate solution ``x``, same shape as *b*.
    """
    shape = b.shape
    b_flat = b.reshape(-1).detach()
    n = b_flat.numel()
    max_iter = max(1, min(max_iter, n))

    beta = torch.linalg.norm(b_flat)
    if beta < tol:
        return torch.zeros_like(b)

    def mv(v_flat: torch.Tensor) -> torch.Tensor:
        return matvec(v_flat.reshape(shape)).reshape(-1).detach()

    Q = [b_flat / beta]
    H = torch.zeros(max_iter + 1, max_iter, dtype=b_flat.dtype, device=b_flat.device)
    x_flat = torch.zeros_like(b_flat)

    for k in range(max_iter):
        v = mv(Q[k])
        for j in range(k + 1):
            H[j, k] = torch.dot(Q[j], v)
            v = v - H[j, k] * Q[j]
        H[k + 1, k] = torch.linalg.norm(v)
        last = k == max_iter - 1
        if H[k + 1, k] > 1e-14 and not last:
            Q.append(v / H[k + 1, k])
        else:
            Q.append(torch.zeros_like(v))

        e1 = torch.zeros(k + 2, dtype=b_flat.dtype, device=b_flat.device)
        e1[0] = beta
        H_k = H[: k + 2, : k + 1]
        y = torch.linalg.lstsq(H_k, e1.unsqueeze(1)).solution.squeeze(1)
        resid = torch.linalg.norm(e1 - H_k @ y)

        if resid < tol * beta or last:
            Q_k = torch.stack(Q[: k + 1], dim=1)
            x_flat = Q_k @ y
            break

    return x_flat.reshape(shape)


class _CachedForward:
    """Model wrapper that returns a single cached forward-pass tensor.

    ``objective_fn`` and ``pde_residual_fn`` only receive ``(model, x_col)``
    and evaluate ``u = model(x_col)`` themselves, so a ``u`` computed
    separately by the caller is never actually part of their autograd
    graph -- a fresh call is a distinct graph node, even though it is
    numerically identical.  Wrapping the model so that calling it again
    with the *same* ``x_col`` tensor returns the already-computed ``u``
    keeps everything on one graph, so Jacobian-vector probes against
    ``u`` (``dJ/du``, ``dR/du``) actually find it as an ancestor.
    """

    def __init__(self, model: nn.Module, x_ref: torch.Tensor, u_ref: torch.Tensor) -> None:
        self._model = model
        self._x_ref = x_ref
        self._u_ref = u_ref

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x is self._x_ref:
            return self._u_ref
        return self._model(x)

    def __getattr__(self, name):
        return getattr(self._model, name)


# ---------------------------------------------------------------------------
# Continuous adjoint solver
# ---------------------------------------------------------------------------


class ContinuousAdjointSolver:
    """Continuous adjoint solver using automatic differentiation.

    Given:
    - **Forward model** – a PINN solving the primal PDE: ``u = model(x)``
    - **Objective**     – scalar ``J(u, x_col)``
    - **PDE residual**  – ``R(u, x_col) ≈ 0``

    ``compute_adjoint`` solves the adjoint equation

        (dR/du)^T lambda = dJ/du

    with a matrix-free Krylov (GMRES) solve driven by
    ``torch.autograd.grad`` Jacobian-vector products -- useful when ``u``
    is the solution of a primal solve that is *not* differentiated through
    directly (e.g. a black-box CFD solver).

    ``shape_sensitivity`` (used by ``optimize``) does **not** go through
    the adjoint equation at all: in this module ``u = model(x_deformed(s))``
    is a direct, fully differentiable forward pass, so ``dJ/ds`` is simply
    the total derivative obtained by backpropagating through
    ``deform_mesh`` and the model -- exact, and cheaper than any adjoint
    correction.

    Parameters
    ----------
    primal_model:
        Trained (or being trained) PINN, callable ``(x) -> u``.
    pde_residual_fn:
        Callable ``(model, x_col) -> residual_tensor``.  The residual should
        be differentiable with respect to the shape parameters embedded in
        *x_col* or *model*.
    objective_fn:
        Callable ``(model, x_col) -> scalar_tensor``.
    """

    def __init__(
        self,
        primal_model: nn.Module,
        pde_residual_fn: Callable,
        objective_fn: Callable,
    ) -> None:
        self.primal = primal_model
        self.pde_res_fn = pde_residual_fn
        self.objective = objective_fn

    # ------------------------------------------------------------------
    # Core adjoint computation
    # ------------------------------------------------------------------

    def compute_adjoint(
        self,
        x_col: torch.Tensor,
        shape_params: ShapeParametrization,
        tol: float = 1e-6,
        max_iter: int = 50,
    ) -> torch.Tensor:
        """Compute the adjoint variable ``lambda`` at *x_col*.

        Solves ``(dR/du)^T lambda = dJ/du`` with a matrix-free GMRES
        Krylov solve: both the right-hand side ``dJ/du`` and the
        ``(dR/du)^T v`` Jacobian-vector products used by GMRES are
        evaluated with ``torch.autograd.grad`` -- the Jacobian ``dR/du``
        is never assembled explicitly.

        Strategy
        --------
        1. Evaluate ``u = model(x_col)`` with ``requires_grad``.
        2. Compute ``R(u, x_col)`` and ``J(u, x_col)``.
        3. Compute ``dJ/du`` via ``torch.autograd.grad``.
        4. If ``R`` is differentiably connected to ``u`` (i.e. a
           vector-Jacobian product against ``u`` is obtainable), solve
           ``(dR/du)^T lambda = dJ/du`` with GMRES. Otherwise (``R`` was
           produced independently of ``u``, so ``dR/du`` cannot be probed)
           fall back to ``lambda = dJ/du``, i.e. the ``dR/du = I``
           approximation.

        Returns
        -------
        torch.Tensor
            Adjoint variable, shape matching ``u``.
        """
        x_col = x_col.detach().requires_grad_(True)
        u = self.primal(x_col)
        # objective_fn / pde_residual_fn re-evaluate ``model(x_col)``
        # internally; route them through the cached forward pass so the
        # probes below see ``u`` on their graph instead of a disconnected
        # duplicate.
        cached_model = _CachedForward(self.primal, x_col, u)

        J = self.objective(cached_model, x_col)
        dJ_du = torch.autograd.grad(J, u, retain_graph=True,
                                    create_graph=False,
                                    allow_unused=True)[0]
        if dJ_du is None:
            return torch.zeros_like(u)

        R = self.pde_res_fn(cached_model, x_col)
        if R is None or R.shape != u.shape or not R.requires_grad:
            return dJ_du

        def rmatvec(v: torch.Tensor) -> Optional[torch.Tensor]:
            return torch.autograd.grad(
                R, u, grad_outputs=v, retain_graph=True,
                create_graph=False, allow_unused=True,
            )[0]

        probe = rmatvec(torch.ones_like(u))
        if probe is None:
            # R was recomputed from x_col independently of this ``u``
            # (no edge back to it in the autograd graph) -- dR/du cannot
            # be probed, so fall back to the dR/du = I approximation.
            return dJ_du

        def rmatvec_safe(v: torch.Tensor) -> torch.Tensor:
            out = rmatvec(v)
            return torch.zeros_like(v) if out is None else out

        lam = _gmres(rmatvec_safe, dJ_du, tol=tol, max_iter=max_iter)
        return lam

    def shape_sensitivity(
        self,
        x_col: torch.Tensor,
        shape_params: ShapeParametrization,
    ) -> torch.Tensor:
        """Compute the shape sensitivity ``dJ/d(control_points)``.

        ``u = model(x_deformed(control_points))`` is a direct, fully
        differentiable forward pass through ``deform_mesh`` and the PINN --
        it is not the implicit solution of a separate primal solve that
        would need correcting via the adjoint equation. Reverse-mode
        autodiff through that composition therefore already gives the
        *exact* total derivative ``dJ/ds``, with no adjoint term to add.

        Returns
        -------
        torch.Tensor
            Gradient w.r.t. ``shape_params.control_points``,
            same shape as ``control_points``.
        """
        # Deform collocation points using current shape (kept differentiable
        # w.r.t. control_points, so the objective below backpropagates
        # through the deformation as well as through the model).
        x_deformed = shape_params.deform_mesh(x_col.detach())

        J = self.objective(self.primal, x_deformed)

        dJ_ds = torch.autograd.grad(
            J, shape_params.control_points,
            retain_graph=False, create_graph=False,
            allow_unused=True
        )[0]
        if dJ_ds is None:
            dJ_ds = torch.zeros_like(shape_params.control_points)

        return dJ_ds

    # ------------------------------------------------------------------
    # Optimisation loop
    # ------------------------------------------------------------------

    def optimize(
        self,
        shape_params: ShapeParametrization,
        x_col: torch.Tensor,
        n_steps: int = 100,
        lr: float = 0.01,
        callback: Optional[Callable] = None,
    ) -> Dict:
        """Run gradient-based shape optimisation.

        Parameters
        ----------
        shape_params:
            Initial shape parametrisation (modified in-place).
        x_col:
            Collocation points ``(N, d)`` on the reference (un-deformed) domain.
        n_steps:
            Number of gradient steps.
        lr:
            Learning rate (Adam optimiser).
        callback:
            Optional callable ``(step, J, shape_params)`` called each step.

        Returns
        -------
        dict
            Keys: ``"best_objective"``, ``"best_control_points"``,
            ``"history_objective"``.
        """
        optimizer = optim.Adam(shape_params.parameters(), lr=lr)

        history: List[float] = []
        best_J = float("inf")
        best_cp = shape_params.control_points.detach().clone()

        for step in range(n_steps):
            optimizer.zero_grad()

            x_deformed = shape_params.deform_mesh(x_col.detach())
            J = self.objective(self.primal, x_deformed)

            # Use adjoint sensitivity as gradient
            sens = self.shape_sensitivity(x_col, shape_params)
            # Manually set the gradient (adjoint method replaces backprop)
            if shape_params.control_points.grad is None:
                shape_params.control_points.grad = sens.clone()
            else:
                shape_params.control_points.grad.copy_(sens)

            optimizer.step()

            J_val = float(J.item())
            history.append(J_val)
            if J_val < best_J:
                best_J = J_val
                best_cp = shape_params.control_points.detach().clone()

            if callback is not None:
                callback(step, J_val, shape_params)

        return {
            "best_objective": best_J,
            "best_control_points": best_cp,
            "history_objective": history,
        }


# ---------------------------------------------------------------------------
# Drag objective
# ---------------------------------------------------------------------------


class DragAdjointObjective:
    """Drag force objective for aerodynamic shape optimisation.

    ``J = integral_surface (pressure + viscous_drag) dS``

    In the PINN setting the integral is approximated as a weighted mean over
    supplied surface collocation points.

    Parameters
    ----------
    surface_pts:
        Boundary/surface collocation points ``(M, d)``.
    nu:
        Kinematic viscosity.
    alpha:
        Angle of attack in radians (used to project force to drag direction).
    """

    def __init__(
        self,
        surface_pts: torch.Tensor,
        nu: float,
        alpha: float = 0.0,
    ) -> None:
        self.surface_pts = surface_pts
        self.nu = nu
        self.alpha = alpha
        # Drag direction unit vector (cos α, -sin α) for 2-D
        self._drag_dir = torch.tensor(
            [math.cos(alpha), -math.sin(alpha)],
            dtype=torch.float32,
        )

    def __call__(self, model: nn.Module, x_col: torch.Tensor) -> torch.Tensor:
        """Evaluate drag objective.

        The model is expected to return ``(u, v, p)`` channels at minimum.
        If a 1-D scalar field is returned it is treated as pressure.

        Returns
        -------
        torch.Tensor
            Scalar drag proxy.
        """
        s_pts = self.surface_pts.to(x_col.device).requires_grad_(True)
        u_s = model(s_pts)

        if u_s.shape[-1] >= 3:
            # Assume layout: (u_vel, v_vel, p, ...)
            vel = u_s[..., :2]           # (M, 2)
            p = u_s[..., 2]              # (M,)
        elif u_s.shape[-1] == 2:
            vel = u_s
            p = torch.zeros(u_s.shape[0], device=u_s.device)
        else:
            vel = torch.zeros(u_s.shape[0], 2, device=u_s.device)
            p = u_s[..., 0]

        # Local tangent/outward-normal from the ordered, closed surface
        # contour (surface_pts traces the body boundary, e.g. as produced
        # by naca_parametric: upper surface LE->TE then lower surface
        # TE->LE, i.e. clockwise) via a periodic central difference.
        tangent = s_pts.roll(-1, dims=0) - s_pts.roll(1, dims=0)
        tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)

        drag_dir = self._drag_dir.to(p.device)

        # Pressure traction on the body is -p * n (pressure pushes inward);
        # its contribution to drag is its component along the drag direction.
        pressure_drag = -p * (normal @ drag_dir)

        # Viscous (skin-friction) drag: wall shear stress
        # tau_w = nu * d(u_tangential)/dn. Under no-slip, velocity itself
        # vanishes at the wall, so it is the normal derivative -- not the
        # raw velocity magnitude -- that carries the skin-friction signal.
        u_tangential = (vel * tangent).sum(-1)
        grad_ut = torch.autograd.grad(
            u_tangential.sum(), s_pts, retain_graph=True,
            create_graph=True, allow_unused=True,
        )[0]
        if grad_ut is None:
            tau_w = torch.zeros_like(u_tangential)
        else:
            tau_w = self.nu * (grad_ut * normal).sum(-1)
        viscous_drag = tau_w * (tangent @ drag_dir)

        drag = torch.mean(pressure_drag + viscous_drag)
        return drag
