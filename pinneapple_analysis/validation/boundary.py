"""Boundary condition checks for trained PINN models.

Implements Dirichlet, Neumann, and periodicity checks using PyTorch autograd.
All checks accept NumPy arrays or PyTorch Tensors for point specifications.
"""
from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Tensor

from .core import CheckResult

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

ArrayLike = Union[Tensor, "np.ndarray"]  # noqa: F821


def _as_tensor(x: ArrayLike, device: str, requires_grad: bool = False) -> Tensor:
    """Convert *x* to a ``float32`` Tensor on *device*."""
    if not isinstance(x, Tensor):
        try:
            import numpy as np  # type: ignore
            x = torch.from_numpy(np.asarray(x, dtype="float32"))
        except Exception:
            x = torch.tensor(x, dtype=torch.float32)
    t = x.to(device=device, dtype=torch.float32)
    if requires_grad:
        t = t.requires_grad_(True)
    return t


def _forward_tensor(model: object, x: Tensor) -> Tensor:
    """Call the model and return a plain ``Tensor``."""
    out = model(x)  # type: ignore[operator]
    if isinstance(out, Tensor):
        return out
    if hasattr(out, "y"):
        return out.y
    raise TypeError(
        f"Model returned {type(out)!r}; expected Tensor or an object with a `.y` attribute."
    )


# ---------------------------------------------------------------------------
# BoundaryCheck
# ---------------------------------------------------------------------------


class BoundaryCheck:
    """Checks boundary conditions by querying the model at boundary points.

    Parameters
    ----------
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, …).
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    # ------------------------------------------------------------------
    # Dirichlet
    # ------------------------------------------------------------------

    def check_dirichlet(
        self,
        model: object,
        boundary_points: ArrayLike,
        expected_values: ArrayLike,
        field_idx: int = 0,
        tolerance: float = 1e-3,
        name: str = "dirichlet_bc",
    ) -> CheckResult:
        """Check a Dirichlet (essential) boundary condition.

        Verifies that the mean absolute error ``|u(x_bc)[field_idx] - u_expected|``
        is below *tolerance*.

        Parameters
        ----------
        model:
            Trained model.
        boundary_points:
            Boundary sample points, shape ``(N, D)``.
        expected_values:
            Target field values at those points, shape ``(N,)`` or ``(N, 1)``.
        field_idx:
            Output field component to compare (for multi-field models).
        tolerance:
            Acceptance threshold on the mean absolute error.
        name:
            Check identifier.

        Returns
        -------
        CheckResult
        """
        x_bc = _as_tensor(boundary_points, self.device)
        u_exp = _as_tensor(expected_values, self.device).squeeze()

        with torch.no_grad():
            u_pred = _forward_tensor(model, x_bc)

        if u_pred.dim() == 1:
            u_hat = u_pred
        else:
            u_hat = u_pred[:, field_idx]

        mae = (u_hat - u_exp).abs().mean().item()
        passed = mae <= tolerance
        return CheckResult(
            name=name,
            passed=passed,
            value=mae,
            threshold=tolerance,
            description=(
                f"Mean |u[{field_idx}](x_bc) - u_expected| over "
                f"{x_bc.shape[0]} boundary points"
            ),
        )

    # ------------------------------------------------------------------
    # Neumann
    # ------------------------------------------------------------------

    def check_neumann(
        self,
        model: object,
        boundary_points: ArrayLike,
        normals: ArrayLike,
        expected_flux: ArrayLike,
        field_idx: int = 0,
        tolerance: float = 1e-3,
        name: str = "neumann_bc",
    ) -> CheckResult:
        """Check a Neumann (natural) boundary condition via autograd.

        Verifies that ``|∂u/∂n(x_bc) - g|`` is below *tolerance* on average,
        where ``∂u/∂n = ∇u · n`` is computed by PyTorch autograd.

        Parameters
        ----------
        model:
            Trained model.
        boundary_points:
            Boundary sample points, shape ``(N, D)``.
        normals:
            Outward unit normals at those points, shape ``(N, D)``.
        expected_flux:
            Target normal derivative ``g``, shape ``(N,)`` or scalar.
        field_idx:
            Output field component to differentiate.
        tolerance:
            Acceptance threshold on the mean absolute error.
        name:
            Check identifier.

        Returns
        -------
        CheckResult
        """
        x_bc = _as_tensor(boundary_points, self.device, requires_grad=True)
        n_vecs = _as_tensor(normals, self.device)
        g = _as_tensor(expected_flux, self.device).squeeze()

        u_pred = _forward_tensor(model, x_bc)
        if u_pred.dim() == 1:
            u_field = u_pred
        else:
            u_field = u_pred[:, field_idx]

        grad_u = torch.autograd.grad(
            u_field.sum(), x_bc, create_graph=False
        )[0]  # (N, D)

        # Normal derivative: dot(∇u, n) for each point
        du_dn = (grad_u * n_vecs).sum(dim=-1)  # (N,)

        if g.dim() == 0:
            g = g.expand_as(du_dn)

        mae = (du_dn - g).abs().mean().item()
        passed = mae <= tolerance
        return CheckResult(
            name=name,
            passed=passed,
            value=mae,
            threshold=tolerance,
            description=(
                f"Mean |∂u[{field_idx}]/∂n - g| over "
                f"{x_bc.shape[0]} boundary points (autograd)"
            ),
        )

    # ------------------------------------------------------------------
    # Periodicity
    # ------------------------------------------------------------------

    def check_periodicity(
        self,
        model: object,
        x_left: ArrayLike,
        x_right: ArrayLike,
        field_idx: int = 0,
        tolerance: float = 1e-3,
        name: str = "periodicity",
    ) -> CheckResult:
        """Check periodic boundary conditions: u(x_left) ≈ u(x_right).

        Parameters
        ----------
        model:
            Trained model.
        x_left:
            Points on the "left" periodic boundary, shape ``(N, D)``.
        x_right:
            Corresponding points on the "right" periodic boundary, shape ``(N, D)``.
        field_idx:
            Output field component to compare.
        tolerance:
            Acceptance threshold on the mean absolute difference.
        name:
            Check identifier.

        Returns
        -------
        CheckResult
        """
        xl = _as_tensor(x_left, self.device)
        xr = _as_tensor(x_right, self.device)

        with torch.no_grad():
            u_l = _forward_tensor(model, xl)
            u_r = _forward_tensor(model, xr)

        def _extract(u: Tensor) -> Tensor:
            return u if u.dim() == 1 else u[:, field_idx]

        mae = (_extract(u_l) - _extract(u_r)).abs().mean().item()
        passed = mae <= tolerance
        return CheckResult(
            name=name,
            passed=passed,
            value=mae,
            threshold=tolerance,
            description=(
                f"Mean |u[{field_idx}](x_left) - u[{field_idx}](x_right)| over "
                f"{xl.shape[0]} periodic boundary pairs"
            ),
        )
