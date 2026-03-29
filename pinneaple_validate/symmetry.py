"""Symmetry checks for trained PINN models.

Implements reflection and rotational symmetry checks in 2-D and higher
dimensions using plain forward passes (no autograd required).
"""
from __future__ import annotations

import math
from typing import Union

import torch
from torch import Tensor

from .core import CheckResult

# ---------------------------------------------------------------------------
# Internal helpers (shared with boundary.py pattern)
# ---------------------------------------------------------------------------

ArrayLike = Union[Tensor, "np.ndarray"]  # noqa: F821


def _as_tensor(x: ArrayLike, device: str) -> Tensor:
    if not isinstance(x, Tensor):
        try:
            import numpy as np  # type: ignore
            x = torch.from_numpy(np.asarray(x, dtype="float32"))
        except Exception:
            x = torch.tensor(x, dtype=torch.float32)
    return x.to(device=device, dtype=torch.float32)


def _forward_tensor(model: object, x: Tensor) -> Tensor:
    out = model(x)  # type: ignore[operator]
    if isinstance(out, Tensor):
        return out
    if hasattr(out, "y"):
        return out.y
    raise TypeError(
        f"Model returned {type(out)!r}; expected Tensor or an object with a `.y` attribute."
    )


def _extract_field(u: Tensor, field_idx: int) -> Tensor:
    return u if u.dim() == 1 else u[:, field_idx]


# ---------------------------------------------------------------------------
# SymmetryCheck
# ---------------------------------------------------------------------------


class SymmetryCheck:
    """Checks spatial symmetry properties of a trained model.

    Parameters
    ----------
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, …).
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    # ------------------------------------------------------------------
    # Reflection symmetry
    # ------------------------------------------------------------------

    def check_reflection(
        self,
        model: object,
        x_points: ArrayLike,
        axis: int = 0,
        field_idx: int = 0,
        expected_sign: float = 1.0,
        tolerance: float = 1e-3,
        name: str = "reflection_symmetry",
    ) -> CheckResult:
        """Check reflection symmetry: u(x) ≈ sign · u(reflect(x, axis)).

        The reflection maps coordinate *axis* to its negative while leaving
        all other coordinates unchanged.

        Parameters
        ----------
        model:
            Trained model.
        x_points:
            Sample points, shape ``(N, D)``.
        axis:
            Coordinate axis to reflect (0-indexed).
        field_idx:
            Output field component to compare.
        expected_sign:
            ``+1.0`` for even (symmetric) fields, ``-1.0`` for odd (anti-symmetric).
        tolerance:
            Acceptance threshold on the mean absolute error.
        name:
            Check identifier.

        Returns
        -------
        CheckResult
        """
        x = _as_tensor(x_points, self.device)

        # Build reflected points
        x_ref = x.clone()
        x_ref[:, axis] = -x_ref[:, axis]

        with torch.no_grad():
            u_orig = _forward_tensor(model, x)
            u_refl = _forward_tensor(model, x_ref)

        u0 = _extract_field(u_orig, field_idx)
        u1 = _extract_field(u_refl, field_idx)

        mae = (u0 - expected_sign * u1).abs().mean().item()
        passed = mae <= tolerance
        sign_str = "+" if expected_sign >= 0 else "-"
        return CheckResult(
            name=name,
            passed=passed,
            value=mae,
            threshold=tolerance,
            description=(
                f"Mean |u[{field_idx}](x) - ({sign_str}1)·u[{field_idx}](reflect(x, axis={axis}))| "
                f"over {x.shape[0]} points"
            ),
        )

    # ------------------------------------------------------------------
    # Rotational symmetry (2-D)
    # ------------------------------------------------------------------

    def check_rotational(
        self,
        model: object,
        x_points: ArrayLike,
        angle: float,
        field_idx: int = 0,
        tolerance: float = 1e-3,
        name: str = "rotational_symmetry",
    ) -> CheckResult:
        """Check 2-D rotational symmetry: u(x) ≈ u(R(angle) · x).

        The rotation is applied to the first two coordinates (x₀, x₁).
        Additional coordinates (e.g. time) are left unchanged.

        Parameters
        ----------
        model:
            Trained model.
        x_points:
            Sample points, shape ``(N, D)`` with D ≥ 2.
        angle:
            Rotation angle in **radians**.
        field_idx:
            Output field component to compare.
        tolerance:
            Acceptance threshold on the mean absolute error.
        name:
            Check identifier.

        Returns
        -------
        CheckResult

        Raises
        ------
        ValueError
            If ``x_points`` has fewer than 2 spatial dimensions.
        """
        x = _as_tensor(x_points, self.device)
        if x.shape[1] < 2:
            raise ValueError(
                f"Rotational symmetry requires at least 2 spatial dimensions; "
                f"got {x.shape[1]}."
            )

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        x0 = x[:, 0]
        x1 = x[:, 1]
        x0_rot = cos_a * x0 - sin_a * x1
        x1_rot = sin_a * x0 + cos_a * x1

        x_rot = x.clone()
        x_rot[:, 0] = x0_rot
        x_rot[:, 1] = x1_rot

        with torch.no_grad():
            u_orig = _forward_tensor(model, x)
            u_rot = _forward_tensor(model, x_rot)

        u0 = _extract_field(u_orig, field_idx)
        u1 = _extract_field(u_rot, field_idx)

        mae = (u0 - u1).abs().mean().item()
        passed = mae <= tolerance
        angle_deg = math.degrees(angle)
        return CheckResult(
            name=name,
            passed=passed,
            value=mae,
            threshold=tolerance,
            description=(
                f"Mean |u[{field_idx}](x) - u[{field_idx}](R({angle_deg:.1f}°)·x)| "
                f"over {x.shape[0]} points (2-D rotation on axes 0,1)"
            ),
        )
