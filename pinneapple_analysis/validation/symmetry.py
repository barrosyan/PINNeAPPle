"""Symmetry checks for trained PINN models.

Implements reflection and rotational symmetry checks in 2-D and higher
dimensions using plain forward passes (no autograd required).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Union

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
        mirror: float = 0.0,
    ) -> CheckResult:
        """Check reflection symmetry: u(x) ≈ sign · u(reflect(x, axis)).

        The reflection maps coordinate *axis* to ``2*mirror - x[:, axis]``
        (i.e. reflection about the plane ``x[axis] == mirror``), leaving all
        other coordinates unchanged.

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
        mirror:
            Coordinate value of the mirror plane along *axis*, i.e. the
            reflection is ``x_ref = 2*mirror - x``. Defaults to ``0.0``,
            which only keeps the reflected points inside the domain when the
            domain is symmetric about zero along *axis*. For a domain such
            as ``x ∈ [0, 1]``, pass the domain midpoint (``mirror=0.5``)
            explicitly — otherwise the reflected points fall outside the
            trained domain and this measures extrapolation error, not
            symmetry.

        Returns
        -------
        CheckResult
        """
        x = _as_tensor(x_points, self.device)

        # Build reflected points: reflect about the plane x[axis] == mirror
        x_ref = x.clone()
        x_ref[:, axis] = 2.0 * mirror - x_ref[:, axis]

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
        vector_field_idx: Optional[Tuple[int, int]] = None,
    ) -> CheckResult:
        """Check 2-D rotational symmetry.

        By default (``vector_field_idx=None``) this compares a single scalar
        output column directly: ``u[field_idx](x) ≈ u[field_idx](R(angle)·x)``.
        This scalar-comparison form is only valid for a genuinely
        scalar/rotation-invariant output field (e.g. pressure magnitude,
        temperature) — it does **not** apply to a vector-valued output
        component such as a velocity that must itself be rotated to match a
        rotationally symmetric field.

        When ``vector_field_idx=(i, j)`` is given instead, the check compares
        the *rotated vector*: the output columns ``(i, j)`` at the original
        points are rotated by ``R(angle)`` and compared against the model's
        own vector output evaluated at the rotated points, i.e.
        ``R(angle) · u[(i,j)](x) ≈ u[(i,j)](R(angle) · x)``. Use this form for
        a rotationally symmetric vector field (e.g. a swirl velocity field).

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
            Output field component to compare when ``vector_field_idx`` is
            ``None`` (scalar/rotation-invariant field path).
        tolerance:
            Acceptance threshold on the mean absolute error.
        name:
            Check identifier.
        vector_field_idx:
            Optional ``(i, j)`` pair of output column indices holding a
            rotationally-symmetric **vector** field (e.g. velocity
            components). When given, both components are rotated by
            ``R(angle)`` before comparison instead of comparing a raw scalar
            column. Mutually exclusive in effect with ``field_idx`` (which is
            ignored when this is set).

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

        if vector_field_idx is not None:
            i, j = vector_field_idx
            ui_orig = u_orig[:, i]
            uj_orig = u_orig[:, j]
            # Rotate the vector at the original points by R(angle); this is
            # what the vector *should* equal at the rotated points if the
            # field is rotationally symmetric.
            ui_orig_rot = cos_a * ui_orig - sin_a * uj_orig
            uj_orig_rot = sin_a * ui_orig + cos_a * uj_orig

            ui_rot = u_rot[:, i]
            uj_rot = u_rot[:, j]

            mae = (
                (ui_orig_rot - ui_rot).abs().mean()
                + (uj_orig_rot - uj_rot).abs().mean()
            ).item() / 2.0
            passed = mae <= tolerance
            angle_deg = math.degrees(angle)
            return CheckResult(
                name=name,
                passed=passed,
                value=mae,
                threshold=tolerance,
                description=(
                    f"Mean |R({angle_deg:.1f}°)·u[{vector_field_idx}](x) - "
                    f"u[{vector_field_idx}](R({angle_deg:.1f}°)·x)| over "
                    f"{x.shape[0]} points (2-D vector rotation on axes 0,1)"
                ),
            )

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
                f"over {x.shape[0]} points (2-D rotation on axes 0,1; scalar/"
                f"rotation-invariant field comparison only)"
            ),
        )
