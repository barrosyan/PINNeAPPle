from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


class ConstraintBase(ABC):
    """Abstract base for design constraints."""

    @abstractmethod
    def penalty(self, theta: Tensor, u: Tensor) -> Tensor:
        """Return a non-negative scalar penalty; zero when satisfied."""

    @abstractmethod
    def satisfied(self, theta: Tensor, u: Tensor) -> bool:
        """Return True when the constraint is satisfied up to numerical noise."""


class BoxConstraint(ConstraintBase):
    """Bound constraint: ``theta_min <= theta <= theta_max``.

    Penalty uses a one-sided ReLU barrier so the gradient remains informative
    even deep inside the feasible region.

    Parameters
    ----------
    theta_min, theta_max:
        Per-component lower / upper bounds, shape (p,).
    weight:
        Penalty coefficient.
    """

    def __init__(
        self,
        theta_min: Tensor,
        theta_max: Tensor,
        weight: float = 100.0,
    ) -> None:
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.weight = weight

    def penalty(self, theta: Tensor, u: Tensor) -> Tensor:
        lo = self.theta_min.to(theta)
        hi = self.theta_max.to(theta)
        below = F.relu(lo - theta)
        above = F.relu(theta - hi)
        return self.weight * (below.sum() + above.sum())

    def satisfied(self, theta: Tensor, u: Tensor) -> bool:
        lo = self.theta_min.to(theta)
        hi = self.theta_max.to(theta)
        return bool(torch.all(theta >= lo) and torch.all(theta <= hi))


class MassConservationConstraint(ConstraintBase):
    """Divergence-free (mass conservation) constraint.

    Estimates the velocity divergence via first-order finite differences and
    penalises its mean squared value.  Accepts *u* of shape ``(N, d)``
    (point cloud) or ``(nx, ny, d)`` (structured 2-D grid).

    Parameters
    ----------
    weight:
        Penalty coefficient.
    """

    def __init__(self, weight: float = 10.0) -> None:
        self.weight = weight

    def _divergence(self, u: Tensor) -> Tensor:
        """Estimate divergence; works for (N, d) and (nx, ny, d)."""
        if u.ndim == 2:
            # Unstructured: treat rows as spatial points in sequence.
            # div ≈ sum of central differences across spatial channels.
            N, d = u.shape
            div = torch.zeros(N, device=u.device, dtype=u.dtype)
            for ch in range(min(d, u.shape[0])):
                # Finite difference along the point index as a proxy for ∂u/∂x.
                padded = F.pad(u[:, ch].unsqueeze(0).unsqueeze(0), (1, 1), mode="replicate")
                grad = (padded[0, 0, 2:] - padded[0, 0, :-2]) / 2.0
                div = div + grad
            return div
        elif u.ndim == 3:
            # Structured grid (nx, ny, d): use spatial dims 0 and 1.
            nx, ny, d = u.shape
            div = torch.zeros(nx, ny, device=u.device, dtype=u.dtype)
            if d >= 1:
                ux = u[..., 0]
                pad_x = F.pad(ux.unsqueeze(0).unsqueeze(0), (0, 0, 1, 1), mode="replicate")
                div = div + (pad_x[0, 0, 2:, :] - pad_x[0, 0, :-2, :]) / 2.0
            if d >= 2:
                uy = u[..., 1]
                pad_y = F.pad(uy.unsqueeze(0).unsqueeze(0), (1, 1, 0, 0), mode="replicate")
                div = div + (pad_y[0, 0, :, 2:] - pad_y[0, 0, :, :-2]) / 2.0
            return div
        else:
            raise ValueError(f"u must be 2-D or 3-D, got shape {tuple(u.shape)}")

    def penalty(self, theta: Tensor, u: Tensor) -> Tensor:
        div_u = self._divergence(u)
        return self.weight * torch.mean(div_u ** 2)

    def satisfied(self, theta: Tensor, u: Tensor) -> bool:
        div_u = self._divergence(u)
        return bool(torch.mean(div_u ** 2).item() < 1e-4)


class GeometricConstraint(ConstraintBase):
    """Aspect-ratio constraint on pairs of design parameters.

    For each pair ``(i, j)`` in *param_pairs*, enforces:
    ``min_val <= theta[i] / theta[j] <= max_val``.

    Parameters
    ----------
    min_val, max_val:
        Acceptable ratio range.
    param_pairs:
        List of ``(i, j)`` index tuples; ratio ``theta[i] / theta[j]``
        is checked for each pair.
    weight:
        Penalty coefficient.
    """

    def __init__(
        self,
        min_val: float = 0.1,
        max_val: float = 10.0,
        param_pairs: List[Tuple[int, int]] | None = None,
        weight: float = 50.0,
    ) -> None:
        self.min_val = min_val
        self.max_val = max_val
        self.param_pairs = param_pairs or []
        self.weight = weight

    def penalty(self, theta: Tensor, u: Tensor) -> Tensor:
        pen = torch.zeros((), dtype=theta.dtype, device=theta.device)
        eps = torch.tensor(1e-8, dtype=theta.dtype, device=theta.device)
        for i, j in self.param_pairs:
            ratio = theta[i] / (torch.abs(theta[j]) + eps)
            pen = pen + F.relu(self.min_val - ratio) + F.relu(ratio - self.max_val)
        return self.weight * pen

    def satisfied(self, theta: Tensor, u: Tensor) -> bool:
        return self.penalty(theta, u).item() == 0.0


class ManufacturabilityConstraint(ConstraintBase):
    """Penalise high-frequency (non-smooth) variation in the design vector.

    Uses total-variation of *theta* as a smoothness measure; high TV implies
    rapid parameter oscillations that are hard to manufacture.

    Parameters
    ----------
    smoothness_weight:
        Penalty coefficient.
    """

    def __init__(self, smoothness_weight: float = 1.0) -> None:
        self.smoothness_weight = smoothness_weight

    def penalty(self, theta: Tensor, u: Tensor) -> Tensor:
        if theta.numel() < 2:
            return torch.zeros((), dtype=theta.dtype, device=theta.device)
        tv = torch.sum(torch.abs(theta[1:] - theta[:-1]))
        return self.smoothness_weight * tv

    def satisfied(self, theta: Tensor, u: Tensor) -> bool:
        return True  # smoothness is always a soft preference, not a hard gate


@dataclass
class ConstraintSet:
    """Container that aggregates multiple constraints.

    Attributes
    ----------
    constraints:
        List of :class:`ConstraintBase` instances.
    """

    constraints: List[ConstraintBase] = field(default_factory=list)

    def add(self, constraint: ConstraintBase) -> None:
        """Append a constraint to the set."""
        self.constraints.append(constraint)

    def total_penalty(self, theta: Tensor, u: Tensor) -> Tensor:
        """Sum all constraint penalties into a single scalar tensor."""
        if not self.constraints:
            return torch.zeros((), dtype=theta.dtype, device=theta.device)
        total = torch.zeros((), dtype=theta.dtype, device=theta.device)
        for c in self.constraints:
            total = total + c.penalty(theta, u)
        return total

    def all_satisfied(self, theta: Tensor, u: Tensor) -> bool:
        """Return True only when every constraint reports satisfaction."""
        return all(c.satisfied(theta, u) for c in self.constraints)
