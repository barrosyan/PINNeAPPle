from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from torch import Tensor


class ObjectiveBase(ABC):
    """Abstract base class for design objectives."""

    @abstractmethod
    def __call__(self, theta: Tensor, u: Tensor) -> Tensor:
        """Evaluate the objective and return a scalar tensor.

        Parameters
        ----------
        theta:
            Design-parameter vector, shape (p,).
        u:
            Physics field predicted by the surrogate, arbitrary shape but
            at least 1-D.

        Returns
        -------
        Tensor
            Scalar tensor (to be minimized).
        """


class DragObjective(ObjectiveBase):
    """Drag coefficient proxy from a velocity / pressure field.

    Works without explicit surface normals: uses the energy-like combination
    ``mean(u_mag^2 + pressure_weight * p^2)`` over the full field.  The last
    channel of *u* is treated as pressure; all preceding channels are velocity.

    Parameters
    ----------
    pressure_weight:
        Weighting for the pressure contribution.
    viscous_weight:
        Weighting for the viscous (velocity-gradient) term.  Applied as a
        scalar multiple of the velocity-magnitude mean; kept separate so
        callers can tune viscous vs. pressure drag balance.
    """

    def __init__(self, pressure_weight: float = 1.0, viscous_weight: float = 0.1) -> None:
        self.pressure_weight = pressure_weight
        self.viscous_weight = viscous_weight

    def __call__(self, theta: Tensor, u: Tensor) -> Tensor:
        u_flat = u.reshape(-1, u.shape[-1]) if u.ndim > 1 else u.unsqueeze(0)
        velocity = u_flat[:, :-1]  # all channels except last
        pressure = u_flat[:, -1]   # last channel is pressure proxy

        vel_term = self.viscous_weight * torch.mean(velocity ** 2)
        pres_term = self.pressure_weight * torch.mean(pressure ** 2)
        return vel_term + pres_term


class ThermalEfficiencyObjective(ObjectiveBase):
    """Thermal efficiency objective (to be minimized via negation).

    Efficiency = heat_flux / (1 + pressure_drop_weight * delta_p)

    where heat_flux = mean(|u[heat_field_idx]|) and delta_p = std(u[..., -1])
    as a pressure-drop proxy.  Returns the *negative* efficiency so that the
    optimizer minimizes it.

    Parameters
    ----------
    heat_field_idx:
        Channel index of the heat-flux / temperature field.
    pressure_drop_weight:
        Relative importance of pressure-drop penalty.
    """

    def __init__(self, heat_field_idx: int = 0, pressure_drop_weight: float = 0.5) -> None:
        self.heat_field_idx = heat_field_idx
        self.pressure_drop_weight = pressure_drop_weight

    def __call__(self, theta: Tensor, u: Tensor) -> Tensor:
        u_flat = u.reshape(-1, u.shape[-1]) if u.ndim > 1 else u.unsqueeze(0)
        heat_flux = torch.mean(torch.abs(u_flat[:, self.heat_field_idx]))
        delta_p = torch.std(u_flat[:, -1])
        efficiency = heat_flux / (1.0 + self.pressure_drop_weight * delta_p)
        return -efficiency  # negated → minimization


class StructuralObjective(ObjectiveBase):
    """Minimize a von-Mises stress proxy.

    Uses the mean of squared field values as a surrogate for peak stress;
    avoids explicit stress-tensor decomposition when field type is unknown.
    """

    def __call__(self, theta: Tensor, u: Tensor) -> Tensor:
        return torch.mean(u ** 2)


class WeightMinimizationObjective(ObjectiveBase):
    """Minimize geometry volume via L2 norm of the design-parameter vector.

    This penalizes large parameter magnitudes, which typically correlate with
    a larger material volume when the parameterization is length/area-based.
    """

    def __call__(self, theta: Tensor, u: Tensor) -> Tensor:
        return torch.norm(theta)


class CompositeObjective(ObjectiveBase):
    """Weighted sum of multiple objectives.

    Parameters
    ----------
    terms:
        Initial list of ``(weight, objective)`` pairs.  More can be added
        later via :py:meth:`add`.
    """

    def __init__(self, terms: List[Tuple[float, ObjectiveBase]] | None = None) -> None:
        self._terms: List[Tuple[float, ObjectiveBase]] = list(terms) if terms else []

    def add(self, weight: float, obj: ObjectiveBase) -> None:
        """Append a new weighted objective."""
        self._terms.append((weight, obj))

    def __call__(self, theta: Tensor, u: Tensor) -> Tensor:
        if not self._terms:
            raise RuntimeError("CompositeObjective has no terms; call .add() first.")
        total = torch.zeros((), dtype=theta.dtype, device=theta.device)
        for w, obj in self._terms:
            total = total + w * obj(theta, u)
        return total
