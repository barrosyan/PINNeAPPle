"""Observation operators for inverse problems.

An *observation operator* H maps the full model state u(x) to the set of
quantities that can actually be measured.  This abstraction is central to
Bayesian inference and data assimilation.

Available operators
-------------------
PointObsOperator          y = u(x_obs)         sparse point measurements
LinearObsOperator         y = A u              linear map (tomography, etc.)
IntegralObsOperator       y = ∫ k(x) u(x) dx  spatial / temporal averages
BoundaryObsOperator       y = u|_{∂Ω}          boundary measurements
ComposedObsOperator       [H₁; H₂; …]          stack multiple operators

All operators expose:
- ``__call__(u, x)`` → predicted observations (torch)
- ``apply_numpy(u_arr, x_arr)`` → numpy interface
- ``jacobian(model, theta, x)`` → ∂H/∂θ for sensitivity analysis
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class ObsOperatorBase(ABC):
    """Abstract observation operator H: state → observables."""

    @abstractmethod
    def __call__(
        self,
        model: nn.Module,
        x_obs: torch.Tensor,
        *,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate H(u) at observation locations.

        Parameters
        ----------
        model : nn.Module
            Forward PINN / surrogate model u(x; θ).
        x_obs : torch.Tensor, shape (N_obs, d)
            Observation locations in physical space.
        params : torch.Tensor, optional
            Physical parameter vector θ (passed to model if needed).

        Returns
        -------
        torch.Tensor, shape (N_obs,) or (N_obs, k)
            Predicted observable values.
        """

    def jacobian_theta(
        self,
        model: nn.Module,
        x_obs: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        """Jacobian ∂H/∂θ via forward-mode AD.

        Returns shape (N_obs * k, p) where p = len(theta).
        """
        def _f(t):
            return self(model, x_obs, params=t).reshape(-1)

        return torch.autograd.functional.jacobian(_f, theta, create_graph=False)


# ──────────────────────────────────────────────────────────────────────────────
# Point observations (collocation at sparse sensor locations)
# ──────────────────────────────────────────────────────────────────────────────

class PointObsOperator(ObsOperatorBase):
    """Evaluate the model at a fixed set of sensor/measurement locations.

    H(u) = u(x_sensor₁), u(x_sensor₂), …, u(x_sensorₙ)

    This is the most common observation operator for PINN-based inverse
    problems where sparse sensor data is available.

    Parameters
    ----------
    sensor_locations : torch.Tensor, shape (N_sensors, d)
        Fixed spatial (and optionally temporal) coordinates of sensors.
    output_indices : list of int, optional
        Indices of model outputs to observe (for multi-field models).
        Default: all outputs.
    """

    def __init__(
        self,
        sensor_locations: torch.Tensor,
        output_indices: Optional[List[int]] = None,
    ) -> None:
        self.sensor_locations = sensor_locations
        self.output_indices = output_indices

    def __call__(
        self,
        model: nn.Module,
        x_obs: Optional[torch.Tensor] = None,
        *,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate model at sensor locations.

        If ``x_obs`` is provided it overrides ``sensor_locations``.
        """
        x = x_obs if x_obs is not None else self.sensor_locations
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        x = x.to(device)
        out = model(x)
        # Unwrap PINNOutput / OperatorOutput
        if hasattr(out, "y"):
            out = out.y
        if self.output_indices is not None:
            out = out[:, self.output_indices]
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Linear observation operator  H(u) = A u
# ──────────────────────────────────────────────────────────────────────────────

class LinearObsOperator(ObsOperatorBase):
    """Linear observation operator H(u) = A · u_vec.

    Useful for:
    - Projection onto measurement basis (e.g. Fourier coefficients)
    - Tomographic projections (Radon transform)
    - Aggregated measurements (spatial averages via all-ones row)

    Parameters
    ----------
    A : torch.Tensor, shape (k, n)
        Observation matrix.  Applied to the flattened model output of size n.
    x_eval : torch.Tensor, shape (n, d)
        Grid at which to evaluate the model before applying A.
    """

    def __init__(self, A: torch.Tensor, x_eval: torch.Tensor) -> None:
        self.A = A
        self.x_eval = x_eval

    def __call__(
        self,
        model: nn.Module,
        x_obs: Optional[torch.Tensor] = None,
        *,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        x = self.x_eval.to(dev)
        out = model(x)
        if hasattr(out, "y"):
            out = out.y
        u_vec = out.reshape(-1)                  # flatten model output
        A = self.A.to(dev)
        return A @ u_vec                         # (k,)


# ──────────────────────────────────────────────────────────────────────────────
# Integral / quadrature observation operator
# ──────────────────────────────────────────────────────────────────────────────

class IntegralObsOperator(ObsOperatorBase):
    """Integral functional: y = ∫ k(x) u(x) dx ≈ Σᵢ wᵢ k(xᵢ) u(xᵢ).

    Represents spatial or temporal averages, flux integrals, moments, etc.

    Parameters
    ----------
    x_quad : torch.Tensor, shape (N_q, d)
        Quadrature points.
    weights : torch.Tensor, shape (N_q,) or (N_obs, N_q)
        Quadrature weights wᵢ.  If 2D, each row corresponds to one observable.
    kernel_fn : callable, optional
        k(x) → scalar weight at each point.  If None, uses uniform kernel.
    """

    def __init__(
        self,
        x_quad: torch.Tensor,
        weights: torch.Tensor,
        kernel_fn: Optional[Callable] = None,
    ) -> None:
        self.x_quad = x_quad
        self.weights = weights
        self.kernel_fn = kernel_fn

    def __call__(
        self,
        model: nn.Module,
        x_obs: Optional[torch.Tensor] = None,
        *,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        x = self.x_quad.to(dev)
        out = model(x)
        if hasattr(out, "y"):
            out = out.y

        u = out.squeeze(-1)  # (N_q,)
        w = self.weights.to(dev)

        if self.kernel_fn is not None:
            k = self.kernel_fn(x)
            u = u * k

        if w.ndim == 1:
            return (w * u).sum(dim=-1)     # scalar integral
        else:
            return w @ u                   # (N_obs,) multiple integrals


# ──────────────────────────────────────────────────────────────────────────────
# Composed operator  [H₁; H₂; …]
# ──────────────────────────────────────────────────────────────────────────────

class ComposedObsOperator(ObsOperatorBase):
    """Stack multiple observation operators into a single vector.

    H_composed(u) = [H₁(u); H₂(u); …; Hₙ(u)]

    Parameters
    ----------
    operators : list of ObsOperatorBase
    x_obs_list : list of torch.Tensor or None
        Sensor locations per operator.  Pass None to use operator defaults.
    """

    def __init__(
        self,
        operators: List[ObsOperatorBase],
        x_obs_list: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> None:
        self.operators = operators
        self.x_obs_list = x_obs_list or [None] * len(operators)

    def __call__(
        self,
        model: nn.Module,
        x_obs: Optional[torch.Tensor] = None,
        *,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        results = []
        for op, x in zip(self.operators, self.x_obs_list):
            xi = x_obs if x is None else x
            results.append(op(model, xi, params=params).reshape(-1))
        return torch.cat(results, dim=0)

    def n_obs(self) -> Optional[int]:
        """Return total number of observables (if all operators have known size)."""
        return None


__all__ = [
    "ObsOperatorBase",
    "PointObsOperator",
    "LinearObsOperator",
    "IntegralObsOperator",
    "ComposedObsOperator",
]
