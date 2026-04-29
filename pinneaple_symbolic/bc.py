"""Boundary condition enforcement for PINNeAPPle symbolic PDE module.

Provides hard (by-construction) and soft (penalty) BC implementations:

- HardBC    : distance-function ansatz — BCs are satisfied exactly at every step.
- PeriodicBC: coordinate-embedding trick to enforce periodicity exactly.
- DirichletBC: soft Dirichlet penalty loss.
- NeumannBC : soft Neumann penalty loss via autograd normal derivative.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Hard BC via distance-function ansatz
# ---------------------------------------------------------------------------

class HardBC:
    """Hard boundary condition enforcement via a distance-function ansatz.

    The network output is modified so that the BC is satisfied *exactly* at
    every evaluation, without any penalty term in the loss:

        u(x) = phi(x) * net(x) + g_bc(x)

    where

    * ``phi(x)`` is a smooth distance function that is 0 on the boundary
      (and positive inside the domain).
    * ``g_bc(x)`` is the boundary value lifted into the domain (can be the
      function that equals the BC on the boundary; a simple choice is the BC
      value extended by any smooth function, or just a constant).

    Parameters
    ----------
    distance_fn : callable(x: Tensor) -> (N, 1) Tensor
        Smooth function that is 0 on Gamma and positive inside the domain.
        Example for the unit square [0,1]^2::

            distance_fn = lambda x: x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2])

    bc_value_fn : callable(x: Tensor) -> (N, 1) Tensor
        The known boundary value extended to the full domain.
        For homogeneous Dirichlet BCs (u=0) this is simply ``lambda x: 0``.

    Examples
    --------
    >>> import torch
    >>> from pinneaple_symbolic.bc import HardBC
    >>> phi = lambda x: x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2])
    >>> bc = HardBC(distance_fn=phi, bc_value_fn=lambda x: torch.zeros(x.shape[0], 1))
    >>> wrapped = bc.wrap_model(model)
    >>> u = wrapped(coords)   # automatically satisfies u = 0 on the boundary
    """

    def __init__(
        self,
        distance_fn: Callable[[torch.Tensor], torch.Tensor],
        bc_value_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self.distance_fn = distance_fn
        self.bc_value_fn = bc_value_fn

    def apply(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply hard BC transformation.

        Returns
        -------
        (N, F) tensor satisfying the BC on the boundary.
        """
        phi = self.distance_fn(x)          # (N, 1) or (N,)
        if phi.ndim == 1:
            phi = phi[:, None]
        g = self.bc_value_fn(x)            # (N, F) or (N, 1) or scalar
        net_out = model(x)
        if net_out.ndim == 1:
            net_out = net_out[:, None]
        if isinstance(g, (int, float)):
            g = torch.full_like(net_out, float(g))
        elif g.ndim == 1:
            g = g[:, None]
        return phi * net_out + g

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Return a new Module whose forward() applies the hard BC ansatz.

        The returned module has the same parameters as *model* and is
        differentiable w.r.t. them.

        Parameters
        ----------
        model : the base neural network.

        Returns
        -------
        HardBCModel wrapping *model*.
        """
        return _HardBCModel(model, self.distance_fn, self.bc_value_fn)


class _HardBCModel(nn.Module):
    """Internal wrapper that enforces a HardBC by construction."""

    def __init__(
        self,
        base: nn.Module,
        distance_fn: Callable[[torch.Tensor], torch.Tensor],
        bc_value_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()
        self.base = base
        self.distance_fn = distance_fn
        self.bc_value_fn = bc_value_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.distance_fn(x)
        if phi.ndim == 1:
            phi = phi[:, None]
        g = self.bc_value_fn(x)
        net_out = self.base(x)
        if net_out.ndim == 1:
            net_out = net_out[:, None]
        if isinstance(g, (int, float)):
            g = torch.full_like(net_out, float(g))
        elif g.ndim == 1:
            g = g[:, None]
        return phi * net_out + g


# ---------------------------------------------------------------------------
# Periodic BC via coordinate embedding
# ---------------------------------------------------------------------------

class PeriodicBC:
    """Periodic boundary condition enforcement via coordinate transformation.

    Maps one coordinate dimension x_i to (cos(2*pi*x_i/L), sin(2*pi*x_i/L)),
    replacing a scalar coordinate with two features on the unit circle.
    This guarantees that the model output at x_i = 0 and x_i = L are
    identical (by construction), without any penalty term.

    Parameters
    ----------
    period : float
        Period L of the coordinate.
    dim : int
        Index of the coordinate dimension that should be periodic (0-based).

    Examples
    --------
    >>> pbc = PeriodicBC(period=1.0, dim=0)
    >>> wrapped = pbc.wrap_model(model)
    >>> u = wrapped(coords)   # coords[:,0] is the periodic coordinate
    """

    def __init__(self, period: float, dim: int = 0) -> None:
        self.period = float(period)
        self.dim = int(dim)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Transform coordinates: replace column *dim* with (cos, sin) pair.

        Parameters
        ----------
        x : (N, D) input tensor.

        Returns
        -------
        (N, D + 1) tensor — the periodic column is replaced by two columns.
        """
        col = x[:, self.dim : self.dim + 1]  # (N, 1)
        theta = 2.0 * math.pi * col / self.period
        cos_col = torch.cos(theta)
        sin_col = torch.sin(theta)
        parts = []
        if self.dim > 0:
            parts.append(x[:, : self.dim])
        parts.append(cos_col)
        parts.append(sin_col)
        if self.dim + 1 < x.shape[1]:
            parts.append(x[:, self.dim + 1 :])
        return torch.cat(parts, dim=1)

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Return a new Module that applies the coordinate transformation before forward.

        The base *model* must accept a (N, D+1) input (one extra dimension for
        the periodic embedding).

        Returns
        -------
        PeriodicBCModel wrapping *model*.
        """
        return _PeriodicBCModel(model, self)


class _PeriodicBCModel(nn.Module):
    """Internal wrapper that applies the periodic coordinate transform."""

    def __init__(self, base: nn.Module, pbc: PeriodicBC) -> None:
        super().__init__()
        self.base = base
        self.pbc = pbc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_transformed = self.pbc.transform(x)
        return self.base(x_transformed)


# ---------------------------------------------------------------------------
# Soft Dirichlet BC (penalty)
# ---------------------------------------------------------------------------

class DirichletBC:
    """Soft Dirichlet boundary condition via an MSE penalty term.

    Computes: loss = weight * mean((u(x_bc) - g(x_bc))^2)

    Parameters
    ----------
    boundary_fn : callable(x: Tensor) -> bool mask Tensor  [optional]
        If provided, applied to filter which points in x_bc are on this boundary.
        When None, all points in x_bc are used.
    value_fn : callable(x: Tensor) -> (N, 1) Tensor
        Target boundary value.
    weight : float
        Loss weight (default 1.0).

    Examples
    --------
    >>> dbc = DirichletBC(value_fn=lambda x: torch.zeros(x.shape[0], 1), weight=100.0)
    >>> loss = dbc.loss(model, x_boundary)
    """

    def __init__(
        self,
        value_fn: Callable[[torch.Tensor], torch.Tensor],
        boundary_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        weight: float = 1.0,
    ) -> None:
        self.value_fn = value_fn
        self.boundary_fn = boundary_fn
        self.weight = float(weight)

    def loss(self, model: nn.Module, x_bc: torch.Tensor) -> torch.Tensor:
        """Compute the Dirichlet BC penalty loss.

        Parameters
        ----------
        model : nn.Module
        x_bc : (N, D) boundary collocation points.

        Returns
        -------
        Scalar loss tensor.
        """
        if self.boundary_fn is not None:
            mask = self.boundary_fn(x_bc).bool()
            x_bc = x_bc[mask]

        if x_bc.numel() == 0:
            return torch.tensor(0.0, device=x_bc.device, dtype=x_bc.dtype)

        u_pred = model(x_bc)
        if u_pred.ndim == 1:
            u_pred = u_pred[:, None]
        g = self.value_fn(x_bc)
        if isinstance(g, (int, float)):
            g = torch.full_like(u_pred, float(g))
        elif g.ndim == 1:
            g = g[:, None]
        return self.weight * torch.mean((u_pred - g) ** 2)


# ---------------------------------------------------------------------------
# Soft Neumann BC (penalty)
# ---------------------------------------------------------------------------

class NeumannBC:
    """Soft Neumann boundary condition via autograd normal derivative.

    Computes: loss = weight * mean((n·∇u(x_bc) - g(x_bc))^2)

    Parameters
    ----------
    flux_fn : callable(x: Tensor) -> (N, 1) Tensor
        Target normal flux value g(x).
    normal_fn : callable(x: Tensor) -> (N, D) Tensor
        Outward unit normal at each boundary point.
    boundary_fn : callable(x: Tensor) -> bool mask Tensor  [optional]
        If provided, used to filter boundary points.
    weight : float
        Loss weight (default 1.0).

    Examples
    --------
    >>> # Zero Neumann on a boundary where the normal is +x direction
    >>> nbc = NeumannBC(
    ...     flux_fn=lambda x: torch.zeros(x.shape[0], 1),
    ...     normal_fn=lambda x: torch.cat([torch.ones(x.shape[0],1), torch.zeros(x.shape[0],1)], dim=1),
    ... )
    >>> loss = nbc.loss(model, x_boundary)
    """

    def __init__(
        self,
        flux_fn: Callable[[torch.Tensor], torch.Tensor],
        normal_fn: Callable[[torch.Tensor], torch.Tensor],
        boundary_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        weight: float = 1.0,
    ) -> None:
        self.flux_fn = flux_fn
        self.normal_fn = normal_fn
        self.boundary_fn = boundary_fn
        self.weight = float(weight)

    def loss(self, model: nn.Module, x_bc: torch.Tensor) -> torch.Tensor:
        """Compute the Neumann BC penalty loss.

        Parameters
        ----------
        model : nn.Module
        x_bc : (N, D) boundary collocation points.

        Returns
        -------
        Scalar loss tensor.
        """
        if self.boundary_fn is not None:
            mask = self.boundary_fn(x_bc).bool()
            x_bc = x_bc[mask]

        if x_bc.numel() == 0:
            return torch.tensor(0.0, device=x_bc.device, dtype=x_bc.dtype)

        x_bc = x_bc.clone().requires_grad_(True)
        u_pred = model(x_bc)
        if u_pred.ndim == 1:
            u_pred = u_pred[:, None]

        # Compute gradient ∇u  (N, D)
        g_u = torch.autograd.grad(
            outputs=u_pred,
            inputs=x_bc,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]

        # Normal vector at boundary points (N, D)
        normals = self.normal_fn(x_bc.detach())
        if normals.device != x_bc.device:
            normals = normals.to(x_bc.device)

        # n·∇u  (N, 1)
        flux_pred = torch.sum(g_u * normals, dim=1, keepdim=True)

        # Target flux (N, 1)
        g_target = self.flux_fn(x_bc.detach())
        if isinstance(g_target, (int, float)):
            g_target = torch.full_like(flux_pred, float(g_target))
        elif g_target.ndim == 1:
            g_target = g_target[:, None]

        return self.weight * torch.mean((flux_pred - g_target) ** 2)
