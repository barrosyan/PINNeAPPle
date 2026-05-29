"""Composite loss functions for training differentiable co-simulation graphs.

Three loss components:

DataLoss
    MSE between predicted port outputs and labelled observations.
    Keys use the dotted 'node.port' format.

PhysicsLoss
    Aggregates physics residuals from every PINNNode in the graph.
    Each PINNNode returns a scalar from ``physics_loss()``.

CouplingLoss
    Penalises inconsistencies at node interfaces: for each connection,
    checks that the source output matches the destination input value.
    Useful when the engine runs in a differentiable training loop.

CoSimLoss
    Weighted sum of the three components; returns the total scalar and a
    breakdown dict for logging.

Usage::

    criterion = CoSimLoss(data_weight=1.0, physics_weight=0.5, coupling_weight=0.1)

    total, info = criterion(
        port_values=engine.port_values,
        graph=graph,
        targets={"mass.x": x_observed, "mass.v": v_observed},
    )
    total.backward()
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import CoSimGraph


# ---------------------------------------------------------------------------
# DataLoss
# ---------------------------------------------------------------------------

class DataLoss(nn.Module):
    """MSE between predicted port values and observations.

    Args:
        weight: scalar multiplier for this loss component.
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute mean MSE over all (node.port, target) pairs present in both dicts.

        Args:
            predictions: ``{"node.port": tensor}`` — flattened port values.
            targets:     ``{"node.port": tensor}`` — observed values (same shape).

        Returns:
            Scalar tensor (0.0 if no key overlaps).
        """
        losses = []
        for key, target in targets.items():
            if key in predictions:
                losses.append(F.mse_loss(predictions[key], target))
        if not losses:
            return torch.tensor(0.0, requires_grad=False)
        return self.weight * torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# PhysicsLoss
# ---------------------------------------------------------------------------

class PhysicsLoss(nn.Module):
    """Sum of physics residuals from all PINNNodes in the graph.

    Args:
        weight: scalar multiplier for this loss component.
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, graph: CoSimGraph) -> torch.Tensor:
        """Iterate all nodes, collect non-None ``physics_loss()`` values.

        Returns:
            Scalar tensor (0.0 if no PINN nodes or no step was taken yet).
        """
        residuals = []
        for node in graph.nodes.values():
            r = node.physics_loss()
            if r is not None:
                residuals.append(r)
        if not residuals:
            return torch.tensor(0.0, requires_grad=False)
        return self.weight * torch.stack(residuals).sum()


# ---------------------------------------------------------------------------
# CouplingLoss
# ---------------------------------------------------------------------------

class CouplingLoss(nn.Module):
    """Penalise interface inconsistencies between connected ports.

    For each connection ``src_node.src_port -> dst_node.dst_port``, computes
    MSE between the source output and the currently stored destination input.
    Useful when differentiating through multiple engine steps.

    Args:
        weight: scalar multiplier for this loss component.
    """

    def __init__(self, weight: float = 0.1) -> None:
        super().__init__()
        self.weight = weight

    def forward(
        self,
        port_values: Dict[str, Dict[str, torch.Tensor]],
        graph: CoSimGraph,
    ) -> torch.Tensor:
        """Compute coupling residuals over all graph connections.

        Args:
            port_values: ``{node_name: {port_name: tensor}}`` — engine state.
            graph:       the ``CoSimGraph`` (for connection list).

        Returns:
            Scalar tensor (0.0 if no overlapping values found).
        """
        residuals = []
        for conn in graph.connections:
            src = port_values.get(conn.src_node, {}).get(conn.src_port)
            dst = port_values.get(conn.dst_node, {}).get(conn.dst_port)
            if src is not None and dst is not None and src.shape == dst.shape:
                residuals.append(F.mse_loss(src, dst))
        if not residuals:
            return torch.tensor(0.0, requires_grad=False)
        return self.weight * torch.stack(residuals).mean()


# ---------------------------------------------------------------------------
# CoSimLoss — composite
# ---------------------------------------------------------------------------

class CoSimLoss(nn.Module):
    """Composite loss: data fidelity + physics residuals + coupling consistency.

    Args:
        data_weight:     weight for :class:`DataLoss`.
        physics_weight:  weight for :class:`PhysicsLoss`.
        coupling_weight: weight for :class:`CouplingLoss`.

    Example::

        criterion = CoSimLoss(data_weight=1.0, physics_weight=0.5)
        total, info = criterion(port_values, graph, targets={"mass.x": x_ref})
        total.backward()
        optimizer.step()
    """

    def __init__(
        self,
        data_weight: float = 1.0,
        physics_weight: float = 1.0,
        coupling_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.data_loss     = DataLoss(weight=data_weight)
        self.physics_loss  = PhysicsLoss(weight=physics_weight)
        self.coupling_loss = CouplingLoss(weight=coupling_weight)

    def forward(
        self,
        port_values: Dict[str, Dict[str, torch.Tensor]],
        graph: CoSimGraph,
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute weighted composite loss.

        Args:
            port_values: current engine port buffer ``{node: {port: tensor}}``.
            graph:       the ``CoSimGraph`` (for PINN nodes and connections).
            targets:     optional ``{"node.port": tensor}`` observations.

        Returns:
            ``(total_loss, breakdown)`` where breakdown is a plain dict with
            keys ``"data"``, ``"physics"``, ``"coupling"``, ``"total"``.
        """
        # Flatten port values to "node.port" keys for DataLoss
        flat: Dict[str, torch.Tensor] = {
            f"{node}.{port}": val
            for node, ports in port_values.items()
            for port, val in ports.items()
        }

        data_l     = self.data_loss(flat, targets or {})
        physics_l  = self.physics_loss(graph)
        coupling_l = self.coupling_loss(port_values, graph)

        total = data_l + physics_l + coupling_l

        breakdown: Dict[str, float] = {
            "data":     float(data_l.detach()),
            "physics":  float(physics_l.detach()),
            "coupling": float(coupling_l.detach()),
            "total":    float(total.detach()),
        }
        return total, breakdown
