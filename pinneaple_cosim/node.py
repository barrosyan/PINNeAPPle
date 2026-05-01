"""Co-simulation node abstractions.

Hierarchy:
    CoSimNode (abstract base)
    ├── TorchNode       — wraps nn.Module; gradients flow through step()
    ├── AnalyticalNode  — pure Python/NumPy callable; no autograd
    ├── PINNNode        — TorchNode + physics residual loss
    └── BlackBoxNode    — opaque callable; detached from autograd

Port conventions:
    Each node declares ``input_ports`` and ``output_ports`` by name.
    The engine resolves connections by matching port names across nodes.
"""
from __future__ import annotations

import abc
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class CoSimNode(abc.ABC):
    """Abstract base class for all co-simulation nodes.

    Subclasses implement ``step()`` to advance the node by one time increment
    given a dict of input tensors and return a dict of output tensors.
    """

    def __init__(
        self,
        name: str,
        input_ports: List[str],
        output_ports: List[str],
    ) -> None:
        self.name = name
        self.input_ports: List[str] = list(input_ports)
        self.output_ports: List[str] = list(output_ports)
        self._state: Dict[str, torch.Tensor] = {}

    @abc.abstractmethod
    def step(
        self,
        inputs: Dict[str, torch.Tensor],
        t: float,
        dt: float,
    ) -> Dict[str, torch.Tensor]:
        """Advance the node by one time step.

        Args:
            inputs: ``{port_name: tensor}`` for every input port.
            t:      current simulation time (seconds or non-dimensional).
            dt:     time step size.

        Returns:
            ``{port_name: tensor}`` for every output port.
        """

    def reset(self) -> None:
        """Clear internal state. Call before each new simulation run."""
        self._state.clear()

    def physics_loss(self) -> Optional[torch.Tensor]:
        """Return a scalar physics-residual loss, or None for non-PINN nodes."""
        return None

    def parameters(self) -> Iterable[nn.Parameter]:
        """Yield trainable parameters (empty for non-torch nodes)."""
        return iter([])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"in={self.input_ports}, "
            f"out={self.output_ports})"
        )


# ---------------------------------------------------------------------------
# TorchNode
# ---------------------------------------------------------------------------

class TorchNode(CoSimNode):
    """Wraps a ``nn.Module``; gradients flow through ``step()``.

    The module receives the concatenation of all input-port tensors (in port
    order) along the last dimension.  Outputs are split into equal chunks, one
    per output port.  For asymmetric splits, override ``step()``.
    """

    def __init__(
        self,
        name: str,
        model: nn.Module,
        input_ports: List[str],
        output_ports: List[str],
    ) -> None:
        super().__init__(name, input_ports, output_ports)
        self.model = model

    def step(
        self,
        inputs: Dict[str, torch.Tensor],
        t: float,
        dt: float,
    ) -> Dict[str, torch.Tensor]:
        x = torch.cat([inputs[p] for p in self.input_ports], dim=-1)
        out = self.model(x)
        n = len(self.output_ports)
        if n == 1:
            return {self.output_ports[0]: out}
        chunks = torch.chunk(out, n, dim=-1)
        return {p: c for p, c in zip(self.output_ports, chunks)}

    def parameters(self) -> Iterable[nn.Parameter]:
        return self.model.parameters()


# ---------------------------------------------------------------------------
# AnalyticalNode
# ---------------------------------------------------------------------------

class AnalyticalNode(CoSimNode):
    """Wraps a pure Python/NumPy callable.  No autograd — outputs are detached.

    The callable signature must be::

        fn(inputs: Dict[str, Tensor], t: float, dt: float) -> Dict[str, Tensor]
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[Dict[str, torch.Tensor], float, float], Dict[str, torch.Tensor]],
        input_ports: List[str],
        output_ports: List[str],
    ) -> None:
        super().__init__(name, input_ports, output_ports)
        self._fn = fn

    def step(
        self,
        inputs: Dict[str, torch.Tensor],
        t: float,
        dt: float,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            return self._fn(inputs, t, dt)


# ---------------------------------------------------------------------------
# PINNNode
# ---------------------------------------------------------------------------

class PINNNode(TorchNode):
    """``TorchNode`` extended with a physics-residual loss.

    Args:
        name, model, input_ports, output_ports: same as TorchNode.
        physics_fn: callable with signature::

            physics_fn(node, inputs, t, dt) -> scalar Tensor

            where ``node`` is this PINNNode (giving access to ``node.model``).

        physics_weight: scalar multiplier for the returned residual.

    The residual is recomputed each time ``physics_loss()`` is called,
    using the inputs and time from the most recent ``step()`` call.
    """

    def __init__(
        self,
        name: str,
        model: nn.Module,
        input_ports: List[str],
        output_ports: List[str],
        physics_fn: Callable[["PINNNode", Dict[str, torch.Tensor], float, float], torch.Tensor],
        physics_weight: float = 1.0,
    ) -> None:
        super().__init__(name, model, input_ports, output_ports)
        self._physics_fn = physics_fn
        self.physics_weight = physics_weight
        self._last_inputs: Optional[Dict[str, torch.Tensor]] = None
        self._last_t: float = 0.0
        self._last_dt: float = 0.0

    def step(
        self,
        inputs: Dict[str, torch.Tensor],
        t: float,
        dt: float,
    ) -> Dict[str, torch.Tensor]:
        self._last_inputs = {k: v for k, v in inputs.items()}
        self._last_t = t
        self._last_dt = dt
        return super().step(inputs, t, dt)

    def physics_loss(self) -> Optional[torch.Tensor]:
        if self._last_inputs is None:
            return None
        residual = self._physics_fn(
            self, self._last_inputs, self._last_t, self._last_dt
        )
        return self.physics_weight * residual

    def reset(self) -> None:
        super().reset()
        self._last_inputs = None


# ---------------------------------------------------------------------------
# BlackBoxNode
# ---------------------------------------------------------------------------

class BlackBoxNode(CoSimNode):
    """Opaque callable node — fully detached from autograd.

    Use for legacy simulators, look-up tables, or any non-differentiable
    component. The callable signature is the same as AnalyticalNode.
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[Dict[str, torch.Tensor], float, float], Dict[str, torch.Tensor]],
        input_ports: List[str],
        output_ports: List[str],
    ) -> None:
        super().__init__(name, input_ports, output_ports)
        self._fn = fn

    def step(
        self,
        inputs: Dict[str, torch.Tensor],
        t: float,
        dt: float,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            result = self._fn({k: v.detach() for k, v in inputs.items()}, t, dt)
        return result
