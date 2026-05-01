"""Co-simulation engine: time-stepped execution with algebraic-loop iteration.

Execution model:
  1. ``execution_order()`` on the graph yields SCCs in topological order.
  2. Single-node SCCs execute once (no loop).
  3. Multi-node SCCs (algebraic / feedback loops) iterate with Gauss-Seidel
     or Jacobi until port values converge or ``max_iter`` is reached.
  4. The recorder captures outputs at each time step.

Differentiability:
  Torch computational graphs are preserved through every ``step()`` call by
  default.  Set ``retain_graph=True`` if you need BPTT across time steps
  (required when unrolling a training loop over multiple simulation steps).

Usage::

    engine = CoSimEngine(graph, recorder=recorder, loop_solver="gauss_seidel")
    engine.reset()
    engine.initialize_ports({"spring": {"x": torch.zeros(1)}})
    recorder = engine.run(T=10.0, dt=0.01)
    traj = recorder.get("mass", "x")
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .graph import CoSimGraph
from .recorder import TrajectoryRecorder


class CoSimEngine:
    """Time-stepped co-simulation engine over a ``CoSimGraph``.

    Args:
        graph:        The ``CoSimGraph`` to simulate.
        recorder:     Optional ``TrajectoryRecorder``; one is created automatically
                      by ``run()`` if not provided.
        loop_solver:  ``"gauss_seidel"`` (sequential, faster convergence) or
                      ``"jacobi"`` (parallel, more faithful to true parallelism).
        max_iter:     Maximum iterations per algebraic loop per time step.
        tol:          Convergence tolerance (max absolute change across all ports).
        retain_graph: Keep PyTorch computational graph after each step (needed
                      for BPTT; increases memory usage).
    """

    def __init__(
        self,
        graph: CoSimGraph,
        recorder: Optional[TrajectoryRecorder] = None,
        loop_solver: str = "gauss_seidel",
        max_iter: int = 50,
        tol: float = 1e-6,
        retain_graph: bool = False,
    ) -> None:
        if loop_solver not in ("gauss_seidel", "jacobi"):
            raise ValueError(
                f"loop_solver must be 'gauss_seidel' or 'jacobi', got {loop_solver!r}."
            )
        self.graph = graph
        self.recorder = recorder
        self.loop_solver = loop_solver
        self.max_iter = max_iter
        self.tol = tol
        self.retain_graph = retain_graph

        # Current port values: {node_name: {port_name: tensor}}
        self._vals: Dict[str, Dict[str, torch.Tensor]] = {
            name: {} for name in graph.nodes
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset engine state and all nodes. Call before each new simulation."""
        self.graph.reset_all()
        self._vals = {name: {} for name in self.graph.nodes}
        if self.recorder is not None:
            self.recorder.reset()

    def initialize_ports(
        self, initial: Dict[str, Dict[str, torch.Tensor]]
    ) -> None:
        """Set initial port values before the first ``step()``.

        Args:
            initial: ``{node_name: {port_name: tensor}}``
        """
        for node_name, ports in initial.items():
            if node_name not in self._vals:
                self._vals[node_name] = {}
            self._vals[node_name].update(ports)

    def step(
        self, t: float, dt: float
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Advance all nodes by one time step.

        Args:
            t:  current simulation time.
            dt: time step size.

        Returns:
            Current port values ``{node_name: {port_name: tensor}}``.
        """
        for scc in self.graph.execution_order():
            if len(scc) == 1:
                self._step_single(scc[0], t, dt)
            else:
                self._step_loop(scc, t, dt)

        if self.recorder is not None:
            for name, ports in self._vals.items():
                self.recorder.record(t + dt, name, ports)

        return {n: dict(p) for n, p in self._vals.items()}

    def run(
        self,
        T: float,
        dt: float,
        t0: float = 0.0,
        record_initial: bool = True,
    ) -> TrajectoryRecorder:
        """Run the simulation from *t0* to *T* with fixed step *dt*.

        Args:
            T:               end time (exclusive).
            dt:              time step size.
            t0:              start time (default 0.0).
            record_initial:  if True, record port values at t=t0 before stepping.

        Returns:
            The attached (or newly created) ``TrajectoryRecorder``.
        """
        if self.recorder is None:
            self.recorder = TrajectoryRecorder()

        if record_initial:
            for name, ports in self._vals.items():
                self.recorder.record(t0, name, ports)

        n_steps = max(1, int(round((T - t0) / dt)))
        t = t0
        for _ in range(n_steps):
            self.step(t, dt)
            t += dt

        return self.recorder

    @property
    def port_values(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Current snapshot of all port values."""
        return {n: dict(p) for n, p in self._vals.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gather_inputs(self, node_name: str) -> Dict[str, torch.Tensor]:
        """Collect input tensors for *node_name* from the current port buffer."""
        inputs: Dict[str, torch.Tensor] = {}
        for conn in self.graph.incoming(node_name):
            src_ports = self._vals.get(conn.src_node, {})
            if conn.src_port in src_ports:
                inputs[conn.dst_port] = src_ports[conn.src_port]
        return inputs

    def _step_single(self, node_name: str, t: float, dt: float) -> None:
        node = self.graph.node(node_name)
        inputs = self._gather_inputs(node_name)
        self._vals[node_name] = node.step(inputs, t, dt)

    def _step_loop(self, scc: List[str], t: float, dt: float) -> None:
        if self.loop_solver == "gauss_seidel":
            self._gauss_seidel(scc, t, dt)
        else:
            self._jacobi(scc, t, dt)

    def _gauss_seidel(self, scc: List[str], t: float, dt: float) -> None:
        """Sequential update — each node immediately sees neighbours' new values."""
        for _ in range(self.max_iter):
            prev = {n: dict(self._vals[n]) for n in scc}
            for name in scc:
                self._step_single(name, t, dt)
            if self._converged(scc, prev):
                break

    def _jacobi(self, scc: List[str], t: float, dt: float) -> None:
        """Parallel update — all nodes see last iteration's values."""
        for _ in range(self.max_iter):
            prev = {n: dict(self._vals[n]) for n in scc}
            new: Dict[str, Dict[str, torch.Tensor]] = {}
            for name in scc:
                node = self.graph.node(name)
                inputs = self._gather_inputs(name)
                new[name] = node.step(inputs, t, dt)
            for name, vals in new.items():
                self._vals[name] = vals
            if self._converged(scc, prev):
                break

    def _converged(
        self,
        scc: List[str],
        prev: Dict[str, Dict[str, torch.Tensor]],
    ) -> bool:
        for name in scc:
            for port, val in self._vals[name].items():
                old = prev[name].get(port)
                if old is None:
                    return False
                with torch.no_grad():
                    if float(torch.max(torch.abs(val.detach() - old.detach()))) > self.tol:
                        return False
        return True
