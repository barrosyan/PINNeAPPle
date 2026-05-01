"""Co-simulation graph: node registry, connection management, topology analysis.

Usage::

    from pinneaple_cosim import CoSimGraph, AnalyticalNode, PINNNode

    g = CoSimGraph()
    g.add_node(forcing_node)
    g.add_node(mass_node)
    g.connect("forcing.F", "mass.F_ext")   # "node.port" -> "node.port"
    g.connect("mass.x",    "mass.x_prev")  # feedback loop

    order = g.execution_order()   # [[forcing_node], [mass_node]] or SCC groups
    loops = g.algebraic_loops()   # [[mass_node]] if self-loop
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch.nn as nn

from .connection import Connection
from .node import CoSimNode
from .utils import condensation_order, tarjan_scc


class CoSimGraph:
    """Registry of nodes and connections with lazy topology caching.

    Methods follow a builder pattern so calls can be chained::

        g = CoSimGraph().add_node(a).add_node(b).connect("a.out", "b.in")
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, CoSimNode] = {}
        self._connections: List[Connection] = []
        self._topo_cache: Optional[List[List[str]]] = None

    # ------------------------------------------------------------------
    # Builder API
    # ------------------------------------------------------------------

    def add_node(self, node: CoSimNode) -> "CoSimGraph":
        """Register a node.  Raises ValueError if the name is already taken."""
        if node.name in self._nodes:
            raise ValueError(
                f"Node {node.name!r} already registered. "
                "Use a unique name for each node."
            )
        self._nodes[node.name] = node
        self._topo_cache = None
        return self

    def connect(self, src: str, dst: str) -> "CoSimGraph":
        """Add a directed connection from 'src_node.src_port' to 'dst_node.dst_port'.

        Raises:
            ValueError: if either node or port name is not registered.
        """
        conn = Connection.parse(src, dst)
        self._validate_connection(conn)
        self._connections.append(conn)
        self._topo_cache = None
        return self

    def remove_node(self, name: str) -> "CoSimGraph":
        """Remove a node and all its connections from the graph."""
        if name not in self._nodes:
            raise KeyError(f"Node {name!r} not in graph.")
        del self._nodes[name]
        self._connections = [
            c for c in self._connections
            if c.src_node != name and c.dst_node != name
        ]
        self._topo_cache = None
        return self

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    def execution_order(self) -> List[List[str]]:
        """Return node groups in topological execution order.

        Each group is a list of node names.  Single-element groups have no
        internal cycles; multi-element groups are algebraic loops that the
        engine must iterate to convergence.

        The result is cached and invalidated whenever the graph changes.
        """
        if self._topo_cache is not None:
            return self._topo_cache
        adj = self._adjacency()
        _, order = condensation_order(adj)
        self._topo_cache = order
        return order

    def has_cycles(self) -> bool:
        """Return True if the graph contains any algebraic (feedback) loop."""
        sccs = tarjan_scc(self._adjacency())
        return any(len(s) > 1 for s in sccs)

    def algebraic_loops(self) -> List[List[str]]:
        """Return all SCCs of size > 1 (feedback loops requiring iteration)."""
        sccs = tarjan_scc(self._adjacency())
        return [s for s in sccs if len(s) > 1]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> Dict[str, CoSimNode]:
        return dict(self._nodes)

    @property
    def connections(self) -> List[Connection]:
        return list(self._connections)

    def node(self, name: str) -> CoSimNode:
        """Retrieve a node by name.  Raises KeyError if not found."""
        if name not in self._nodes:
            raise KeyError(f"Node {name!r} not in graph.")
        return self._nodes[name]

    def incoming(self, node_name: str) -> List[Connection]:
        """All connections whose destination is *node_name*."""
        return [c for c in self._connections if c.dst_node == node_name]

    def outgoing(self, node_name: str) -> List[Connection]:
        """All connections whose source is *node_name*."""
        return [c for c in self._connections if c.src_node == node_name]

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        """Yield all ``nn.Parameter`` objects from every node in the graph."""
        for node in self._nodes.values():
            yield from node.parameters()

    def reset_all(self) -> None:
        """Call ``reset()`` on every node (clear internal states)."""
        for node in self._nodes.values():
            node.reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adjacency(self) -> Dict[str, List[str]]:
        """Build adjacency list {src_node: [dst_node, ...]} over all nodes."""
        adj: Dict[str, List[str]] = {name: [] for name in self._nodes}
        for conn in self._connections:
            adj[conn.src_node].append(conn.dst_node)
        return adj

    def _validate_connection(self, conn: Connection) -> None:
        if conn.src_node not in self._nodes:
            raise ValueError(
                f"Source node {conn.src_node!r} not registered in graph. "
                f"Registered nodes: {list(self._nodes.keys())}"
            )
        if conn.dst_node not in self._nodes:
            raise ValueError(
                f"Destination node {conn.dst_node!r} not registered in graph. "
                f"Registered nodes: {list(self._nodes.keys())}"
            )
        src = self._nodes[conn.src_node]
        dst = self._nodes[conn.dst_node]
        if conn.src_port not in src.output_ports:
            raise ValueError(
                f"Port {conn.src_port!r} not in {conn.src_node!r}.output_ports "
                f"= {src.output_ports}"
            )
        if conn.dst_port not in dst.input_ports:
            raise ValueError(
                f"Port {conn.dst_port!r} not in {conn.dst_node!r}.input_ports "
                f"= {dst.input_ports}"
            )

    def __repr__(self) -> str:
        return (
            f"CoSimGraph("
            f"nodes={list(self._nodes.keys())}, "
            f"connections={len(self._connections)}, "
            f"cycles={self.has_cycles()})"
        )
