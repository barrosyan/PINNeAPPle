"""Graph algorithms for co-simulation topology analysis.

Provides:
  - Tarjan's iterative SCC algorithm (safe for large graphs, no recursion limit)
  - Condensation DAG construction and topological sort
  - Simple topological sort for pure DAGs
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# Tarjan's SCC (iterative — avoids Python recursion limit)
# ---------------------------------------------------------------------------

def tarjan_scc(adj: Dict[str, List[str]]) -> List[List[str]]:
    """Find all Strongly Connected Components using Tarjan's algorithm.

    Args:
        adj: adjacency list {node: [neighbor, ...]}. Every node that appears
             as a neighbor must also be a key (add it with an empty list if
             it has no outgoing edges).

    Returns:
        List of SCCs in *reverse* topological order (sources last).
        Each SCC is a list of node names.
    """
    index: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    on_stack: Dict[str, bool] = {}
    stack: List[str] = []
    sccs: List[List[str]] = []
    counter = [0]

    # Iterative DFS using an explicit call stack
    # Each frame: (node, iterator_over_neighbors, already_entered)
    for start in list(adj.keys()):
        if start in index:
            continue

        # (node, neighbor_iter, entered)
        call_stack = [(start, iter(adj.get(start, [])), False)]

        while call_stack:
            v, it, entered = call_stack[-1]

            if not entered:
                # First visit
                index[v] = lowlink[v] = counter[0]
                counter[0] += 1
                on_stack[v] = True
                stack.append(v)
                call_stack[-1] = (v, it, True)

            try:
                w = next(it)
                if w not in index:
                    # Tree edge — push w
                    call_stack.append((w, iter(adj.get(w, [])), False))
                elif on_stack.get(w, False):
                    # Back edge — update lowlink
                    lowlink[v] = min(lowlink[v], index[w])
            except StopIteration:
                # All neighbors processed — pop frame
                call_stack.pop()
                if call_stack:
                    parent = call_stack[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[v])

                # Check if v is a root of an SCC
                if lowlink[v] == index[v]:
                    scc: List[str] = []
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        scc.append(w)
                        if w == v:
                            break
                    sccs.append(scc)

    return sccs


# ---------------------------------------------------------------------------
# Condensation + topological sort
# ---------------------------------------------------------------------------

def condensation_order(
    adj: Dict[str, List[str]],
) -> Tuple[List[List[str]], List[List[str]]]:
    """Compute execution order via SCC condensation.

    Steps:
      1. Find all SCCs with Tarjan.
      2. Build condensation DAG (SCCs as super-nodes).
      3. Topological sort the condensation (Kahn's algorithm).

    Args:
        adj: adjacency list where every referenced node is a key.

    Returns:
        (sccs, order):
            sccs  — all SCCs (each a list of node names)
            order — SCCs in topological execution order (sources first)
    """
    sccs = tarjan_scc(adj)

    # Map node → SCC index
    node_to_scc: Dict[str, int] = {}
    for i, scc in enumerate(sccs):
        for node in scc:
            node_to_scc[node] = i

    n = len(sccs)
    scc_adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for src, neighbors in adj.items():
        s = node_to_scc[src]
        for dst in neighbors:
            d = node_to_scc[dst]
            if s != d:
                scc_adj[s].add(d)

    # Kahn's topological sort on condensation
    in_degree = {i: 0 for i in range(n)}
    for s, dsts in scc_adj.items():
        for d in dsts:
            in_degree[d] += 1

    queue = [i for i in range(n) if in_degree[i] == 0]
    topo: List[int] = []
    while queue:
        node_idx = queue.pop(0)
        topo.append(node_idx)
        for d in scc_adj[node_idx]:
            in_degree[d] -= 1
            if in_degree[d] == 0:
                queue.append(d)

    if len(topo) != n:
        raise RuntimeError(
            "Cycle found in condensation DAG — this is a bug in tarjan_scc."
        )

    return sccs, [sccs[i] for i in topo]


def topological_sort(adj: Dict[str, List[str]]) -> List[str]:
    """Topological sort for pure DAGs (no cycles).

    Raises:
        RuntimeError: if the graph contains a cycle.
    """
    all_nodes = set(adj.keys())
    for neighbors in adj.values():
        all_nodes.update(neighbors)

    in_degree: Dict[str, int] = {v: 0 for v in all_nodes}
    for v in adj:
        for w in adj[v]:
            in_degree[w] += 1

    queue = [v for v, d in in_degree.items() if d == 0]
    order: List[str] = []
    while queue:
        v = queue.pop(0)
        order.append(v)
        for w in adj.get(v, []):
            in_degree[w] -= 1
            if in_degree[w] == 0:
                queue.append(w)

    if len(order) != len(in_degree):
        missing = set(in_degree) - set(order)
        raise RuntimeError(
            f"Graph has cycles involving nodes: {missing}. "
            "Use condensation_order() for graphs with feedback loops."
        )
    return order
