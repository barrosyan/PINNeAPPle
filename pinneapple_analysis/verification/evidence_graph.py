"""A run-level Evidence Graph -- ties one analysis run's
:class:`~pinneapple_analysis.verification.provenance.ProvenanceRecord`
together with its (optional) real
:class:`~pinneapple_analysis.verification.physics_confidence_score.PhysicsConfidenceScore`
and any solver-vs-solver
:class:`~pinneapple_analysis.verification.solver_orchestration.ComparisonReport`
objects into one small, queryable claim -> evidence graph.

Why this is a distinct module from ``knowledge_graph``
--------------------------------------------------------
``knowledge_graph.py`` builds ONE graph describing PINNeAPPle's general
physics knowledge (phenomena, equations, presets, references) -- it is
the same graph regardless of which run asked a question. This module
builds a DIFFERENT graph per RUN: "why should THIS specific result be
trusted", grounded only in the real checks that specific run actually
produced. It never re-derives new numbers (exactly like
``physics_confidence_score``'s anti-fabrication design) -- it is a
graph *view* over evidence objects a caller already has in hand.

Node kinds
----------
``claim``
    Exactly one root node per graph: "the result of run <run_id> is
    trustworthy". Its ``trustworthy`` attribute is ``True``/``False``/
    ``None`` (unknown -- e.g. no guardrail ran at all), taken directly
    from ``provenance.guardrail_trustworthy`` -- never computed by this
    module.
``run``
    One node carrying the run's identifying facts (problem description,
    architecture, drafted preset) straight from the ``ProvenanceRecord``.
``guardrail_check``
    One node per entry in ``provenance.guardrail_checks`` (each a real
    dict with ``name``/``passed``/``detail``, the shape every
    ``core/pipeline.py``-style caller in this codebase already produces).
``confidence_component``
    One node per :class:`ConfidenceComponent` in an optional supplied
    ``PhysicsConfidenceScore`` -- only present when a caller actually
    computed one; never fabricated here.
``solver_comparison``
    One node per optional supplied ``ComparisonReport``.

Edges always point FROM the claim node (directly or transitively via
``run``) TO the evidence that supports or contradicts it, labeled
``supports`` or ``contradicts`` based on that evidence's own real
pass/fail status -- this module never decides truth, it only routes
already-decided verdicts into a queryable shape.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx

__all__ = ["build_evidence_graph", "evidence_summary", "explain_trust"]


def build_evidence_graph(
    provenance,
    confidence=None,
    comparisons: Optional[List[Any]] = None,
) -> "nx.DiGraph":
    """Build a small evidence graph for one run.

    Parameters
    ----------
    provenance : pinneapple_analysis.verification.provenance.ProvenanceRecord
        The run's real provenance record. Required -- this is the only
        thing every run in this codebase always has.
    confidence : Optional[pinneapple_analysis.verification.physics_confidence_score.PhysicsConfidenceScore]
        A real, already-computed confidence score (from
        ``compute_physics_confidence(...)``), if the caller has one.
    comparisons : Optional[List[pinneapple_analysis.verification.solver_orchestration.ComparisonReport]]
        Real solver-vs-solver comparisons, if the caller ran any.

    Returns
    -------
    networkx.DiGraph
        Never raises for "nothing but provenance was supplied" -- a
        graph built from provenance alone is still a valid, if smaller,
        evidence graph (mirrors ``PhysicsConfidenceScore``'s own
        ``coverage`` honesty mechanism: less evidence is represented as
        fewer nodes, never as a fabricated one).
    """
    g = nx.DiGraph()

    run_id = provenance.run_id
    claim_id = f"claim::{run_id}"
    run_node_id = f"run::{run_id}"

    g.add_node(
        claim_id,
        kind="claim",
        run_id=run_id,
        trustworthy=provenance.guardrail_trustworthy,
        statement=f"The result of run {run_id!r} is trustworthy",
    )
    g.add_node(
        run_node_id,
        kind="run",
        run_id=run_id,
        problem_description=provenance.problem_description,
        architecture=provenance.architecture,
        drafted_preset=provenance.drafted_preset,
        flow_regime=provenance.flow_regime,
    )
    g.add_edge(claim_id, run_node_id, relation="grounded_in")

    for i, check in enumerate(provenance.guardrail_checks):
        node_id = f"guardrail_check::{run_id}::{i}::{check.get('name', 'unnamed')}"
        passed = bool(check.get("passed"))
        g.add_node(
            node_id,
            kind="guardrail_check",
            name=check.get("name"),
            passed=passed,
            detail=check.get("detail"),
        )
        g.add_edge(run_node_id, node_id, relation="supports" if passed else "contradicts")

    if confidence is not None:
        for comp in confidence.components:
            node_id = f"confidence_component::{run_id}::{comp.name}"
            supports = comp.score >= 0.5
            g.add_node(
                node_id,
                kind="confidence_component",
                name=comp.name,
                score=comp.score,
                source_summary=comp.source_summary,
            )
            g.add_edge(run_node_id, node_id, relation="supports" if supports else "contradicts")

    if comparisons:
        for i, comparison in enumerate(comparisons):
            node_id = f"solver_comparison::{run_id}::{i}"
            # A comparison "supports" the claim when the two independent
            # solvers agree closely (small relative RMSE) -- no fixed
            # universal threshold exists for "close enough" across every
            # possible PDE, so this uses the same 0.5 relative-error
            # anchor `physics_confidence_score._component_from_benchmark`
            # already uses for its own benchmark_agreement component
            # (relative_rmse=1.0 means "no better than a naive zero guess").
            supports = comparison.relative_rmse < 0.5
            g.add_node(
                node_id,
                kind="solver_comparison",
                label_a=comparison.label_a,
                label_b=comparison.label_b,
                relative_rmse=comparison.relative_rmse,
                summary=comparison.summary,
            )
            g.add_edge(run_node_id, node_id, relation="supports" if supports else "contradicts")

    return g


def evidence_summary(g: "nx.DiGraph") -> Dict[str, Any]:
    """A structured summary of an evidence graph -- counts of evidence
    nodes by kind and by supports/contradicts, plus the claim's own
    stated trustworthiness. Never re-derives trustworthiness itself."""
    claim_nodes = [n for n, d in g.nodes(data=True) if d.get("kind") == "claim"]
    if len(claim_nodes) != 1:
        raise ValueError(
            f"evidence_summary expects a graph with exactly one 'claim' node "
            f"(got {len(claim_nodes)}) -- pass a graph built by build_evidence_graph()."
        )
    claim_id = claim_nodes[0]
    claim = g.nodes[claim_id]

    evidence_by_kind: Dict[str, int] = {}
    n_supports = 0
    n_contradicts = 0
    for _, _, data in g.edges(data=True):
        relation = data.get("relation")
        if relation == "supports":
            n_supports += 1
        elif relation == "contradicts":
            n_contradicts += 1

    for _, data in g.nodes(data=True):
        kind = data.get("kind")
        if kind and kind not in ("claim", "run"):
            evidence_by_kind[kind] = evidence_by_kind.get(kind, 0) + 1

    return {
        "run_id": claim.get("run_id"),
        "trustworthy": claim.get("trustworthy"),
        "n_evidence_nodes": sum(evidence_by_kind.values()),
        "evidence_by_kind": evidence_by_kind,
        "n_supports": n_supports,
        "n_contradicts": n_contradicts,
    }


def explain_trust(g: "nx.DiGraph") -> str:
    """A plain-text "why should I trust this" rendering of an evidence
    graph -- mirrors ``knowledge_graph.explain_preset``'s pattern of
    producing an LLM-context-ready string from real graph contents, but
    grounded in one run's actual evidence instead of general physics
    knowledge. Never invents a reason not present in the graph."""
    summary = evidence_summary(g)
    lines = [f"Evidence graph for run {summary['run_id']!r}:"]

    trustworthy = summary["trustworthy"]
    if trustworthy is None:
        lines.append("  Overall trustworthiness: UNKNOWN (no guardrail check recorded for this run)")
    else:
        lines.append(f"  Overall trustworthiness: {'TRUSTWORTHY' if trustworthy else 'NOT TRUSTWORTHY'}")

    lines.append(
        f"  Evidence: {summary['n_evidence_nodes']} item(s) "
        f"({summary['n_supports']} supporting, {summary['n_contradicts']} contradicting)"
    )
    for kind, count in sorted(summary["evidence_by_kind"].items()):
        lines.append(f"    - {count} {kind} node(s)")

    if summary["n_evidence_nodes"] == 0:
        lines.append("  (no evidence beyond provenance itself was supplied to build_evidence_graph)")

    for _, data in g.nodes(data=True):
        if data.get("kind") == "guardrail_check" and not data.get("passed"):
            lines.append(f"  CONTRADICTING: guardrail check {data.get('name')!r} failed -- {data.get('detail')}")
        elif data.get("kind") == "confidence_component" and data.get("score", 1.0) < 0.5:
            lines.append(
                f"  CONTRADICTING: confidence component {data.get('name')!r} scored "
                f"{data.get('score'):.3g} -- {data.get('source_summary')}"
            )
        elif data.get("kind") == "solver_comparison" and data.get("relative_rmse", 0.0) >= 0.5:
            lines.append(
                f"  CONTRADICTING: solver comparison {data.get('label_a')!r} vs "
                f"{data.get('label_b')!r} -- {data.get('summary')}"
            )

    return "\n".join(lines)
