"""Tests for pinneapple_analysis.verification.evidence_graph.

Uses real dataclasses (ProvenanceRecord, ConfidenceComponent,
PhysicsConfidenceScore, ComparisonReport) as fixtures -- not mocks --
since these are plain, cheap-to-construct dataclasses already used
throughout the sibling verification modules.
"""
from __future__ import annotations

import pytest

from pinneapple_analysis.verification.provenance import ProvenanceRecord
from pinneapple_analysis.verification.physics_confidence_score import (
    ConfidenceComponent,
    PhysicsConfidenceScore,
)
from pinneapple_analysis.verification.solver_orchestration import ComparisonReport
from pinneapple_analysis.verification.evidence_graph import (
    build_evidence_graph,
    evidence_summary,
    explain_trust,
)


def _provenance(trustworthy=True, checks=None) -> ProvenanceRecord:
    return ProvenanceRecord(
        run_id="run-001",
        problem_description="steady 2D heat conduction in a plate",
        architecture="vanilla_pinn",
        drafted_preset="heat_equation_steady",
        flow_regime=None,
        guardrail_trustworthy=trustworthy,
        guardrail_checks=checks if checks is not None else [
            {"name": "pde_residual", "passed": True, "detail": "residual=1.2e-4"},
            {"name": "parameter_sanity", "passed": True, "detail": "all params positive"},
        ],
    )


def test_build_from_provenance_only():
    g = build_evidence_graph(_provenance())
    summary = evidence_summary(g)

    assert summary["run_id"] == "run-001"
    assert summary["trustworthy"] is True
    assert summary["n_evidence_nodes"] == 2  # the 2 guardrail checks
    assert summary["evidence_by_kind"] == {"guardrail_check": 2}
    assert summary["n_supports"] == 2
    assert summary["n_contradicts"] == 0


def test_build_with_a_failed_guardrail_check_is_a_contradiction():
    checks = [
        {"name": "pde_residual", "passed": True, "detail": "residual=1.2e-4"},
        {"name": "conservation_balance", "passed": False, "detail": "mass not conserved: 8% drift"},
    ]
    g = build_evidence_graph(_provenance(trustworthy=False, checks=checks))
    summary = evidence_summary(g)

    assert summary["trustworthy"] is False
    assert summary["n_supports"] == 1
    assert summary["n_contradicts"] == 1

    text = explain_trust(g)
    assert "NOT TRUSTWORTHY" in text
    assert "conservation_balance" in text
    assert "mass not conserved" in text


def test_build_with_no_guardrail_checks_has_no_evidence():
    g = build_evidence_graph(_provenance(trustworthy=None, checks=[]))
    summary = evidence_summary(g)

    assert summary["trustworthy"] is None
    assert summary["n_evidence_nodes"] == 0
    text = explain_trust(g)
    assert "UNKNOWN" in text
    assert "no evidence beyond provenance" in text


def test_build_with_confidence_score():
    confidence = PhysicsConfidenceScore(
        overall_score=0.9,
        components=[
            ConfidenceComponent(name="physics_guardrail", score=1.0, source_summary="2/2 passed"),
            ConfidenceComponent(name="numerical_convergence", score=0.8, source_summary="GCI_fine=0.01"),
        ],
        coverage=0.5,
    )
    g = build_evidence_graph(_provenance(checks=[]), confidence=confidence)
    summary = evidence_summary(g)

    assert summary["evidence_by_kind"] == {"confidence_component": 2}
    assert summary["n_supports"] == 2
    assert summary["n_contradicts"] == 0


def test_build_with_low_confidence_component_is_a_contradiction():
    confidence = PhysicsConfidenceScore(
        overall_score=0.2,
        components=[ConfidenceComponent(name="benchmark_agreement", score=0.2, source_summary="rel_l2=0.8")],
        coverage=0.25,
    )
    g = build_evidence_graph(_provenance(checks=[]), confidence=confidence)
    summary = evidence_summary(g)

    assert summary["n_contradicts"] == 1
    text = explain_trust(g)
    assert "benchmark_agreement" in text
    assert "0.2" in text


def test_build_with_solver_comparisons():
    close = ComparisonReport(
        label_a="pinn", label_b="openfoam", n_points=100, max_abs_diff=0.01, mean_abs_diff=0.005,
        rmse=0.006, relative_rmse=0.02, worst_index=5, worst_abs_diff=0.01, summary="close agreement",
    )
    far = ComparisonReport(
        label_a="pinn", label_b="fenicsx", n_points=100, max_abs_diff=5.0, mean_abs_diff=2.0,
        rmse=2.5, relative_rmse=0.9, worst_index=7, worst_abs_diff=5.0, summary="large disagreement",
    )
    g = build_evidence_graph(_provenance(checks=[]), comparisons=[close, far])
    summary = evidence_summary(g)

    assert summary["evidence_by_kind"] == {"solver_comparison": 2}
    assert summary["n_supports"] == 1
    assert summary["n_contradicts"] == 1
    text = explain_trust(g)
    assert "large disagreement" in text


def test_build_with_everything_combined():
    confidence = PhysicsConfidenceScore(
        overall_score=1.0,
        components=[ConfidenceComponent(name="physics_guardrail", score=1.0, source_summary="2/2 passed")],
        coverage=0.25,
    )
    comparison = ComparisonReport(
        label_a="pinn", label_b="openfoam", n_points=50, max_abs_diff=0.02, mean_abs_diff=0.01,
        rmse=0.012, relative_rmse=0.03, worst_index=1, worst_abs_diff=0.02, summary="close agreement",
    )
    g = build_evidence_graph(_provenance(), confidence=confidence, comparisons=[comparison])
    summary = evidence_summary(g)

    assert summary["n_evidence_nodes"] == 2 + 1 + 1  # 2 guardrail checks + 1 confidence + 1 comparison
    assert summary["n_contradicts"] == 0


def test_evidence_summary_rejects_a_graph_without_a_claim_node():
    import networkx as nx
    g = nx.DiGraph()
    g.add_node("not_a_claim", kind="run")
    with pytest.raises(ValueError, match="exactly one 'claim' node"):
        evidence_summary(g)
