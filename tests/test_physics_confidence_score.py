"""Tests for ``pinneapple_analysis.verification.physics_confidence_score``
(``ConfidenceComponent``/``PhysicsConfidenceScore``/
``compute_physics_confidence``), the transparent, componentized confidence
aggregator built on top of this repo's real verification checks.

Design under test, restated: this module never runs a check itself -- it
only turns real result objects a caller already has
(``pinneapple_llm.guardrail.GuardrailReport``,
``pinneapple_analysis.verification.convergence.ConvergenceResult``, real
``CalibrationMetrics`` numbers wrapped in a ``CalibrationSummary``,
``pinneapple_data.physics_case.BenchmarkComparison``) into one auditable
score, with ``coverage`` always reported alongside ``overall_score``. Where
practical these tests use REAL (not mocked) instances of the referenced
types -- a real ``richardson_extrapolate`` computation on a known-order
synthetic problem (mirroring ``tests/test_grid_convergence.py``), real
``CalibrationMetrics`` static-method outputs, a real
``lane_emden_n1.5`` benchmark comparison via ``PhysicsCase`` (mirroring
``tests/test_physics_case.py``), and real ``GuardrailReport``/
``CheckResult`` dataclass instances (mirroring the check-result shapes
``PhysicsGuardrail.check()`` itself produces, per
``tests/test_physics_guardrail.py``).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pinneapple_analysis.verification.physics_confidence_score import (
    CalibrationSummary,
    ConfidenceComponent,
    N_POSSIBLE_COMPONENTS,
    PhysicsConfidenceScore,
    compute_physics_confidence,
)
from pinneapple_analysis.verification.convergence import richardson_extrapolate
from pinneapple_analysis.uncertainty.calibration import CalibrationMetrics
from pinneapple_llm.guardrail import CheckResult, GuardrailReport
from pinneapple_data.physics_case import PhysicsCase
from pinneapple_pdb.benchmarks import get_benchmark


# ---------------------------------------------------------------------------
# Real building blocks shared across tests
# ---------------------------------------------------------------------------

def _real_guardrail_report_all_pass() -> GuardrailReport:
    return GuardrailReport(checks=[
        CheckResult(name="parameter_sanity", passed=True, detail="all recognised physical parameters are positive"),
        CheckResult(name="pde_residual", passed=True, detail="mean-squared PDE residual = 1e-4 (<= threshold 1e-2)",
                    value=1e-4, threshold=1e-2),
    ])


def _real_guardrail_report_partial_fail() -> GuardrailReport:
    return GuardrailReport(checks=[
        CheckResult(name="parameter_sanity", passed=True, detail="all recognised physical parameters are positive"),
        CheckResult(name="pde_residual", passed=False, detail="mean-squared PDE residual = 5e-1 (> threshold 1e-2)",
                    value=5e-1, threshold=1e-2),
    ])


def _real_convergence_result_asymptotic():
    """Second-order-accurate central-difference approximation of
    f'(0.6)=cos(0.6) for f=sin -- same synthetic known-order construction
    ``tests/test_grid_convergence.py`` uses, so the observed order/GCI are
    real numbers from a real, analytically-understood computation."""
    x0 = 0.6

    def central_difference(n_points: int) -> float:
        h = 2.0 / n_points
        return (np.sin(x0 + h) - np.sin(x0 - h)) / (2.0 * h)

    f_coarse = central_difference(20)
    f_medium = central_difference(40)
    f_fine = central_difference(80)
    return richardson_extrapolate(f_coarse, f_medium, f_fine, r=2.0)


def _real_calibration_summary_well_calibrated() -> CalibrationSummary:
    torch.manual_seed(0)
    y_pred = torch.randn(2000)
    y_std = torch.full((2000,), 1.0)
    y_true = y_pred + y_std * torch.randn(2000)  # residuals genuinely ~ N(0, y_std^2): well calibrated
    ece = CalibrationMetrics.expected_calibration_error(y_pred, y_true, y_std, n_bins=15)
    cov = CalibrationMetrics.coverage_at_level(y_pred, y_true, y_std, alpha=0.1)
    sharp = CalibrationMetrics.sharpness(y_std)
    return CalibrationSummary(ece=ece, coverage=cov, target_coverage=0.9, sharpness=sharp)


def _real_calibration_summary_poorly_calibrated() -> CalibrationSummary:
    torch.manual_seed(0)
    y_pred = torch.randn(2000)
    y_true = y_pred + 5.0 * torch.randn(2000)  # residual std is 5x the reported y_std: badly miscalibrated
    y_std = torch.full((2000,), 1.0)
    ece = CalibrationMetrics.expected_calibration_error(y_pred, y_true, y_std, n_bins=15)
    return CalibrationSummary(ece=ece)


def _real_benchmark_comparison_close():
    entry = get_benchmark("lane_emden_n1.5")
    results = {
        "xi": entry.reference_x[:, 0],
        "theta": entry.reference_y[:, 0],
        "phi": entry.reference_y[:, 1],
    }
    case = PhysicsCase(reference_benchmark="lane_emden_n1.5", results=results)
    return case.validate_against_benchmark()


def _real_benchmark_comparison_wrong():
    entry = get_benchmark("lane_emden_n1.5")
    results = {
        "xi": entry.reference_x[:, 0],
        "theta": np.zeros_like(entry.reference_y[:, 0]),
        "phi": np.zeros_like(entry.reference_y[:, 1]),
    }
    case = PhysicsCase(reference_benchmark="lane_emden_n1.5", results=results)
    return case.validate_against_benchmark()


# ---------------------------------------------------------------------------
# 0 components
# ---------------------------------------------------------------------------

def test_zero_components_never_fabricates_a_score():
    result = compute_physics_confidence()
    assert isinstance(result, PhysicsConfidenceScore)
    assert result.overall_score is None
    assert result.components == []
    assert result.coverage == 0.0
    # summary() must not raise, and must be explicit about the "None" case.
    assert "None" in result.summary()


# ---------------------------------------------------------------------------
# 1 component: physics_guardrail alone
# ---------------------------------------------------------------------------

def test_one_component_guardrail_all_pass():
    report = _real_guardrail_report_all_pass()
    result = compute_physics_confidence(guardrail_report=report)
    assert result.coverage == pytest.approx(1 / N_POSSIBLE_COMPONENTS)
    assert len(result.components) == 1
    comp = result.components[0]
    assert comp.name == "physics_guardrail"
    assert comp.score == pytest.approx(1.0)
    assert result.overall_score == pytest.approx(1.0)
    assert "parameter_sanity=PASS" in comp.source_summary
    assert "pde_residual=PASS" in comp.source_summary
    assert "trustworthy=True" in comp.source_summary


def test_one_component_guardrail_partial_fail_uses_finer_resolution_than_bool():
    """The score reflects the FRACTION of checks passed (1/2 = 0.5), not a
    collapsed 0.0/1.0 -- this is the documented finer-resolution behaviour."""
    report = _real_guardrail_report_partial_fail()
    result = compute_physics_confidence(guardrail_report=report)
    comp = result.components[0]
    assert comp.score == pytest.approx(0.5)
    assert "1/2 guardrail checks passed" in comp.source_summary
    assert "trustworthy=False" in comp.source_summary
    assert result.overall_score == pytest.approx(0.5)


def test_guardrail_component_empty_checks_is_vacuously_one():
    report = GuardrailReport(checks=[])
    result = compute_physics_confidence(guardrail_report=report)
    assert result.components[0].score == pytest.approx(1.0)
    assert "0 checks ran" in result.components[0].source_summary


# ---------------------------------------------------------------------------
# 1 component: numerical_convergence alone
# ---------------------------------------------------------------------------

def test_convergence_component_asymptotic_scores_one():
    convergence = _real_convergence_result_asymptotic()
    assert convergence.is_asymptotic  # sanity: this synthetic problem really is asymptotic
    result = compute_physics_confidence(convergence_result=convergence)
    comp = result.components[0]
    assert comp.name == "numerical_convergence"
    assert comp.score == pytest.approx(1.0)
    assert f"{convergence.gci_fine:.4g}" in comp.source_summary
    assert f"{convergence.asymptotic_ratio:.4g}" in comp.source_summary
    assert result.coverage == pytest.approx(1 / N_POSSIBLE_COMPONENTS)


def test_convergence_component_non_asymptotic_scores_between_zero_and_one():
    from pinneapple_analysis.verification.convergence import ConvergenceResult

    # Fabricate a ConvergenceResult (a plain dataclass) whose asymptotic_ratio
    # is exactly 1.5 with tolerance 0.10 -> deviation=0.5 -> far outside the
    # pass band but not at the 2x-tolerance floor, so score is strictly
    # between 0 and 1.
    convergence = ConvergenceResult(
        observed_order=1.9, extrapolated_value=1.0, gci_fine=0.05, gci_coarse=0.09,
        asymptotic_ratio=1.15, is_asymptotic=False, refinement_ratio=2.0,
        asymptotic_tolerance=0.10,
    )
    result = compute_physics_confidence(convergence_result=convergence)
    comp = result.components[0]
    # deviation=0.15, tol=0.10 -> score = 1 - (0.15-0.10)/0.10 = 0.5
    assert comp.score == pytest.approx(0.5)
    assert 0.0 < comp.score < 1.0


def test_convergence_component_far_outside_asymptotic_range_clamps_to_zero():
    from pinneapple_analysis.verification.convergence import ConvergenceResult

    convergence = ConvergenceResult(
        observed_order=0.1, extrapolated_value=1.0, gci_fine=0.9, gci_coarse=0.95,
        asymptotic_ratio=5.0, is_asymptotic=False, refinement_ratio=2.0,
        asymptotic_tolerance=0.10,
    )
    result = compute_physics_confidence(convergence_result=convergence)
    assert result.components[0].score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 1 component: uq_calibration alone
# ---------------------------------------------------------------------------

def test_calibration_component_well_calibrated_scores_high():
    calibration = _real_calibration_summary_well_calibrated()
    result = compute_physics_confidence(calibration_metrics=calibration)
    comp = result.components[0]
    assert comp.name == "uq_calibration"
    assert comp.score == pytest.approx(1.0 - calibration.ece)
    assert f"ECE={calibration.ece:.4g}" in comp.source_summary
    assert "coverage=" in comp.source_summary
    assert "sharpness=" in comp.source_summary


def test_calibration_component_poorly_calibrated_scores_lower():
    good = _real_calibration_summary_well_calibrated()
    bad = _real_calibration_summary_poorly_calibrated()
    good_result = compute_physics_confidence(calibration_metrics=good)
    bad_result = compute_physics_confidence(calibration_metrics=bad)
    assert bad.ece > good.ece
    assert bad_result.components[0].score < good_result.components[0].score


def test_calibration_component_without_optional_fields_omits_them_from_summary():
    calibration = CalibrationSummary(ece=0.05)
    result = compute_physics_confidence(calibration_metrics=calibration)
    comp = result.components[0]
    assert comp.score == pytest.approx(0.95)
    assert "coverage=" not in comp.source_summary
    assert "sharpness=" not in comp.source_summary


# ---------------------------------------------------------------------------
# 1 component: benchmark_agreement alone
# ---------------------------------------------------------------------------

def test_benchmark_component_close_match_scores_high():
    comparison = _real_benchmark_comparison_close()
    result = compute_physics_confidence(benchmark_comparison=comparison)
    comp = result.components[0]
    assert comp.name == "benchmark_agreement"
    assert comp.score == pytest.approx(1.0 - comparison.relative_l2_error, abs=1e-6)
    assert comp.score > 0.999  # exact-grid match: essentially zero error
    assert "lane_emden_n1.5" in comp.source_summary


def test_benchmark_component_wrong_profile_scores_low():
    comparison = _real_benchmark_comparison_wrong()
    result = compute_physics_confidence(benchmark_comparison=comparison)
    comp = result.components[0]
    assert comparison.relative_l2_error > 0.1
    assert comp.score == pytest.approx(1.0 - comparison.relative_l2_error, abs=1e-6)
    assert comp.score < 0.9


def test_benchmark_component_relative_error_beyond_one_clamps_to_zero():
    from pinneapple_data.physics_case import BenchmarkComparison

    comparison = BenchmarkComparison(
        benchmark="fake", x_vars=("xi",), y_vars=("theta",), n_reference_points=10,
        n_compared_points=10, rmse=2.0, relative_l2_error=1.7,
        per_variable_rmse={"theta": 2.0}, per_variable_relative_error={"theta": 1.7},
    )
    result = compute_physics_confidence(benchmark_comparison=comparison)
    assert result.components[0].score == pytest.approx(0.0)


def test_benchmark_component_nan_relative_error_raises_instead_of_fabricating():
    from pinneapple_data.physics_case import BenchmarkComparison

    comparison = BenchmarkComparison(
        benchmark="degenerate", x_vars=("xi",), y_vars=("theta",), n_reference_points=1,
        n_compared_points=1, rmse=0.0, relative_l2_error=float("nan"),
        per_variable_rmse={"theta": 0.0}, per_variable_relative_error={"theta": float("nan")},
    )
    with pytest.raises(ValueError):
        compute_physics_confidence(benchmark_comparison=comparison)


# ---------------------------------------------------------------------------
# 2 components
# ---------------------------------------------------------------------------

def test_two_components_guardrail_and_benchmark():
    report = _real_guardrail_report_all_pass()
    comparison = _real_benchmark_comparison_close()
    result = compute_physics_confidence(guardrail_report=report, benchmark_comparison=comparison)
    assert result.coverage == pytest.approx(2 / N_POSSIBLE_COMPONENTS)
    names = {c.name for c in result.components}
    assert names == {"physics_guardrail", "benchmark_agreement"}
    expected_mean = (1.0 + (1.0 - comparison.relative_l2_error)) / 2.0
    assert result.overall_score == pytest.approx(expected_mean, abs=1e-6)


def test_two_components_ordering_is_fixed_regardless_of_kwarg_order():
    report = _real_guardrail_report_all_pass()
    convergence = _real_convergence_result_asymptotic()
    r1 = compute_physics_confidence(guardrail_report=report, convergence_result=convergence)
    r2 = compute_physics_confidence(convergence_result=convergence, guardrail_report=report)
    assert [c.name for c in r1.components] == [c.name for c in r2.components] == [
        "physics_guardrail", "numerical_convergence",
    ]


# ---------------------------------------------------------------------------
# All 4 components
# ---------------------------------------------------------------------------

def test_all_four_components_full_coverage():
    report = _real_guardrail_report_all_pass()
    convergence = _real_convergence_result_asymptotic()
    calibration = _real_calibration_summary_well_calibrated()
    comparison = _real_benchmark_comparison_close()

    result = compute_physics_confidence(
        guardrail_report=report,
        convergence_result=convergence,
        calibration_metrics=calibration,
        benchmark_comparison=comparison,
    )

    assert result.coverage == pytest.approx(1.0)
    assert len(result.components) == 4
    assert [c.name for c in result.components] == [
        "physics_guardrail", "numerical_convergence", "uq_calibration", "benchmark_agreement",
    ]
    expected_mean = sum(c.score for c in result.components) / 4
    assert result.overall_score == pytest.approx(expected_mean)
    assert 0.0 <= result.overall_score <= 1.0

    text = result.summary()
    assert "coverage=1.00" in text
    for c in result.components:
        assert c.name in text


def test_coverage_reflects_partial_availability_distinctly_from_full():
    """The load-bearing honesty check: a score built from 1/4 checks must
    report a different (lower) coverage than the same-valued score built
    from 4/4 checks -- coverage must never be silently dropped or
    conflated with overall_score."""
    report_all_pass = _real_guardrail_report_all_pass()
    partial = compute_physics_confidence(guardrail_report=report_all_pass)
    assert partial.overall_score == pytest.approx(1.0)
    assert partial.coverage == pytest.approx(0.25)

    convergence = _real_convergence_result_asymptotic()
    calibration = CalibrationSummary(ece=0.0)
    comparison = _real_benchmark_comparison_close()
    full = compute_physics_confidence(
        guardrail_report=report_all_pass,
        convergence_result=convergence,
        calibration_metrics=calibration,
        benchmark_comparison=comparison,
    )
    assert full.coverage == pytest.approx(1.0)
    # Both scores are near 1.0 here, but coverage tells them apart -- that is
    # the entire point of exposing coverage as a first-class field.
    assert partial.coverage != full.coverage


# ---------------------------------------------------------------------------
# ConfidenceComponent is a plain, inspectable dataclass
# ---------------------------------------------------------------------------

def test_confidence_component_fields_are_all_real_values():
    report = _real_guardrail_report_partial_fail()
    result = compute_physics_confidence(guardrail_report=report)
    comp = result.components[0]
    assert isinstance(comp, ConfidenceComponent)
    assert isinstance(comp.score, float)
    assert 0.0 <= comp.score <= 1.0
    assert isinstance(comp.source_summary, str) and comp.source_summary
    assert not math.isnan(comp.score)
