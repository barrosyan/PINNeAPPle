"""``PhysicsConfidenceScore``: a transparent, componentized aggregate score
built ONLY from real, already-executed checks -- never a black-box number.

Why this module exists, and why it is not a contradiction of this repo's
anti-fabrication design
------------------------------------------------------------------------
This repo has an explicit, deliberate anti-hallucination principle, stated
outright in ``saas/physics_verification_engine/core/uncertainty_inverse.py``
(now living at ``pinneapple_analysis/verification/``, alongside this
module): :class:`~saas.physics_verification_engine.core.uncertainty_inverse.UncertaintyReport`
is documented as "a structured UQ report distinguishing uncertainty
*types*, not a single opaque confidence score", and
``quantify_uncertainty``'s own docstring says its numbers come from real
PINNeAPPle UQ classes "-- never a fabricated confidence number." A prior
capability audit flagged "no aggregate physics confidence score exists" as
a **deliberate** choice, not a gap, for exactly that reason: collapsing
several independent, differently-scoped signals into one opaque number
invites exactly the false confidence this repo's whole verification stack
(``PhysicsGuardrail``, Richardson/GCI convergence, calibration metrics,
benchmark comparison) exists to prevent.

This module honors that principle rather than overriding it. It is a
**transparent aggregator over real checks a caller already ran** -- never
a check-running black box, and never a source of new numbers of its own.
Concretely:

* :func:`compute_physics_confidence` never runs a check itself. It only
  accepts the real result *objects* that ``PhysicsGuardrail.check()``,
  ``richardson_extrapolate``/``mesh_independence_study``,
  ``CalibrationMetrics``'s static methods, and
  ``PhysicsCase.validate_against_benchmark()`` already produced, and turns
  each one that was actually supplied into one auditable
  :class:`ConfidenceComponent` via a plain, documented arithmetic formula
  (never a fitted/learned/opaque function -- see each component's
  docstring below for its exact derivation).
* A check that was not run is never faked, guessed, or defaulted to a
  neutral value -- it is simply absent from
  :attr:`PhysicsConfidenceScore.components`, exactly like
  ``GuardrailReport.checked_names``/``skipped`` never fakes a skipped
  check as a pass.
* :attr:`PhysicsConfidenceScore.coverage` (``components_used / 4``) is the
  load-bearing honesty mechanism of the whole module and MUST always be
  read alongside :attr:`PhysicsConfidenceScore.overall_score`. A score of
  1.0 built from ``coverage=0.25`` (one real check, e.g. only the
  benchmark comparison happened to be very close) is emphatically **not**
  the same claim as a score of 1.0 built from ``coverage=1.0`` (every one
  of the four checks this module knows how to aggregate actually ran and
  agreed). Callers and UIs displaying ``overall_score`` in isolation,
  without ``coverage`` right next to it, are misrepresenting what the
  number means -- this is precisely the failure mode ``UncertaintyReport``
  and ``PhysicsGuardrail`` were built to avoid, and this module is
  designed so that reproducing that failure requires actively discarding
  data this module hands you (``coverage``), not merely using the API as
  intended.
* If zero components are available, :attr:`PhysicsConfidenceScore.
  overall_score` is ``None`` -- never a fabricated number computed from
  nothing (mirrors ``GuardrailReport.trustworthy``'s refusal to silently
  treat "not evaluated" as "passed", and ``UncertaintyReport``'s
  ``aleatoric_std=None`` when that method genuinely cannot report one).

The four checks this module knows how to aggregate
---------------------------------------------------
1. ``physics_guardrail`` -- from a real
   ``pinneapple_llm.guardrail.GuardrailReport`` (``PhysicsGuardrail.check()``'s
   return value): re-evaluated PDE residual, dimensional/parameter
   sanity, conservation balance (when applicable), and reference-data
   match (when supplied).
2. ``numerical_convergence`` -- from a real
   ``pinneapple_analysis.verification.convergence.ConvergenceResult``
   (``richardson_extrapolate``/``mesh_independence_study``'s return
   value): Roache (1994) Grid Convergence Index / asymptotic-range check
   from three systematically-refined solves.
3. ``uq_calibration`` -- from a real, already-computed
   :class:`CalibrationSummary` wrapping
   ``pinneapple_analysis.uncertainty.calibration.CalibrationMetrics``'s
   static-method outputs (``expected_calibration_error``, and optionally
   ``coverage_at_level``/``sharpness``, which that class documents as a
   pure static-method namespace with no bundled result object of its
   own -- see :class:`CalibrationSummary`'s docstring for why this small
   container exists).
4. ``benchmark_agreement`` -- from a real
   ``pinneapple_data.physics_case.BenchmarkComparison``
   (``PhysicsCase.validate_against_benchmark()``'s return value): relative
   L2 error against a named, independently-verified reference dataset.

Every derivation is a plain, inspectable, documented formula (see each
component's docstring immediately below), never a fitted or learned
function, and every :class:`ConfidenceComponent` carries a
``source_summary`` quoting the real numbers that produced its score so
nobody has to trust the arithmetic without being able to check it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

__all__ = [
    "ConfidenceComponent",
    "CalibrationSummary",
    "PhysicsConfidenceScore",
    "compute_physics_confidence",
]

# The fixed set of checks this module knows how to aggregate. Used only to
# compute `coverage = len(components) / N_POSSIBLE_COMPONENTS` -- never to
# fabricate a placeholder entry for one that wasn't run.
N_POSSIBLE_COMPONENTS = 4


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _require_finite(value: float, what: str) -> float:
    """Refuse to silently turn a non-finite real-check output (NaN/inf --
    e.g. ``BenchmarkComparison.relative_l2_error`` is documented to be
    ``nan`` when the reference has zero norm) into a fabricated score.
    Raising here is the honest behaviour: a component score derived from
    ``1 - nan`` would itself be ``nan``, silently poisoning
    ``overall_score``'s arithmetic mean rather than surfacing the problem."""
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(
            f"compute_physics_confidence: {what}={v!r} is not finite -- cannot derive a "
            "real confidence component from it. This is a property of the input result "
            "object itself (e.g. a benchmark comparison whose reference had zero norm), "
            "not something this module can paper over without fabricating a number."
        )
    return v


# ---------------------------------------------------------------------------
# CalibrationSummary: a minimal container for CalibrationMetrics' outputs
# ---------------------------------------------------------------------------

@dataclass
class CalibrationSummary:
    """A small, honest container for the real numbers a caller already
    computed via ``pinneapple_analysis.uncertainty.calibration
    .CalibrationMetrics``'s static methods.

    ``CalibrationMetrics`` is documented as "a pure static-method namespace"
    (see its own docstring / ``tests/test_breadth_six_packages.py``'s
    ``test_breadth_calibration_metrics``) with no instantiation and no
    single bundled result object -- callers call
    ``CalibrationMetrics.expected_calibration_error(...)``,
    ``.coverage_at_level(...)``, and ``.sharpness(...)`` separately. This
    dataclass does not compute anything itself and does not call
    ``CalibrationMetrics`` -- it only carries the real numbers a caller
    already obtained from those real static methods, so
    :func:`compute_physics_confidence` has one typed object to accept
    exactly like it accepts a real ``GuardrailReport``/``ConvergenceResult``
    /``BenchmarkComparison``.

    Attributes
    ----------
    ece : float
        ``CalibrationMetrics.expected_calibration_error(...)``'s real
        output. Required -- this is the only number the
        ``uq_calibration`` component's score is derived from (see
        :func:`compute_physics_confidence`'s docstring for the exact
        formula).
    coverage : Optional[float]
        ``CalibrationMetrics.coverage_at_level(...)``'s real output, if the
        caller computed it. Not used in the score formula (ECE alone
        drives it, per this module's design) -- carried through only so
        ``source_summary`` can quote it for human inspection.
    target_coverage : Optional[float]
        The nominal coverage ``coverage`` was evaluated against (i.e.
        ``1 - alpha`` for whatever ``alpha`` was passed to
        ``coverage_at_level``), so ``coverage`` can be read in context.
    sharpness : Optional[float]
        ``CalibrationMetrics.sharpness(...)``'s real output, if the caller
        computed it. Also not used in the score formula (sharpness is a
        precision measure, not a calibration-correctness measure -- see
        that method's own docstring: "a model can be sharp but
        miscalibrated, or well-calibrated but not sharp") -- carried
        through for ``source_summary`` only.
    """

    ece: float
    coverage: Optional[float] = None
    target_coverage: Optional[float] = None
    sharpness: Optional[float] = None


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceComponent:
    """One real, already-executed check turned into a 0-1 score by a
    plain, documented arithmetic formula (never fitted/learned/opaque).

    Attributes
    ----------
    name : str
        One of ``"physics_guardrail"``, ``"numerical_convergence"``,
        ``"uq_calibration"``, ``"benchmark_agreement"``.
    score : float
        A real 0-1 value derived from that check's actual output -- see
        :func:`compute_physics_confidence`'s docstring for the exact
        formula used per component.
    source_summary : str
        A short, human-readable string quoting the real numbers that
        produced this component's score (e.g.
        ``"GCI_fine=0.003, asymptotic_ratio=1.02, is_asymptotic=True"``),
        so the score is never opaque -- anyone reading a
        :class:`PhysicsConfidenceScore` can see exactly which real
        measurement drove each component without re-running anything.
    """

    name: str
    score: float
    source_summary: str


@dataclass
class PhysicsConfidenceScore:
    """A transparent aggregate over whichever of the four real checks a
    caller actually ran. See this module's docstring for the full design
    rationale -- in short: ``overall_score`` is a plain arithmetic mean of
    ``components``' scores (never a hidden weighting), ``components``
    contains only the checks that were actually computed (never a
    placeholder for one that wasn't), and ``coverage`` MUST be read
    alongside ``overall_score`` -- a high score built from a low
    ``coverage`` is not the same claim as one built from ``coverage=1.0``.

    Attributes
    ----------
    overall_score : Optional[float]
        ``mean(c.score for c in components)`` when ``coverage > 0``;
        ``None`` if zero components were available (never fabricated from
        nothing).
    components : List[ConfidenceComponent]
        Only the components that were actually computed, in the fixed
        order ``physics_guardrail``, ``numerical_convergence``,
        ``uq_calibration``, ``benchmark_agreement`` (whichever subset was
        supplied), never a placeholder entry for a check that wasn't run.
    coverage : float
        ``len(components) / 4`` -- what fraction of the four checks this
        module knows how to aggregate actually contributed to
        ``overall_score``. This is the honesty mechanism of the whole
        module: it must be shown prominently alongside ``overall_score``,
        never as an afterthought, and never conflated with it (a
        ``coverage=0.25`` score and a ``coverage=1.0`` score of the same
        numeric value are not equivalent claims).
    """

    overall_score: Optional[float]
    components: List[ConfidenceComponent]
    coverage: float

    def summary(self) -> str:
        """A human-readable report mirroring
        ``GuardrailReport.summary()``'s style -- states the overall score
        AND coverage together (never one without the other), plus every
        component's real source numbers."""
        lines = ["PhysicsConfidenceScore report:"]
        if self.overall_score is None:
            lines.append("  overall_score: None (no real checks were supplied)")
        else:
            lines.append(f"  overall_score: {self.overall_score:.4g}  (coverage={self.coverage:.2f}, "
                          f"i.e. {len(self.components)}/{N_POSSIBLE_COMPONENTS} possible checks)")
        for c in self.components:
            lines.append(f"  [{c.name}] score={c.score:.4g} -- {c.source_summary}")
        if not self.components:
            lines.append("  (no components: guardrail_report, convergence_result, calibration_metrics, "
                          "and benchmark_comparison were all None)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-component derivations
# ---------------------------------------------------------------------------

def _component_from_guardrail(report) -> ConfidenceComponent:
    """``physics_guardrail`` component.

    Derivation: ``GuardrailReport`` exposes per-check granularity (a list
    of ``CheckResult``, each with its own ``passed`` bool) rather than
    only the single aggregate ``trustworthy`` bool -- per this module's
    design (use finer resolution when it's available, don't collapse to a
    single bool), the score is the FRACTION of checks that actually ran
    and passed::

        score = (# CheckResult with passed=True) / (# CheckResult that ran)

    This is strictly more informative than ``1.0 if trustworthy else 0.0``
    when several checks ran and only some failed (e.g. 2/3 checks
    passing scores 0.667, not 0.0), while still reducing to exactly that
    0.0/1.0 behaviour when every check passes or none do. A check that
    could not run at all (e.g. no reference data supplied, or this
    ``pde_kind`` has no known conservation law) is absent from
    ``report.checks`` entirely (see ``GuardrailReport.checked_names``/
    ``skipped``) and therefore never enters this fraction either way --
    exactly mirroring ``GuardrailReport.trustworthy``'s own "absent counts
    as neither pass nor fail" semantics.

    Edge case: if ``report.checks`` is empty (no check ran at all -- not
    expected from a normal ``PhysicsGuardrail.check()`` call, which always
    runs at least ``parameter_sanity``/``dimensional_analysis`` and
    ``pde_residual``), the fraction is vacuously defined as 1.0, matching
    Python's own ``all([]) == True`` that ``GuardrailReport.trustworthy``
    relies on for the same empty-list case; ``source_summary`` states this
    explicitly rather than leaving it implicit.
    """
    checks = list(report.checks)
    n_total = len(checks)
    if n_total == 0:
        return ConfidenceComponent(
            name="physics_guardrail",
            score=1.0,
            source_summary="0 checks ran (vacuously trustworthy, matching GuardrailReport.trustworthy's "
                            "all([])==True semantics for an empty check list)",
        )
    n_passed = sum(1 for c in checks if c.passed)
    score = n_passed / n_total
    per_check = ", ".join(f"{c.name}={'PASS' if c.passed else 'FAIL'}" for c in checks)
    return ConfidenceComponent(
        name="physics_guardrail",
        score=score,
        source_summary=f"{n_passed}/{n_total} guardrail checks passed ({per_check}); "
                        f"trustworthy={report.trustworthy}",
    )


def _component_from_convergence(result) -> ConfidenceComponent:
    """``numerical_convergence`` component.

    Derivation, per this module's design (``1.0 if is_asymptotic else a
    value derived from how far asymptotic_ratio is from 1``)::

        deviation = |asymptotic_ratio - 1.0|
        if is_asymptotic:                      # i.e. deviation <= asymptotic_tolerance
            score = 1.0
        else:
            score = clamp(1.0 - (deviation - asymptotic_tolerance) / asymptotic_tolerance, 0, 1)

    Rationale: ``is_asymptotic`` is itself defined (in
    ``richardson_extrapolate``) as ``deviation <= asymptotic_tolerance``,
    so scoring the pass region flatly at 1.0 exactly matches what the real
    check already certified. For the fail region, a "penalty band" equal
    in width to the pass band itself (``asymptotic_tolerance``) is the
    simplest possible extension that is (a) continuous at the boundary --
    a deviation of exactly ``asymptotic_tolerance`` scores 1.0 from both
    sides, no discontinuous jump -- and (b) reaches exactly 0.0 once the
    deviation is twice the tolerance, a plain, symmetric, easily-restated
    rule rather than a tuned decay curve. Deviations beyond that are
    clamped at 0.0, never a negative or fabricated number.

    Edge case: if ``asymptotic_tolerance <= 0`` (degenerate; the default
    is 0.10), the ratio-based formula above is undefined (division by
    zero) -- this falls back to the strict boundary itself: score is 1.0
    only if ``deviation == 0`` exactly, else 0.0, and ``source_summary``
    states this explicitly.
    """
    deviation = abs(result.asymptotic_ratio - 1.0)
    tol = result.asymptotic_tolerance
    if result.is_asymptotic:
        score = 1.0
    elif tol > 0.0:
        score = _clamp01(1.0 - (deviation - tol) / tol)
    else:
        score = 1.0 if deviation == 0.0 else 0.0
    summary = (
        f"observed_order={result.observed_order:.4g}, GCI_fine={result.gci_fine:.4g}, "
        f"GCI_coarse={result.gci_coarse:.4g}, asymptotic_ratio={result.asymptotic_ratio:.4g}, "
        f"asymptotic_tolerance={tol:.4g}, is_asymptotic={result.is_asymptotic}"
    )
    return ConfidenceComponent(name="numerical_convergence", score=score, source_summary=summary)


def _component_from_calibration(calibration: CalibrationSummary) -> ConfidenceComponent:
    """``uq_calibration`` component.

    Derivation: ``CalibrationMetrics.expected_calibration_error`` is
    documented to return "ECE in [0, 1]. A perfectly calibrated model
    returns 0." Since ECE is *already* naturally bounded to exactly the
    same [0, 1] range a confidence score needs, the score is the simplest
    possible linear complement, with no extra reference threshold to
    justify or tune::

        score = clamp(1.0 - ece, 0.0, 1.0)

    ECE=0 (perfect calibration) maps to score=1.0; ECE=1 (the theoretical
    maximum possible miscalibration under that metric's own definition)
    maps to score=0.0; linear in between. ``coverage_at_level`` and
    ``sharpness`` (if supplied on the ``CalibrationSummary``) are
    deliberately NOT part of this formula -- ``CalibrationMetrics.
    sharpness``'s own docstring warns "a model can be sharp but
    miscalibrated, or well-calibrated but not sharp", i.e. sharpness
    measures precision, not correctness, so folding it into a
    calibration-correctness score would conflate two different axes this
    repo's own docs are careful to keep separate; they are carried
    through into ``source_summary`` for context only.
    """
    ece = _require_finite(calibration.ece, "calibration.ece")
    score = _clamp01(1.0 - ece)
    parts = [f"ECE={ece:.4g}"]
    if calibration.coverage is not None:
        if calibration.target_coverage is not None:
            parts.append(f"coverage={calibration.coverage:.4g} (target={calibration.target_coverage:.4g})")
        else:
            parts.append(f"coverage={calibration.coverage:.4g}")
    if calibration.sharpness is not None:
        parts.append(f"sharpness={calibration.sharpness:.4g}")
    return ConfidenceComponent(name="uq_calibration", score=score, source_summary=", ".join(parts))


def _component_from_benchmark(comparison) -> ConfidenceComponent:
    """``benchmark_agreement`` component.

    Derivation: ``BenchmarkComparison.relative_l2_error`` is
    ``rmse(diff) / rmse(reference)`` -- 0.0 means an exact match, and 1.0
    means the prediction differs from the reference by as much as the
    reference's own magnitude (i.e. no more informative than a naive
    all-zero guess). Those are natural, self-defining 0/1 anchors, so
    (exactly as for ``uq_calibration`` above) the score is the simplest
    possible linear complement, clamped rather than extrapolated beyond
    those anchors::

        score = clamp(1.0 - relative_l2_error, 0.0, 1.0)

    A relative error beyond 1.0 (worse than the naive-zero baseline)
    clamps at score=0.0 rather than going negative.
    """
    rel_err = _require_finite(comparison.relative_l2_error, "benchmark_comparison.relative_l2_error")
    score = _clamp01(1.0 - rel_err)
    summary = (
        f"benchmark={comparison.benchmark}, relative_l2_error={rel_err:.4g}, "
        f"rmse={comparison.rmse:.4g}, n_compared_points={comparison.n_compared_points}"
    )
    return ConfidenceComponent(name="benchmark_agreement", score=score, source_summary=summary)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_physics_confidence(
    *,
    guardrail_report=None,
    convergence_result=None,
    calibration_metrics: Optional[CalibrationSummary] = None,
    benchmark_comparison=None,
) -> PhysicsConfidenceScore:
    """Aggregate whichever real, already-executed checks a caller supplies
    into one transparent :class:`PhysicsConfidenceScore`.

    This function never runs a check itself -- it only accepts real result
    objects a caller already obtained by actually running the
    corresponding checks, and never fabricates a component for one that
    wasn't supplied. See this module's docstring for the full
    anti-fabrication design rationale.

    Parameters
    ----------
    guardrail_report : Optional[pinneapple_llm.guardrail.GuardrailReport]
        A real report from ``PhysicsGuardrail.check(model, ...)``.
    convergence_result : Optional[pinneapple_analysis.verification.convergence.ConvergenceResult]
        A real result from ``richardson_extrapolate(...)`` or
        ``mesh_independence_study(...).convergence``.
    calibration_metrics : Optional[CalibrationSummary]
        A real bundle of numbers a caller already computed via
        ``pinneapple_analysis.uncertainty.calibration.CalibrationMetrics``'s
        static methods (``expected_calibration_error`` required;
        ``coverage_at_level``/``sharpness`` optional context). See
        :class:`CalibrationSummary`'s docstring for why this small
        container exists (``CalibrationMetrics`` itself has no bundled
        result object).
    benchmark_comparison : Optional[pinneapple_data.physics_case.BenchmarkComparison]
        A real result from ``PhysicsCase.validate_against_benchmark()``.

    Returns
    -------
    PhysicsConfidenceScore
        ``components`` contains exactly one :class:`ConfidenceComponent`
        per non-``None`` argument (in the fixed order
        ``physics_guardrail``, ``numerical_convergence``,
        ``uq_calibration``, ``benchmark_agreement``), ``coverage`` is
        ``len(components) / 4``, and ``overall_score`` is the plain
        arithmetic mean of ``components``' scores (documented, unweighted
        -- see :class:`PhysicsConfidenceScore`'s docstring) when
        ``coverage > 0``, else ``None``.

    This function never raises for "nothing was supplied": if all four
    arguments are ``None``, it returns
    ``PhysicsConfidenceScore(overall_score=None, components=[], coverage=0.0)``
    rather than fabricating a result or raising.
    """
    components: List[ConfidenceComponent] = []

    if guardrail_report is not None:
        components.append(_component_from_guardrail(guardrail_report))
    if convergence_result is not None:
        components.append(_component_from_convergence(convergence_result))
    if calibration_metrics is not None:
        components.append(_component_from_calibration(calibration_metrics))
    if benchmark_comparison is not None:
        components.append(_component_from_benchmark(benchmark_comparison))

    coverage = len(components) / N_POSSIBLE_COMPONENTS
    overall_score = (
        sum(c.score for c in components) / len(components) if components else None
    )
    return PhysicsConfidenceScore(overall_score=overall_score, components=components, coverage=coverage)
