from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from pinneapple_analysis.inverse_problems.counterfactual_attribution import (
    CandidateCause,
    ExactShapleyAttributor,
    classify_ambiguity,
)


@dataclass(frozen=True)
class ToyState:
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0


def _value_fn(state: ToyState) -> float:
    # Deliberately includes an interaction term (a*b) so exact-closure
    # actually exercises something a naive one-factor-at-a-time sweep
    # would get wrong.
    return 2.0 * state.a + 3.0 * state.b + state.a * state.b + state.c


def _causes():
    return [
        CandidateCause("a", apply_baseline=lambda s: replace(s, a=0.0), apply_current=lambda s: replace(s, a=1.0)),
        CandidateCause("b", apply_baseline=lambda s: replace(s, b=0.0), apply_current=lambda s: replace(s, b=1.0)),
        CandidateCause("c", apply_baseline=lambda s: replace(s, c=0.0), apply_current=lambda s: replace(s, c=1.0)),
    ]


def test_shapley_closes_exactly_including_interaction_term():
    attributor = ExactShapleyAttributor(_value_fn, ToyState())
    result = attributor.attribute(_causes())
    assert abs(result.residual) < 1e-9
    assert result.measured_delta == pytest.approx(sum(result.contributions.values()), abs=1e-9)


def test_shapley_measured_delta_matches_direct_evaluation():
    attributor = ExactShapleyAttributor(_value_fn, ToyState())
    result = attributor.attribute(_causes())
    assert result.measured_delta == pytest.approx(_value_fn(ToyState(1.0, 1.0, 1.0)) - _value_fn(ToyState()))


def test_shapley_single_cause_gets_the_whole_delta():
    attributor = ExactShapleyAttributor(_value_fn, ToyState())
    only_c = [CandidateCause("c", apply_baseline=lambda s: replace(s, c=0.0), apply_current=lambda s: replace(s, c=1.0))]
    result = attributor.attribute(only_c)
    assert result.contributions["c"] == pytest.approx(result.measured_delta, rel=1e-9)


def test_shapley_symmetric_causes_get_equal_contribution():
    # a and b enter v() identically except for their own linear coefficient
    # being swapped in a symmetric variant -- use a genuinely symmetric
    # value function here to check the symmetry axiom holds numerically.
    def symmetric_v(state: ToyState) -> float:
        return state.a + state.b + 0.5 * state.a * state.b

    attributor = ExactShapleyAttributor(symmetric_v, ToyState())
    causes = [
        CandidateCause("a", apply_baseline=lambda s: replace(s, a=0.0), apply_current=lambda s: replace(s, a=1.0)),
        CandidateCause("b", apply_baseline=lambda s: replace(s, b=0.0), apply_current=lambda s: replace(s, b=1.0)),
    ]
    result = attributor.attribute(causes)
    assert result.contributions["a"] == pytest.approx(result.contributions["b"], abs=1e-9)


def test_classify_ambiguity_flags_overlapping_causes_missing_measurement():
    contributions = {"cause_a": 500.0, "cause_b": 480.0}
    uncertainty = {"cause_a": 100.0, "cause_b": 100.0}
    signatures = {"cause_a": ("signal_x",), "cause_b": ("signal_y",)}
    result = classify_ambiguity(contributions, uncertainty, signatures, available_measurements=set())
    assert result.indeterminate is True
    assert result.recommended_measurement in ("signal_x", "signal_y")


def test_classify_ambiguity_resolves_with_measurement_available():
    contributions = {"cause_a": 500.0, "cause_b": 480.0}
    uncertainty = {"cause_a": 100.0, "cause_b": 100.0}
    signatures = {"cause_a": ("signal_x",), "cause_b": ("signal_y",)}
    result = classify_ambiguity(contributions, uncertainty, signatures, available_measurements={"signal_x", "signal_y"})
    assert result.indeterminate is False


def test_classify_ambiguity_not_indeterminate_when_well_separated():
    contributions = {"cause_a": 900.0, "cause_b": 100.0}
    uncertainty = {"cause_a": 50.0, "cause_b": 50.0}
    signatures = {"cause_a": ("signal_x",), "cause_b": ("signal_y",)}
    result = classify_ambiguity(contributions, uncertainty, signatures, available_measurements=set())
    assert result.indeterminate is False
