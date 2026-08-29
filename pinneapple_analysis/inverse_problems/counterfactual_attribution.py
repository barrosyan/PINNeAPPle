"""pinneapple_analysis.inverse_problems.counterfactual_attribution --
exact Shapley-value attribution of a measured change in a scalar outcome
across a small set of candidate causes, evaluated through counterfactual
runs of ANY caller-supplied model (a physics simulator, a trained
surrogate, or any other value function) -- plus an ambiguity classifier
for when two causes cannot be told apart from the available evidence.

SELECTED FORMULATION: exact Shapley-value attribution (Shapley, "A Value
for n-Person Games", in Contributions to the Theory of Games II, Annals
of Mathematics Studies 28, Princeton University Press, 1953; the same
cooperative-game-theory construction underlying SHAP -- Lundberg & Lee,
"A Unified Approach to Interpreting Model Predictions", NeurIPS 2017).
Given n candidate causes, each togglable between a BASELINE and a
CURRENT value, and a value function v(S) = the model's scalar output
with every cause in subset S set to its current value and every other
cause at baseline:

    phi_i = sum over subsets S of (N minus {i}):
                [|S|! (n-|S|-1)! / n!] * (v(S union {i}) - v(S))

Shapley's own axioms (efficiency, symmetry, additivity, null player)
make this the UNIQUE allocation with the EFFICIENCY property: the
attributed contributions sum EXACTLY to v(full set) - v(empty set),
including every pairwise/higher-order interaction between causes. A
naive one-factor-at-a-time sensitivity sweep does not have this exact-
closure property whenever causes interact.

COST: exact computation requires 2^n evaluations of v() for n candidate
causes (e.g. 32 for n=5) -- each evaluation is whatever the caller's
value function costs (a full physics solve, a forward pass, ...). This
is appropriate for an on-demand diagnostic, not a real-time control loop;
`ExactShapleyAttributor` caches by subset so repeated calls (e.g. after
changing which causes are "current") don't re-evaluate v() needlessly.

AMBIGUITY / IDENTIFIABILITY: `classify_ambiguity` addresses a common
follow-on question -- can the data actually tell two overlapping causes
apart? Two causes are treated as INSEPARABLE if (a) their attributed
contributions overlap within their stated uncertainty bands, AND (b) the
available measurement set includes no channel unique to either cause's
own signature. When both hold, the result is explicitly marked
indeterminate together with the specific measurement that would resolve
it, rather than silently picking a winner.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

State = TypeVar("State")


@dataclass(frozen=True)
class CandidateCause(Generic[State]):
    name: str
    apply_baseline: Callable[[State], State]
    apply_current: Callable[[State], State]
    signature_measurements: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShapleyAttributionResult:
    measured_delta: float
    contributions: dict[str, float]
    residual: float  # should be ~0 (float precision) -- the efficiency-property closure check
    baseline_value: float
    current_value: float


class ExactShapleyAttributor(Generic[State]):
    """Wraps a value function v: State -> float and a baseline state;
    `attribute(causes)` returns the exact Shapley decomposition of
    v(all-current) - v(all-baseline) across the given causes."""

    def __init__(self, value_fn: Callable[[State], float], baseline_state: State):
        self.value_fn = value_fn
        self.baseline_state = baseline_state

    def attribute(self, causes: list[CandidateCause[State]]) -> ShapleyAttributionResult:
        n = len(causes)
        cache: dict[frozenset, float] = {}

        def v(subset: frozenset) -> float:
            if subset in cache:
                return cache[subset]
            state = self.baseline_state
            for i, cause in enumerate(causes):
                state = cause.apply_current(state) if i in subset else cause.apply_baseline(state)
            cache[subset] = self.value_fn(state)
            return cache[subset]

        v_empty, v_full = v(frozenset()), v(frozenset(range(n)))

        contributions = {}
        all_indices = list(range(n))
        for i in range(n):
            others = [j for j in all_indices if j != i]
            phi_i = 0.0
            for k in range(len(others) + 1):
                for subset in itertools.combinations(others, k):
                    S = frozenset(subset)
                    weight = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                    phi_i += weight * (v(S | {i}) - v(S))
            contributions[causes[i].name] = phi_i

        measured_delta = v_full - v_empty
        residual = measured_delta - sum(contributions.values())
        return ShapleyAttributionResult(measured_delta, contributions, residual, v_empty, v_full)


@dataclass(frozen=True)
class AmbiguityResult:
    indeterminate: bool
    reason: str
    recommended_measurement: str | None


def classify_ambiguity(
    contributions: dict[str, float],
    contribution_uncertainty: dict[str, float],
    signature_measurements: dict[str, tuple[str, ...]],
    available_measurements: set[str],
) -> AmbiguityResult:
    """See module docstring's AMBIGUITY / IDENTIFIABILITY section."""
    ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    if len(ranked) < 2:
        return AmbiguityResult(False, "fewer than two candidate causes", None)

    (name_a, val_a), (name_b, val_b) = ranked[0], ranked[1]
    band_a = contribution_uncertainty.get(name_a, 0.0)
    band_b = contribution_uncertainty.get(name_b, 0.0)
    if abs(val_a - val_b) >= (band_a + band_b):
        return AmbiguityResult(False, f"{name_a} and {name_b} separated beyond their combined uncertainty band", None)

    sig_a = set(signature_measurements.get(name_a, ()))
    sig_b = set(signature_measurements.get(name_b, ()))
    unresolved_a = (sig_a - sig_b) - available_measurements
    unresolved_b = (sig_b - sig_a) - available_measurements

    if not unresolved_a and not unresolved_b:
        return AmbiguityResult(False, f"{name_a} vs {name_b} overlap in magnitude but are separable via available measurements", None)

    missing = next(iter(unresolved_a), None) or next(iter(unresolved_b), None)
    return AmbiguityResult(
        True,
        f"top two candidate causes ({name_a}, {name_b}) overlap within their combined uncertainty "
        f"({abs(val_a - val_b):.4g} vs {band_a + band_b:.4g} combined band) and the measurement "
        f"that would distinguish them is not available",
        missing,
    )
