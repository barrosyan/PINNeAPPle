"""pinneapple_tools.surrogate_policy — measured-runtime "does this problem
need a trained surrogate, or can we just call its solver directly?"
policy.

Deliberately measurement-driven, not a dimension-based guess: a 3D solve
is not always slower than a 2D one (a closed-form/analytic 3D solver can
run in microseconds while an iterative 2D CFD solve takes seconds), so a
"3D = always needs a surrogate" heuristic is wrong often enough to be
worth avoiding entirely. The caller supplies a zero-argument thunk that
runs one real solve; this module only handles timing, a small JSON cache
so the same problem isn't re-timed every call, and thresholding.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class SurrogatePolicyResult:
    problem_id: str
    needs_model: bool
    reason: str
    measured_s: Optional[float] = None


class SurrogatePolicy:
    def __init__(self, cache_path: str, *, slow_threshold_s: float = 1.0):
        self.cache_path = cache_path
        self.slow_threshold_s = slow_threshold_s
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".", exist_ok=True)
        self._cache: Dict[str, float] = {}
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                self._cache = json.load(f)

    def _save_cache(self) -> None:
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f, indent=2)

    def measure_solver_time(self, problem_id: str, run_once: Callable[[], Any], *, force: bool = False) -> float:
        """Runs ``run_once()`` once and caches the elapsed wall-clock
        seconds under ``problem_id``. Returns the cached value if already
        measured, unless ``force=True``."""
        if not force and problem_id in self._cache:
            return self._cache[problem_id]
        t0 = time.perf_counter()
        run_once()
        elapsed = time.perf_counter() - t0
        self._cache[problem_id] = elapsed
        self._save_cache()
        return elapsed

    def classify(
        self,
        problem_id: str,
        *,
        run_once: Optional[Callable[[], Any]] = None,
        has_direct_solver: bool = True,
        assumed_expensive_geometry: bool = False,
    ) -> SurrogatePolicyResult:
        """Priority order, matching the reasons a sibling internal
        project's policy used: (1) no direct solver exists at all -> needs
        a model regardless of timing; (2) the geometry step alone is
        assumed expensive (e.g. real CAD meshing) -> needs a model without
        timing the full solve; (3) otherwise, measure (or reuse a cached
        measurement of) the actual solve and threshold on that."""
        if not has_direct_solver:
            return SurrogatePolicyResult(problem_id, True, "no_direct_solver")
        if assumed_expensive_geometry:
            return SurrogatePolicyResult(problem_id, True, "assumed_expensive_geometry")

        measured = self._cache.get(problem_id)
        if measured is None:
            if run_once is None:
                return SurrogatePolicyResult(problem_id, True, "unmeasured_assume_slow")
            measured = self.measure_solver_time(problem_id, run_once)

        if measured >= self.slow_threshold_s:
            return SurrogatePolicyResult(problem_id, True, "measured_slow", measured_s=measured)
        return SurrogatePolicyResult(problem_id, False, "measured_fast", measured_s=measured)

    def recommend_architecture(self, result: SurrogatePolicyResult, *, n_observations: int = 0) -> Optional[str]:
        """A small, honest heuristic — not a learned recommender. Returns
        ``None`` if no model is needed at all."""
        if not result.needs_model:
            return None
        return "vanilla_pinn" if n_observations < 50 else "modified_mlp"
