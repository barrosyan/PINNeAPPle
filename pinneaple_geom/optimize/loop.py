"""Geometry optimization loop helpers.

ParamSpace, OptState: re-exported from pinneaple_design_opt.optimizer (canonical location).
GeometryOptimizer: UI-friendly loop wrapper that delegates CMA-ES/GA to
EvolutionaryDesignOptimizer — no duplicate optimizer logic here.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

# Re-export canonical types from pinneaple_design_opt
from pinneaple_design_opt.optimizer import (  # noqa: F401
    ParamSpace,
    OptState,
    EvolutionaryDesignOptimizer,
    DesignOptimizerConfig,
)

EvalFn = Callable[[Dict[str, float]], float]


class GeometryOptimizer:
    """UI-friendly geometry optimizer backed by EvolutionaryDesignOptimizer.

    Converts between the dict-of-params interface used by UI/CLI callers and
    the numpy-array interface used by EvolutionaryDesignOptimizer internally.
    No CMA-ES or GA logic lives here — all of that is in EvolutionaryDesignOptimizer.

    Parameters
    ----------
    space:
        ParamSpace defining bounds and initial point.
    seed:
        Random seed.
    sigma0:
        Initial CMA-ES step size (fraction of parameter range).
    """

    def __init__(
        self,
        space: ParamSpace,
        *,
        seed: int = 0,
        sigma0: float = 0.2,
    ):
        self.space = space
        self._keys = list(space.bounds.keys())
        self._bounds_arr = np.array(
            [space.bounds[k] for k in self._keys], dtype=np.float64
        )
        cfg = DesignOptimizerConfig(method="evolutionary", sigma0=sigma0)
        self._opt = EvolutionaryDesignOptimizer(cfg, seed=seed)

    # ------------------------------------------------------------------
    # Dict ↔ array conversion helpers
    # ------------------------------------------------------------------

    def _to_dict(self, arr: np.ndarray) -> Dict[str, float]:
        return self.space.clip({k: float(v) for k, v in zip(self._keys, arr)})

    def _to_arr(self, d: Dict[str, float]) -> np.ndarray:
        return np.array([d[k] for k in self._keys], dtype=np.float64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, n: int = 1) -> List[Dict[str, float]]:
        """Return *n* candidate parameter dicts."""
        candidates_arr = self._opt.ask(self._bounds_arr, n)
        return [self._to_dict(a) for a in candidates_arr]

    def tell(self, xs: List[Dict[str, float]], ys: List[float]) -> None:
        """Inform the optimizer of objective values for the last candidates."""
        xs_arr = [self._to_arr(x) for x in xs]
        self._opt.tell(xs_arr, ys)

    def run(
        self,
        evaluate: EvalFn,
        *,
        iters: int = 30,
        batch: int = 4,
        on_step: Optional[Callable[[OptState], Any]] = None,
    ) -> OptState:
        """Run the full optimization loop.

        Parameters
        ----------
        evaluate:
            Callable ``(params: dict) -> float`` (lower is better).
        iters:
            Number of iterations.
        batch:
            Candidates evaluated per iteration.
        on_step:
            Optional callback ``(OptState) -> Any`` called after each iteration.
        """
        best_x = dict(self.space.x0)
        best_y = float("inf")
        last_x = dict(self.space.x0)
        last_y = float("inf")

        for t in range(int(iters)):
            cand = self.ask(n=int(batch))
            ys = [float(evaluate(x)) for x in cand]
            self.tell(cand, ys)

            j = int(np.argmin(ys))
            last_x, last_y = cand[j], float(ys[j])
            if last_y < best_y:
                best_x, best_y = last_x, last_y

            st = OptState(step=t, best_x=best_x, best_y=best_y, last_x=last_x, last_y=last_y)
            if on_step is not None:
                on_step(st)

        return OptState(step=int(iters), best_x=best_x, best_y=best_y, last_x=last_x, last_y=last_y)


__all__ = ["ParamSpace", "OptState", "GeometryOptimizer"]
