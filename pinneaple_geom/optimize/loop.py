
"""Geometry optimization loop helpers.

This is a lightweight, UI-friendly optimization scaffold:

  - user defines `param_space` (bounds + initial params)
  - user defines `evaluate(params)` that returns a scalar score (lower is better)
  - the loop proposes candidates (random / CMA-ES if available)
  - the UI can visualize geometry at each iteration and let the user intervene

The goal is NOT to enforce a single optimizer, but to provide a common API that
both the CLI examples and the web app can reuse.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any

import numpy as np


@dataclass
class ParamSpace:
    """Box-bounded parameter space."""
    bounds: Dict[str, Tuple[float, float]]
    x0: Dict[str, float]

    def clip(self, x: Dict[str, float]) -> Dict[str, float]:
        out = {}
        for k, v in x.items():
            lo, hi = self.bounds[k]
            out[k] = float(np.clip(v, lo, hi))
        return out


EvalFn = Callable[[Dict[str, float]], float]


@dataclass
class OptState:
    step: int
    best_x: Dict[str, float]
    best_y: float
    last_x: Dict[str, float]
    last_y: float


class GeometryOptimizer:
    """Simple optimizer with optional CMA-ES.

    If `cma` package is installed, uses CMA-ES, otherwise random-search + local jitter.
    """

    def __init__(
        self,
        space: ParamSpace,
        *,
        seed: int = 0,
        sigma0: float = 0.2,
    ):
        self.space = space
        self.rng = np.random.default_rng(int(seed))
        self.sigma0 = float(sigma0)

        self._use_cma = False
        self._cma = None
        try:
            import cma  # type: ignore
            self._use_cma = True
            self._cma = cma
        except Exception:
            self._use_cma = False

        self._cma_es = None
        if self._use_cma:
            x0 = np.array([space.x0[k] for k in space.bounds.keys()], dtype=np.float64)
            self._cma_es = self._cma.CMAEvolutionStrategy(x0, self.sigma0, {"seed": int(seed)})

    def ask(self, n: int = 1) -> list[Dict[str, float]]:
        keys = list(self.space.bounds.keys())
        if self._use_cma and self._cma_es is not None:
            xs = self._cma_es.ask(number=n)
            out = []
            for xi in xs:
                d = {k: float(v) for k, v in zip(keys, xi)}
                out.append(self.space.clip(d))
            return out

        # fallback: random around x0 with decaying sigma
        out = []
        for _ in range(n):
            d = {}
            for k, (lo, hi) in self.space.bounds.items():
                mu = self.space.x0[k]
                sig = self.sigma0 * (hi - lo)
                v = float(mu + self.rng.normal(0.0, sig))
                d[k] = float(np.clip(v, lo, hi))
            out.append(d)
        return out

    def tell(self, xs: list[Dict[str, float]], ys: list[float]) -> None:
        if self._use_cma and self._cma_es is not None:
            keys = list(self.space.bounds.keys())
            X = [np.array([x[k] for k in keys], dtype=np.float64) for x in xs]
            self._cma_es.tell(X, ys)
            # update x0 to current best for better UX in fallback mode
            best = self._cma_es.best.x
            self.space.x0 = {k: float(v) for k, v in zip(keys, best)}

    def run(
        self,
        evaluate: EvalFn,
        *,
        iters: int = 30,
        batch: int = 4,
        on_step: Optional[Callable[[OptState], Any]] = None,
    ) -> OptState:
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
