"""Multi-restart PINN training with NaN watchdog and best-model selection.

Many PINNs are sensitive to initialisation: a bad random seed leads to divergence
or a poor local minimum.  ``MultiRestartTrainer`` runs the training function N
times with different seeds, keeps a NaN watchdog, and returns the best result
according to a user-supplied (or default) score function.

The pattern is taken directly from industrial PINN workflows (e.g. Rate-PINN for
blast-furnace wear modelling) and is model-agnostic: you supply a factory and a
training function; the trainer handles seeds, watchdog, and selection.

Usage
-----
::

    from pinneaple_neural.trainer.multistart import MultiRestartTrainer, MultiRestartConfig

    def make_model(seed):
        torch.manual_seed(seed)
        return MyPINN(...)

    def train_fn(model, seed):
        # Returns (history_dict, final_loss) or (history_dict,)
        ...
        return history, final_loss

    mrt = MultiRestartTrainer(make_model, train_fn,
                              config=MultiRestartConfig(n_restarts=10))
    best = mrt.fit()
    best_model = best.model
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config & result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MultiRestartConfig:
    """Configuration for ``MultiRestartTrainer``.

    Attributes
    ----------
    n_restarts        : number of independent restarts to attempt
    seed_base         : random seed for restart 0; restart k uses seed_base + k*100
    nan_patience      : abort a restart immediately when NaN is detected;
                        after this many consecutive NaN restarts, stop entirely
    max_failed        : raise RuntimeError if this many consecutive restarts fail
    verbose           : print per-restart summary lines
    keep_all          : if True, store all valid RestartResult objects in the
                        ``all_results`` list on the returned ``MultiRestartResult``
    """
    n_restarts: int = 5
    seed_base: int = 42
    nan_patience: int = 3
    max_failed: int = 10
    verbose: bool = True
    keep_all: bool = False


@dataclass
class RestartResult:
    """Outcome of a single training restart.

    Attributes
    ----------
    model       : best model state (deep-copied from training)
    history     : dict of metric lists (whatever train_fn returns)
    score       : lower-is-better scalar used for selection
    restart_idx : zero-based restart number
    seed        : random seed used
    nan_detected: True if training was aborted due to NaN/Inf
    """
    model: nn.Module
    history: Dict[str, List[float]]
    score: float
    restart_idx: int
    seed: int
    nan_detected: bool = False


@dataclass
class MultiRestartResult:
    """Aggregated output of ``MultiRestartTrainer.fit()``.

    Attributes
    ----------
    best        : the ``RestartResult`` with the lowest score
    all_results : populated when ``config.keep_all=True``
    n_valid     : number of restarts that completed without NaN
    n_nan       : number of restarts aborted by NaN watchdog
    """
    best: RestartResult
    all_results: List[RestartResult] = field(default_factory=list)
    n_valid: int = 0
    n_nan: int = 0

    @property
    def model(self) -> nn.Module:
        return self.best.model

    @property
    def history(self) -> Dict[str, List[float]]:
        return self.best.history

    @property
    def score(self) -> float:
        return self.best.score


# ─────────────────────────────────────────────────────────────────────────────
# NaN watchdog helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_finite(value: Any) -> bool:
    """Return True if value is finite (handles tensors, floats, dicts, lists)."""
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, float):
        return not (np.isnan(value) or np.isinf(value))
    if isinstance(value, dict):
        return all(_check_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_check_finite(v) for v in value)
    return True  # unknown type — pass through


def _default_score(
    history: Dict[str, List[float]],
    model: nn.Module,
) -> float:
    """Default score: last value of ``loss`` key, or mean of all last values."""
    if "loss" in history and history["loss"]:
        return float(history["loss"][-1])
    # Try any key ending with "loss"
    for k, v in history.items():
        if "loss" in k.lower() and v:
            return float(v[-1])
    # Fallback: mean of all last scalar values
    vals = [v[-1] for v in history.values() if isinstance(v, list) and v]
    return float(np.mean(vals)) if vals else float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# MultiRestartTrainer
# ─────────────────────────────────────────────────────────────────────────────

class MultiRestartTrainer:
    """Runs a training function N times with different seeds, keeps the best.

    Parameters
    ----------
    model_factory : ``Callable[[int], nn.Module]``
        Called as ``model_factory(seed)`` for each restart.  Must return a
        freshly-initialised ``nn.Module``.

    train_fn : ``Callable[[nn.Module, int], Tuple[Dict, float] | Tuple[Dict] | Dict]``
        Called as ``train_fn(model, seed)``.  Must return either:
        - ``(history_dict, score_float)``  — history + explicit score
        - ``(history_dict,)`` or ``history_dict`` — score inferred via ``score_fn``

        ``history_dict`` values should be lists of scalars (one per epoch/step).
        If the function raises an exception or returns NaN values, the restart
        is marked as failed.

    config : MultiRestartConfig

    score_fn : ``Callable[[Dict, nn.Module], float]`` | None
        Used when ``train_fn`` does not return an explicit score.
        Signature: ``score_fn(history, model) -> float`` (lower is better).
        Default: last value of ``history["loss"]``.

    on_restart_end : optional callback invoked after each restart with
        ``(restart_idx, result_or_None, is_best_so_far)``.
    """

    def __init__(
        self,
        model_factory: Callable[[int], nn.Module],
        train_fn: Callable,
        *,
        config: Optional[MultiRestartConfig] = None,
        score_fn: Optional[Callable] = None,
        on_restart_end: Optional[Callable] = None,
    ) -> None:
        self.model_factory = model_factory
        self.train_fn = train_fn
        self.cfg = config or MultiRestartConfig()
        self.score_fn = score_fn or _default_score
        self.on_restart_end = on_restart_end

    # ------------------------------------------------------------------

    def fit(self, *args, **kwargs) -> MultiRestartResult:
        """Run all restarts and return the aggregated result.

        Any extra ``*args`` / ``**kwargs`` are forwarded to ``train_fn`` after
        ``(model, seed)``.
        """
        cfg = self.cfg
        best: Optional[RestartResult] = None
        all_results: List[RestartResult] = []
        n_valid = 0
        n_nan = 0
        consecutive_failures = 0

        for restart in range(cfg.n_restarts):
            seed = cfg.seed_base + restart * 100
            torch.manual_seed(seed)
            np.random.seed(seed)

            if cfg.verbose:
                logger.info(f"[MultiRestartTrainer] restart {restart + 1}/{cfg.n_restarts}  seed={seed}")

            model = self.model_factory(seed)
            result = self._run_one(model, seed, restart, args, kwargs)

            if result.nan_detected:
                n_nan += 1
                consecutive_failures += 1
                if cfg.verbose:
                    logger.warning(
                        f"  restart {restart + 1}: NaN/Inf detected — skipped."
                    )
                if consecutive_failures >= cfg.nan_patience:
                    logger.warning(
                        f"  {consecutive_failures} consecutive NaN restarts — stopping early."
                    )
                    break
                self._notify(restart, None, False)
                continue

            consecutive_failures = 0
            n_valid += 1
            is_best = best is None or result.score < best.score
            if is_best:
                best = result

            if cfg.keep_all:
                all_results.append(result)

            if cfg.verbose:
                logger.info(
                    f"  restart {restart + 1}: score={result.score:.6f}"
                    + ("  ← best" if is_best else "")
                )

            self._notify(restart, result, is_best)

        if best is None:
            raise RuntimeError(
                f"[MultiRestartTrainer] All {cfg.n_restarts} restarts failed (NaN/error). "
                "Check your model_factory and train_fn."
            )

        if cfg.verbose:
            logger.info(
                f"[MultiRestartTrainer] best restart #{best.restart_idx + 1}  "
                f"score={best.score:.6f}  ({n_valid}/{cfg.n_restarts} valid)"
            )

        return MultiRestartResult(
            best=best,
            all_results=all_results,
            n_valid=n_valid,
            n_nan=n_nan,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_one(
        self,
        model: nn.Module,
        seed: int,
        restart_idx: int,
        extra_args: tuple,
        extra_kwargs: dict,
    ) -> RestartResult:
        nan_detected = False
        history: Dict[str, List[float]] = {}
        score = float("inf")

        try:
            out = self.train_fn(model, seed, *extra_args, **extra_kwargs)
        except Exception as exc:
            logger.warning(f"  restart {restart_idx + 1}: train_fn raised {exc!r}")
            return RestartResult(
                model=model, history={}, score=float("inf"),
                restart_idx=restart_idx, seed=seed, nan_detected=True,
            )

        # Parse output
        if isinstance(out, tuple):
            if len(out) >= 2:
                history, score_raw = out[0], out[1]
                if _check_finite(score_raw):
                    score = float(score_raw)
                else:
                    nan_detected = True
            elif len(out) == 1:
                history = out[0]
            else:
                history = {}
        elif isinstance(out, dict):
            history = out
        else:
            history = {}

        # NaN watchdog on history
        if not nan_detected:
            for k, v in (history or {}).items():
                if isinstance(v, list) and v and not _check_finite(v[-1]):
                    nan_detected = True
                    break

        if nan_detected:
            return RestartResult(
                model=model, history=history, score=float("inf"),
                restart_idx=restart_idx, seed=seed, nan_detected=True,
            )

        # Infer score if not provided explicitly
        if score == float("inf"):
            try:
                score = float(self.score_fn(history, model))
            except Exception:
                score = float("inf")

        if not _check_finite(score):
            nan_detected = True
            score = float("inf")

        return RestartResult(
            model=copy.deepcopy(model),
            history=history,
            score=score,
            restart_idx=restart_idx,
            seed=seed,
            nan_detected=nan_detected,
        )

    def _notify(self, restart_idx: int, result: Optional[RestartResult], is_best: bool) -> None:
        if self.on_restart_end is not None:
            try:
                self.on_restart_end(restart_idx, result, is_best)
            except Exception:
                pass
