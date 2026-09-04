"""Automatic (adaptive) hyperparameter search.

``pinneapple_neural.trainer.parallel.run_parallel_sweep``/``SweepConfig``
already exist and work, but they are a **grid** sweep: the caller supplies
every combination to try up front, evaluates all of them in parallel, and
there is no mechanism that uses early trial results to decide what to try
next, or to stop a clearly-bad trial early. That is not "automatic" in the
sense normally meant by hyperparameter tuning -- it is parallel brute
force over a manually-specified grid.

``AdaptiveSweep`` is the actual adaptive search: a TPE (TreeParzenEstimator)
Bayesian sampler over a continuous/categorical parameter *space*
(``suggest_float``/``suggest_int``/``suggest_categorical`` ranges, not a
fixed grid) with a median pruner that can stop an unpromising trial after
only a few of its intermediate reports -- both via Optuna (optional
dependency, ``pip install "pinneapple[tuning]"``). A dependency-free
fallback (random search, no pruning) is used automatically if Optuna is
not installed, so the feature isn't entirely gated behind an extra.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

ParamSpec = Tuple[str, tuple]  # ("float"|"int"|"categorical", args)


@dataclass
class AdaptiveSweepConfig:
    param_space: Dict[str, ParamSpec]
    n_trials: int = 30
    direction: str = "minimize"  # or "maximize"
    seed: Optional[int] = None
    pruning: bool = True  # only has an effect with the Optuna backend


def _suggest_optuna(trial, name: str, spec: ParamSpec):
    """``spec`` is ``("float"|"int", (lo, hi))`` or ``("float"|"int",
    (lo, hi, "log"))`` for a log-uniform range, or ``("categorical",
    (choices,))``. The optional third element is a plain marker string,
    not a positional argument to Optuna's own ``suggest_float``/
    ``suggest_int`` (whose ``log`` parameter is keyword-only) -- passing
    it straight through via ``*args`` raises a ``TypeError`` (too many
    positional arguments), which is what an earlier version of this
    function did.
    """
    kind, args = spec
    if kind == "float":
        lo, hi = args[0], args[1]
        log = len(args) > 2 and args[2] == "log"
        return trial.suggest_float(name, lo, hi, log=log)
    if kind == "int":
        lo, hi = args[0], args[1]
        log = len(args) > 2 and args[2] == "log"
        return trial.suggest_int(name, lo, hi, log=log)
    if kind == "categorical":
        return trial.suggest_categorical(name, args[0] if len(args) == 1 else list(args))
    raise ValueError(f"unknown param kind '{kind}' for '{name}' (expected 'float'/'int'/'categorical')")


def _suggest_random(rng: random.Random, name: str, spec: ParamSpec):
    kind, args = spec
    if kind == "float":
        lo, hi = args[0], args[1]
        log = len(args) > 2 and args[2] == "log"
        if log:
            import math
            return math.exp(rng.uniform(math.log(lo), math.log(hi)))
        return rng.uniform(lo, hi)
    if kind == "int":
        return rng.randint(int(args[0]), int(args[1]))
    if kind == "categorical":
        choices = args[0] if len(args) == 1 else list(args)
        return rng.choice(list(choices))
    raise ValueError(f"unknown param kind '{kind}' for '{name}'")


def run_adaptive_sweep(
    trial_fn: Callable[..., float],
    cfg: AdaptiveSweepConfig,
) -> List[Dict[str, Any]]:
    """Run an adaptive (Bayesian/TPE via Optuna, or random-search fallback)
    hyperparameter search.

    Parameters
    ----------
    trial_fn : ``trial_fn(**params) -> float`` (the metric to
        minimize/maximize per ``cfg.direction``). With the Optuna backend,
        ``trial_fn`` may optionally accept a keyword-only ``report_fn:
        Callable[[float, int], None]`` argument to report intermediate
        values for pruning (``report_fn(value, step)``); it is simply not
        passed under the random-search fallback (no pruning support
        there), so ``trial_fn`` should default that argument to a no-op.
    cfg : the parameter space + trial budget.

    Returns
    -------
    List of ``{"params": {...}, "value": float}`` dicts, sorted best-first
    (also includes pruned/failed trials with ``"pruned": True`` /
    ``"error": str`` instead of ``"value"``, rather than silently dropping
    them -- an adaptive search's failure trials are informative too).
    """
    try:
        import optuna
        return _run_with_optuna(trial_fn, cfg, optuna)
    except ImportError:
        return _run_with_random_search(trial_fn, cfg)


def _run_with_optuna(trial_fn, cfg: AdaptiveSweepConfig, optuna) -> List[Dict[str, Any]]:
    import optuna as _optuna  # noqa: F401  (already imported by caller; re-imported for clarity)

    sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    pruner = optuna.pruners.MedianPruner() if cfg.pruning else optuna.pruners.NopPruner()
    study = optuna.create_study(direction=cfg.direction, sampler=sampler, pruner=pruner)

    results: List[Dict[str, Any]] = []

    import inspect
    accepts_report_fn = "report_fn" in inspect.signature(trial_fn).parameters

    def objective(trial):
        params = {name: _suggest_optuna(trial, name, spec) for name, spec in cfg.param_space.items()}

        def report_fn(value: float, step: int) -> None:
            trial.report(value, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if accepts_report_fn:
            value = trial_fn(**params, report_fn=report_fn)
        else:
            value = trial_fn(**params)
        results.append({"params": params, "value": value})
        return value

    def _run_one(trial):
        try:
            return objective(trial)
        except optuna.TrialPruned:
            results.append({"params": trial.params, "pruned": True})
            raise
        except Exception as e:
            # Record a genuine trial_fn failure -- not caught by
            # study.optimize(catch=...) recording anything of its own, so
            # without this the trial would simply vanish from `results`
            # instead of showing up as an informative failure.
            results.append({"params": trial.params, "error": str(e)})
            raise

    study.optimize(_run_one, n_trials=cfg.n_trials, catch=(Exception,))

    reverse = cfg.direction == "maximize"
    scored = [r for r in results if "value" in r]
    scored.sort(key=lambda r: r["value"], reverse=reverse)
    unscored = [r for r in results if "value" not in r]
    return scored + unscored


def _run_with_random_search(trial_fn, cfg: AdaptiveSweepConfig) -> List[Dict[str, Any]]:
    rng = random.Random(cfg.seed)
    results: List[Dict[str, Any]] = []
    for _ in range(cfg.n_trials):
        params = {name: _suggest_random(rng, name, spec) for name, spec in cfg.param_space.items()}
        try:
            value = trial_fn(**params)
        except Exception as e:
            results.append({"params": params, "error": str(e)})
            continue
        results.append({"params": params, "value": value})

    reverse = cfg.direction == "maximize"
    scored = [r for r in results if "value" in r]
    scored.sort(key=lambda r: r["value"], reverse=reverse)
    failed = [r for r in results if "value" not in r]
    return scored + failed
