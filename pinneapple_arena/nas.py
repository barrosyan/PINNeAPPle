"""Architecture + hyperparameter search over the existing model catalog.

**What this is**: a *search-based* joint selection of (architecture family,
architecture hyperparameters, learning rate) over a fixed, already-registered
catalog (``pinneapple_neural.architectures.ModelRegistry``), driven by the
same Optuna TPE-sampler-with-median-pruner / random-search-fallback
machinery that ``pinneapple_neural.trainer.adaptive_sweep`` already uses.
Each Optuna "trial" both picks an architecture *and* samples that
architecture's hyperparameters (architecture choice is treated as just
another categorical hyperparameter — a flat search space, exactly the shape
``AdaptiveSweepConfig.param_space`` already expects), builds it via
``ModelRegistry.build()``, trains it for a small number of epochs on a real
``ArenaProblem`` via the real ``pinneapple_arena.trainer.train_pinn`` path,
and scores it from the real evaluation metrics (+ optionally the real
``TrainResult.physics_residual``).

**What this is not**: this is *not* a differentiable / weight-sharing NAS
(no DARTS-style supernet, no shared weights across candidates, no gradient
w.r.t. architecture choice). Every trial trains an independent model from
scratch. That is a deliberate scope decision — a true weight-sharing NAS
would be a much larger undertaking than the existing infrastructure (a
fixed architecture catalog + a black-box Optuna sweep) naturally supports,
and "search over the catalog + its hyperparameters" is what is actually
useful on top of what already exists (:mod:`pinneapple_arena.arena`'s
Arena, :mod:`pinneapple_neural.trainer.adaptive_sweep`'s AdaptiveSweep).

Why only re-implement part of ``adaptive_sweep``'s Optuna wiring
------------------------------------------------------------------
``adaptive_sweep.run_adaptive_sweep`` samples one *flat, unconditional*
parameter space every trial and returns a plain list of trial dicts — no
access to the underlying ``optuna.Study`` object. That flat-space model is
exactly what we want here too (architecture name + every candidate
architecture's hyperparameters, all sampled every trial; the trial function
below simply ignores whichever candidates' hyperparameters weren't the ones
selected for that trial — wasted samples, but harmless with the small
trial budgets a search inner-loop like this uses). But ``NASResult`` is
required to expose the raw ``optuna.Study`` (useful for external
introspection, e.g. ``optuna.visualization``), which
``run_adaptive_sweep``'s return type cannot carry. Since touching
``adaptive_sweep.py`` is out of scope for this module, :func:`search_architecture`
reuses ``adaptive_sweep``'s actual parameter-sampling primitives
(``_suggest_optuna``/``_suggest_random`` — the part that knows how to turn a
``("float"|"int"|"categorical", args)`` spec into a sampled value, including
log-uniform ranges) and follows the exact same
``TPESampler`` + ``MedianPruner`` / "``ImportError`` -> dependency-free
random-search fallback" pattern, rather than writing a second, parallel
parsing of the parameter-spec mini-language.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch.nn as nn

from pinneapple_neural.architectures import ModelRegistry
from pinneapple_neural.trainer.adaptive_sweep import (
    ParamSpec,
    _suggest_optuna,
    _suggest_random,
)

from .config import ModelConfig, NetworkConfig, TrainingConfig
from .problems import ArenaProblem
from .trainer import evaluate_model, train_pinn


# ── per-architecture hyperparameter search spaces ──────────────────────────
#
# Default candidate set is deliberately a small, tractable subset of the 12
# registered architecture families, not all of them:
#
#   - vanilla_pinn : plain MLP PINN (pinneapple_neural.architectures.pinns
#                    .vanilla.VanillaPINN) — constructor takes only
#                    (in_dim, out_dim, hidden: List[int], activation: str).
#   - modified_mlp : Wang et al. "modified MLP" PINN with a Fourier
#                    embedding gate — constructor takes
#                    (in_dim, out_dim, hidden_dim: int, n_layers: int,
#                    activation: Type[nn.Module]).
#   - siren        : sinusoidal-activation PINN (Sitzmann et al.) —
#                    constructor takes (in_dim, out_dim, hidden_dim: int,
#                    n_layers: int, omega_0: float).
#
# These three were picked because: (a) they all train through the same
# `train_pinn` autograd-residual path used elsewhere in Arena (no grids,
# graphs, or branch/trunk operator splits to assemble), (b) their
# constructors expose a handful of simple scalar/categorical
# hyperparameters with no cross-parameter constraints, and (c) they span a
# genuinely different inductive bias each (plain MLP vs. Fourier-embedded
# MLP vs. sinusoidal activations), so the search is not just re-discovering
# the same architecture three times. Other registered families (FNO,
# DeepONet, MeshGraphNet, transformer variants, Noether-equivariant
# operators, ...) need structurally different training data (grids, graphs,
# branch/trunk pairs) and were left out of the *default* space rather than
# forced through a one-size-fits-all trial loop; a caller can still search
# over them by passing a custom `ArchitectureSearchSpace` with an
# appropriate `build_kwargs_fn` (see `ArchitectureCandidate`).

_ACTIVATIONS = ["tanh", "silu", "gelu"]
_ACTIVATION_CLASSES = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU}


def _vanilla_pinn_kwargs(params: Dict[str, Any], in_dim: int, out_dim: int) -> Dict[str, Any]:
    hidden = [params["width"]] * params["n_layers"]
    return dict(in_dim=in_dim, out_dim=out_dim, hidden=hidden, activation=params["activation"])


def _modified_mlp_kwargs(params: Dict[str, Any], in_dim: int, out_dim: int) -> Dict[str, Any]:
    return dict(
        in_dim=in_dim, out_dim=out_dim,
        hidden_dim=params["hidden_dim"], n_layers=params["n_layers"],
        activation=_ACTIVATION_CLASSES[params["activation"]],
    )


def _siren_kwargs(params: Dict[str, Any], in_dim: int, out_dim: int) -> Dict[str, Any]:
    return dict(
        in_dim=in_dim, out_dim=out_dim,
        hidden_dim=params["hidden_dim"], n_layers=params["n_layers"],
        omega_0=params["omega_0"],
    )


@dataclass
class ArchitectureCandidate:
    """One searchable architecture: its ``ModelRegistry`` key, its own
    hyperparameter search space (in the same ``ParamSpec`` mini-language
    ``AdaptiveSweepConfig.param_space`` uses — ``("float"|"int", (lo, hi))``,
    optionally ``(..., "log")``, or ``("categorical", (choices,))``), and a
    function that turns one sampled-hyperparameter dict into the exact
    kwargs ``ModelRegistry.build(name, **kwargs)`` needs. The build-kwargs
    indirection exists because constructors are not uniform — e.g.
    ``modified_mlp`` wants an actual ``Type[nn.Module]`` for ``activation``
    where ``vanilla_pinn`` wants a plain string; ``instantiate()`` filters
    *unknown* kwargs but does not type-convert known ones, so this mapping
    has to be exact per architecture rather than generic.
    """
    name: str
    param_space: Dict[str, ParamSpec]
    build_kwargs_fn: Callable[[Dict[str, Any], int, int], Dict[str, Any]]


DEFAULT_CANDIDATES: Dict[str, ArchitectureCandidate] = {
    "vanilla_pinn": ArchitectureCandidate(
        name="vanilla_pinn",
        param_space={
            "n_layers": ("int", (2, 5)),
            "width": ("int", (16, 128)),
            "activation": ("categorical", (_ACTIVATIONS,)),
        },
        build_kwargs_fn=_vanilla_pinn_kwargs,
    ),
    "modified_mlp": ArchitectureCandidate(
        name="modified_mlp",
        param_space={
            "n_layers": ("int", (2, 6)),
            "hidden_dim": ("int", (16, 128)),
            "activation": ("categorical", (_ACTIVATIONS,)),
        },
        build_kwargs_fn=_modified_mlp_kwargs,
    ),
    "siren": ArchitectureCandidate(
        name="siren",
        param_space={
            "n_layers": ("int", (2, 6)),
            "hidden_dim": ("int", (16, 128)),
            "omega_0": ("float", (5.0, 60.0)),
        },
        build_kwargs_fn=_siren_kwargs,
    ),
}

_DEFAULT_LR_SPACE: ParamSpec = ("float", (1e-4, 1e-2, "log"))


@dataclass
class ArchitectureSearchSpace:
    """Which architectures are candidates, plus each one's hyperparameter
    search space and a shared learning-rate range applied to every
    candidate (learning rate isn't architecture-specific, so it lives
    outside `candidates` rather than being duplicated in each one).
    """
    candidates: Dict[str, ArchitectureCandidate] = field(
        default_factory=lambda: dict(DEFAULT_CANDIDATES))
    lr_param_space: ParamSpec = _DEFAULT_LR_SPACE

    @property
    def architecture_names(self) -> List[str]:
        return sorted(self.candidates)

    def _flat_param_space(self) -> Dict[str, ParamSpec]:
        """Flatten into the shape ``AdaptiveSweepConfig.param_space``
        expects: one unconditional entry per parameter, architecture
        candidates' hyperparameters namespaced by ``"<arch>__<param>"`` so
        names never collide across architectures."""
        space: Dict[str, ParamSpec] = {
            "architecture": ("categorical", (self.architecture_names,)),
            "lr": self.lr_param_space,
        }
        for arch_name, cand in self.candidates.items():
            for pname, pspec in cand.param_space.items():
                space[f"{arch_name}__{pname}"] = pspec
        return space


# ── scoring ──────────────────────────────────────────────────────────────

def _accuracy_score(eval_out: Dict[str, Any], field_names: List[str]) -> float:
    """Mean relative-L2 error across fields — lower is better. Mirrors
    ``pinneapple_arena.arena._accuracy_score`` exactly (same metric Arena's
    own ranking uses), computed locally here rather than imported since
    that helper is a private symbol of a different module."""
    m = eval_out.get("metrics", {})
    rels = [m[f"rel_{f}"] for f in field_names if f"rel_{f}" in m]
    if not rels:
        return float("nan")
    return float(np.mean(rels))


def _score(eval_out: Dict[str, Any], field_names: List[str],
          physics_residual: Optional[float], physics_aware: bool, lam: float) -> "tuple[float, float]":
    """The search objective (lower is better). Returns ``(score, accuracy)``
    — the plain accuracy component is returned alongside the combined score
    so callers (and ``all_trials`` records) can see how much of the score
    came from accuracy vs. the physics-residual penalty.

    Plain mode: ``score == accuracy`` (mean relative-L2 error), the same
    notion Arena's default ranking already uses.

    ``physics_aware=True``: reuses the *actual* ``TrainResult.physics_residual``
    that ``train_pinn`` already computes (never re-derived — see
    ``pinneapple_arena.trainer.TrainResult.physics_residual``'s own
    docstring) and combines it with accuracy as
    ``score = accuracy + lambda * physics_residual``. This is a simple
    additive penalty, not a principled multi-objective scalarization: a
    model that fits the *evaluation points* well but violates the PDE
    between them (the exact failure mode ``physics_aware_rank`` in
    ``arena.py`` flags post-hoc) scores worse here and is less likely to be
    selected as "best" by the search. ``lambda`` (default 1.0) trades the
    two off — larger values bias the search toward low-residual (physically
    consistent) models even at some accuracy cost; there is no universally
    "right" value since accuracy and residual are typically on different
    scales, so callers with unusual scaling (e.g. very small/large PDE
    residual magnitudes) should override it. Falls back to pure accuracy
    for any trial where no residual is available (e.g. non-PINN candidates,
    or `physics_aware=False`).
    """
    acc = _accuracy_score(eval_out, field_names)
    if physics_aware and physics_residual is not None:
        return float(acc + lam * physics_residual), acc
    return float(acc), acc


# ── result container ────────────────────────────────────────────────────

@dataclass
class NASResult:
    best_architecture: Optional[str]
    best_hyperparams: Dict[str, Any]
    best_score: float
    all_trials: List[Dict[str, Any]]
    study: Optional[Any] = None  # raw optuna.Study, or None under the random-search fallback


# ── the search loop ─────────────────────────────────────────────────────

def search_architecture(
    problem: ArenaProblem,
    search_space: Optional[ArchitectureSearchSpace] = None,
    n_trials: int = 20,
    physics_aware: bool = False,
    lam: float = 1.0,
    epochs_per_trial: int = 300,
    n_train: int = 64,
    n_bc: int = 32,
    n_col: int = 128,
    grid_n: int = 12,
    device: str = "cpu",
    seed: Optional[int] = None,
    timeout: Optional[float] = None,
    **train_kwargs: Any,
) -> NASResult:
    """Search over ``search_space`` (architecture name + each architecture's
    hyperparameters, jointly) for the model that best solves ``problem``.

    Each trial: samples an architecture name and that architecture's
    hyperparameters (+ a shared learning rate) from ``search_space``,
    builds it via ``ModelRegistry.build``, trains it for
    ``epochs_per_trial`` epochs via the real ``train_pinn`` path (small
    epoch budget by design — this is a search *inner loop*, not a full
    training run; raise ``epochs_per_trial`` for a more accurate search at
    the cost of wall-clock time), evaluates it on ``problem``'s held-out
    grid via the real ``evaluate_model``, and scores it via :func:`_score`.

    Uses Optuna's TPE sampler + median pruner when Optuna is installed
    (same as ``pinneapple_neural.trainer.adaptive_sweep``), else falls back
    to dependency-free random search — no pruning in that case, matching
    ``adaptive_sweep``'s own fallback behaviour.

    Parameters
    ----------
    problem : an ``ArenaProblem`` (e.g. from ``pinneapple_arena.problems.get_problem``).
        Must support ``supervised_data`` + ``pinn_residuals`` (all built-in
        Arena problems do) since every default candidate architecture
        trains through ``train_pinn``.
    search_space : candidate architectures + their hyperparameter ranges.
        Defaults to :data:`DEFAULT_CANDIDATES` (``vanilla_pinn``,
        ``modified_mlp``, ``siren``) wrapped in an ``ArchitectureSearchSpace``.
    n_trials : number of architecture+hyperparameter combinations to try.
    physics_aware : if True, penalize accuracy by the trained model's
        ``TrainResult.physics_residual`` (see :func:`_score`).
    lam : the accuracy/physics-residual tradeoff weight, only used when
        ``physics_aware=True``.
    epochs_per_trial, n_train, n_bc, n_col, grid_n : kept small by default
        (this is a search mechanism, not a full training run) — override
        for a more faithful (slower) search.
    timeout : optional wall-clock budget in seconds for the whole search
        (Optuna backend: passed straight to ``study.optimize(timeout=...)``;
        random-search fallback: checked between trials).
    **train_kwargs : forwarded to ``TrainingConfig`` for every trial (e.g.
        ``grad_clip``, ``scheduler``, ``optimizer``, ``weight_decay``) —
        unrecognized keys are silently dropped, ``epochs``/``lr`` are
        ignored here since they are controlled by ``epochs_per_trial`` and
        the per-trial sampled learning rate respectively.

    Returns
    -------
    NASResult
    """
    space = search_space or ArchitectureSearchSpace()
    if not space.candidates:
        raise ValueError("search_space has no architecture candidates to search over.")
    flat_space = space._flat_param_space()

    tc_kwargs = {k: v for k, v in train_kwargs.items()
                if k in TrainingConfig.__dataclass_fields__ and k not in ("epochs", "lr")}

    # Data is architecture/hyperparameter-independent — built once, reused
    # by every trial (mirrors Arena's own `_prepare_data`: dense PINN
    # collocation points sampled separately from the supervised (X, Y)
    # points, over the same bounding box).
    xy_int, _Y_int, xy_bc, Y_bc, xy_eval, Y_eval, field_names = problem.supervised_data(
        n_train=n_train, n_bc=n_bc, grid_n=grid_n)
    data_rng = np.random.default_rng(seed if seed is not None else 0)
    lo, hi = xy_int.min(axis=0), xy_int.max(axis=0)
    xy_col = data_rng.uniform(lo, hi, (n_col, problem.input_dim))

    all_trials: List[Dict[str, Any]] = []

    def _eval_params(params: Dict[str, Any]) -> float:
        arch_name = params["architecture"]
        lr = params["lr"]
        cand = space.candidates[arch_name]
        arch_params = {p: params[f"{arch_name}__{p}"] for p in cand.param_space}
        record: Dict[str, Any] = {"architecture": arch_name,
                                  "hyperparams": dict(arch_params, lr=lr)}
        try:
            build_kwargs = cand.build_kwargs_fn(arch_params, problem.input_dim, problem.output_dim)
            model = ModelRegistry.build(arch_name, **build_kwargs)
            mcfg = ModelConfig(
                name=f"nas__{arch_name}", type=arch_name,
                network=NetworkConfig(),
                training=TrainingConfig(epochs=epochs_per_trial, lr=lr, **tc_kwargs),
            )
            result = train_pinn(
                model, mcfg,
                pinn_residuals_fn=problem.pinn_residuals,
                xy_int=xy_col, xy_bc=xy_bc, uv_bc=Y_bc,
                problem_params={}, device=device,
            )
            eval_out = evaluate_model(result, mcfg, xy_eval, Y_eval, field_names, device=device)
            score, accuracy = _score(eval_out, field_names, result.physics_residual, physics_aware, lam)
        except Exception as e:
            record["score"] = None
            record["error"] = str(e)
            all_trials.append(record)
            raise
        record["score"] = score
        record["accuracy"] = accuracy
        record["physics_residual"] = result.physics_residual
        all_trials.append(record)
        return score

    try:
        import optuna
        study = _run_optuna(_eval_params, flat_space, n_trials, seed, timeout, optuna)
    except ImportError:
        study = None
        _run_random_search(_eval_params, flat_space, n_trials, seed, timeout)

    scored = [t for t in all_trials if t.get("score") is not None]
    if not scored:
        return NASResult(best_architecture=None, best_hyperparams={},
                         best_score=float("nan"), all_trials=all_trials, study=study)

    best = min(scored, key=lambda t: t["score"])
    return NASResult(
        best_architecture=best["architecture"],
        best_hyperparams=best["hyperparams"],
        best_score=best["score"],
        all_trials=all_trials,
        study=study,
    )


def _run_optuna(objective_fn: Callable[[Dict[str, Any]], float],
                flat_space: Dict[str, ParamSpec], n_trials: int,
                seed: Optional[int], timeout: Optional[float], optuna) -> Any:
    """Same sampler/pruner construction as
    ``adaptive_sweep._run_with_optuna``: ``TPESampler`` + ``MedianPruner``.
    Trials here never call ``trial.report``/pruning (no natural
    intermediate metric — a training run's whole point is a single final
    score), so the pruner is effectively inert; kept for parity with
    ``adaptive_sweep`` and so a caller inspecting ``NASResult.study`` sees
    the same setup they'd get from an ``AdaptiveSweep`` run."""
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    def _objective(trial):
        params = {name: _suggest_optuna(trial, name, spec) for name, spec in flat_space.items()}
        return objective_fn(params)

    study.optimize(_objective, n_trials=n_trials, timeout=timeout, catch=(Exception,))
    return study


def _run_random_search(objective_fn: Callable[[Dict[str, Any]], float],
                       flat_space: Dict[str, ParamSpec], n_trials: int,
                       seed: Optional[int], timeout: Optional[float]) -> None:
    """Dependency-free fallback, matching
    ``adaptive_sweep._run_with_random_search``'s behaviour (uniform/random
    sampling, no pruning)."""
    rng = random.Random(seed)
    t0 = time.time()
    for _ in range(n_trials):
        if timeout is not None and (time.time() - t0) > timeout:
            break
        params = {name: _suggest_random(rng, name, spec) for name, spec in flat_space.items()}
        try:
            objective_fn(params)
        except Exception:
            continue
