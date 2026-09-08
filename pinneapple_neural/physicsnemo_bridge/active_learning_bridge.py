"""PINNeAPPle active learning -> PhysicsNeMo retrain bridge.

Promotes "Padrao 2" from ``examples/vs_physicsnemo/README.md``::

    PINNeAPPle ResidualBasedAL -> pontos hard de collocation
    PhysicsNeMo retreina FNO nesses pontos (GPU rapido)
    -> loop: AL seleciona -> PhysicsNeMo retreina -> AL seleciona...

That README documents the pattern with a diagram but, unlike patterns 1 and
3, neither ``examples/pinneapple_and_physicsnemo/`` nor
``examples/vs_physicsnemo/`` contains a runnable script that actually wires
it up. This module builds the real loop directly on top of
:mod:`pinneapple_data.active_learning`'s real, already-tested
``ResidualBasedAL`` / ``ActiveLearningConfig`` strategy classes -- the same
ones ``AdaptiveCollocationTrainer`` uses -- so point *selection* is fully
delegated rather than reimplemented; only the *retrain* half (a plain
Adam-optimizer training loop, mirroring the style of ``train_fno`` in
``examples/vs_physicsnemo/04_physicsnemo_fno_operator/example.py`` and
``train_surrogate`` in
``examples/vs_physicsnemo/05_combined_fno_digital_twin/example.py``) is new
glue.

``physicsnemo`` is a genuinely optional dependency here: the retrain loop
operates on any ``torch.nn.Module`` -- a plain PINNeAPPle model or a trained
physicsnemo one, since physicsnemo models are themselves ``nn.Module``
instances -- so nothing in this module requires physicsnemo to be
importable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from pinneapple_data.active_learning import ActiveLearningConfig, ResidualBasedAL


# ---------------------------------------------------------------------------
# Config / result types
# ---------------------------------------------------------------------------

@dataclass
class RetrainConfig:
    """Configuration for the "retrain" half of the AL <-> retrain loop."""

    n_rounds: int = 5
    n_epochs_per_round: int = 100
    batch_size: int = 512
    lr: float = 1e-3
    selection_mode: str = "weighted"  # "weighted" (RAD) | "top_k" (RAR)
    device: str = "cpu"
    verbose: bool = True


@dataclass
class ALRoundResult:
    """Bookkeeping for a single AL-select + retrain round."""

    round_index: int
    n_points_added: int
    n_points_total: int
    mean_train_loss: float
    train_losses: List[float] = field(default_factory=list)


@dataclass
class ALRetrainResult:
    """Full result of :func:`active_learning_retrain_loop`."""

    rounds: List[ALRoundResult]
    x_collocation: np.ndarray
    final_loss: float


# ---------------------------------------------------------------------------
# Initial collocation seeding
# ---------------------------------------------------------------------------

def _seed_initial_points(
    bounds: Dict[str, Tuple[float, float]], n: int, seed: int
) -> np.ndarray:
    """Latin Hypercube seed points for the initial collocation set.

    This is plain, generic sampling infrastructure -- not the active-learning
    selection logic, which is fully delegated to
    :class:`~pinneapple_data.active_learning.ResidualBasedAL` below. It is
    kept local (mirroring the same tiny LHS routine already duplicated
    between ``pinneapple_data.collocation`` and
    ``pinneapple_data.active_learning``) purely to seed round 0 before any
    residual information exists.
    """
    rng = np.random.default_rng(seed)
    names = list(bounds.keys())
    d = len(names)
    lo = np.array([bounds[k][0] for k in names], dtype=np.float64)
    hi = np.array([bounds[k][1] for k in names], dtype=np.float64)
    out = np.zeros((n, d), dtype=np.float32)
    for j in range(d):
        perm = rng.permutation(n)
        out[:, j] = (lo[j] + (perm + rng.random(n)) / n * (hi[j] - lo[j])).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# The AL <-> retrain loop
# ---------------------------------------------------------------------------

def active_learning_retrain_loop(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    residual_fn: Callable[[np.ndarray], np.ndarray],
    bounds: Dict[str, Tuple[float, float]],
    *,
    x_initial: Optional[np.ndarray] = None,
    al_config: Optional[ActiveLearningConfig] = None,
    retrain_config: Optional[RetrainConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> ALRetrainResult:
    """Run the "AL selects -> retrains -> AL selects -> ..." loop (Padrao 2).

    Each round:

    1. Retrain ``model`` for ``retrain_config.n_epochs_per_round`` epochs on
       the current collocation set ``x_collocation``, minimizing
       ``loss_fn(model, x_batch)``.
    2. Ask :class:`~pinneapple_data.active_learning.ResidualBasedAL` to
       select ``al_config.n_select`` new "hard" points using
       ``residual_fn`` (RAR/RAD, per ``retrain_config.selection_mode``).
    3. Append the new points to the collocation set and repeat.

    Parameters
    ----------
    model:
        Any ``torch.nn.Module`` -- a plain PINNeAPPle model or a trained
        physicsnemo one (physicsnemo models ARE ``nn.Module`` instances, so
        no adapter is required for training).
    loss_fn:
        ``(model, x_batch) -> scalar loss tensor``. Mirrors the training
        step used by ``train_fno``/``train_surrogate`` in the
        ``examples/vs_physicsnemo`` PhysicsNeMo-operator examples.
    residual_fn:
        ``(N, D) array -> (N,) array`` of ``|residual|``, passed straight
        through to ``ResidualBasedAL.select``.
    bounds:
        Coordinate bounds ``{name: (lo, hi)}`` -- the same format used by
        ``ResidualBasedAL``.
    x_initial:
        Optional initial collocation set. If omitted, an LHS seed set of
        ``al_config.n_initial`` points is generated.
    al_config:
        :class:`~pinneapple_data.active_learning.ActiveLearningConfig`
        controlling pool size, points selected per round, and the initial
        seed count.
    retrain_config:
        :class:`RetrainConfig` controlling the retrain loop itself.
    optimizer:
        Optional pre-built optimizer. Defaults to
        ``Adam(model.parameters(), lr=retrain_config.lr)``.

    Returns
    -------
    ALRetrainResult
        Per-round bookkeeping plus the final accumulated collocation set.
    """
    al_cfg = al_config or ActiveLearningConfig()
    rc = retrain_config or RetrainConfig()
    strategy = ResidualBasedAL(al_cfg, bounds)

    x_col = (
        np.asarray(x_initial, dtype=np.float32)
        if x_initial is not None
        else _seed_initial_points(bounds, al_cfg.n_initial, al_cfg.seed)
    )

    model.to(rc.device)
    opt = optimizer or torch.optim.Adam(model.parameters(), lr=rc.lr)
    rng = np.random.default_rng(al_cfg.seed)

    rounds: List[ALRoundResult] = []
    final_loss = float("nan")

    for round_idx in range(1, rc.n_rounds + 1):
        model.train()
        epoch_losses: List[float] = []
        n = len(x_col)
        for _epoch in range(rc.n_epochs_per_round):
            perm = rng.permutation(n)
            batch_losses: List[float] = []
            for start in range(0, n, rc.batch_size):
                idx = perm[start:start + rc.batch_size]
                xb = torch.from_numpy(x_col[idx]).to(rc.device)
                opt.zero_grad()
                loss = loss_fn(model, xb)
                loss.backward()
                opt.step()
                batch_losses.append(float(loss.detach().cpu().item()))
            epoch_losses.append(float(np.mean(batch_losses)))
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        final_loss = epoch_losses[-1] if epoch_losses else final_loss

        model.eval()
        new_pts = strategy.select(residual_fn, n_select=al_cfg.n_select, mode=rc.selection_mode)
        model.train()

        prev_n = len(x_col)
        x_col = np.concatenate([x_col, new_pts], axis=0)

        rounds.append(
            ALRoundResult(
                round_index=round_idx,
                n_points_added=len(new_pts),
                n_points_total=len(x_col),
                mean_train_loss=mean_loss,
                train_losses=epoch_losses,
            )
        )
        if rc.verbose:
            print(
                f"[AL round {round_idx}/{rc.n_rounds}] "
                f"{prev_n} -> {len(x_col)} points  mean_loss={mean_loss:.4e}"
            )

    return ALRetrainResult(rounds=rounds, x_collocation=x_col, final_loss=final_loss)
