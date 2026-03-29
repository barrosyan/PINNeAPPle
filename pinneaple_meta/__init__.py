from __future__ import annotations
"""pinneaple_meta — Meta-Learning for parametric PDE families.

Trains a meta-initialisation that can adapt to any member of a PDE family
(e.g. Burgers with varying ν, Navier-Stokes with varying Re) in just a few
gradient steps — without retraining from scratch.

Algorithms
----------
MAML (Model-Agnostic Meta-Learning)
    Bi-level optimisation: inner loop adapts to each task; outer loop updates
    the shared initialisation so inner adaptation is maximally effective.
    Uses first-order approximation (FOMAML) by default for efficiency.

Reptile
    Simpler first-order method: train on each task independently, then
    interpolate the shared weights toward the task-specific weights.
    No second-order gradients; scales to larger models.

Components
----------
PDETaskSampler
    Samples tasks (PDE instances with specific parameters) for meta-training.
MAMLTrainer
    Full MAML training loop with fast-adaptation inner loop and meta-update.
ReptileTrainer
    Reptile training loop (simpler, no higher-order gradients).
MetaModel
    Wraps a trained meta-model for easy adaptation and prediction.

Quick start
-----------
>>> from pinneaple_meta import ReptileTrainer, ReptileConfig, PDETaskSampler
>>>
>>> sampler = PDETaskSampler(
...     param_ranges={"nu": (0.001, 0.1)},
...     physics_fn_factory=lambda p: make_burgers_loss(nu=p["nu"]),
... )
>>> cfg = ReptileConfig(n_inner_steps=10, n_meta_epochs=500)
>>> trainer = ReptileTrainer(model, cfg, sampler)
>>> trainer.train()
>>> adapted = trainer.adapt(new_task, n_steps=20)
"""

from .config import MAMLConfig, ReptileConfig
from .task_sampler import PDETaskSampler
from .maml import MAMLTrainer

try:
    from .reptile import ReptileTrainer
except Exception:  # pragma: no cover
    ReptileTrainer = None  # type: ignore[assignment,misc]

try:
    from .meta_model import MetaModel
except Exception:  # pragma: no cover
    MetaModel = None  # type: ignore[assignment,misc]


def meta_train(model, sampler, *, algorithm: str = "reptile", **cfg_kwargs):
    """Convenience entry point for meta-training.

    Parameters
    ----------
    model : nn.Module — base model to meta-train
    sampler : PDETaskSampler
    algorithm : "reptile" (default) or "maml"
    **cfg_kwargs : passed to ReptileConfig / MAMLConfig

    Returns
    -------
    Trained trainer object with .adapt() method.
    """
    if algorithm == "maml":
        cfg = MAMLConfig(**cfg_kwargs)
        return MAMLTrainer(model, cfg, sampler)
    cfg = ReptileConfig(**cfg_kwargs)
    if ReptileTrainer is None:
        raise ImportError("ReptileTrainer not available — check pinneaple_meta.reptile")
    return ReptileTrainer(model, cfg, sampler)


def meta_adapt(meta_trainer, task: dict, n_steps: int = 10):
    """Fast-adapt a trained meta-model to a new task.

    Parameters
    ----------
    meta_trainer : MAMLTrainer or ReptileTrainer (already trained)
    task : dict with keys "support", "physics_fn", optional "params"
    n_steps : number of inner gradient steps

    Returns
    -------
    Adapted nn.Module copy (does not modify meta_trainer.model).
    """
    return meta_trainer.adapt(task, n_steps=n_steps)


__all__ = [
    "MAMLConfig",
    "ReptileConfig",
    "PDETaskSampler",
    "MAMLTrainer",
    "ReptileTrainer",
    "MetaModel",
    "meta_train",
    "meta_adapt",
]
