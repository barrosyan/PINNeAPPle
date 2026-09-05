from __future__ import annotations
"""pinneapple_meta — Meta-Learning for parametric PDE families.

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
>>> from pinneapple_adaptation.meta_learning import ReptileTrainer, ReptileConfig, PDETaskSampler
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

from .reptile import ReptileTrainer
from .meta_model import MetaModel


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
    # Both trainers require an explicit .train() call to actually run the
    # meta-training loop -- this function previously constructed the
    # trainer and returned it immediately without training, silently
    # contradicting its own docstring above ("Returns: Trained trainer
    # object"). .adapt() on an untrained trainer "worked" (no exception)
    # but adapted from the model's random initialization instead of a
    # real meta-learned one -- a use-after-"training" bug easy to miss
    # since nothing ever raised.
    if algorithm == "maml":
        cfg = MAMLConfig(**cfg_kwargs)
        trainer = MAMLTrainer(model, cfg, sampler)
    else:
        cfg = ReptileConfig(**cfg_kwargs)
        if ReptileTrainer is None:
            raise ImportError("ReptileTrainer not available — check pinneapple_meta.reptile")
        trainer = ReptileTrainer(model, cfg, sampler)
    trainer.train()
    return trainer


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
