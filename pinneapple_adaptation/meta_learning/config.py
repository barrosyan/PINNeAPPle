"""Meta-learning configuration dataclasses for MAML and Reptile trainers."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MAMLConfig:
    """Configuration for Model-Agnostic Meta-Learning (MAML) training.

    Parameters
    ----------
    n_inner_steps:
        Number of gradient steps taken on the support set during the inner
        loop (fast adaptation) for each task.
    inner_lr:
        Learning rate used inside the inner loop.
    outer_lr:
        Meta (outer-loop) learning rate applied when updating the shared
        initialization parameters.
    n_tasks_per_batch:
        Number of tasks sampled from the task distribution per meta-update
        step.
    n_meta_epochs:
        Total number of outer-loop meta-update iterations.
    first_order:
        When ``True``, use first-order MAML (FOMAML): gradients through the
        inner-loop update step are ignored.  This is faster and typically
        achieves similar performance to full second-order MAML.
    device:
        PyTorch device string, e.g. ``"cpu"``, ``"cuda"``, ``"cuda:0"``.
    seed:
        Random seed for reproducibility.  ``0`` is a valid seed.
    checkpoint_every:
        Persist a checkpoint to disk every this many meta-epochs.  Set to
        ``0`` to disable automatic checkpointing.
    """

    n_inner_steps: int = 5
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    n_tasks_per_batch: int = 4
    n_meta_epochs: int = 1000
    first_order: bool = True
    device: str = "cpu"
    seed: int = 0
    checkpoint_every: int = 100

    def __post_init__(self) -> None:
        if self.n_inner_steps < 1:
            raise ValueError("MAMLConfig.n_inner_steps must be >= 1.")
        if self.n_tasks_per_batch < 1:
            raise ValueError("MAMLConfig.n_tasks_per_batch must be >= 1.")
        if self.n_meta_epochs < 1:
            raise ValueError("MAMLConfig.n_meta_epochs must be >= 1.")
        if self.inner_lr <= 0:
            raise ValueError("MAMLConfig.inner_lr must be > 0.")
        if self.outer_lr <= 0:
            raise ValueError("MAMLConfig.outer_lr must be > 0.")
        if self.checkpoint_every < 0:
            raise ValueError("MAMLConfig.checkpoint_every must be >= 0.")


@dataclass
class ReptileConfig:
    """Configuration for the Reptile meta-learning algorithm.

    Reptile is a simpler first-order meta-learning algorithm that avoids
    second-order gradients entirely.  The outer update is a weighted
    interpolation between the current meta-parameters and the task-adapted
    parameters: ``θ ← θ + ε * (θ_task − θ)``.

    Parameters
    ----------
    n_inner_steps:
        Number of SGD steps taken on each task during the inner loop.
    inner_lr:
        Learning rate for the inner-loop SGD optimizer.
    outer_lr:
        Reptile outer step size.  The outer update is
        ``θ ← θ + (outer_lr * schedule) * epsilon * (θ_task − θ)``, where
        ``schedule`` linearly anneals from ``1`` down to ``0`` over
        ``n_meta_epochs`` (as in Nichol et al.'s experiments), so the
        actual per-epoch step shrinks from ``outer_lr`` toward ``0`` by
        the end of training.
    n_tasks_per_batch:
        Number of tasks to process per meta-update.  Averaging the
        task-adapted parameters over several tasks before interpolating
        is what makes the Reptile update approximate the MAML gradient
        (Nichol et al. 2018); a single task per batch degenerates to
        sequential single-task fine-tuning.
    n_meta_epochs:
        Total number of meta-update iterations.
    device:
        PyTorch device string, e.g. ``"cpu"``, ``"cuda"``.
    seed:
        Random seed for reproducibility.
    epsilon:
        Interpolation factor in the Reptile weight update
        (``θ ← θ + epsilon * (θ_task − θ)``), applied on top of the
        annealed ``outer_lr`` schedule described above.  Kept at ``1.0``
        by default so ``outer_lr`` alone controls the effective step
        size; set to a value in ``(0, 1)`` for additional damping.
    checkpoint_every:
        Persist a checkpoint to disk every this many meta-epochs.  Set to
        ``0`` to disable.
    """

    n_inner_steps: int = 10
    inner_lr: float = 0.01
    outer_lr: float = 0.1
    n_tasks_per_batch: int = 4
    n_meta_epochs: int = 1000
    device: str = "cpu"
    seed: int = 0
    epsilon: float = 1.0
    checkpoint_every: int = 100

    def __post_init__(self) -> None:
        if self.n_inner_steps < 1:
            raise ValueError("ReptileConfig.n_inner_steps must be >= 1.")
        if self.n_tasks_per_batch < 1:
            raise ValueError("ReptileConfig.n_tasks_per_batch must be >= 1.")
        if self.n_meta_epochs < 1:
            raise ValueError("ReptileConfig.n_meta_epochs must be >= 1.")
        if self.inner_lr <= 0:
            raise ValueError("ReptileConfig.inner_lr must be > 0.")
        if self.outer_lr <= 0:
            raise ValueError("ReptileConfig.outer_lr must be > 0.")
        if not (0.0 < self.epsilon <= 1.0):
            raise ValueError("ReptileConfig.epsilon must be in (0, 1].")
        if self.checkpoint_every < 0:
            raise ValueError("ReptileConfig.checkpoint_every must be >= 0.")
