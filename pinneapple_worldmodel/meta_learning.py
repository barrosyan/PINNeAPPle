"""Meta-learning for the physics world model.

Uses both Pinneapple-native meta-learning (``pinneapple_meta``) and built-in
MAML/Reptile implementations to train a model that can rapidly adapt to a new
physics domain with only a handful of trajectory examples.

Key classes
-----------
:class:`TaskDistribution`
    Samples physics tasks (scenario + parameters) for meta-training.  Each
    task is a small (support, query) pair of WorldModelDataset slices.

:class:`MAMLWorldModel`
    Wraps :class:`~.model.PhysicsWorldModel` with a MAML inner-loop
    (via ``higher`` if available, otherwise manual parameter copies).

:class:`ReptileWorldModel`
    Lightweight Reptile variant — no second-order gradients required;
    suitable for large models.

:class:`MetaLearner`
    Orchestrates task sampling, inner-loop adaptation, and outer-loop
    meta-gradient updates.  Falls back to ``pinneapple_meta.MAMLTrainer``
    and ``pinneapple_meta.ReptileTrainer`` when available.

Quick start::

    from pinneapple_worldmodel.meta_learning import MetaLearner, MetaConfig
    from pinneapple_worldmodel.dataset_factory import DatasetCatalog

    learner = MetaLearner(MetaConfig(
        algorithm="reptile",
        n_meta_epochs=500,
        n_inner_steps=5,
        inner_lr=1e-3,
        outer_lr=1e-4,
        n_tasks_per_batch=4,
        device="cuda",
    ))
    meta_model = learner.meta_train(catalog)
    # Fast-adapt to a new task:
    adapted = learner.adapt(meta_model, support_dataset, n_steps=10)
"""
from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from .model import PhysicsWorldModel, WorldModelConfig
from .dataset import WorldModelDataset
from .dataset_factory import DatasetCatalog

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MetaConfig
# ---------------------------------------------------------------------------

@dataclass
class MetaConfig:
    """Configuration for :class:`MetaLearner`.

    Parameters
    ----------
    algorithm : ``"maml"`` | ``"reptile"`` | ``"auto"``
        ``"auto"`` prefers pinneapple_meta if available, else reptile.
    n_meta_epochs : int — outer-loop steps.
    n_inner_steps : int — gradient steps inside each task.
    inner_lr : float — fast-adaptation learning rate.
    outer_lr : float — meta-update learning rate.
    n_tasks_per_batch : int — tasks sampled per meta-batch.
    n_support : int — support-set size (samples per task for inner loop).
    n_query : int — query-set size (for meta-gradient).
    first_order : bool — use first-order MAML (ignore second derivatives).
    device : str
    log_every : int
    save_path : str or None — path to save the meta-trained model.
    verbose : bool
    """
    algorithm: str = "reptile"
    n_meta_epochs: int = 500
    n_inner_steps: int = 5
    inner_lr: float = 1e-3
    outer_lr: float = 1e-4
    n_tasks_per_batch: int = 4
    n_support: int = 32
    n_query: int = 32
    first_order: bool = True
    device: str = "cpu"
    log_every: int = 50
    save_path: Optional[str] = None
    verbose: bool = True


# ---------------------------------------------------------------------------
# TaskDistribution
# ---------------------------------------------------------------------------

class TaskDistribution:
    """Sample physics tasks from a :class:`~.dataset_factory.DatasetCatalog`.

    A *task* is a ``(support_batch, query_batch)`` pair drawn from one
    dataset entry in the catalog.  Different tasks correspond to different
    physics scenarios / parameter regimes.

    Parameters
    ----------
    catalog : DatasetCatalog
    n_support : int — number of samples in the support set.
    n_query : int — number of samples in the query set.
    device : str
    """

    def __init__(
        self,
        catalog: DatasetCatalog,
        n_support: int = 32,
        n_query: int = 32,
        device: str = "cpu",
    ) -> None:
        self.datasets: List[WorldModelDataset] = []
        for entry in catalog.entries:
            if len(entry.dataset) >= n_support + n_query:
                self.datasets.append(entry.dataset)

        if not self.datasets:
            raise ValueError(
                "No catalog entries with enough samples for meta-learning. "
                "Increase n_samples_per_scenario."
            )
        self.n_support = n_support
        self.n_query = n_query
        self.device = device
        log.info("TaskDistribution: %d task sources", len(self.datasets))

    def sample(
        self, n_tasks: int = 4
    ) -> List[Tuple[Dict[str, Tensor], Dict[str, Tensor]]]:
        """Sample *n_tasks* (support, query) batch pairs.

        Returns
        -------
        list of (support_batch, query_batch) tuples.
        """
        tasks = []
        sources = random.choices(self.datasets, k=n_tasks)
        for ds in sources:
            n = len(ds)
            idx = torch.randperm(n)[:self.n_support + self.n_query].tolist()
            sup_idx = idx[:self.n_support]
            qry_idx = idx[self.n_support:]
            support = self._collate(ds, sup_idx)
            query = self._collate(ds, qry_idx)
            tasks.append((support, query))
        return tasks

    def _collate(self, ds: WorldModelDataset, indices: List[int]) -> Dict[str, Tensor]:
        sub = Subset(ds, indices)
        loader = DataLoader(sub, batch_size=len(indices), shuffle=False)
        batch = next(iter(loader))
        return {k: v.to(self.device) if isinstance(v, Tensor) else v
                for k, v in batch.items()}


# ---------------------------------------------------------------------------
# MetaLearner
# ---------------------------------------------------------------------------

class MetaLearner:
    """MAML / Reptile meta-learning for physics world models.

    Parameters
    ----------
    config : MetaConfig
    model_config : WorldModelConfig or None — architecture (auto-inferred from catalog).

    Attributes
    ----------
    meta_model : PhysicsWorldModel — the meta-initialised model after training.
    meta_history : list of dicts — one per meta-epoch with loss metrics.

    Example
    -------
    >>> learner = MetaLearner(MetaConfig(algorithm="reptile", n_meta_epochs=300))
    >>> meta_model = learner.meta_train(catalog)
    >>> adapted = learner.adapt(meta_model, support_dataset, n_steps=5)
    """

    def __init__(
        self,
        config: MetaConfig,
        model_config: Optional[WorldModelConfig] = None,
    ) -> None:
        self.config = config
        self.model_config = model_config or WorldModelConfig()
        self.meta_model: Optional[PhysicsWorldModel] = None
        self.meta_history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def meta_train(
        self,
        catalog: DatasetCatalog,
        *,
        warm_start: Optional[PhysicsWorldModel] = None,
    ) -> PhysicsWorldModel:
        """Run the full meta-training loop.

        Parameters
        ----------
        catalog : DatasetCatalog
        warm_start : optional pre-trained model to meta-fine-tune.

        Returns
        -------
        PhysicsWorldModel — meta-initialised (ready for fast adaptation).
        """
        cfg = self.config
        device = torch.device(cfg.device)

        # Attempt to use pinneapple_meta
        if cfg.algorithm in ("auto", "reptile", "maml"):
            result = self._try_pinneapple_meta(catalog)
            if result is not None:
                self.meta_model = result
                return result

        # Fallback: built-in implementation
        task_dist = TaskDistribution(
            catalog,
            n_support=cfg.n_support,
            n_query=cfg.n_query,
            device=cfg.device,
        )

        # Build model from first catalog entry
        ref_entry = next(iter(catalog.entries))
        ref_ds = ref_entry.dataset
        model = warm_start or self._build_model(ref_ds)
        model = model.to(device)

        if cfg.algorithm in ("reptile", "auto"):
            model = self._reptile_loop(model, task_dist)
        else:
            model = self._maml_loop(model, task_dist)

        self.meta_model = model

        if cfg.save_path:
            torch.save(
                {"model": model.state_dict(), "config": model.config,
                 "n_fields": model.n_fields, "grid_shape": model.grid_shape},
                cfg.save_path,
            )
            if cfg.verbose:
                print(f"[MetaLearner] Saved meta-model → {cfg.save_path}")

        return model

    # ------------------------------------------------------------------
    # Fast adaptation
    # ------------------------------------------------------------------

    def adapt(
        self,
        model: PhysicsWorldModel,
        support_dataset: WorldModelDataset,
        *,
        n_steps: int = 10,
        lr: Optional[float] = None,
    ) -> PhysicsWorldModel:
        """Fast-adapt *model* to *support_dataset* (inner-loop only).

        Parameters
        ----------
        model : PhysicsWorldModel — meta-initialised model.
        support_dataset : WorldModelDataset — support set for new task.
        n_steps : int — gradient steps for adaptation.
        lr : float or None — use ``config.inner_lr`` if None.

        Returns
        -------
        PhysicsWorldModel — adapted copy (original unchanged).
        """
        adapted = copy.deepcopy(model)
        adapted.train()
        device = next(adapted.parameters()).device

        opt = torch.optim.SGD(adapted.parameters(), lr=lr or self.config.inner_lr)
        loader = DataLoader(support_dataset, batch_size=min(32, len(support_dataset)),
                            shuffle=True)

        for step in range(n_steps):
            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, Tensor) else v
                         for k, v in batch.items()}
                pred = adapted(batch["state_t"], batch.get("context"))
                loss = torch.mean((pred - batch["state_tp1"]) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
            break  # one pass per step

        adapted.eval()
        return adapted

    # ------------------------------------------------------------------
    # Reptile loop (built-in)
    # ------------------------------------------------------------------

    def _reptile_loop(
        self,
        model: PhysicsWorldModel,
        task_dist: TaskDistribution,
    ) -> PhysicsWorldModel:
        cfg = self.config
        device = torch.device(cfg.device)

        for epoch in range(1, cfg.n_meta_epochs + 1):
            tasks = task_dist.sample(cfg.n_tasks_per_batch)
            query_losses = []

            for support, query in tasks:
                # Inner loop: clone and adapt
                inner = copy.deepcopy(model)
                inner.train()
                inner_opt = torch.optim.SGD(inner.parameters(), lr=cfg.inner_lr)

                for _ in range(cfg.n_inner_steps):
                    pred = inner(support["state_t"], support.get("context"))
                    loss = torch.mean((pred - support["state_tp1"]) ** 2)
                    inner_opt.zero_grad()
                    loss.backward()
                    inner_opt.step()

                # Reptile outer update: interpolate toward inner params
                with torch.no_grad():
                    for p_meta, p_inner in zip(
                        model.parameters(), inner.parameters()
                    ):
                        p_meta.data += cfg.outer_lr * (p_inner.data - p_meta.data)

                # Query loss for monitoring
                inner.eval()
                with torch.no_grad():
                    q_pred = inner(query["state_t"], query.get("context"))
                    q_loss = torch.mean((q_pred - query["state_tp1"]) ** 2).item()
                    query_losses.append(q_loss)

            record = {
                "epoch": float(epoch),
                "query_loss": sum(query_losses) / max(len(query_losses), 1),
            }
            self.meta_history.append(record)

            if cfg.verbose and (epoch % cfg.log_every == 0 or epoch == 1):
                print(f"[MetaLearner/Reptile] epoch={epoch}/{cfg.n_meta_epochs}  "
                      f"query_loss={record['query_loss']:.4g}")

        return model

    # ------------------------------------------------------------------
    # MAML loop (first-order, built-in)
    # ------------------------------------------------------------------

    def _maml_loop(
        self,
        model: PhysicsWorldModel,
        task_dist: TaskDistribution,
    ) -> PhysicsWorldModel:
        cfg = self.config
        meta_opt = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)

        for epoch in range(1, cfg.n_meta_epochs + 1):
            tasks = task_dist.sample(cfg.n_tasks_per_batch)
            meta_loss_total = torch.tensor(0.0, device=next(model.parameters()).device)

            for support, query in tasks:
                # Inner loop on a copy
                inner = copy.deepcopy(model)
                inner.train()
                inner_opt = torch.optim.SGD(inner.parameters(), lr=cfg.inner_lr)

                for _ in range(cfg.n_inner_steps):
                    pred = inner(support["state_t"], support.get("context"))
                    loss = torch.mean((pred - support["state_tp1"]) ** 2)
                    inner_opt.zero_grad()
                    loss.backward()
                    inner_opt.step()

                # Query loss from adapted model
                inner.eval()
                with torch.enable_grad():
                    q_pred = inner(query["state_t"], query.get("context"))
                    q_loss = torch.mean((q_pred - query["state_tp1"]) ** 2)

                if cfg.first_order:
                    # First-order MAML: meta-gradient is ∂L_query/∂φ evaluated at
                    # the support-adapted params φ, used directly as a proxy for
                    # ∂L_query/∂θ (drops the inner-loop Jacobian dφ/dθ).
                    inner.zero_grad(set_to_none=True)
                    q_loss.backward()
                    with torch.no_grad():
                        for p_meta, p_inner in zip(
                            model.parameters(), inner.parameters()
                        ):
                            if p_meta.grad is None:
                                p_meta.grad = torch.zeros_like(p_meta)
                            if p_inner.grad is not None:
                                p_meta.grad += p_inner.grad / cfg.n_tasks_per_batch
                else:
                    meta_loss_total = meta_loss_total + q_loss / cfg.n_tasks_per_batch

            if cfg.first_order:
                meta_opt.step()
                meta_opt.zero_grad()
            else:
                meta_opt.zero_grad()
                meta_loss_total.backward()
                meta_opt.step()

            record = {
                "epoch": float(epoch),
                "meta_loss": meta_loss_total.item() if not cfg.first_order else 0.0,
            }
            self.meta_history.append(record)

            if cfg.verbose and (epoch % cfg.log_every == 0 or epoch == 1):
                print(f"[MetaLearner/MAML] epoch={epoch}/{cfg.n_meta_epochs}  "
                      f"meta_loss={record['meta_loss']:.4g}")

        return model

    # ------------------------------------------------------------------
    # pinneapple_meta integration
    # ------------------------------------------------------------------

    def _try_pinneapple_meta(
        self, catalog: DatasetCatalog
    ) -> Optional[PhysicsWorldModel]:
        """Attempt to use pinneapple_meta.MAMLTrainer or ReptileTrainer."""
        try:
            from pinneapple_adaptation.meta_learning import (  # type: ignore
                MAMLTrainer, ReptileTrainer, PDETaskSampler,
            )
        except ImportError:
            return None

        cfg = self.config

        # Build task sampler from catalog
        datasets_by_scenario = catalog.datasets_by_scenario()
        tasks: List[Any] = []
        for scenario, datasets in datasets_by_scenario.items():
            for ds in datasets:
                tasks.append(ds)

        if not tasks:
            return None

        # Build model
        ref_ds = tasks[0]
        model = self._build_model(ref_ds)

        try:
            sampler = PDETaskSampler(tasks)
        except Exception:
            return None

        TrainerCls = ReptileTrainer if cfg.algorithm != "maml" else MAMLTrainer

        try:
            trainer = TrainerCls(
                model=model,
                task_sampler=sampler,
                inner_lr=cfg.inner_lr,
                outer_lr=cfg.outer_lr,
                n_inner_steps=cfg.n_inner_steps,
                device=cfg.device,
            )
            meta_model = trainer.train(n_epochs=cfg.n_meta_epochs)
            if cfg.verbose:
                print(f"[MetaLearner] pinneapple_meta.{TrainerCls.__name__} complete.")
            return meta_model
        except Exception as exc:
            log.debug("pinneapple_meta trainer failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_model(self, ref_ds: WorldModelDataset) -> PhysicsWorldModel:
        from dataclasses import replace
        cfg = self.model_config
        if cfg.context_dim != ref_ds.context_dim:
            cfg = replace(cfg, context_dim=ref_ds.context_dim)
        return PhysicsWorldModel(
            cfg, n_fields=ref_ds.n_fields, grid_shape=ref_ds.grid_shape
        )
