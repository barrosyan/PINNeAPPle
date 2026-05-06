"""Specialist model registry and checkpoint management.

:class:`ModelZoo` stores trained *specialist* physics models — one model per
physics domain (heat, Burgers, Navier-Stokes, …) — and exposes a unified API
for loading, querying, and ensembling them.

Each entry in the zoo is a :class:`ZooEntry` that bundles the model, its
training metadata, physics tags, and performance metrics.  The zoo supports:

* **Register** — add a new specialist after training.
* **Query** — retrieve by scenario name, physics tag, or performance threshold.
* **Load / save** — persist the full zoo to disk (PyTorch checkpoints).
* **Ensemble** — combine multiple specialists via parameter averaging or
  inference voting for a quick multi-domain ensemble.
* **Adapter hooks** — slot any ``pinneaple_models`` architecture
  (SIREN, AFNO, MeshGraphNet, …) alongside the default FNO-based model.

Quick start::

    from pinneaple_worldmodel.model_zoo import ModelZoo, ZooEntry

    zoo = ModelZoo.load("./zoo_checkpoints")
    heat_model = zoo.get("heat_2d")
    ns_models   = zoo.by_tag("navier_stokes")
    ensemble    = zoo.ensemble(tags=["2d"])
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .model import PhysicsWorldModel, WorldModelConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ZooEntry
# ---------------------------------------------------------------------------

@dataclass
class ZooEntry:
    """One entry in the :class:`ModelZoo`.

    Parameters
    ----------
    name : str — unique identifier (typically scenario name).
    model : PhysicsWorldModel
    scenario : str — primary scenario this model was trained on.
    physics_tags : list of str — e.g. ``["diffusion", "2d", "parabolic"]``.
    metrics : dict — validation metrics from specialist training.
    n_params : int
    trained_at : float — Unix timestamp of when training finished.
    checkpoint_path : str or None — path to saved ``.pt`` file (if persisted).
    metadata : dict — any extra info (architecture variant, dataset size, …).
    """
    name: str
    model: PhysicsWorldModel
    scenario: str
    physics_tags: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    n_params: int = 0
    trained_at: float = field(default_factory=time.time)
    checkpoint_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_params == 0:
            self.n_params = self.model.parameter_count()


# ---------------------------------------------------------------------------
# EnsembleModel
# ---------------------------------------------------------------------------

class EnsembleModel(nn.Module):
    """Averaging ensemble of multiple PhysicsWorldModel specialists.

    Parameters
    ----------
    models : list of PhysicsWorldModel — constituent specialists.
    weights : optional list of floats — per-model weights (default: uniform).
    aggregation : ``"mean"`` or ``"vote"`` (element-wise majority).
    """

    def __init__(
        self,
        models: List[PhysicsWorldModel],
        weights: Optional[List[float]] = None,
        aggregation: str = "mean",
    ) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]
        self.aggregation = aggregation

    def forward(self, state: Tensor, context: Optional[Tensor] = None) -> Tensor:
        preds = [m(state, context) for m in self.models]
        if self.aggregation == "mean":
            stacked = torch.stack(preds, dim=0)  # (K, B, C, *grid)
            w = torch.tensor(self.weights, device=state.device).view(-1, *([1] * (stacked.dim() - 1)))
            return (stacked * w).sum(0)
        else:
            stacked = torch.stack(preds, dim=0)
            return stacked.median(0).values

    def parameter_count(self) -> int:
        return sum(m.parameter_count() for m in self.models)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# WaMAModel — parameter-averaged model (Weights Averaging / Model Merging)
# ---------------------------------------------------------------------------

class WaMaModel(PhysicsWorldModel):
    """Model created by averaging the weights of multiple specialist models.

    This is the Weights Averaging (WaMA / model soup) technique: simply
    average the parameters of all constituent models.  Produces better
    generalisation than any single specialist when the specialists are
    sufficiently diverse.

    Parameters
    ----------
    base_config : WorldModelConfig — shared architecture (must match all specialists).
    n_fields : int
    grid_shape : tuple
    specialists : list of PhysicsWorldModel — models to average.
    weights : optional per-model float weights.
    """

    @classmethod
    def from_specialists(
        cls,
        specialists: List[PhysicsWorldModel],
        weights: Optional[List[float]] = None,
    ) -> "WaMaModel":
        if not specialists:
            raise ValueError("Need at least one specialist.")

        if weights is None:
            weights = [1.0 / len(specialists)] * len(specialists)
        total = sum(weights)
        weights = [w / total for w in weights]

        ref = specialists[0]
        model = cls(ref.config, n_fields=ref.n_fields, grid_shape=ref.grid_shape)
        avg_sd: Dict[str, Tensor] = {}

        for key in ref.state_dict():
            avg_sd[key] = sum(
                w * s.state_dict()[key].float()
                for w, s in zip(weights, specialists)
            )

        model.load_state_dict(avg_sd)  # type: ignore[arg-type]
        return model


# ---------------------------------------------------------------------------
# ModelZoo
# ---------------------------------------------------------------------------

class ModelZoo:
    """Registry of trained specialist physics models.

    Parameters
    ----------
    root_dir : str or None — optional default directory for save/load.

    Attributes
    ----------
    entries : dict mapping name → ZooEntry.

    Example
    -------
    >>> zoo = ModelZoo()
    >>> zoo.register(entry)
    >>> heat = zoo.get("heat_2d")
    >>> zoo.save("./zoo_checkpoints")
    >>> zoo2 = ModelZoo.load("./zoo_checkpoints")
    """

    def __init__(self, root_dir: Optional[str] = None) -> None:
        self.root_dir = root_dir
        self.entries: Dict[str, ZooEntry] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, entry: ZooEntry, *, overwrite: bool = False) -> None:
        """Add a new entry to the zoo.

        Parameters
        ----------
        entry : ZooEntry
        overwrite : bool — if False and name already exists, raises ValueError.
        """
        if entry.name in self.entries and not overwrite:
            raise ValueError(
                f"Zoo already has a model named '{entry.name}'. "
                f"Use overwrite=True to replace it."
            )
        self.entries[entry.name] = entry
        log.info("Registered zoo entry '%s' (%d params, tags=%s)",
                 entry.name, entry.n_params, entry.physics_tags)

    def deregister(self, name: str) -> None:
        self.entries.pop(name, None)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, name: str) -> PhysicsWorldModel:
        """Return the model for *name* (raises KeyError if not found)."""
        return self.entries[name].model

    def get_entry(self, name: str) -> ZooEntry:
        return self.entries[name]

    def by_tag(self, tag: str) -> List[ZooEntry]:
        return [e for e in self.entries.values() if tag in e.physics_tags]

    def by_scenario(self, scenario: str) -> List[ZooEntry]:
        return [e for e in self.entries.values() if e.scenario == scenario]

    def best_by_metric(self, metric: str, *, lower_is_better: bool = True) -> ZooEntry:
        """Return the entry with the best value for *metric*."""
        candidates = [e for e in self.entries.values() if metric in e.metrics]
        if not candidates:
            raise ValueError(f"No entry has metric '{metric}'.")
        return min(candidates, key=lambda e: e.metrics[metric]) if lower_is_better \
            else max(candidates, key=lambda e: e.metrics[metric])

    def list_names(self) -> List[str]:
        return list(self.entries.keys())

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def __iter__(self):
        return iter(self.entries.values())

    # ------------------------------------------------------------------
    # Ensemble
    # ------------------------------------------------------------------

    def ensemble(
        self,
        names: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        *,
        aggregation: str = "mean",
        metric_weights: Optional[str] = None,
        lower_is_better: bool = True,
    ) -> EnsembleModel:
        """Create an ensemble from a subset of zoo specialists.

        Parameters
        ----------
        names : specific model names to include (overrides tags).
        tags : filter by physics tags — includes entries matching ANY tag.
        aggregation : ``"mean"`` or ``"vote"``.
        metric_weights : name of a metric to use as per-model weight.
            If provided the entry's metric value is converted to a weight
            (higher metric → higher weight for ``lower_is_better=False``).
        lower_is_better : direction for metric_weights.

        Returns
        -------
        EnsembleModel
        """
        if names is not None:
            entries = [self.entries[n] for n in names if n in self.entries]
        elif tags is not None:
            seen = set()
            entries = []
            for tag in tags:
                for e in self.by_tag(tag):
                    if e.name not in seen:
                        entries.append(e)
                        seen.add(e.name)
        else:
            entries = list(self.entries.values())

        if not entries:
            raise ValueError("No entries match the requested filter.")

        models = [e.model for e in entries]

        weights = None
        if metric_weights is not None:
            raw = [e.metrics.get(metric_weights, 1.0) for e in entries]
            if lower_is_better:
                # invert so smaller error → higher weight
                safe = [1.0 / (v + 1e-8) for v in raw]
            else:
                safe = raw
            weights = safe

        return EnsembleModel(models, weights=weights, aggregation=aggregation)

    # ------------------------------------------------------------------
    # Weight averaging (model soup)
    # ------------------------------------------------------------------

    def soup(
        self,
        names: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        *,
        metric_weights: Optional[str] = None,
        lower_is_better: bool = True,
    ) -> WaMaModel:
        """Create a weight-averaged model from selected specialists.

        Parameters
        ----------
        names / tags : same as :meth:`ensemble`.
        metric_weights : optional metric for per-model weighting.
        lower_is_better : direction for metric_weights.

        Returns
        -------
        WaMaModel
        """
        if names is not None:
            entries = [self.entries[n] for n in names if n in self.entries]
        elif tags is not None:
            seen: set = set()
            entries = []
            for tag in tags:
                for e in self.by_tag(tag):
                    if e.name not in seen:
                        entries.append(e)
                        seen.add(e.name)
        else:
            entries = list(self.entries.values())

        if not entries:
            raise ValueError("No entries match the requested filter.")

        models = [e.model for e in entries]
        weights = None
        if metric_weights is not None:
            raw = [e.metrics.get(metric_weights, 1.0) for e in entries]
            if lower_is_better:
                safe = [1.0 / (v + 1e-8) for v in raw]
            else:
                safe = raw
            weights = safe

        return WaMaModel.from_specialists(models, weights=weights)

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def save(self, directory: Optional[str] = None) -> Path:
        """Save all models and index to *directory*.

        Structure::

            directory/
                index.json          ← metadata for all entries
                <name>.pt           ← model checkpoint for each entry

        Returns
        -------
        Path — the directory path.
        """
        root = Path(directory or self.root_dir or "./model_zoo")
        root.mkdir(parents=True, exist_ok=True)

        index = {}
        for name, entry in self.entries.items():
            ckpt_path = root / f"{name}.pt"
            torch.save({
                "model_state": entry.model.state_dict(),
                "model_config": entry.model.config,
                "n_fields": entry.model.n_fields,
                "grid_shape": entry.model.grid_shape,
                "scenario": entry.scenario,
                "physics_tags": entry.physics_tags,
                "metrics": entry.metrics,
                "n_params": entry.n_params,
                "trained_at": entry.trained_at,
                "metadata": entry.metadata,
            }, ckpt_path)
            entry.checkpoint_path = str(ckpt_path)
            index[name] = {
                "checkpoint": str(ckpt_path),
                "scenario": entry.scenario,
                "physics_tags": entry.physics_tags,
                "metrics": entry.metrics,
                "n_params": entry.n_params,
                "trained_at": entry.trained_at,
                "metadata": entry.metadata,
            }

        with open(root / "index.json", "w") as f:
            json.dump(index, f, indent=2)

        log.info("Zoo saved to %s (%d entries)", root, len(self.entries))
        return root

    @classmethod
    def load(
        cls,
        directory: str,
        *,
        map_location: str = "cpu",
        names: Optional[List[str]] = None,
    ) -> "ModelZoo":
        """Load a zoo from *directory*.

        Parameters
        ----------
        directory : str — path saved by :meth:`save`.
        map_location : str — passed to torch.load.
        names : optional subset of model names to load (default: all).

        Returns
        -------
        ModelZoo
        """
        root = Path(directory)
        index_path = root / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"No zoo index at {index_path}")

        with open(index_path) as f:
            index = json.load(f)

        zoo = cls(root_dir=str(root))
        for name, meta in index.items():
            if names is not None and name not in names:
                continue
            ckpt_path = Path(meta["checkpoint"])
            if not ckpt_path.exists():
                log.warning("Checkpoint missing: %s", ckpt_path)
                continue
            ckpt = torch.load(str(ckpt_path), map_location=map_location)
            model = PhysicsWorldModel(
                ckpt["model_config"],
                n_fields=ckpt["n_fields"],
                grid_shape=ckpt["grid_shape"],
            )
            model.load_state_dict(ckpt["model_state"])
            entry = ZooEntry(
                name=name,
                model=model,
                scenario=ckpt.get("scenario", name),
                physics_tags=ckpt.get("physics_tags", []),
                metrics=ckpt.get("metrics", {}),
                n_params=ckpt.get("n_params", model.parameter_count()),
                trained_at=ckpt.get("trained_at", 0.0),
                checkpoint_path=str(ckpt_path),
                metadata=ckpt.get("metadata", {}),
            )
            zoo.entries[name] = entry
            log.info("Loaded '%s' from %s", name, ckpt_path)

        return zoo

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        print(f"\n[ModelZoo] {len(self.entries)} specialist(s):")
        for name, entry in self.entries.items():
            metrics_str = ", ".join(f"{k}={v:.4g}" for k, v in entry.metrics.items())
            print(f"  {name:25s}  tags={entry.physics_tags}  "
                  f"params={entry.n_params:,}  metrics=[{metrics_str}]")
        print()
