"""pinneapple_registry.model_store — ``ModelStore``: a local, filesystem-
backed, versioned store for trained models, paralleling (not replacing)
``pinneapple_hub`` — see the package docstring in
``pinneapple_registry/__init__.py`` for why both exist.

Layout on disk::

    <root>/<problem_id>/v20260908_101530/model.pt
    <root>/<problem_id>/v20260908_101530/metadata.json
    <root>/<problem_id>/latest.json           -> {"version": "v20260908_101530"}
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

VALID_STAGES = ("development", "staging", "production", "archived")


class ModelStore:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    # -- paths -----------------------------------------------------------

    def _problem_dir(self, problem_id: str) -> str:
        return os.path.join(self.root, problem_id)

    def _version_dir(self, problem_id: str, version: str) -> str:
        return os.path.join(self._problem_dir(problem_id), version)

    def _latest_path(self, problem_id: str) -> str:
        return os.path.join(self._problem_dir(problem_id), "latest.json")

    # -- save / load -------------------------------------------------------

    def save(
        self,
        problem_id: str,
        model: Any,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        stage: str = "development",
        sample_input: Optional[Any] = None,
    ) -> str:
        """Save a trained model as a new timestamped version. If ``model``
        has a ``save_checkpoint`` (any ``BaseModel``, including
        ``ComponentModel``), that is used so the checkpoint stays
        reconstructable the same way ``BaseModel.load_checkpoint``/
        ``ComponentModel.load_checkpoint`` already expect; otherwise falls
        back to a bare ``state_dict``.

        Returns the new version string (e.g. ``"v20260908_101530"``).
        """
        version = f"v{datetime.now():%Y%m%d_%H%M%S}"
        vdir = self._version_dir(problem_id, version)
        os.makedirs(vdir, exist_ok=True)

        ckpt_path = os.path.join(vdir, "model.pt")
        if hasattr(model, "save_checkpoint"):
            model.save_checkpoint(ckpt_path, metadata=metadata)
        else:
            import torch
            torch.save({"state_dict": model.state_dict(), "metadata": metadata or {}}, ckpt_path)

        if stage not in VALID_STAGES:
            raise ValueError(f"stage must be one of {VALID_STAGES}, got '{stage}'")

        meta = dict(metadata or {})
        meta.update({"problem_id": problem_id, "version": version, "stage": stage,
                     "created_at": datetime.now().isoformat()})
        with open(os.path.join(vdir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        self._set_latest(problem_id, version)
        return version

    def _set_latest(self, problem_id: str, version: str) -> None:
        with open(self._latest_path(problem_id), "w") as f:
            json.dump({"version": version}, f)

    def latest(self, problem_id: str) -> Optional[str]:
        path = self._latest_path(problem_id)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)["version"]

    def versions(self, problem_id: str) -> List[str]:
        d = self._problem_dir(problem_id)
        if not os.path.isdir(d):
            return []
        return sorted(
            v for v in os.listdir(d)
            if v.startswith("v") and os.path.isdir(os.path.join(d, v))
        )

    def _resolve_version(self, problem_id: str, version: Optional[str]) -> str:
        v = version or self.latest(problem_id)
        if v is None:
            raise KeyError(f"No versions saved for problem_id='{problem_id}'")
        return v

    def metadata(self, problem_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        v = self._resolve_version(problem_id, version)
        with open(os.path.join(self._version_dir(problem_id, v), "metadata.json")) as f:
            return json.load(f)

    def checkpoint_path(self, problem_id: str, version: Optional[str] = None) -> str:
        v = self._resolve_version(problem_id, version)
        return os.path.join(self._version_dir(problem_id, v), "model.pt")

    def load(self, problem_id: str, version: Optional[str] = None, model_cls: Optional[Any] = None, **init_kwargs: Any):
        """Load a checkpoint back into a live model. If ``model_cls`` is
        given, calls ``model_cls.load_checkpoint(path, **init_kwargs)``
        (works for any ``BaseModel``/``ComponentModel`` subclass);
        otherwise returns the raw ``torch.load`` dict."""
        path = self.checkpoint_path(problem_id, version)
        if model_cls is not None:
            return model_cls.load_checkpoint(path, **init_kwargs)
        import torch
        return torch.load(path, map_location="cpu", weights_only=False)

    # -- stage promotion ---------------------------------------------------

    def promote(self, problem_id: str, version: str, stage: str) -> None:
        if stage not in VALID_STAGES:
            raise ValueError(f"stage must be one of {VALID_STAGES}, got '{stage}'")
        meta = self.metadata(problem_id, version)
        meta["stage"] = stage
        meta["promoted_at"] = datetime.now().isoformat()
        with open(os.path.join(self._version_dir(problem_id, version), "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def by_stage(self, problem_id: str, stage: str) -> List[str]:
        return [v for v in self.versions(problem_id) if self.metadata(problem_id, v).get("stage") == stage]

    # -- comparison ----------------------------------------------------

    def compare(self, problem_id: str, metric: Optional[str] = None) -> List[Dict[str, Any]]:
        """Every version's metadata, sorted by ``metric`` (ascending) if
        given and present, else by version string (== creation order,
        since versions are timestamp-sortable)."""
        rows = [self.metadata(problem_id, v) for v in self.versions(problem_id)]
        if metric:
            def _key(row: Dict[str, Any]) -> Any:
                metrics = row.get("metrics", {})
                val = metrics.get(metric, row.get(metric))
                return (val is None, val)
            rows.sort(key=_key)
        else:
            rows.sort(key=lambda r: r.get("version", ""))
        return rows

    # -- export ----------------------------------------------------------

    def export_onnx(
        self,
        problem_id: str,
        sample_input: Any,
        *,
        version: Optional[str] = None,
        model_cls: Optional[Any] = None,
        **init_kwargs: Any,
    ) -> str:
        """Exports the given (or latest) version's model to ONNX next to
        its checkpoint. Reuses ``pinneapple_tools.model_export.export_onnx``
        rather than re-implementing ONNX tracing.

        Many PINN-family models (e.g. anything built on ``PINNBase``, which
        ``ComponentModel`` wraps) return a ``PINNOutput``/``ModelOutput``
        dataclass from ``forward()``, not a plain tensor — the traced ONNX
        exporter only accepts tensors/tuples/lists as outputs. Wraps the
        model the same way ``PINNBase.physics_loss``'s internal ``_Adapter``
        already does, purely for the duration of tracing.
        """
        import torch.nn as nn
        from pinneapple_tools.model_export import export_onnx as _export_onnx

        v = self._resolve_version(problem_id, version)
        model = self.load(problem_id, v, model_cls=model_cls, **init_kwargs)

        class _TensorOutputAdapter(nn.Module):
            def __init__(self, base: Any):
                super().__init__()
                self.base = base

            def forward(self, x):
                out = self.base(x)
                return out.y if hasattr(out, "y") else out

        onnx_path = os.path.join(self._version_dir(problem_id, v), "model.onnx")
        _export_onnx(_TensorOutputAdapter(model), onnx_path, sample_input)
        return onnx_path
