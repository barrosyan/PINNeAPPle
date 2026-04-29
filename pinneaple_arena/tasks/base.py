from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from pinneaple_arena.bundle.loader import BundleData


@dataclass(frozen=True)
class TaskResult:
    """Standard output for a task evaluation."""
    metrics: Dict[str, float]
    artifacts: Optional[Dict[str, Any]] = None


class ArenaTask:
    """Base class for Arena tasks."""
    task_id: str = "base"

    def compute_metrics(self, bundle: BundleData, backend_outputs: Dict[str, Any]) -> Dict[str, float] | TaskResult:
        import numpy as np

        def _to_numpy(val):
            if hasattr(val, "detach"):
                val = val.detach().cpu().numpy()
            return np.asarray(val, dtype=np.float64)

        if isinstance(backend_outputs, dict):
            if not backend_outputs:
                return {"l2_rel": float("nan"), "linf": float("nan")}
            y_pred = _to_numpy(next(iter(backend_outputs.values())))
        else:
            y_pred = _to_numpy(backend_outputs)

        bundle_dict = bundle if isinstance(bundle, dict) else vars(bundle)
        y_true = None
        for key in ("y_true", "u_true"):
            if key in bundle_dict:
                y_true = _to_numpy(bundle_dict[key])
                break
        if y_true is None:
            skip_keys = {"x", "t"}
            for key, val in bundle_dict.items():
                if key not in skip_keys:
                    y_true = _to_numpy(val)
                    break

        if y_true is None:
            return {"l2_rel": float("nan"), "linf": float("nan")}

        diff = y_pred - y_true
        norm_true = np.linalg.norm(y_true.ravel())
        l2_rel = float(np.linalg.norm(diff.ravel()) / norm_true) if norm_true != 0.0 else float("nan")
        linf = float(np.max(np.abs(diff)))
        return {"l2_rel": l2_rel, "linf": linf}