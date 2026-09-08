"""pinneapple_registry.dataset_store — ``DatasetStore``: local, per-
problem/scenario storage for simulation datasets (coords + field arrays),
with auto-computed summary statistics.

Not a replacement for ``pinneapple_data``'s Zarr-sharded UPD pipeline
(that is the right tool for large, shardable, streaming datasets) — this
is the lightweight complement for "one simulation run's worth of arrays,
versioned by problem+scenario", the shape ``pinneapple_registry``'s other
stores (model/experiment) already assume.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


class DatasetStore:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _scenario_dir(self, problem_id: str, scenario_id: str) -> str:
        return os.path.join(self.root, problem_id, scenario_id)

    def save(
        self,
        problem_id: str,
        scenario_id: str,
        coords: np.ndarray,
        fields: Dict[str, np.ndarray],
        *,
        observations: Optional[np.ndarray] = None,
        observation_columns: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        d = self._scenario_dir(problem_id, scenario_id)
        os.makedirs(d, exist_ok=True)

        np.savez_compressed(os.path.join(d, "simulation.npz"), coords=coords, **fields)

        if observations is not None:
            import csv
            cols = observation_columns or [f"col{i}" for i in range(observations.shape[1])]
            with open(os.path.join(d, "observations.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(cols)
                writer.writerows(observations.tolist())

        stats = {
            name: {
                "min": float(np.min(arr)), "max": float(np.max(arr)),
                "mean": float(np.mean(arr)), "std": float(np.std(arr)),
                "shape": list(arr.shape),
            }
            for name, arr in {"coords": coords, **fields}.items()
        }

        meta = dict(metadata or {})
        meta.update({
            "problem_id": problem_id, "scenario_id": scenario_id,
            "created_at": datetime.now().isoformat(),
            "field_names": list(fields.keys()),
            "stats": stats,
        })
        with open(os.path.join(d, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return d

    def load(self, problem_id: str, scenario_id: str) -> Dict[str, Any]:
        d = self._scenario_dir(problem_id, scenario_id)
        if not os.path.isdir(d):
            raise KeyError(f"No dataset for problem_id='{problem_id}', scenario_id='{scenario_id}'")
        npz = np.load(os.path.join(d, "simulation.npz"))
        out: Dict[str, Any] = {"coords": npz["coords"], "fields": {k: npz[k] for k in npz.files if k != "coords"}}
        with open(os.path.join(d, "metadata.json")) as f:
            out["metadata"] = json.load(f)
        obs_path = os.path.join(d, "observations.csv")
        if os.path.exists(obs_path):
            import csv
            with open(obs_path) as f:
                reader = csv.reader(f)
                header = next(reader)
                out["observations"] = {"columns": header, "rows": [row for row in reader]}
        return out

    def list_scenarios(self, problem_id: str) -> List[str]:
        d = os.path.join(self.root, problem_id)
        if not os.path.isdir(d):
            return []
        return sorted(s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s)))

    def stats(self, problem_id: str, scenario_id: str) -> Dict[str, Any]:
        with open(os.path.join(self._scenario_dir(problem_id, scenario_id), "metadata.json")) as f:
            return json.load(f)["stats"]
