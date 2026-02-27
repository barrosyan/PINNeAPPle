"""03 — Custom backend that only exposes `predict_fn`.

This shows a lightweight integration path:
- You don't need to return a torch model.
- If you return a numpy callable `predict_fn(xy)->uvp`, the task can still
  compute BC + (optional) sensor metrics.

Run from repo root:
    python examples/arena/03_custom_predict_backend.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from pinneaple_arena.bundle.loader import BundleData
from pinneaple_arena.registry import register_backend
from pinneaple_arena.runner.run_benchmark import run_benchmark


@register_backend
class ConstantFieldBackend:
    """A silly backend: u=1, v=0, p=0 everywhere.

Useful as a baseline to validate that:
- the arena can run with *predict_fn-only* backends
- metrics/leaderboard/artifacts still work
"""

    name = "constant_field"

    def train(self, bundle: BundleData, run_cfg: Dict[str, Any]) -> Dict[str, Any]:
        def predict_fn(xy: np.ndarray) -> np.ndarray:
            xy = np.asarray(xy, dtype=np.float32)
            out = np.zeros((xy.shape[0], 3), dtype=np.float32)
            out[:, 0] = 1.0  # u
            out[:, 1] = 0.0  # v
            out[:, 2] = 0.0  # p
            return out

        return {
            "predict_fn": predict_fn,
            "metrics": {"backend": "constant_field"},
        }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    artifacts_dir = repo_root / "artifacts" / "examples" / "custom_predict_backend"

    # We create a run config dynamically (so the example is self-contained)
    run_cfg = artifacts_dir / "run_constant_field.yaml"
    run_cfg.parent.mkdir(parents=True, exist_ok=True)
    run_cfg.write_text(
        "\n".join(
            [
                "run_name: constant_field_demo",
                "",
                "backend:",
                "  name: constant_field",
                "",
            ]
        ),
        encoding="utf-8",
    )

    out = run_benchmark(
        artifacts_dir=artifacts_dir,
        task_cfg_path=repo_root / "examples" / "arena" / "configs" / "task_flow_obstacle_2d.yaml",
        run_cfg_path=run_cfg,
        bundle_schema_path=repo_root / "configs" / "data" / "bundle_schema.yaml",
    )

    print("\n=== DONE ===")
    print("summary:", out["summary"]["key_metrics"])
    print("(expect PDE metrics = NaN, because predict_fn has no gradients)")
    print("leaderboard:", artifacts_dir / "leaderboard.json")


if __name__ == "__main__":
    main()
