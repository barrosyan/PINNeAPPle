from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Dict

from pinneaple_arena.io.yamlx import load_yaml
from pinneaple_arena.bundle.schema import load_bundle_schema
from pinneaple_arena.bundle.loader import load_bundle
from pinneaple_arena.runner.metrics import ensure_float_dict
from pinneaple_arena.runner.report import write_run_artifacts
from pinneaple_arena.runner.leaderboard import update_leaderboard
from pinneaple_arena.registry import get_backend, get_task

# garante registro via decorators
from pinneaple_arena import tasks as _tasks  # noqa: F401
from pinneaple_arena import backends as _backends  # noqa: F401


def _make_run_id(task_id: str, run_name: str) -> str:
    t = str(time.time()).encode("utf-8")
    h = hashlib.sha1(t + task_id.encode("utf-8") + run_name.encode("utf-8")).hexdigest()[:12]
    return f"{task_id}-{run_name}-{h}"


def _safe_git_sha(repo_root: Path) -> str | None:
    try:
        head = repo_root / ".git" / "HEAD"
        if not head.exists():
            return None
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref:"):
            ref_path = repo_root / ".git" / ref.split(" ", 1)[1].strip()
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()[:12]
            return None
        return ref[:12]
    except Exception:
        return None


def run_benchmark(
    *,
    artifacts_dir: str | Path,
    task_cfg_path: str | Path,
    run_cfg_path: str | Path,
    bundle_schema_path: str | Path,
) -> Dict[str, Any]:
    artifacts_dir = str(artifacts_dir)

    task_cfg = load_yaml(task_cfg_path)
    run_cfg = load_yaml(run_cfg_path)
    schema = load_bundle_schema(bundle_schema_path)

    task_id = str(task_cfg.get("task_id", "flow_obstacle_2d"))
    bundle_root = str(task_cfg.get("bundle_root"))
    require_sensors = bool(task_cfg.get("require_sensors", False))
    run_name = str(run_cfg.get("run_name", Path(run_cfg_path).stem))

    if not bundle_root:
        raise RuntimeError("task config must define bundle_root")

    bundle = load_bundle(bundle_root, schema=schema, require_sensors=require_sensors)

    task_cls = get_task(task_id)
    task = task_cls()

    backend_name = str(run_cfg.get("backend", {}).get("name", "pinneaple_native"))
    backend_cls = get_backend(backend_name)
    backend = backend_cls()

    run_id = _make_run_id(task_id, run_name)

    backend_outputs = backend.train(bundle, run_cfg)
    task_metrics = task.compute_metrics(bundle, backend_outputs)

    merged: Dict[str, float] = {}
    merged.update(ensure_float_dict(backend_outputs.get("metrics", {})))
    merged.update(ensure_float_dict(task_metrics))

    report = {
        "run_id": run_id,
        "run_name": run_name,
        "task_id": task_id,
        "backend": backend_name,
        "bundle_root": bundle_root,
        "timestamp_unix": time.time(),
        "run_cfg_path": str(run_cfg_path),
        "task_cfg_path": str(task_cfg_path),
        "bundle_schema_version": getattr(schema, "schema_version", ""),
        "git_sha": _safe_git_sha(Path.cwd()),
        "python": __import__("sys").version.split()[0],
    }

    summary = {
        "run_id": run_id,
        "run_name": run_name,
        "task_id": task_id,
        "backend": backend_name,
        "key_metrics": {
            "test_pde_rms": merged.get("test_pde_rms", float("nan")),
            "test_div_rms": merged.get("test_div_rms", float("nan")),
            "bc_mse": merged.get("bc_mse", float("nan")),
            "test_l2_uv": merged.get("test_l2_uv", float("nan")),
        },
    }

    run_dir = write_run_artifacts(
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        report=report,
        metrics=merged,
        summary=summary,
    )

    leaderboard_row = {
        "run_id": run_id,
        "run_name": run_name,
        "task_id": task_id,
        "backend": backend_name,
        **summary["key_metrics"],
    }
    update_leaderboard(Path(artifacts_dir) / "leaderboard.json", leaderboard_row)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "report": report,
        "metrics": merged,
        "summary": summary,
    }