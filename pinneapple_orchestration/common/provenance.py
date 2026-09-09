"""pinneapple_orchestration.common.provenance — ``flow_run_manifest()``: a
context manager every flow in this package wraps its body in, writing
``<manifest_dir>/<flow_name>/<run_id>.json`` on completion (success or
failure): parameters, this repo's own git SHA at run time, a per-task
status log, and any named artifacts — independent of whatever a real
Prefect server's own database retains (and available even when Prefect
isn't installed at all, since ``_prefect_compat`` may be running the flow
as a plain function).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional


def _git_sha(repo_dir: Optional[str] = None) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


@dataclass
class FlowRunManifest:
    flow_name: str
    run_id: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    git_sha: Optional[str] = None
    started_at: str = ""
    finished_at: Optional[str] = None
    status: str = "running"
    task_log: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def log_task(self, task_name: str, status: str, **extra: Any) -> None:
        self.task_log.append({"task": task_name, "status": status, "at": datetime.now().isoformat(), **extra})

    def add_artifact(self, name: str, value: Any) -> None:
        self.artifacts[name] = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flow_name": self.flow_name, "run_id": self.run_id, "parameters": self.parameters,
            "git_sha": self.git_sha, "started_at": self.started_at, "finished_at": self.finished_at,
            "status": self.status, "task_log": self.task_log, "artifacts": self.artifacts, "error": self.error,
        }


@contextmanager
def flow_run_manifest(
    flow_name: str,
    parameters: Optional[Dict[str, Any]] = None,
    *,
    manifest_dir: str = "data/manifests",
    repo_dir: Optional[str] = None,
) -> Iterator[FlowRunManifest]:
    """Usage::

        with flow_run_manifest("flow_02_training", {"component": "PipeVanillaPINN"}) as manifest:
            manifest.log_task("build_component", "ok")
            ...
            manifest.add_artifact("model_version", version)
    """
    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    manifest = FlowRunManifest(
        flow_name=flow_name, run_id=run_id, parameters=dict(parameters or {}),
        git_sha=_git_sha(repo_dir), started_at=datetime.now().isoformat(),
    )
    t0 = time.time()
    try:
        yield manifest
        manifest.status = "success"
    except Exception as exc:
        manifest.status = "failed"
        manifest.error = str(exc)
        raise
    finally:
        manifest.finished_at = datetime.now().isoformat()
        manifest.artifacts.setdefault("duration_s", time.time() - t0)
        out_dir = os.path.join(manifest_dir, flow_name)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{run_id}.json"), "w") as f:
            json.dump(manifest.to_dict(), f, indent=2, default=str)
