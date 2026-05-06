"""Run a benchmark sweep over multiple models.

This module implements *between-model* parallelism. Within each model you can
still use DDP via the Trainer.

MVP strategy
------------
  - If N_GPUs >= N_models: allocate 1 GPU per model process
  - Else: queue models and run sequentially per GPU slot
  - If ddp_per_model=True: allocate a group of GPUs per model and launch with torchrun

The sweep produces one run directory per model.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _available_gpus() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class SweepJob:
    model_name: str
    run_cfg: Dict[str, Any]


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def run_sweep(
    *,
    models: Sequence[Dict[str, Any]],
    base_run_cfg: Dict[str, Any],
    bundle_root: str,
    out_dir: str = "runs/arena_sweep",
    parallelism: str = "process",  # process | sequential
    gpus: str | int = "auto",
    ddp_per_model: bool = False,
    ddp_world_size: int = 2,
    python_exe: str = "python",
) -> List[Path]:
    """Run a sweep.

    Parameters
    ----------
    models:
        List of dicts, each at least {"name": "...", "model": {...}}.
        Anything else is merged into run_cfg.
    base_run_cfg:
        Baseline config shared by all models.
    bundle_root:
        Path to an Arena bundle directory on disk.
    out_dir:
        Parent directory for model runs.
    gpus:
        "auto" or int (#gpus to use).
    ddp_per_model:
        If True, allocate ddp_world_size GPUs per model and launch using torchrun.
    """

    out_root = _ensure_dir(out_dir)

    n_gpu_total = _available_gpus() if gpus == "auto" else int(gpus)
    if n_gpu_total < 0:
        n_gpu_total = 0

    jobs: List[SweepJob] = []
    for m in models:
        name = str(m.get("name") or m.get("model_name") or "").strip()
        if not name:
            raise ValueError("Each model entry must include 'name'.")

        cfg = json.loads(json.dumps(base_run_cfg))  # deep copy
        # merge model-specific
        cfg.setdefault("model", {})
        cfg["model"].update(dict(m.get("model", {})))
        cfg.setdefault("arena", {})
        cfg.setdefault("train", {})
        cfg.setdefault("backend", {})
        if isinstance(cfg["backend"], str):
            cfg["backend"] = {"name": cfg["backend"]}
        cfg["backend"].setdefault("name", "pinneaple_models")
        cfg["model"]["name"] = name

        # allow extra keys like train overrides
        for k, v in m.items():
            if k in ("name", "model"):
                continue
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v

        jobs.append(SweepJob(model_name=name, run_cfg=cfg))

    # --- Execution
    run_dirs: List[Path] = []

    if parallelism == "sequential" or n_gpu_total == 0:
        for job in jobs:
            run_dir = _ensure_dir(out_root / job.model_name)
            _write_json(run_dir / "run_cfg.json", job.run_cfg)
            cmd = [python_exe, "-m", "pinneaple_arena.runner.run_benchmark", "--bundle", bundle_root, "--run", str(run_dir)]
            env = os.environ.copy()
            env["PINNEAPLE_ARENA_CFG"] = str(run_dir / "run_cfg.json")
            subprocess.check_call(cmd, env=env)
            run_dirs.append(run_dir)
        return run_dirs

    # Process parallelism
    # Assign GPU slots; optionally group GPUs for ddp_per_model.
    if ddp_per_model:
        if ddp_world_size <= 0:
            raise ValueError("ddp_world_size must be >= 1")
        group = int(ddp_world_size)
        max_parallel = max(1, n_gpu_total // group)
    else:
        group = 1
        max_parallel = min(len(jobs), max(1, n_gpu_total))

    active: List[subprocess.Popen] = []
    queue = list(jobs)

    def launch(job: SweepJob, gpu_ids: List[int]):
        run_dir = _ensure_dir(out_root / job.model_name)
        _write_json(run_dir / "run_cfg.json", job.run_cfg)

        env = os.environ.copy()
        env["PINNEAPLE_ARENA_CFG"] = str(run_dir / "run_cfg.json")
        if gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)

        if ddp_per_model and len(gpu_ids) > 1:
            cmd = [
                "torchrun",
                f"--nproc_per_node={len(gpu_ids)}",
                "-m",
                "pinneaple_arena.runner.run_benchmark",
                "--bundle",
                bundle_root,
                "--run",
                str(run_dir),
            ]
            # backend will read train.ddp=True from cfg
        else:
            cmd = [python_exe, "-m", "pinneaple_arena.runner.run_benchmark", "--bundle", bundle_root, "--run", str(run_dir)]

        p = subprocess.Popen(cmd, env=env)
        return p, run_dir

    # Allocate GPU groups round-robin.
    slots: List[List[int]] = []
    if n_gpu_total > 0:
        ids = list(range(n_gpu_total))
        for i in range(max_parallel):
            slots.append(ids[i * group : (i + 1) * group])
    else:
        slots = [[] for _ in range(max_parallel)]

    run_dirs = []
    while queue or active:
        # fill slots
        while queue and len(active) < len(slots):
            job = queue.pop(0)
            slot = slots[len(active) % len(slots)]
            p, run_dir = launch(job, slot)
            active.append(p)
            run_dirs.append(run_dir)

        # wait for any to finish
        done_idx = None
        for i, p in enumerate(active):
            ret = p.poll()
            if ret is not None:
                if ret != 0:
                    raise RuntimeError(f"Sweep job failed (return code {ret}).")
                done_idx = i
                break
        if done_idx is None:
            # avoid busy loop
            import time

            time.sleep(0.2)
        else:
            active.pop(done_idx)

    return run_dirs
