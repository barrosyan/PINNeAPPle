from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class RunResult:
    ok: bool
    step: str
    returncode: int
    stdout: str
    stderr: str


def _run(cmd: List[str], cwd: str, timeout_s: int) -> RunResult:
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        ok = (p.returncode == 0)
        return RunResult(ok=ok, step=" ".join(cmd), returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except subprocess.TimeoutExpired as e:
        return RunResult(ok=False, step=" ".join(cmd), returncode=124, stdout=e.stdout or "", stderr=str(e))


def project_tree(cwd: str, max_files: int = 400) -> str:
    root = Path(cwd)
    paths = []
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        rel = str(p.relative_to(root))
        if any(rel.endswith(ext) for ext in (".pyc", ".pkl", ".pt", ".pth", ".png", ".jpg", ".pdf", ".zip")):
            continue
        paths.append(rel)
        if len(paths) >= max_files:
            break
    return "\n".join(paths)


def py_compile_all(cwd: str) -> RunResult:
    root = Path(cwd)

    py_files = [str(p.relative_to(root)) for p in root.rglob("*.py")]

    if not py_files:
        return RunResult(ok=True, step="py_compile (no files)", returncode=0, stdout="", stderr="")

    cmd = ["python", "-m", "py_compile", *py_files]
    return _run(cmd, cwd=cwd, timeout_s=120)

def smoke_run(cwd: str, timeout_s: int = 180) -> RunResult:
    # Prefer run.py if present
    rp = Path(cwd) / "run.py"
    if rp.exists():
        return _run(["python", "run.py"], cwd=cwd, timeout_s=timeout_s)

    # Else try main.py
    mp = Path(cwd) / "main.py"
    if mp.exists():
        return _run(["python", "main.py"], cwd=cwd, timeout_s=timeout_s)

    # Else no runnable entry
    return RunResult(ok=True, step="smoke (no entrypoint)", returncode=0, stdout="", stderr="")
