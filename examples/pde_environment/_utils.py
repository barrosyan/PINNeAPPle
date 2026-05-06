from __future__ import annotations

from pathlib import Path
import sys


def ensure_repo_on_path() -> Path:
    """Best-effort: add repo root to sys.path if PINNeAPPle isn't installed editable."""
    here = Path(__file__).resolve()
    # expected layout: <repo>/.../examples_env/_utils.py
    candidates = [here.parents[i] for i in range(1, min(6, len(here.parents)))]
    for c in candidates:
        if (c / 'pyproject.toml').exists() and (c / 'pinneaple_environment').exists():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            return c
    # fallback: current working dir
    cwd = Path.cwd().resolve()
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))
    return cwd
