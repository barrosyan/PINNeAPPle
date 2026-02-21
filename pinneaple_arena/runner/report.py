from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


def _write_json(path: Path, obj: Any) -> None:
    """
    Write a Python object to a JSON file with UTF-8 encoding.

    Parameters
    ----------
    path : Path
        Target file path.
    obj : Any
        Serializable Python object to be written as JSON.

    Notes
    -----
    Parent directories are created automatically if they do not exist.
    JSON is written with indentation and Unicode preservation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_run_artifacts(
    *,
    artifacts_dir: str | Path,
    run_id: str,
    report: Dict[str, Any],
    metrics: Dict[str, Any],
    summary: Dict[str, Any],
) -> Path:
    """
    Persist run artifacts (report, metrics, summary) to disk.

    Parameters
    ----------
    artifacts_dir : str | Path
        Base directory where artifacts are stored.
    run_id : str
        Unique identifier for the run.
    report : Dict[str, Any]
        Detailed run report dictionary.
    metrics : Dict[str, Any]
        Training or evaluation metrics dictionary.
    summary : Dict[str, Any]
        High-level summary of the run.

    Returns
    -------
    Path
        Path to the root directory containing the saved artifacts.

    Notes
    -----
    Artifacts are stored under:
        artifacts_dir/runs/{run_id}/
    """
    root = Path(artifacts_dir) / "runs" / run_id
    _write_json(root / "report.json", report)
    _write_json(root / "metrics.json", metrics)
    _write_json(root / "summary.json", summary)
    return root