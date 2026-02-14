from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


def _write_json(path: Path, obj: Any) -> None:
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
    root = Path(artifacts_dir) / "runs" / run_id
    _write_json(root / "report.json", report)
    _write_json(root / "metrics.json", metrics)
    _write_json(root / "summary.json", summary)
    return root
