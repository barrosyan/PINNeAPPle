from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_leaderboard(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return data


def update_leaderboard(path: str | Path, row: Dict[str, Any]) -> None:
    p = Path(path)
    rows = load_leaderboard(p)
    rows.append(row)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
