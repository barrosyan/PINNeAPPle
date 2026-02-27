from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List


def load_leaderboard(path: str | Path) -> List[Dict[str, Any]]:
    """
    Load a leaderboard JSON file.

    Returns [] if the file does not exist, is empty, or is invalid JSON.
    """
    p = Path(path)
    if not p.exists():
        return []

    txt = p.read_text(encoding="utf-8").strip()
    if not txt:
        return []

    try:
        data = json.loads(txt)
    except JSONDecodeError:
        # corrupted/partial file -> ignore to keep sweep running
        return []

    if not isinstance(data, list):
        return []
    # keep only dict rows
    return [r for r in data if isinstance(r, dict)]


def update_leaderboard(path: str | Path, row: Dict[str, Any]) -> None:
    """
    Append a new entry to the leaderboard JSON file.
    """
    p = Path(path)
    rows = load_leaderboard(p)
    rows.append(row)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")