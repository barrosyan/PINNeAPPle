from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_leaderboard(path: str | Path) -> List[Dict[str, Any]]:
    """
    Load a leaderboard JSON file.

    Parameters
    ----------
    path : str | Path
        Path to the leaderboard JSON file.

    Returns
    -------
    List[Dict[str, Any]]
        A list of leaderboard entries. If the file does not exist
        or does not contain a valid list, an empty list is returned.
    """
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return data


def update_leaderboard(path: str | Path, row: Dict[str, Any]) -> None:
    """
    Append a new entry to the leaderboard JSON file.

    Parameters
    ----------
    path : str | Path
        Path to the leaderboard JSON file.
    row : Dict[str, Any]
        Dictionary representing a new leaderboard entry.

    Notes
    -----
    If the file does not exist, it will be created.
    Existing entries are preserved and the new row is appended.
    """
    p = Path(path)
    rows = load_leaderboard(p)
    rows.append(row)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")