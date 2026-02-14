from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


def ensure_float_dict(d: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            # skip non-numeric
            continue
    return out
