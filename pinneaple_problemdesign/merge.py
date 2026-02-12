from __future__ import annotations

from typing import Any, Dict
from .schema import ProblemSpec


_EMPTY = (None, "", [], {})


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for k, v in src.items():
        if v in _EMPTY:
            continue

        if k not in dst:
            dst[k] = v
            continue

        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
            continue

        # Only overwrite if destination is empty (protect confirmed user data)
        if dst.get(k) in _EMPTY:
            dst[k] = v


def merge_into_spec(spec: ProblemSpec, partial: Dict[str, Any]) -> None:
    # dataclass -> mutable dict view
    dst = spec.__dict__
    _deep_merge(dst, partial)
