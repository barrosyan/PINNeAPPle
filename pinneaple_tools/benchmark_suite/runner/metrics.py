from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


def ensure_float_dict(d: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert dictionary values to floats when possible.

    Parameters
    ----------
    d : Dict[str, Any]
        Input dictionary with arbitrary value types.

    Returns
    -------
    Dict[str, float]
        New dictionary containing only the keys whose values
        could be successfully converted to float. Keys are
        coerced to strings. Non-numeric values are silently skipped.
    """
    out: Dict[str, float] = {}
    for k, v in d.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            # skip non-numeric
            continue
    return out