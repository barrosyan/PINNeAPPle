"""Ranking heuristics: lexical overlap, freshness, quality scores."""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple

from .models import RankedItem


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "") if t.strip()]


def lexical_overlap_score(query: str, doc: str) -> float:
    """
    v0.1 lexical overlap score (0..1).
    Simple: fraction of unique query tokens present in doc tokens.
    """
    q = set(tokenize(query))
    if not q:
        return 0.0
    d = set(tokenize(doc))
    hit = len(q.intersection(d))
    return float(hit) / float(len(q))


def safe_parse_iso8601(dt: str) -> Optional[datetime]:
    if not dt or not isinstance(dt, str):
        return None
    s = dt.strip()
    try:
        # common GitHub: 2025-01-02T03:04:05Z
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        return datetime.fromisoformat(s)
    except Exception:
        return None


def freshness_score(dt: Optional[datetime], *, half_life_days: float = 365.0) -> float:
    """
    Exponential decay freshness score (0..1).
    score=1 now, ~0.5 at half_life_days, decays afterwards.
    """
    if dt is None:
        return 0.0
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now - dt).total_seconds() / (3600.0 * 24.0))
    lam = math.log(2.0) / max(1e-9, half_life_days)
    return float(math.exp(-lam * age_days))


def normalize(xs: List[float]) -> List[float]:
    if not xs:
        return xs
    mn, mx = min(xs), max(xs)
    if abs(mx - mn) < 1e-12:
        return [1.0 for _ in xs]
    return [(x - mn) / (mx - mn) for x in xs]


def sort_ranked(items: List[RankedItem]) -> List[RankedItem]:
    return sorted(items, key=lambda x: x.score, reverse=True)
