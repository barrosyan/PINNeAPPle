"""Model selection helpers.

Given a dataset/task IO spec, pick candidate models from ModelRegistry.

MVP: filter by input_kind + required batch keys (expects).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pinneaple_models.registry import ModelRegistry


def select_models(
    *,
    input_kind: str,
    expects: Optional[List[str]] = None,
    supports_physics_loss: Optional[bool] = None,
    tags_any: Optional[List[str]] = None,
) -> List[str]:
    expects = list(expects or [])
    tags_any = [t.lower() for t in (tags_any or [])]

    out: List[str] = []
    for name in ModelRegistry.list():
        spec = ModelRegistry.spec(name)
        if str(spec.input_kind) != str(input_kind):
            continue
        if supports_physics_loss is not None and bool(spec.supports_physics_loss) != bool(supports_physics_loss):
            continue
        if expects and any(k not in (spec.expects or []) for k in expects):
            continue
        if tags_any:
            if not any(t in [x.lower() for x in (spec.tags or [])] for t in tags_any):
                continue
        out.append(name)
    return out
