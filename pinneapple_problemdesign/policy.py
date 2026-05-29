"""Non-invention policy: apply assumptions and gaps to design state."""
from __future__ import annotations

from typing import Any, Dict, List

from .schema import Gap, Assumption


def _dedup_by_id(gaps: List[Gap]) -> List[Gap]:
    out = []
    seen = set()
    for g in gaps:
        if g.id in seen:
            continue
        seen.add(g.id)
        out.append(g)
    return out


def apply_non_invention_policy(
    state,
    *,
    unknown_fields: List[str],
    assumptions_suggested: List[str],
    gaps_suggested: List[Dict[str, Any]],
) -> None:
    # Suggested assumptions become explicit assumptions requiring confirmation
    for s in assumptions_suggested:
        if not s or not isinstance(s, str):
            continue
        state.spec.assumptions.append(
            Assumption(text=s.strip(), confidence=0.3, needs_confirmation=True)
        )

    # Unknown important fields become gaps
    for uf in unknown_fields:
        if not uf or not isinstance(uf, str):
            continue
        gid = f"unknown.{uf.strip()}"
        state.gaps.append(
            Gap(
                id=gid,
                question=f"Please clarify: '{uf.strip()}' is not specified. What is it?",
                severity="important",
                rationale="This field is needed to complete the problem specification without guessing.",
                how_to_obtain="Provide a concrete value/definition (names, units, policy, or constraints).",
            )
        )

    # LLM-suggested gaps (if present) are added but never marked resolved automatically
    for g in gaps_suggested:
        try:
            state.gaps.append(
                Gap(
                    id=str(g.get("id", "")).strip() or "gap.unnamed",
                    question=str(g.get("question", "")).strip() or "Clarify missing requirement.",
                    severity=str(g.get("severity", "important")).strip() or "important",
                    rationale=str(g.get("rationale", "")).strip(),
                    how_to_obtain=str(g.get("how_to_obtain", "")).strip(),
                    resolved=False,
                )
            )
        except Exception:
            continue

    state.gaps = _dedup_by_id(state.gaps)
