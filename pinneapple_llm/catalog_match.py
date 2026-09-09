"""pinneapple_llm.catalog_match — ``resolve_problem()``: a two-stage,
never-raises front end to ``draft_problem()``.

``draft_problem()`` already sends the *entire* preset catalog (name +
accepted kwargs) to the LLM in one shot and mechanically rejects any
hallucinated name/kwarg — safe, but it (a) sends the whole catalog on
every call even when most of it is obviously irrelevant, and (b) raises
on any failure (missing API key, network error, malformed response),
with no fallback. ``resolve_problem()`` adds:

1. A cheap keyword-overlap pre-filter narrowing the full catalog down to
   a shortlist before ever calling an LLM (cheaper prompts, and a second,
   independent signal the caller can inspect even when the LLM step
   succeeds).
2. A deterministic, LLM-free fallback — the top keyword match — used
   automatically if the LLM step fails for any reason at all.

Never raises: if you need ``draft_problem()``'s strict "raise on
hallucination" behavior instead, call that directly.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .draft import DraftResult, _preset_catalog

_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "to", "with", "and", "or",
    "is", "are", "i", "want", "need", "problem", "simulate", "solve", "model",
}


def _tokenize(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS and len(t) > 2}


@dataclass
class ResolveResult:
    preset: Optional[str]
    kwargs: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    used_llm: bool = False
    shortlist: List[str] = field(default_factory=list)


def _keyword_shortlist(description: str, catalog: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    query_tokens = _tokenize(description)
    scored = []
    for entry in catalog:
        name_tokens = _tokenize(entry["name"].replace("_", " "))
        scored.append((len(query_tokens & name_tokens), entry))
    scored.sort(key=lambda t: t[0], reverse=True)
    nonzero = [e for score, e in scored if score > 0]
    return (nonzero or [e for _, e in scored])[:top_k]


def _resolve_via_llm(
    description: str,
    shortlist: List[Dict[str, Any]],
    provider: str,
    model: Optional[str],
    api_key: Optional[str],
    conversation_store: Any,
) -> DraftResult:
    from ._dispatch import call_llm
    from .draft import _SYSTEM_PROMPT

    catalog_names = {c["name"] for c in shortlist}
    prompt = (
        f"USER REQUEST:\n{description}\n\n"
        f"AVAILABLE PRESETS (name and accepted kwargs):\n{json.dumps(shortlist, indent=2)}\n"
    )
    raw = call_llm(
        prompt, provider=provider, model=model, api_key=api_key, system=_SYSTEM_PROMPT,
        json_mode=True, module="resolve_problem", conversation_store=conversation_store,
    )
    parsed = json.loads(raw)  # ValueError/json.JSONDecodeError on malformed response — caught by the caller
    preset = parsed.get("preset")
    kwargs = parsed.get("kwargs", {}) or {}
    reasoning = parsed.get("reasoning", "")

    if preset is not None:
        # Same mechanical anti-hallucination check draft_problem() itself
        # does, scoped to the shortlist here rather than the full catalog.
        if preset not in catalog_names:
            raise ValueError(f"LLM named preset '{preset}' not in shortlist {sorted(catalog_names)}")
        accepted = next(c["accepted_kwargs"] for c in shortlist if c["name"] == preset)
        unknown = set(kwargs) - set(accepted)
        if unknown:
            raise ValueError(f"LLM proposed unknown kwargs {sorted(unknown)} for preset '{preset}'")

    return DraftResult(preset=preset, kwargs=kwargs, reasoning=reasoning, raw_response=raw)


def resolve_problem(
    description: str,
    *,
    provider: str = "anthropic",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    conversation_store: Any = None,
    top_k: int = 12,
) -> ResolveResult:
    """Resolve ``description`` to a registered PDE preset — the shortlist
    step always runs; the LLM step is attempted next and its result used
    if it succeeds, else the top keyword match is returned instead.
    Never raises."""
    catalog = _preset_catalog()
    shortlist = _keyword_shortlist(description, catalog, top_k)
    shortlist_names = [e["name"] for e in shortlist]

    try:
        result = _resolve_via_llm(description, shortlist, provider, model, api_key, conversation_store)
        return ResolveResult(
            preset=result.preset, kwargs=result.kwargs, reasoning=result.reasoning,
            used_llm=True, shortlist=shortlist_names,
        )
    except Exception as exc:
        top = shortlist_names[0] if shortlist_names else None
        return ResolveResult(
            preset=top, kwargs={},
            reasoning=f"LLM unavailable/failed ({exc}); falling back to top keyword match.",
            used_llm=False, shortlist=shortlist_names,
        )
