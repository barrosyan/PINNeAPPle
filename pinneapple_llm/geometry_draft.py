"""LLM-assisted geometry drafting -- same constrained pattern as
``draft.py``'s ``draft_problem``, applied to CAD/domain generation instead
of PDE problem specs: the LLM selects and parametrises one of
``pinneapple_design.geometry``'s own already-implemented generators
(``list_domains()``/``list_domains_3d()``'s named physics domains, or
``naca_parametric`` for an airfoil profile), never free-form CAD code or
mesh coordinates it invented itself. Every choice is checked against the
real registry/signature before being returned, exactly like
``draft_problem``.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ._dispatch import call_llm
from .draft import _SYSTEM_PROMPT as _BASE_SYSTEM_PROMPT

_SYSTEM_PROMPT = _BASE_SYSTEM_PROMPT.replace(
    '"preset"', '"generator"'
).replace(
    "physics problem preset", "geometry generator"
)


@dataclass
class GeometryDraftResult:
    generator: Optional[str]
    kind: str  # "domain2d" | "domain3d" | "airfoil"
    kwargs: Dict[str, Any]
    reasoning: str
    raw_response: str


def _geometry_catalog() -> List[Dict[str, Any]]:
    from pinneapple_design.geometry import list_domains, list_domains_3d, naca_parametric

    catalog = []
    for name in list_domains():
        catalog.append({"name": name, "kind": "domain2d", "accepted_kwargs": ["<see get_domain(name) docstring>"]})
    for name in list_domains_3d():
        catalog.append({"name": name, "kind": "domain3d", "accepted_kwargs": ["<see get_domain_3d(name) docstring>"]})
    sig = inspect.signature(naca_parametric)
    catalog.append({
        "name": "naca_parametric", "kind": "airfoil",
        "accepted_kwargs": [p for p in sig.parameters],
    })
    return catalog


def draft_geometry(
    description: str,
    *,
    provider: str = "anthropic",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    conversation_store=None,
) -> GeometryDraftResult:
    """Ask an LLM to pick + parametrise a geometry generator for a
    natural-language description (e.g. "a NACA 2412 airfoil" or "a lid
    driven cavity in 3D").

    Does not build the geometry itself -- once you have the result, call
    the real generator yourself: ``get_domain(result.generator, **result
    .kwargs)`` / ``get_domain_3d(...)`` / ``naca_parametric(**result
    .kwargs)`` depending on ``result.kind``. See ``draft_problem``'s
    docstring for why this module stops short of chaining straight into
    use without that look.

    Raises
    ------
    ValueError
        If the LLM names a generator or kwarg not in the real catalog
        (``naca_parametric``'s kwargs ARE checked against its actual
        signature; a named domain's kwargs are not further checked here
        since ``get_domain``/``get_domain_3d`` validate their own kwargs
        when called -- this still guarantees the *name* itself is real).
    """
    catalog = _geometry_catalog()
    catalog_names = {c["name"] for c in catalog}

    prompt = (
        f"USER REQUEST:\n{description}\n\n"
        f"AVAILABLE GEOMETRY GENERATORS:\n{json.dumps(catalog, indent=2)}\n"
    )

    raw = call_llm(
        prompt, provider=provider, model=model, api_key=api_key, system=_SYSTEM_PROMPT,
        json_mode=True, module="draft_geometry", conversation_store=conversation_store,
    )

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON:\n{raw}") from e

    name = parsed.get("generator")
    kwargs = parsed.get("kwargs", {}) or {}
    reasoning = parsed.get("reasoning", "")
    kind = ""

    if name is not None:
        if name not in catalog_names:
            raise ValueError(
                f"LLM named generator '{name}', not in the real catalog ({sorted(catalog_names)}) "
                "-- refusing to use a hallucinated generator name."
            )
        entry = next(c for c in catalog if c["name"] == name)
        kind = entry["kind"]
        if kind == "airfoil":
            unknown = set(kwargs) - set(entry["accepted_kwargs"])
            if unknown:
                raise ValueError(
                    f"LLM proposed kwargs {sorted(unknown)} that '{name}' does not accept "
                    f"(accepted: {entry['accepted_kwargs']})."
                )

    return GeometryDraftResult(generator=name, kind=kind, kwargs=kwargs, reasoning=reasoning, raw_response=raw)
