"""LLM-assisted ``ProblemSpec`` drafting -- deliberately constrained to
selecting and parametrising one of PINNeAPPle's own already-implemented,
already-testable presets, never to inventing new PDE machinery from
scratch.

This constraint is the actual point, not a limitation to work around: an
LLM asked to "write CFD code" freely can silently fabricate boundary
conditions or produce code that runs while violating conservation, with no
mechanism to catch it. Restricted to *choosing among a known preset
catalog* (``pinneapple_physics.list_presets()``) and filling in the
numeric parameters a human would otherwise pass to ``get_preset(name,
**kwargs)`` by hand, the LLM's only possible failure modes are (a) picking
a preset that doesn't match the user's actual problem, or (b) picking
physically-wrong parameter values -- both of which
``pinneapple_llm.guardrail.PhysicsGuardrail`` (a fixed, re-computed
residual + parameter-sanity check, not a language model's confidence) can
catch mechanically, downstream of this module and independent of it. This
module never asks the LLM to write or approve physics; it asks it to name
one.

Provider-agnostic: ``"anthropic"``/``"openai"`` (optional deps, bring your
own API key) or ``"ollama"`` (a local model, no API key, no data leaves
your machine -- see ``local_llm.py``). ``pip install "pinneapple[llm]"``
for the hosted providers' SDKs; ``ollama`` needs only ``requests`` (a
common transitive dependency already) plus a local Ollama install.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ._dispatch import call_llm

_SYSTEM_PROMPT = """You are selecting a physics problem preset for the \
PINNeAPPle library, not writing simulation code. You MUST respond with a \
single JSON object and nothing else, of the exact form:

{"preset": "<one name from the provided list>", "kwargs": {"<param>": <value>, ...}, "reasoning": "<one sentence>"}

Rules:
- "preset" MUST be exactly one of the names in the AVAILABLE PRESETS list \
given to you. Never invent a name that is not in that list.
- "kwargs" MUST only contain parameter names that preset's own signature \
accepts (you are told each preset's accepted kwargs). Never invent a \
parameter name.
- If nothing in the list is a reasonable match for the user's request, \
respond with {"preset": null, "kwargs": {}, "reasoning": "<why nothing fits>"} \
instead of guessing.
"""


@dataclass
class DraftResult:
    preset: Optional[str]
    kwargs: Dict[str, Any]
    reasoning: str
    raw_response: str


def _preset_catalog() -> List[Dict[str, Any]]:
    """Every registered preset name + its factory function's accepted
    kwargs (introspected, not hand-maintained -- stays in sync with
    ``pinneapple_physics.pde_environment.presets`` automatically)."""
    import inspect
    from pinneapple_physics.pde_environment.presets import registry as _registry_mod

    names = _registry_mod.list_presets()  # triggers _auto_register() internally
    catalog = []
    for name in names:
        fn = _registry_mod._REGISTRY.get(name)
        try:
            sig = inspect.signature(fn) if fn is not None else None
            params = [p for p in sig.parameters if p not in ("self",)] if sig is not None else []
        except (TypeError, ValueError):
            params = []
        catalog.append({"name": name, "accepted_kwargs": params})
    return catalog


def draft_problem(
    description: str,
    *,
    provider: str = "anthropic",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    conversation_store=None,
) -> DraftResult:
    """Ask an LLM to pick + parametrise a ``ProblemSpec`` preset for a
    natural-language problem description.

    Does NOT build or return a ``ProblemSpec`` itself -- call
    ``pinneapple_physics.get_preset(result.preset, **result.kwargs)``
    yourself once you've looked at ``result.reasoning`` (this module
    deliberately does not chain straight into training without that
    human-in-the-loop look, since the LLM's *preset choice* -- unlike its
    parameter values -- is not something ``PhysicsGuardrail`` can check
    for you: a residual can be numerically small for a preset that solves
    the wrong problem entirely).

    Parameters
    ----------
    description : natural-language problem description.
    provider : ``"anthropic"``, ``"openai"``, or ``"ollama"`` (local).
    model : provider model name; a reasonable current default is used if
        not given.
    api_key : falls back to ``ANTHROPIC_API_KEY``/``OPENAI_API_KEY`` env
        vars if not given (the respective SDK's own default behaviour);
        unused for ``"ollama"``.
    conversation_store : optional ``ConversationStore`` (see
        ``conversation_store.py``) to log this call to.

    Returns
    -------
    :class:`DraftResult` -- ``preset`` is ``None`` if the model judged no
    preset a reasonable fit; check that before using ``.kwargs``.

    Raises
    ------
    ValueError
        If the LLM's response isn't valid JSON, or names a preset or
        kwarg not in the actual registered catalog (i.e. it hallucinated
        one anyway, despite the constrained prompt -- this is checked
        mechanically here, not trusted).
    """
    catalog = _preset_catalog()
    catalog_names = {c["name"] for c in catalog}

    prompt = (
        f"USER REQUEST:\n{description}\n\n"
        f"AVAILABLE PRESETS (name and accepted kwargs):\n{json.dumps(catalog, indent=2)}\n"
    )

    raw = call_llm(
        prompt, provider=provider, model=model, api_key=api_key, system=_SYSTEM_PROMPT,
        json_mode=True, module="draft_problem", conversation_store=conversation_store,
    )

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON:\n{raw}") from e

    preset = parsed.get("preset")
    kwargs = parsed.get("kwargs", {}) or {}
    reasoning = parsed.get("reasoning", "")

    if preset is not None:
        if preset not in catalog_names:
            raise ValueError(
                f"LLM named preset '{preset}', which is not in the actual registered catalog "
                f"({sorted(catalog_names)}) -- refusing to use a hallucinated preset name."
            )
        accepted = next(c["accepted_kwargs"] for c in catalog if c["name"] == preset)
        unknown = set(kwargs) - set(accepted)
        if unknown:
            raise ValueError(
                f"LLM proposed kwargs {sorted(unknown)} that preset '{preset}' does not accept "
                f"(accepted: {accepted}) -- refusing to use hallucinated parameters."
            )

    return DraftResult(preset=preset, kwargs=kwargs, reasoning=reasoning, raw_response=raw)
