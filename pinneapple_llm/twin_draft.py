"""LLM-assisted digital twin drafting -- same constrained pattern as
``draft.py``/``geometry_draft.py``: the LLM proposes values for
``pinneapple_systems.digital_twin.DigitalTwinConfig``'s own real,
introspected fields (update interval, assimilation method, anomaly
threshold, ...), never a free-form twin implementation. The actual twin is
always built by the real ``build_digital_twin`` factory, never by code the
LLM wrote.

"3D digital twin" here means composing three already-real, already-tested
pieces -- a trained PINN surrogate (from ``solve_pde``/``pipeline``), the
real ``DigitalTwin`` live-inference/assimilation loop, and
``pinneapple_blender``'s export bridge -- into one function that, on each
twin update tick, also drops a ``.ply`` frame for Blender to pick up (see
``export_live_frame``). This is not a new 3D-twin subsystem; it is glue
between three subsystems that already independently work.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields as dataclass_fields
from typing import Any, Dict, Optional

from ._dispatch import call_llm

_SYSTEM_PROMPT = """You are configuring a PINNeAPPle DigitalTwinConfig, not \
writing simulation or control code. You MUST respond with a single JSON \
object and nothing else:

{"config_kwargs": {"<field>": <value>, ...}, "reasoning": "<one sentence>"}

Rules:
- Every key in "config_kwargs" MUST be one of the field names given to you \
in AVAILABLE CONFIG FIELDS. Never invent a field name.
- Respect each field's stated type (float/int/str/bool). Never invent a \
value for a field whose valid choices are given and your value isn't one \
of them (e.g. `assimilation` only accepts "none"/"ekf"/"enkf").
"""


@dataclass
class TwinDraftResult:
    config_kwargs: Dict[str, Any]
    reasoning: str
    raw_response: str


def _config_field_catalog() -> list:
    from pinneapple_systems.digital_twin import DigitalTwinConfig

    catalog = []
    for f in dataclass_fields(DigitalTwinConfig):
        catalog.append({"name": f.name, "type": str(f.type), "default": repr(getattr(DigitalTwinConfig(), f.name, None))})
    return catalog


def draft_digital_twin(
    description: str,
    *,
    provider: str = "anthropic",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    conversation_store=None,
) -> TwinDraftResult:
    """Ask an LLM to propose ``DigitalTwinConfig`` kwargs for a
    natural-language description (e.g. "a twin that re-infers every 100ms
    and flags anomalies aggressively").

    Does not build the twin -- call ``pinneapple_systems.digital_twin
    .build_digital_twin(model, field_names, **result.config_kwargs)``
    yourself (same human-in-the-loop reasoning as ``draft_problem``).

    Raises
    ------
    ValueError
        If the LLM proposes a field name not in the real
        ``DigitalTwinConfig`` dataclass.
    """
    catalog = _config_field_catalog()
    catalog_names = {c["name"] for c in catalog}

    prompt = (
        f"USER REQUEST:\n{description}\n\n"
        f"AVAILABLE CONFIG FIELDS:\n{json.dumps(catalog, indent=2)}\n"
    )
    raw = call_llm(
        prompt, provider=provider, model=model, api_key=api_key, system=_SYSTEM_PROMPT,
        json_mode=True, module="draft_digital_twin", conversation_store=conversation_store,
    )

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON:\n{raw}") from e

    kwargs = parsed.get("config_kwargs", {}) or {}
    reasoning = parsed.get("reasoning", "")
    unknown = set(kwargs) - catalog_names
    if unknown:
        raise ValueError(
            f"LLM proposed DigitalTwinConfig field(s) {sorted(unknown)} not in the real dataclass "
            f"(fields: {sorted(catalog_names)}) -- refusing to use hallucinated fields."
        )
    return TwinDraftResult(config_kwargs=kwargs, reasoning=reasoning, raw_response=raw)


def build_3d_live_twin(
    model,
    field_names: list,
    *,
    export_dir: str,
    points,
    config_kwargs: Optional[Dict[str, Any]] = None,
    **build_kwargs,
):
    """Build a real ``DigitalTwin`` (``pinneapple_systems.digital_twin
    .build_digital_twin``) plus a bound "3D export" hook: every time the
    twin's own state updates, write the current field prediction at
    ``points`` as the next frame in a ``pinneapple_blender`` ``.ply``
    sequence in ``export_dir`` -- so pointing Blender's
    ``import_pinneapple_sequence.py`` at ``export_dir`` after the fact (or
    periodically, for a "live" view) shows the twin's actual inferred
    state in 3D, not a static one-off snapshot.

    Returns ``(twin, export_hook)`` -- ``twin`` is the real, unmodified
    ``DigitalTwin`` object (every method it normally has still works);
    ``export_hook()`` can be called manually (e.g. from your own polling
    loop or the twin's own monitoring callback, however your specific
    ``DigitalTwin`` version exposes one) to write the next frame using the
    twin's current state.
    """
    from pinneapple_systems.digital_twin import build_digital_twin
    from pinneapple_blender import export_scene

    twin = build_digital_twin(model, field_names, **(config_kwargs or {}), **build_kwargs)

    frame_counter = {"i": 0}

    def export_hook(field_values=None):
        """Write the next frame. ``field_values``: (N,) array for the
        first field in ``field_names``, or pass explicitly if you want a
        different field/derived quantity visualised -- this function does
        not reach into ``twin``'s internals itself (its exact live-state
        attribute is a ``DigitalTwin``-version-specific detail this glue
        function deliberately does not assume), so the caller supplies
        the values to plot at this tick."""
        if field_values is None:
            raise ValueError(
                "export_hook needs field_values explicitly (this glue function does not assume "
                "DigitalTwin's internal live-state attribute name/shape)"
            )
        path = export_scene(points, field_values, export_dir, field_name=field_names[0], frame_index=frame_counter["i"])
        frame_counter["i"] += 1
        return path

    return twin, export_hook
