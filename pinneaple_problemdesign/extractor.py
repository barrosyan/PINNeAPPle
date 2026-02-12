from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple

from .protocol import LLMProvider, LLMMessage


SYSTEM_PROMPT = """
You are an engineering requirements analyst.

CRITICAL RULES:
- Extract ONLY information explicitly stated by the user.
- Do NOT guess. Do NOT invent.
- If a field is unknown, omit it from the spec fields and instead add it to unknown_fields.
- If you want to propose a reasonable assumption, put it in assumptions_suggested (plain text),
  and mark it as needing confirmation later.
- Return VALID JSON only. No markdown. No commentary.
""".strip()


USER_TEMPLATE = """
User message:
<<USER_TEXT>>

Return a JSON object with:

- "partial_spec": a PARTIAL ProblemSpec JSON object (only include fields explicitly provided).
- "unknown_fields": list of important missing fields that block progress.
- "assumptions_suggested": list of possible assumptions (uncertain) that need confirmation.
- "gaps_suggested": list of objects with {id, question, severity, rationale, how_to_obtain} if you can.

Allowed fields in partial_spec (omit if unknown):
title, goal, task_type, inputs, outputs, horizon, input_window, frequency, domain_context,
data{sources, format, sampling, variables_observed, target_variables, known_quality_issues, missingness, train_span, val_split_policy, labels_available},
physics{governing_equations, boundary_conditions, initial_conditions, constraints, parameters_known, parameters_unknown, units},
geometry{domain, representation, sensors, coordinate_system},
validation{primary_metrics, acceptance_criteria, robustness_tests, ood_scenarios},
deployment{environment, latency_budget_ms, update_policy, monitoring},
constraints{hardware, max_training_time, interpretability, compliance}

Example output shape:
{
  "partial_spec": {
    "goal": "Forecast outlet temperature 6 hours ahead",
    "task_type": "forecasting",
    "frequency": "10min",
    "horizon": "6h"
  },
  "unknown_fields": ["inputs", "outputs", "validation.primary_metrics"],
  "assumptions_suggested": ["Assume input_window = 64 steps for baseline."],
  "gaps_suggested": [
    {
      "id": "io.inputs",
      "question": "What are the input features and units?",
      "severity": "blocker",
      "rationale": "We need inputs to define the model interface.",
      "how_to_obtain": "List sensor/feature names and units."
    }
  ]
}
""".strip()


def _strip_code_fences(text: str) -> str:
    # removes ```json ... ``` or ``` ... ```
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _extract_first_json_object(text: str) -> str:
    """
    If the model returns extra text, try to extract the first JSON object.
    """
    text = text.strip()
    # find first '{' and attempt to balance braces
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text  # fallback


def extract_from_text(
    llm: LLMProvider,
    user_text: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    content = USER_TEMPLATE.replace("<<USER_TEXT>>", user_text)

    messages = [
        LLMMessage(role="system", content=SYSTEM_PROMPT),
        LLMMessage(role="user", content=content),
    ]

    resp = llm.generate(
        messages,
        temperature=0.0,
        max_tokens=1400,
        json_mode=True,
    )

    raw_text = resp.text or ""
    cleaned = _strip_code_fences(raw_text)
    cleaned = _extract_first_json_object(cleaned)

    try:
        payload = json.loads(cleaned)
    except Exception:
        return {}, {
            "unknown_fields": [],
            "assumptions_suggested": [],
            "gaps_suggested": [],
            "extract_ok": False,
            "raw_text": raw_text,
            "cleaned_text": cleaned,
        }

    partial_spec = payload.get("partial_spec") or {}
    meta = {
        "unknown_fields": payload.get("unknown_fields", []) or [],
        "assumptions_suggested": payload.get("assumptions_suggested", []) or [],
        "gaps_suggested": payload.get("gaps_suggested", []) or [],
        "extract_ok": True,
    }
    return partial_spec, meta
