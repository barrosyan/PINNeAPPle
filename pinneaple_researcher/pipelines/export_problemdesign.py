"""Export extracted solutions to ProblemDesign format."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from ..models import ExtractedProblemSolution


def to_problemdesign_dict(x: ExtractedProblemSolution) -> Dict[str, Any]:
    """
    ProblemDesign-like schema (stable, explicit):
      - source
      - problem
      - solution
      - benchmark
      - strengths
      - weaknesses
      - alternatives
      - future_improvements
    """
    source = {
        "type": x.source_type,
        "id": x.source_id,
        "title": x.title,
    }

    # best-effort enrichment from extracted fields
    benchmark = x.metrics or ""
    strengths = ""
    weaknesses = x.limitations or ""
    alternatives = ""
    future = ""

    # allow agent to place extra structure
    ex = x.extra or {}
    strengths = ex.get("strengths", strengths)
    weaknesses = ex.get("weaknesses", weaknesses)
    alternatives = ex.get("alternatives", alternatives)
    future = ex.get("future_improvements", future)

    return {
        "source": source,
        "problem": x.problem,
        "solution": x.solution,
        "benchmark": benchmark,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "alternatives": alternatives,
        "future_improvements": future,
        "details": {
            "equations": x.equations,
            "data_requirements": x.data_requirements,
            "training_recipe": x.training_recipe,
        },
    }


def export_problemdesign(
    items: List[ExtractedProblemSolution],
    *,
    out_path: str,
) -> str:
    payload = [to_problemdesign_dict(x) for x in items]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Try integrating with pinneaple_problemdesign if there is a known writer.
    # Fallback always writes JSON.
    try:
        import importlib

        pd = importlib.import_module("pinneaple_problemdesign")
        # Common patterns: save(), write_spec(), registry.add(), etc.
        # If you have a canonical function, plug it here later.
        # For now: just write JSON beside it.
    except Exception:
        pass

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return out_path
