"""Export extracted solutions to ProblemDesign format."""
from __future__ import annotations

import json
import os
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

    benchmark = x.metrics or ""
    strengths = ""
    weaknesses = x.limitations or ""
    alternatives = ""
    future = ""

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


def _to_partial_problem_spec(x: ExtractedProblemSolution) -> Dict[str, Any]:
    """Best-effort mapping from researcher findings → partial ProblemSpec dict.

    This is intentionally partial: fields that cannot be reliably inferred from
    literature extraction are left empty. To get a full spec, run the result
    through a DesignAgent elicitation session.
    """
    return {
        "title": x.title or "",
        "goal": x.problem or "",
        "task_type": "other",
        "domain_context": x.solution or "",
        "inputs": [],
        "outputs": [],
        "data": {
            "sources": [],
            "format": x.data_requirements or "",
            "variables_observed": [],
            "target_variables": [],
        },
        "physics": {
            "governing_equations": x.equations or [],
            "constraints": [],
            "parameters_known": [],
            "parameters_unknown": [],
        },
        "validation": {
            "primary_metrics": [x.metrics] if x.metrics else [],
            "acceptance_criteria": "",
        },
        "_researcher_source": {
            "type": x.source_type,
            "id": x.source_id,
        },
        "_note": (
            "Partial spec from pinneapple_researcher. Run through DesignAgent to fill gaps "
            "and call build_pinneapple_spec() to generate runnable code."
        ),
    }


def export_problemdesign(
    items: List[ExtractedProblemSolution],
    *,
    out_path: str,
) -> str:
    """Write researcher findings to *out_path* (JSON) and, when
    ``pinneapple_problemdesign`` is available, also write a companion
    ``*_pinneapple_spec.json`` with partial ``ProblemSpec`` mappings and
    generated Pinneapple API objects for any item that has enough physics info.

    Parameters
    ----------
    items:
        Researcher-extracted solutions.
    out_path:
        Destination path for the main researcher-format JSON.

    Returns
    -------
    str
        The resolved *out_path*.
    """
    payload = [to_problemdesign_dict(x) for x in items]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------------------
    # Companion spec file — generated when pinneapple_problemdesign is available.
    # Each item is mapped to a partial ProblemSpec; if enough physics info exists
    # we also call build_pinneapple_spec() to produce concrete API objects.
    # ---------------------------------------------------------------------------
    try:
        from pinneapple_problemdesign import ProblemSpec, build_pinneapple_spec

        spec_records: List[Dict[str, Any]] = []
        for x in items:
            partial = _to_partial_problem_spec(x)

            pinneapple_spec_dict: Optional[Dict[str, Any]] = None
            if x.equations:
                # Enough physics info to attempt code generation
                try:
                    # Build a minimal ProblemSpec from the partial mapping
                    ps = ProblemSpec(
                        title=partial["title"],
                        goal=partial["goal"],
                        task_type="pde_solution",
                        domain_context=partial["domain_context"],
                    )
                    ps.physics.governing_equations = list(x.equations or [])
                    generated = build_pinneapple_spec(ps)
                    pinneapple_spec_dict = generated.to_dict()
                except Exception:
                    pass

            spec_records.append({
                "partial_problem_spec": partial,
                "pinneapple_spec": pinneapple_spec_dict,
            })

        spec_path = out_path.replace(".json", "_pinneapple_spec.json")
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(spec_records, f, indent=2, ensure_ascii=False)

    except ImportError:
        pass

    return out_path
