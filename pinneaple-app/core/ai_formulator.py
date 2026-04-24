"""
AI-assisted physics problem formulation.

Uses the Anthropic API (Claude) to convert a natural-language description
into a structured physics problem spec.  Falls back to keyword-based
heuristics if the API is unavailable.
"""
from __future__ import annotations
import os
import re
from typing import Dict, Optional


_SYSTEM_PROMPT = """\
You are a physics expert and scientific computing assistant integrated into the
Pinneaple Physics-AI platform.

Given a natural-language description of a physical problem, return a JSON object
with the following keys:
  - name            (string)
  - category        (string — one of: Fluid Dynamics, Heat Transfer, Structural Mechanics,
                     Acoustics, Electromagnetics, Chemical Engineering, Other)
  - description     (string — concise)
  - equations       (list of strings — governing equations in LaTeX-like notation)
  - domain          (dict — coordinate → [min, max])
  - params          (dict — key physical parameters with typical values)
  - bcs             (list of strings — boundary conditions)
  - ics             (list of strings — initial conditions, empty if steady)
  - tags            (list of strings)
  - solvers         (list of strings — suggested: fdm, fem, fvm, lbm, pinn, fno, deeponet)
  - ref             (string — relevant reference if well-known)

Return ONLY valid JSON, no markdown fences.
"""

_KEYWORD_MAP = {
    # Fluid
    ("navier", "stokes"):             "navier_stokes_2d",
    ("flow", "cylinder"):             "cylinder_flow",
    ("cavity", "lid"):                "lid_driven_cavity",
    ("burgers",):                     "burgers_1d",
    # Heat
    ("heat", "conduc"):               "heat_2d",
    ("temperatura", "placa"):         "heat_2d",
    # Wave
    ("wave", "onda"):                 "wave_1d",
    ("acoustic",):                    "wave_1d",
    # Structural
    ("elastic", "plate"):             "elastic_plate",
    ("elasticidade", "placa"):        "elastic_plate",
    # Reaction-diffusion
    ("reaction", "diffusion"):        "reaction_diffusion",
    ("reação", "difusão"):            "reaction_diffusion",
    # Poisson
    ("poisson",):                     "poisson_2d",
}


def _keyword_match(description: str) -> Optional[str]:
    """Return a preset key if we recognise the problem from keywords."""
    low = description.lower()
    for kws, preset_key in _KEYWORD_MAP.items():
        if all(kw in low for kw in kws):
            return preset_key
    return None


def formulate_with_ai(description: str) -> Dict:
    """
    Convert a natural-language description to a problem spec dict.

    Tries Anthropic Claude first; falls back to keyword heuristics + template.
    """
    from .problem_library import get_problem

    # 1. Try keyword matching (fast, no API needed)
    preset_key = _keyword_match(description)
    if preset_key:
        prob = get_problem(preset_key)
        if prob:
            return {**prob, "_source": "keyword_match", "_preset_key": preset_key}

    # 2. Try Anthropic API
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": description}],
            )
            import json
            raw = msg.content[0].text.strip()
            spec = json.loads(raw)
            spec["_source"] = "claude_api"
            return spec
        except Exception as e:
            pass  # fall through to generic template

    # 3. Generic fallback template
    return {
        "name":        f"Custom Problem",
        "category":    "Other",
        "description": description,
        "equations":   ["[Not yet identified — describe more precisely]"],
        "domain":      {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "params":      {},
        "bcs":         ["[Define boundary conditions]"],
        "ics":         [],
        "tags":        ["custom"],
        "solvers":     ["pinn"],
        "ref":         "",
        "_source":     "fallback",
    }


def suggest_models(problem: Dict) -> list:
    """Return ranked model suggestions for a given problem."""
    tags = problem.get("tags", [])
    solvers = problem.get("solvers", [])
    dim = problem.get("dim", 2)
    suggestions = []

    if "lbm" in solvers:
        suggestions.append({"model": "LBM Solver", "type": "lbm",
                             "reason": "Native LBM solver for this CFD problem", "score": 95})
    if "pinn" in solvers or True:
        suggestions.append({"model": "PINN", "type": "pinn",
                             "reason": "Physics-Informed Neural Network — data-efficient", "score": 85})
    if dim >= 2:
        suggestions.append({"model": "FNO", "type": "fno",
                             "reason": "Fourier Neural Operator — fast for structured grids", "score": 80})
    if "nonlinear" in tags:
        suggestions.append({"model": "DeepONet", "type": "deeponet",
                             "reason": "Operator learning for nonlinear dynamics", "score": 75})
    if "cfd" in tags or "lbm" in tags:
        suggestions.append({"model": "TCN", "type": "tcn",
                             "reason": "Temporal Convolutional Network for flow sequences", "score": 65})

    return sorted(suggestions, key=lambda x: -x["score"])
