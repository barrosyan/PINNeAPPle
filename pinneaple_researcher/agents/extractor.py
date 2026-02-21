"""ExtractorAgent for extracting problem/solution from KB artifacts."""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from ..models import ExtractedProblemSolution, KBIndex
from ..providers.gemini_provider import GeminiProvider
from .prompts import EXTRACT_SYSTEM


EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_type": {"type": "string", "enum": ["paper", "repo"]},
                    "source_id": {"type": "string"},
                    "title": {"type": "string"},
                    "problem": {"type": "string"},
                    "solution": {"type": "string"},
                    "equations": {"type": ["string", "null"]},
                    "data_requirements": {"type": ["string", "null"]},
                    "training_recipe": {"type": ["string", "null"]},
                    "metrics": {"type": ["string", "null"]},
                    "limitations": {"type": ["string", "null"]},
                    "extra": {"type": "object"},
                },
                "required": ["source_type", "source_id", "title", "problem", "solution", "extra"],
            },
        }
    },
    "required": ["items"],
}


class ExtractorAgent:
    def __init__(self, provider: GeminiProvider):
        self.provider = provider

    def run(
        self,
        *,
        kb_index: KBIndex,
        max_items: int = 20,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> List[ExtractedProblemSolution]:
        # Load a small sample of chunks (v0.1)
        sample_rows: List[Dict[str, Any]] = []
        with open(kb_index.chunks_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 60:  # keep prompt bounded
                    break
                try:
                    sample_rows.append(json.loads(line))
                except Exception:
                    continue

        prompt = {
            "task": "Extract problems and solutions from the knowledge base chunks.",
            "constraints": {
                "max_items": int(max_items),
                "focus": "Physics AI / PINNs / solvers, but allow adjacent methods if relevant.",
            },
            "kb_sample_chunks": sample_rows,
            "output_schema": EXTRACT_SCHEMA,
        }

        raw = self.provider.generate(
            prompt=json.dumps(prompt, ensure_ascii=False),
            system=EXTRACT_SYSTEM,
            schema=EXTRACT_SCHEMA,   # <-- dict, JSON mode nativo
            temperature=temperature,
            max_tokens=max_tokens,
        )
        data = json.loads(raw)
        items = data.get("items", [])
        out: List[ExtractedProblemSolution] = []
        for it in items[:max_items]:
            out.append(
                ExtractedProblemSolution(
                    source_type=it["source_type"],
                    source_id=it["source_id"],
                    title=it["title"],
                    problem=it["problem"],
                    solution=it["solution"],
                    equations=it.get("equations"),
                    data_requirements=it.get("data_requirements"),
                    training_recipe=it.get("training_recipe"),
                    metrics=it.get("metrics"),
                    limitations=it.get("limitations"),
                    extra=it.get("extra") or {},
                )
            )
        return out
