"""ReproducerAgent for generating reproduction patches from repo/project."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..models import RankedItem
from ..providers.gemini_provider import GeminiProvider
from .prompts import REPRODUCE_SYSTEM


REPRO_SCHEMA = {
    "type": "object",
    "properties": {
        "project_name": {"type": "string"},
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
        "run_instructions": {"type": "string"},
    },
    "required": ["project_name", "files", "run_instructions"],
}


@dataclass
class ReproduceResult:
    project_dir: str
    run_instructions: str


class ReproducerAgent:
    def __init__(self, provider: GeminiProvider):
        self.provider = provider

    def run(
        self,
        *,
        item: RankedItem,
        kb_snippet: str,
        out_dir: str,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> ReproduceResult:
        prompt = {
            "task": "Generate a minimal runnable reproduction project.",
            "item": {
                "type": item.type,
                "id": item.id,
                "title": item.title,
                "url": item.url,
                "meta": item.meta,
            },
            "kb_snippet": kb_snippet,
            "requirements": {
                "language": "python",
                "no_external_private_assets": True,
                "must_run": "at least a smoke training/inference loop (even tiny)",
                "keep_dependencies_minimal": True,
            },
            "output_schema": REPRO_SCHEMA,
        }

        raw = self.provider.generate(
            prompt=json.dumps(prompt, ensure_ascii=False),
            system=REPRODUCE_SYSTEM,
            schema=REPRO_SCHEMA,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        data = json.loads(raw)

        project_name = data["project_name"]
        project_dir = os.path.join(out_dir, project_name)
        os.makedirs(project_dir, exist_ok=True)

        for f in data["files"]:
            path = os.path.join(project_dir, f["path"])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as wf:
                wf.write(f["content"])

        # always write a top README with run instructions
        with open(os.path.join(project_dir, "README.md"), "a", encoding="utf-8") as rf:
            rf.write("\n\n## Run\n\n")
            rf.write(data.get("run_instructions", "").strip() + "\n")

        return ReproduceResult(project_dir=project_dir, run_instructions=data.get("run_instructions", ""))
