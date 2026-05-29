"""VerifierAgent for verifying project runnability and suggesting patches."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..providers.gemini_provider import GeminiProvider
from .prompts import VERIFY_SYSTEM


VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["pass", "fail"]},
        "diagnosis": {"type": "string"},
        "patches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "mode": {"type": "string", "enum": ["overwrite", "append"]},
                    "content": {"type": "string"},
                },
                "required": ["path", "mode", "content"],
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["status", "diagnosis", "patches", "notes"],
}


@dataclass
class VerifyResult:
    status: str
    diagnosis: str
    patches: list[dict]
    notes: str


class VerifierAgent:
    def __init__(self, provider: GeminiProvider):
        self.provider = provider

    def run(
        self,
        *,
        project_tree: str,
        logs: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> VerifyResult:
        prompt = {
            "task": "Verify if the generated project is runnable and propose concrete patches to make it pass.",
            "project_tree": project_tree,
            "logs": logs,
            "output_schema": VERIFY_SCHEMA,
        }
        raw = self.provider.generate(
            prompt=json.dumps(prompt, ensure_ascii=False),
            system=VERIFY_SYSTEM,
            schema=VERIFY_SCHEMA,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        data = json.loads(raw)
        return VerifyResult(
            status=data["status"],
            diagnosis=data["diagnosis"],
            patches=list(data.get("patches", [])),
            notes=data.get("notes", ""),
        )
