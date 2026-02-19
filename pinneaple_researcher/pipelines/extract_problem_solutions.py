from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import List, Optional

from ..agents.extractor import ExtractorAgent
from ..models import ExtractedProblemSolution, KBIndex
from ..providers.gemini_provider import GeminiProvider
from .export_problemdesign import export_problemdesign


def extract_problem_solutions(
    *,
    kb_index: KBIndex,
    out_dir: Optional[str] = None,
    max_items: int = 20,
) -> List[ExtractedProblemSolution]:
    provider = GeminiProvider()
    agent = ExtractorAgent(provider)

    items = agent.run(kb_index=kb_index, max_items=max_items)

    save_dir = out_dir or kb_index.run_dir
    os.makedirs(save_dir, exist_ok=True)

    # raw extracted
    path = os.path.join(save_dir, "extracted_problem_solutions.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(x) for x in items], f, indent=2, ensure_ascii=False)

    # problemdesign export
    export_path = os.path.join(save_dir, "problemdesign_export.json")
    export_problemdesign(items, out_path=export_path)

    return items
