from __future__ import annotations

import os
import argparse
import json

from pinneaple_researcher.pipelines.extract_problem_solutions import extract_problem_solutions
from pinneaple_researcher.models import KBIndex

max_items = 50
run_dir=r"runs/researcher/<topic>/<timestamp>"
os.environ["GEMINI_MODEL"]="models/gemini-2.0-flash"
os.environ["GEMINI_API_KEY"]="your-api-key"

kb = KBIndex(
    topic="pinn thermal instability virtual sensor",
    run_dir=run_dir,
    chunks_path=f"{run_dir}/kb_index/chunks.jsonl",
    manifest_path=f"{run_dir}/manifest.json",
)
items = extract_problem_solutions(kb_index=kb, max_items=max_items)
print(json.dumps([x.__dict__ for x in items], indent=2, ensure_ascii=False))

