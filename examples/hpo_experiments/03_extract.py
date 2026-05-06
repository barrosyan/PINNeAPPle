"""Example 03: extract Problem/Solution pairs from a built KB index.

This step uses an LLM provider (Gemini) through ``pinneaple_researcher.providers``.

Prereqs (env vars):
  - GEMINI_API_KEY

Optional:
  - GEMINI_MODEL (default is provider-defined)

Usage:
  python examples/pinneaple_researcher/03_extract.py --run-dir runs/researcher/<topic>/<timestamp> --max-items 30
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pinneaple_researcher.models import KBIndex
from pinneaple_researcher.pipelines.extract_problem_solutions import extract_problem_solutions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="KB run directory created by build_kb")
    ap.add_argument("--topic", default=None, help="Optional topic label (defaults to folder name)")
    ap.add_argument("--max-items", type=int, default=25)
    args = ap.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit(
            "Missing GEMINI_API_KEY. Set it first, e.g.\n"
            "  export GEMINI_API_KEY='...'\n"
            "Optionally set GEMINI_MODEL too."
        )

    run_dir = Path(args.run_dir)
    topic = args.topic or run_dir.parent.name

    kb = KBIndex(
        topic=topic,
        run_dir=str(run_dir),
        chunks_path=str(run_dir / "kb_index" / "chunks.jsonl"),
        manifest_path=str(run_dir / "manifest.json"),
    )

    items = extract_problem_solutions(kb_index=kb, max_items=args.max_items)
    print(json.dumps([x.__dict__ for x in items], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
