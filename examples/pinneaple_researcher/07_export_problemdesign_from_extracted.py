"""Example 07: export ProblemDesign JSON from a saved extraction file.

If you already ran Example 03/00 with extraction enabled, you'll have:
  - extracted_problem_solutions.json

This script re-exports into a stable ProblemDesign-like schema.

Usage:
  python examples/pinneaple_researcher/07_export_problemdesign_from_extracted.py \
      --extracted runs/researcher/<topic>/<timestamp>/extracted_problem_solutions.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pinneaple_researcher.models import ExtractedProblemSolution
from pinneaple_researcher.pipelines.export_problemdesign import export_problemdesign


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    p = Path(args.extracted)
    raw = json.loads(p.read_text(encoding="utf-8"))
    items = [ExtractedProblemSolution(**x) for x in raw]

    out = Path(args.out) if args.out else p.parent / "problemdesign_export.json"
    export_problemdesign(items, out_path=str(out))
    print(str(out))


if __name__ == "__main__":
    main()
