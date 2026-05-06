"""Example 00: end-to-end researcher quickstart (discover -> build KB -> optional extract).

This is the best single-file demo of what ``pinneaple_researcher`` can do.

What you get at the end:
  - a persisted run folder (runs/researcher/<topic>/<timestamp>/...)
  - a manifest.json with ranked papers/repos
  - paper artifacts with PDF sections (best-effort) and repo artifacts with README/tree/key files
  - a chunks.jsonl that can be used for RAG-style search
  - (optional) extracted problem/solution pairs + a problemdesign_export.json

Usage:
  python examples/pinneaple_researcher/00_quickstart_end_to_end.py \
      --topic "physics-informed neural networks boundary conditions" \
      --k-papers 6 --k-repos 6 --min-stars 50

Tips:
  - Set GITHUB_TOKEN to enable deeper repo-quality signals + file fetching.
  - Set GEMINI_API_KEY to enable extraction (otherwise the script will skip that step).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pinneaple_researcher import ResearcherConfig
from pinneaple_researcher.pipelines.build_kb import build_kb
from pinneaple_researcher.pipelines.discover import discover
from pinneaple_researcher.pipelines.extract_problem_solutions import extract_problem_solutions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--k-papers", type=int, default=6)
    ap.add_argument("--k-repos", type=int, default=6)
    ap.add_argument("--min-stars", type=int, default=50)
    ap.add_argument("--github-languages", nargs="*", default=None)
    ap.add_argument("--arxiv-categories", nargs="*", default=None)
    ap.add_argument("--chunk-chars", type=int, default=1800)
    ap.add_argument("--chunk-overlap", type=int, default=200)
    ap.add_argument("--extract", action="store_true", help="Force extraction (requires GEMINI_API_KEY)")
    ap.add_argument("--max-items", type=int, default=20)
    args = ap.parse_args()

    cfg = ResearcherConfig(
        topic=args.topic,
        k_papers=args.k_papers,
        k_repos=args.k_repos,
        github_min_stars=args.min_stars,
        github_languages=args.github_languages,
        arxiv_categories=args.arxiv_categories,
        chunk_chars=args.chunk_chars,
        chunk_overlap=args.chunk_overlap,
    )

    disc = discover(cfg)
    print("\n=== DISCOVERY (top items) ===")
    print(json.dumps(disc.__dict__, default=lambda o: o.__dict__, indent=2, ensure_ascii=False))

    kb = build_kb(cfg, disc)
    print("\n=== KB BUILT ===")
    print(json.dumps(kb.__dict__, indent=2, ensure_ascii=False))

    run_dir = Path(kb.run_dir)
    print("\nRun folder:", run_dir)
    print("- manifest:", run_dir / "manifest.json")
    print("- chunks:", run_dir / "kb_index" / "chunks.jsonl")

    can_extract = bool(os.environ.get("GEMINI_API_KEY"))
    if args.extract and not can_extract:
        raise SystemExit("--extract requested, but GEMINI_API_KEY is not set")

    if can_extract:
        print("\n=== EXTRACT (LLM) ===")
        items = extract_problem_solutions(kb_index=kb, max_items=args.max_items)
        print(f"Extracted {len(items)} items")
        print("- raw:", run_dir / "extracted_problem_solutions.json")
        print("- problemdesign:", run_dir / "problemdesign_export.json")
    else:
        print("\n(Skipping extract: GEMINI_API_KEY not set. Set it to enable LLM extraction.)")


if __name__ == "__main__":
    main()
