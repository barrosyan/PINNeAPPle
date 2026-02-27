"""Example 02: build a KB run from a discovery result.

This writes:
  - run folder with manifest.json
  - paper/repo artifacts (markdown + metadata, best-effort paper PDF extraction)
  - kb_index/chunks.jsonl for retrieval

Usage:
  python examples/pinneaple_researcher/02_build_kb.py --topic "pinn thermal instability virtual sensor" \
      --k-papers 8 --k-repos 8 --min-stars 50
"""

from __future__ import annotations

import argparse
import json

from pinneaple_researcher import ResearcherConfig
from pinneaple_researcher.pipelines.build_kb import build_kb
from pinneaple_researcher.pipelines.discover import discover


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--k-papers", type=int, default=8)
    ap.add_argument("--k-repos", type=int, default=8)
    ap.add_argument("--min-stars", type=int, default=50)
    ap.add_argument("--github-languages", nargs="*", default=None)
    ap.add_argument("--arxiv-categories", nargs="*", default=None)
    ap.add_argument("--chunk-chars", type=int, default=1800)
    ap.add_argument("--chunk-overlap", type=int, default=200)
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
    idx = build_kb(cfg, disc)
    print(json.dumps(idx.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
