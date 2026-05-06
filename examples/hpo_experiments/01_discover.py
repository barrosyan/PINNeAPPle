"""Example 01: discovery only (papers + repos) with ranking metadata.

Usage:
  python examples/pinneaple_researcher/01_discover.py \
      --topic "pinn thermal instability virtual sensor" \
      --k-papers 8 --k-repos 8 --min-stars 50

Tip: set GITHUB_TOKEN to reduce rate limits and enable deeper repo-quality signals.
"""

from __future__ import annotations

import argparse
import json

from pinneaple_researcher import ResearcherConfig
from pinneaple_researcher.pipelines.discover import discover


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--k-papers", type=int, default=8)
    ap.add_argument("--k-repos", type=int, default=8)
    ap.add_argument("--min-stars", type=int, default=50)
    ap.add_argument("--github-languages", nargs="*", default=None)
    ap.add_argument("--arxiv-categories", nargs="*", default=None)
    args = ap.parse_args()

    cfg = ResearcherConfig(
        topic=args.topic,
        k_papers=args.k_papers,
        k_repos=args.k_repos,
        github_min_stars=args.min_stars,
        github_languages=args.github_languages,
        arxiv_categories=args.arxiv_categories,
    )
    res = discover(cfg)
    print(json.dumps(res, default=lambda o: o.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
