"""Example 04: reproduce a discovered repo/paper into a runnable local project.

This uses the LLM-backed Reproducer+Verifier agents to:
  - scaffold a small project folder from the selected item + KB snippet
  - try to `python -m py_compile` the project
  - run a simple smoke command (see pinneaple_researcher.utils.verify_runtime)
  - iterate patches when needed

Prereqs (env vars):
  - GEMINI_API_KEY

Usage:
  # Reproduce by reading an item from a manifest.json
  python examples/pinneaple_researcher/04_reproduce.py --kb-run runs/researcher/<topic>/<timestamp> --type repo --id owner/repo

  # Or reproduce an ad-hoc item
  python examples/pinneaple_researcher/04_reproduce.py --kb-run runs/researcher/<topic>/<timestamp> --type paper --id 1105.3778v1 \
      --title "Oscillatory thermal instability" --url https://arxiv.org/abs/1105.3778v1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from pinneaple_researcher.models import RankedItem
from pinneaple_researcher.pipelines.reproduce import reproduce


def _load_from_manifest(manifest_path: Path, *, type_: str, id_: str) -> RankedItem | None:
    try:
        data: Dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    key = "papers" if type_ == "paper" else "repos"
    for raw in data.get(key, []) or []:
        if raw.get("id") == id_:
            return RankedItem(
                type=type_,
                id=raw.get("id", ""),
                title=raw.get("title", id_),
                url=raw.get("url", ""),
                score=float(raw.get("score", 1.0)),
                why=list(raw.get("why", []) or []),
                meta=dict(raw.get("meta", {}) or {}),
            )
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb-run", required=True, help="KB run directory created by build_kb")
    ap.add_argument("--type", choices=["paper", "repo"], required=True)
    ap.add_argument("--id", required=True, help="arXiv id (paper) or owner/repo (repo)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--url", default=None)
    ap.add_argument("--max-fix-iters", type=int, default=2)
    args = ap.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("Missing GEMINI_API_KEY. Reproduce uses the GeminiProvider.")

    kb_run = Path(args.kb_run)
    manifest_path = kb_run / "manifest.json"

    item = _load_from_manifest(manifest_path, type_=args.type, id_=args.id)
    if item is None:
        # fall back to ad-hoc item
        if not (args.title and args.url):
            raise SystemExit(
                "Item not found in manifest.json and (title,url) not provided. "
                "Provide --title and --url or pick an id that exists in the manifest."
            )
        item = RankedItem(type=args.type, id=args.id, title=args.title, url=args.url, score=1.0, why=[], meta={})

    project_dir = reproduce(item=item, kb_index_dir=str(kb_run), max_fix_iters=args.max_fix_iters)
    print(project_dir)


if __name__ == "__main__":
    main()
