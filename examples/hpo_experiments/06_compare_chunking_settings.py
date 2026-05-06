"""Example 06: compare KB chunking settings (chunk_chars/overlap) on an existing run.

Why it matters:
  - Smaller chunks improve precision but increase index size.
  - Larger chunks can preserve context but may dilute retrieval.

Usage:
  python examples/pinneaple_researcher/06_compare_chunking_settings.py \
      --run-dir runs/researcher/<topic>/<timestamp> \
      --chunk-chars 1200 1800 2600 \
      --overlap 150 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from pinneaple_researcher.kb.chunking import chunk_text


def _iter_artifact_markdown(run_dir: Path) -> List[Path]:
    out: List[Path] = []
    for base in ("papers", "repos"):
        root = run_dir / base
        if not root.exists():
            continue
        for p in root.rglob("content.md"):
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--chunk-chars", nargs="+", type=int, default=[1200, 1800, 2600])
    ap.add_argument("--overlap", nargs="+", type=int, default=[150, 200])
    ap.add_argument("--max-files", type=int, default=30, help="Limit artifacts to keep this demo fast")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    files = _iter_artifact_markdown(run_dir)[: args.max_files]
    if not files:
        raise SystemExit("No artifacts found. Did you pass the correct run directory?")

    # measure average chunk counts
    report: List[Dict] = []
    for cc in args.chunk_chars:
        for ov in args.overlap:
            total_chunks = 0
            total_chars = 0
            for p in files:
                txt = p.read_text(encoding="utf-8", errors="ignore")
                total_chars += len(txt)
                total_chunks += len(chunk_text(txt, chunk_chars=cc, overlap=ov))
            report.append(
                {
                    "chunk_chars": cc,
                    "overlap": ov,
                    "files": len(files),
                    "total_chars": total_chars,
                    "total_chunks": total_chunks,
                    "avg_chunk_chars": round(total_chars / max(1, total_chunks), 1),
                    "avg_chunks_per_file": round(total_chunks / max(1, len(files)), 1),
                }
            )

    report.sort(key=lambda r: (r["chunk_chars"], r["overlap"]))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
