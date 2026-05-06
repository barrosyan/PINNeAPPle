"""Example 05: local search over KB chunks.jsonl (no embeddings required).

The researcher builds a KB as a JSONL file of overlapping text chunks.
Even with a simple lexical scorer, you can:
  - find the most relevant chunks for a question
  - collect citations (source_id + url)
  - feed top chunks into an agent/prompt elsewhere

Usage:
  python examples/pinneaple_researcher/05_local_search_over_chunks.py \
      --chunks runs/researcher/<topic>/<timestamp>/kb_index/chunks.jsonl \
      --query "how do they enforce boundary conditions?" --topk 8
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


_TOK = re.compile(r"[a-zA-Z0-9_]+")


def _tokens(s: str) -> List[str]:
    return [t.lower() for t in _TOK.findall(s or "") if len(t) >= 3]


def _bm25ish_score(query: str, text: str) -> float:
    # lightweight: tf overlap normalized by query length
    q = _tokens(query)
    if not q:
        return 0.0
    tf = Counter(_tokens(text))
    hit = sum(tf.get(w, 0) for w in q)
    return float(hit) / float(len(q))


@dataclass
class Hit:
    score: float
    source_type: str
    source_id: str
    title: str
    url: str
    chunk_id: str
    text: str


def iter_chunks(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--preview-chars", type=int, default=420)
    args = ap.parse_args()

    p = Path(args.chunks)
    hits: List[Hit] = []
    for row in iter_chunks(p):
        score = _bm25ish_score(args.query, row.get("text", ""))
        if score <= 0:
            continue
        hits.append(
            Hit(
                score=score,
                source_type=row.get("source_type", ""),
                source_id=row.get("source_id", ""),
                title=row.get("title", ""),
                url=row.get("url", ""),
                chunk_id=row.get("chunk_id", ""),
                text=row.get("text", ""),
            )
        )

    hits.sort(key=lambda h: h.score, reverse=True)
    hits = hits[: args.topk]

    print(f"Query: {args.query}")
    print(f"Top {len(hits)} hits:\n")

    for i, h in enumerate(hits, 1):
        preview = (h.text or "").replace("\n", " ")
        if len(preview) > args.preview_chars:
            preview = preview[: args.preview_chars] + "…"
        print(f"[{i}] score={h.score:.3f} | {h.source_type}:{h.source_id} | chunk={h.chunk_id}")
        print(f"    {h.title}")
        print(f"    {h.url}")
        print(f"    {preview}\n")


if __name__ == "__main__":
    main()
