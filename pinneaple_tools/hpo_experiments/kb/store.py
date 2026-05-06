"""KB store for artifacts, indexing, and persistence."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..kb.chunking import chunk_text
from ..models import KBArtifact, KBIndex, RankedItem


_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")


def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = _SAFE.sub("_", s)
    s = s.strip("_")
    return s[:120] if len(s) > 120 else s


def ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p


class KBStore:
    """
    Persists a KB run:
      - manifest.json (ranked items + metadata)
      - artifacts (papers/, repos/)
      - chunks.jsonl (chunked text for RAG)
    """

    def __init__(self, *, out_dir: str):
        self.out_dir = out_dir

    def new_run_dir(self, topic: str) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.out_dir, slugify(topic), ts)
        ensure_dir(run_dir)
        ensure_dir(os.path.join(run_dir, "papers"))
        ensure_dir(os.path.join(run_dir, "repos"))
        ensure_dir(os.path.join(run_dir, "kb_index"))
        return run_dir

    def write_manifest(self, run_dir: str, *, topic: str, papers: List[RankedItem], repos: List[RankedItem]) -> str:
        path = os.path.join(run_dir, "manifest.json")
        payload = {
            "topic": topic,
            "papers": [asdict(x) for x in papers],
            "repos": [asdict(x) for x in repos],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return path

    def write_artifact_text(
        self,
        run_dir: str,
        *,
        item: RankedItem,
        text: str,
    ) -> KBArtifact:
        base = "papers" if item.type == "paper" else "repos"
        item_dir = os.path.join(run_dir, base, slugify(item.id))
        ensure_dir(item_dir)

        meta_path = os.path.join(item_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(item.meta, f, indent=2, ensure_ascii=False)

        content_path = os.path.join(item_dir, "content.md")
        with open(content_path, "w", encoding="utf-8") as f:
            f.write(text or "")

        return KBArtifact(
            type=item.type,
            id=item.id,
            title=item.title,
            url=item.url,
            path=item_dir,
            meta=item.meta,
        )

    def write_chunks_jsonl(
        self,
        run_dir: str,
        *,
        topic: str,
        artifacts: List[KBArtifact],
        chunk_chars: int,
        chunk_overlap: int,
    ) -> str:
        out_path = os.path.join(run_dir, "kb_index", "chunks.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for a in artifacts:
                content_path = os.path.join(a.path, "content.md")
                try:
                    with open(content_path, "r", encoding="utf-8") as rf:
                        text = rf.read()
                except Exception:
                    text = ""

                chunks = chunk_text(text, chunk_chars=chunk_chars, overlap=chunk_overlap)
                for c in chunks:
                    row = {
                        "topic": topic,
                        "source_type": a.type,
                        "source_id": a.id,
                        "title": a.title,
                        "url": a.url,
                        "chunk_id": c.chunk_id,
                        "text": c.text,
                        "meta": a.meta,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return out_path

    def build_index(
        self,
        *,
        run_dir: str,
        topic: str,
        artifacts: List[KBArtifact],
        chunk_chars: int,
        chunk_overlap: int,
    ) -> KBIndex:
        chunks_path = self.write_chunks_jsonl(
            run_dir,
            topic=topic,
            artifacts=artifacts,
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
        )
        manifest_path = os.path.join(run_dir, "manifest.json")
        return KBIndex(topic=topic, run_dir=run_dir, chunks_path=chunks_path, manifest_path=manifest_path)
