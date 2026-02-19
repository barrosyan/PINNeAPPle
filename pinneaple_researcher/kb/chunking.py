from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class TextChunk:
    chunk_id: str
    text: str


def chunk_text(text: str, *, chunk_chars: int = 1800, overlap: int = 200) -> List[TextChunk]:
    text = text or ""
    if chunk_chars <= 0:
        return [TextChunk(chunk_id="0", text=text)]

    out: List[TextChunk] = []
    step = max(1, chunk_chars - max(0, overlap))
    i = 0
    idx = 0
    while i < len(text):
        j = min(len(text), i + chunk_chars)
        out.append(TextChunk(chunk_id=str(idx), text=text[i:j]))
        idx += 1
        i += step
    return out
