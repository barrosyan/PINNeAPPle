from __future__ import annotations

import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional

try:
    from pdfminer.high_level import extract_text
except Exception:  # pragma: no cover
    extract_text = None


SECTION_PATTERNS = [
    ("abstract", re.compile(r"\babstract\b", re.IGNORECASE)),
    ("introduction", re.compile(r"\bintroduction\b", re.IGNORECASE)),
    ("method", re.compile(r"\b(method|methods|methodology|approach)\b", re.IGNORECASE)),
    ("experiments", re.compile(r"\b(experiments?|results?|evaluation)\b", re.IGNORECASE)),
    ("conclusion", re.compile(r"\b(conclusion|conclusions)\b", re.IGNORECASE)),
]


def arxiv_pdf_url(arxiv_id: str, use_export_subdomain: bool = True) -> str:
    # Common PDF URL is /pdf/<id>.pdf; using export.arxiv.org is a common best practice for automated access. :contentReference[oaicite:6]{index=6}
    host = "https://export.arxiv.org" if use_export_subdomain else "https://arxiv.org"
    return f"{host}/pdf/{arxiv_id}.pdf"


def download_pdf(url: str, out_path: str, timeout_s: int = 60) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "pinneaple-researcher/0.2"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        data = r.read()
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path


def extract_pdf_text(pdf_path: str) -> str:
    if extract_text is None:
        raise RuntimeError("pdfminer.six not installed. Install: pip install -e \".[research]\"")  # :contentReference[oaicite:7]{index=7}
    return extract_text(pdf_path) or ""


def split_sections(text: str) -> Dict[str, str]:
    """
    Heuristic: find headings by keyword matches; slice between them.
    Works “ok” for many arXiv PDFs but not perfect (PDF layout can be messy).
    """
    if not text:
        return {"full_text": ""}

    # normalize
    t = re.sub(r"\r\n", "\n", text)
    t = re.sub(r"\n{3,}", "\n\n", t)

    # find occurrences
    hits = []
    for name, pat in SECTION_PATTERNS:
        m = pat.search(t)
        if m:
            hits.append((m.start(), name))
    hits.sort(key=lambda x: x[0])

    if not hits:
        return {"full_text": t}

    # ensure abstract starts near beginning if exists
    sections: Dict[str, str] = {}
    for i, (pos, name) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(t)
        chunk = t[pos:end].strip()
        # Keep chunk bounded
        sections[name] = chunk[:60_000]
    sections["full_text"] = t[:200_000]
    return sections
