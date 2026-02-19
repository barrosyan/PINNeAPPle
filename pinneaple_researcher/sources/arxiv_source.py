from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    summary: str
    url: str
    published: str
    authors: List[str]
    categories: List[str]


class ArxivSource:
    """
    Minimal arXiv Atom API search.
    """
    API = "http://export.arxiv.org/api/query"

    def search(
        self,
        topic: str,
        *,
        k: int = 10,
        categories: Optional[Sequence[str]] = None,
        timeout_s: int = 30,
    ) -> List[ArxivPaper]:
        cat_q = ""
        if categories:
            cat_q = " AND (" + " OR ".join([f"cat:{c}" for c in categories]) + ")"

        query = f"all:{topic}{cat_q}"
        params = {
            "search_query": query,
            "start": "0",
            "max_results": str(int(k)),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        url = self.API + "?" + urllib.parse.urlencode(params)

        req = urllib.request.Request(url, headers={"User-Agent": "pinneaple-researcher/0.2"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = resp.read()

        root = ET.fromstring(data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        out: List[ArxivPaper] = []
        for entry in root.findall("atom:entry", ns):
            entry_id = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
            arxiv_id = entry_id.split("/")[-1]

            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip().replace("\n", " ")
            summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()

            authors = []
            for a in entry.findall("atom:author", ns):
                nm = (a.findtext("atom:name", default="", namespaces=ns) or "").strip()
                if nm:
                    authors.append(nm)

            cats = []
            for c in entry.findall("atom:category", ns):
                term = (c.attrib.get("term", "") or "").strip()
                if term:
                    cats.append(term)

            out.append(
                ArxivPaper(
                    arxiv_id=arxiv_id,
                    title=title,
                    summary=summary,
                    url=entry_id,
                    published=published,
                    authors=authors,
                    categories=cats,
                )
            )
        return out
