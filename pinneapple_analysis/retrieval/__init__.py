"""pinneapple_analysis.retrieval -- real, local semantic search over a
real corpus (verification-module docstrings + real preset knowledge-
graph summaries + optionally real external literature fetched live from
arXiv; see ``corpus.py``'s own docstring for the full source list and
honesty scope). No fabricated documents, no fabricated scores, no paid
embedding API dependency -- a local sentence-transformer model and plain
cosine similarity.

Quickstart
----------
>>> from pinneapple_analysis.retrieval import build_default_index
>>> index = build_default_index()  # internal docs/presets only, instant, offline
>>> index = build_default_index(include_literature=True)  # + real arXiv papers, needs network
>>> for result in index.search("turbulence closure model", top_k=3):
...     print(result.score, result.document.source)

Requires the ``retrieval`` extra: ``pip install pinneapple[retrieval]``.
"""
from __future__ import annotations

from pinneapple_analysis.retrieval.corpus import (
    Document,
    build_corpus,
    fetch_literature_documents,
    build_literature_corpus,
    DEFAULT_LITERATURE_TOPICS,
)
from pinneapple_analysis.retrieval.index import RetrievalIndex, SearchResult, DEFAULT_MODEL_NAME

__all__ = [
    "Document", "build_corpus", "fetch_literature_documents", "build_literature_corpus",
    "DEFAULT_LITERATURE_TOPICS", "RetrievalIndex", "SearchResult", "DEFAULT_MODEL_NAME",
    "build_default_index",
]


def build_default_index(
    *, model_name: str = DEFAULT_MODEL_NAME, include_literature: bool = False,
    literature_topics=None, literature_k_per_topic: int = 5,
) -> RetrievalIndex:
    """Build a real index over the real default corpus in one call --
    convenience wrapper over ``build_corpus()`` + ``RetrievalIndex.build()``.
    See ``build_corpus``'s own docstring for what ``include_literature``
    adds and why it's opt-in."""
    docs = build_corpus(
        include_literature=include_literature, literature_topics=literature_topics,
        literature_k_per_topic=literature_k_per_topic,
    )
    return RetrievalIndex.build(docs, model_name=model_name)
