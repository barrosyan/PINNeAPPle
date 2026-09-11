"""pinneapple_analysis.retrieval -- real, local semantic search over a
real corpus (verification-module docstrings + real preset knowledge-
graph summaries; see ``corpus.py``'s own docstring for the full source
list and honesty scope). No fabricated documents, no fabricated scores,
no paid embedding API dependency -- a local sentence-transformer model
and plain cosine similarity.

Quickstart
----------
>>> from pinneapple_analysis.retrieval import build_default_index
>>> index = build_default_index()
>>> for result in index.search("turbulence closure model", top_k=3):
...     print(result.score, result.document.source)

Requires the ``retrieval`` extra: ``pip install pinneapple[retrieval]``.
"""
from __future__ import annotations

from pinneapple_analysis.retrieval.corpus import Document, build_corpus
from pinneapple_analysis.retrieval.index import RetrievalIndex, SearchResult, DEFAULT_MODEL_NAME

__all__ = [
    "Document", "build_corpus", "RetrievalIndex", "SearchResult", "DEFAULT_MODEL_NAME",
    "build_default_index",
]


def build_default_index(*, model_name: str = DEFAULT_MODEL_NAME) -> RetrievalIndex:
    """Build a real index over the real default corpus in one call --
    convenience wrapper over ``build_corpus()`` + ``RetrievalIndex.build()``."""
    return RetrievalIndex.build(build_corpus(), model_name=model_name)
