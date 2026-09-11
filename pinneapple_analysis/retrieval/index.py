"""A real, local, embedding-based semantic index over a
:class:`~pinneapple_analysis.retrieval.corpus.Document` corpus.

Deliberately a small numpy cosine-similarity index, not a dedicated
vector database (chromadb/faiss) -- at this corpus's real current size
(tens to low hundreds of documents), brute-force cosine similarity over
an in-memory numpy array is exact (no approximation error a dedicated
ANN index would trade for speed) and fast enough; there is nothing a
vector database would meaningfully add yet. See this module's own
docstring in downstream ``ROADMAP.md`` documents for when to actually
introduce one (once the corpus grows into the thousands+).

Uses a small, local, CPU-friendly sentence-embedding model
(``sentence-transformers``, default ``all-MiniLM-L6-v2``) -- no paid
embedding API dependency. Requires the ``retrieval`` extra
(``pip install pinneapple[retrieval]``); importing this module without
it installed raises a clear ``ImportError`` naming the extra, not a
cryptic failure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from pinneapple_analysis.retrieval.corpus import Document

__all__ = ["SearchResult", "RetrievalIndex", "DEFAULT_MODEL_NAME"]

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


_MODEL_CACHE: dict = {}  # model_name -> loaded SentenceTransformer, process-wide, shared across RetrievalIndex instances


def _require_sentence_transformers():
    try:
        import sentence_transformers
    except ImportError as e:
        raise ImportError(
            "pinneapple_analysis.retrieval requires the 'retrieval' extra: "
            "pip install 'pinneapple[retrieval]' (installs sentence-transformers)."
        ) from e
    return sentence_transformers


def _load_model(model_name: str):
    """Load (or reuse an already-loaded) SentenceTransformer -- loading
    from disk/HF cache is real, non-trivial latency; reusing it across
    calls/instances that share the same model_name avoids reloading it
    on every single search()."""
    if model_name not in _MODEL_CACHE:
        st = _require_sentence_transformers()
        _MODEL_CACHE[model_name] = st.SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


@dataclass(frozen=True)
class SearchResult:
    document: Document
    score: float  # cosine similarity, in [-1, 1] -- never a fabricated confidence number


class RetrievalIndex:
    """A built, queryable index over a real document list. Never
    constructed with a partial/fabricated embedding -- ``build()`` embeds
    every document for real before the index is usable."""

    def __init__(self, documents: List[Document], embeddings: np.ndarray, model_name: str) -> None:
        if len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"RetrievalIndex: {len(documents)} documents but {embeddings.shape[0]} embedding rows -- "
                f"these must match 1:1."
            )
        self.documents = documents
        self.embeddings = embeddings
        self.model_name = model_name

    @classmethod
    def build(cls, documents: List[Document], *, model_name: str = DEFAULT_MODEL_NAME) -> "RetrievalIndex":
        """Embed every document for real via a local sentence-transformer
        model. Raises ``ValueError`` for an empty document list -- an
        index over nothing is not a usable index, never silently built
        as one that will always return no results without saying why."""
        if not documents:
            raise ValueError("RetrievalIndex.build: documents is empty -- nothing to index.")

        model = _load_model(model_name)
        texts = [f"{d.title}\n\n{d.text}" for d in documents]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return cls(documents=list(documents), embeddings=np.asarray(embeddings, dtype=np.float32), model_name=model_name)

    def search(self, query: str, *, top_k: int = 5) -> List[SearchResult]:
        """Real cosine-similarity search -- every returned score is
        computed from the real query/document embeddings, never
        fabricated or defaulted. Returns fewer than ``top_k`` results if
        the corpus itself is smaller than that, never pads with
        placeholders."""
        if top_k <= 0:
            raise ValueError(f"RetrievalIndex.search: top_k must be positive, got {top_k}")

        model = _load_model(self.model_name)
        query_embedding = np.asarray(
            model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0], dtype=np.float32,
        )

        scores = self.embeddings @ query_embedding  # both L2-normalized -> dot product == cosine similarity
        top_k = min(top_k, len(self.documents))
        top_indices = np.argsort(-scores)[:top_k]
        return [SearchResult(document=self.documents[i], score=float(scores[i])) for i in top_indices]

    def save(self, path: str) -> None:
        """Persist the real, already-computed embeddings + document
        metadata -- no re-embedding needed on load."""
        doc_ids = [d.doc_id for d in self.documents]
        sources = [d.source for d in self.documents]
        titles = [d.title for d in self.documents]
        texts = [d.text for d in self.documents]
        np.savez_compressed(
            path, embeddings=self.embeddings, doc_ids=doc_ids, sources=sources, titles=titles, texts=texts,
            model_name=self.model_name,
        )

    @classmethod
    def load(cls, path: str) -> "RetrievalIndex":
        data = np.load(path, allow_pickle=False)
        documents = [
            Document(doc_id=str(doc_id), source=str(source), title=str(title), text=str(text))
            for doc_id, source, title, text in zip(data["doc_ids"], data["sources"], data["titles"], data["texts"])
        ]
        return cls(documents=documents, embeddings=data["embeddings"], model_name=str(data["model_name"]))
