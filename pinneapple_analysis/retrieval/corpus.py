"""Real, non-fabricated corpus sources for semantic retrieval.

Every :class:`Document` here comes from content that already exists and
is already tested elsewhere in this codebase -- this module never
invents text. Three real sources:

1. Every ``pinneapple_analysis.verification.*`` module's own docstring
   (the module-level docstring, read via ``inspect.getdoc`` -- the exact
   text a developer reading the source already sees, not a rewrite).
2. Every real preset's ``knowledge_graph.explain_preset(...)['llm_context']``
   -- already a real, structured, citation-aware summary built from
   PINNeAPPle's own live preset registry (see ``knowledge_graph.py``'s
   own docstring for how that's assembled; this module does not
   duplicate that logic, only calls it).
3. Real external literature -- real arXiv papers, fetched live via
   ``pinneapple_tools.hpo_experiments.sources.ArxivSource`` (the exact
   same real-API-backed search ``pinneapple_llm.research.search_literature``
   already uses; see that module's own docstring for why this codebase
   never asks an LLM to "recall" a paper from training data -- the same
   anti-hallucination discipline applies here). Every literature
   :class:`Document`'s ``source`` is a real, dereferenceable
   ``https://arxiv.org/abs/<id>`` URL from the API response itself, and
   its ``text`` is the paper's own real title + abstract, never an LLM
   paraphrase.

Sources 1-2 are always available offline and instant. Source 3 needs
network access and is opt-in (``include_literature=True``) precisely
because it's slower and can fail without connectivity -- see
:func:`build_corpus`.
"""
from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from dataclasses import dataclass
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "Document",
    "build_corpus",
    "fetch_literature_documents",
    "build_literature_corpus",
    "DEFAULT_LITERATURE_TOPICS",
]


@dataclass(frozen=True)
class Document:
    doc_id: str
    source: str  # e.g. "verification_module:architecture_recommendation" or "preset:burgers_1d"
    title: str
    text: str


def _verification_module_documents() -> List[Document]:
    import pinneapple_analysis.verification as verification_pkg

    docs: List[Document] = []
    for _finder, name, is_pkg in pkgutil.iter_modules(verification_pkg.__path__):
        if is_pkg or name.startswith("_"):
            continue
        module = importlib.import_module(f"pinneapple_analysis.verification.{name}")
        docstring = inspect.getdoc(module)
        if not docstring:
            continue
        docs.append(Document(
            doc_id=f"verification_module:{name}",
            source=f"pinneapple_analysis.verification.{name}",
            title=name.replace("_", " "),
            text=docstring,
        ))
    return docs


def _preset_documents() -> List[Document]:
    from pinneapple_analysis.verification.knowledge_graph import build_knowledge_graph, explain_preset

    g = build_knowledge_graph()
    preset_names = sorted(
        data["name"] for _node_id, data in g.nodes(data=True) if data.get("type") == "Preset"
    )
    docs: List[Document] = []
    for preset_name in preset_names:
        try:
            info = explain_preset(g, preset_name)
        except KeyError:
            continue
        context = info.get("llm_context")
        if not context:
            continue
        docs.append(Document(
            doc_id=f"preset:{preset_name}",
            source=f"knowledge_graph.explain_preset({preset_name!r})",
            title=preset_name.replace("_", " "),
            text=context,
        ))
    return docs


# A real, curated starter set of topics covering this platform's actual
# scope (PINNs, neural operators, UQ, digital twins, turbulence closure,
# inverse problems) -- not an arbitrary or exhaustive literature survey,
# just enough real coverage to prove external-literature retrieval works
# end to end. Expand this list (or pass a custom one) as real demand
# for specific topics shows up.
DEFAULT_LITERATURE_TOPICS: List[str] = [
    "physics informed neural networks",
    "fourier neural operator",
    "deep operator network",
    "uncertainty quantification scientific machine learning",
    "turbulence closure machine learning",
    "digital twin physics simulation",
    "inverse problems deep learning PDE",
    "graph neural network mesh simulation",
]


def _arxiv_paper_to_document(paper) -> Document:
    return Document(
        doc_id=f"arxiv:{paper.arxiv_id}",
        source=paper.url,  # a real, dereferenceable https://arxiv.org/abs/<id> URL from the API response
        title=paper.title,
        text=f"{paper.title}\n\n{paper.summary}",
    )


def fetch_literature_documents(query: str, *, k: int = 5, timeout_s: int = 30) -> List[Document]:
    """Fetch real papers from the live arXiv API for one query -- a thin
    wrapper over ``ArxivSource.search`` (the same real search
    ``pinneapple_llm.research.search_literature`` uses), turning each
    real ``ArxivPaper`` into a :class:`Document`. Never fabricates a
    result: a network failure raises (the caller decides whether to
    catch it), and an empty real result set returns an empty list, never
    a placeholder.
    """
    from pinneapple_tools.hpo_experiments.sources import ArxivSource

    papers = ArxivSource().search(query, k=k, timeout_s=timeout_s)
    return [_arxiv_paper_to_document(p) for p in papers]


def build_literature_corpus(
    topics: Optional[Sequence[str]] = None, *, k_per_topic: int = 5, timeout_s: int = 30,
) -> List[Document]:
    """Fetch real papers for several topics from the live arXiv API,
    de-duplicated by arXiv id. Skips (logs a warning, does not raise) a
    topic whose fetch fails -- e.g. no network for that one call -- so
    one bad query doesn't abort the whole corpus build; if EVERY topic
    fails this returns an empty list, which is itself an honest signal
    (never padded with placeholder documents)."""
    topics = list(topics) if topics is not None else DEFAULT_LITERATURE_TOPICS
    docs: List[Document] = []
    seen_ids = set()
    for topic in topics:
        try:
            topic_docs = fetch_literature_documents(topic, k=k_per_topic, timeout_s=timeout_s)
        except Exception as exc:
            logger.warning(f"build_literature_corpus: fetching topic {topic!r} failed, skipping it: {exc}")
            continue
        for doc in topic_docs:
            if doc.doc_id not in seen_ids:
                docs.append(doc)
                seen_ids.add(doc.doc_id)
    return docs


def build_corpus(
    *, include_literature: bool = False, literature_topics: Optional[Sequence[str]] = None,
    literature_k_per_topic: int = 5,
) -> List[Document]:
    """Assemble the real, current corpus -- recomputed fresh every call
    (never a stale cached snapshot), so it always reflects whatever
    modules/presets are actually registered right now.

    ``include_literature=True`` additionally fetches real arXiv papers
    (see :func:`build_literature_corpus`) -- opt-in and off by default
    because it needs network access and is meaningfully slower (several
    real HTTP calls) than the always-available internal sources.
    """
    docs = _verification_module_documents() + _preset_documents()
    if include_literature:
        docs += build_literature_corpus(literature_topics, k_per_topic=literature_k_per_topic)
    return docs
