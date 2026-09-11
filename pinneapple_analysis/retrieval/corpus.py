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

import hashlib
import importlib
import inspect
import logging
import os
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
    "code_repo_documents",
    "iter_repo_files",
    "chunk_text",
    "file_fingerprint",
    "DEFAULT_CODE_EXTENSIONS",
    "DEFAULT_EXCLUDE_DIRS",
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


# ---------------------------------------------------------------------------
# 4. Real local source code -- a fourth Document source, for indexing an
# organisation's own repositories (any local clone, not just this one).
#
# Built for org-management's "knowledge base of all code already written
# across the org" requirement, but deliberately placed here rather than in
# a downstream repo: it's the same Document/RetrievalIndex machinery as
# sources 1-3 above, and belongs next to them rather than duplicated. Every
# Document's ``text`` is a real, unmodified slice of a real file already on
# disk -- never a summary, never LLM-generated, never fabricated.
# ---------------------------------------------------------------------------

# A real, practical default for source-controlled projects in this org
# (Python, TS/JS, docs, config) -- deliberately excludes binary/data/lock
# files, which would either fail to decode as text or add no retrievable
# signal (embedding a lockfile's hash noise is not useful semantic search).
DEFAULT_CODE_EXTENSIONS = frozenset({
    ".py", ".md", ".rst", ".txt",
    ".ts", ".tsx", ".js", ".jsx",
    ".toml", ".yaml", ".yml", ".json", ".cfg", ".ini",
})

# Directories that are either not source (build/dependency output) or
# would otherwise dominate the corpus with irrelevant/duplicated content.
DEFAULT_EXCLUDE_DIRS = frozenset({
    ".git", ".venv", "venv", "__pycache__", "node_modules", "dist", "build",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "egg-info", ".idea", ".vscode",
    "site-packages", ".tox", "htmlcov", ".next", ".turbo",
})


def iter_repo_files(
    repo_root: str,
    *,
    extensions: Sequence[str] = DEFAULT_CODE_EXTENSIONS,
    exclude_dirs: Sequence[str] = DEFAULT_EXCLUDE_DIRS,
    max_file_bytes: int = 250_000,
) -> List[str]:
    """Real ``os.walk`` over *repo_root*, returning every real file path
    whose extension is in *extensions*, skipping *exclude_dirs* (by
    directory *name*, matched at any depth) and any file over
    *max_file_bytes* (a large data/generated file is skipped, not
    truncated silently into a misleading partial document -- callers that
    want partial coverage of a huge file should raise the limit instead).
    Never fabricates a path: every entry returned really exists and passed
    a real ``os.path.getsize`` check at call time."""
    exclude = set(exclude_dirs)
    found: List[str] = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in exclude and not d.startswith(".")]
        for fn in filenames:
            if os.path.splitext(fn)[1] not in extensions:
                continue
            path = os.path.join(dirpath, fn)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if 0 < size <= max_file_bytes:
                found.append(path)
    return sorted(found)


def chunk_text(text: str, *, max_chars: int = 4000, overlap: int = 200) -> List[str]:
    """Simple, real character-window chunker -- splits on a blank-line
    boundary near each window edge when one exists (so a chunk doesn't cut
    a function/class in half more often than necessary), falling back to a
    hard cut otherwise. No chunk is ever padded/rewritten -- each is a
    verbatim slice of *text*. Returns ``[text]`` unchanged if it already
    fits in one chunk."""
    if max_chars <= 0:
        raise ValueError(f"chunk_text: max_chars must be positive, got {max_chars}")
    text = text.strip("\n")
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            # Prefer to break at the last blank line inside this window
            # (keeps a chunk from splitting mid-function where avoidable).
            boundary = text.rfind("\n\n", start, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip("\n")
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)  # always advance, even if overlap >= window
    return chunks


def file_fingerprint(path: str) -> str:
    """Real content hash (sha256, hex) of a file's current bytes on disk --
    the change-detection primitive the org-management code-index refresh
    job uses to decide whether a file needs re-embedding. Distinct from
    mtime: a hash catches a change even if mtime didn't update (e.g. a
    git checkout that preserves timestamps), and is stable across copies."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def code_repo_documents(
    repo_root: str,
    *,
    repo_name: Optional[str] = None,
    extensions: Sequence[str] = DEFAULT_CODE_EXTENSIONS,
    exclude_dirs: Sequence[str] = DEFAULT_EXCLUDE_DIRS,
    max_file_bytes: int = 250_000,
    max_chars_per_chunk: int = 4000,
    only_paths: Optional[Sequence[str]] = None,
) -> List[Document]:
    """Real source-code Documents for one local repository clone.

    Walks *repo_root* (via :func:`iter_repo_files`) and turns every real
    text file found into one or more :class:`Document` chunks (via
    :func:`chunk_text`) -- ``doc_id`` is
    ``"code:{repo_name}:{relative/path}#{chunk_index}"`` and ``source`` is
    the real absolute file path, so every retrieval hit traces back to a
    real, openable file, never a synthesized description of one.

    ``only_paths``, if given, restricts the walk's output to exactly this
    set of already-known file paths (must be absolute paths as returned by
    :func:`iter_repo_files`) instead of re-walking the whole tree -- this
    is what lets a periodic refresh job re-embed only the files it already
    determined (via :func:`file_fingerprint`) have changed, rather than
    rebuilding the entire corpus from scratch every run.
    """
    repo_name = repo_name or os.path.basename(os.path.normpath(repo_root))
    if only_paths is not None:
        paths = list(only_paths)
    else:
        paths = iter_repo_files(
            repo_root, extensions=extensions, exclude_dirs=exclude_dirs, max_file_bytes=max_file_bytes,
        )

    docs: List[Document] = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError as exc:
            logger.warning(f"code_repo_documents: could not read {path!r}, skipping it: {exc}")
            continue
        if not text.strip():
            continue
        relpath = os.path.relpath(path, repo_root)
        for i, chunk in enumerate(chunk_text(text, max_chars=max_chars_per_chunk)):
            docs.append(Document(
                doc_id=f"code:{repo_name}:{relpath}#{i}",
                source=path,
                title=f"{repo_name}/{relpath}",
                text=chunk,
            ))
    return docs
