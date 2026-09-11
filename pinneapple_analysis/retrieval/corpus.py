"""Real, non-fabricated corpus sources for semantic retrieval.

Every :class:`Document` here comes from content that already exists and
is already tested elsewhere in this codebase -- this module never
invents text. Two real sources, both already-tested and already-cited
where citations exist:

1. Every ``pinneapple_analysis.verification.*`` module's own docstring
   (the module-level docstring, read via ``inspect.getdoc`` -- the exact
   text a developer reading the source already sees, not a rewrite).
2. Every real preset's ``knowledge_graph.explain_preset(...)['llm_context']``
   -- already a real, structured, citation-aware summary built from
   PINNeAPPle's own live preset registry (see ``knowledge_graph.py``'s
   own docstring for how that's assembled; this module does not
   duplicate that logic, only calls it).

This corpus is intentionally small (as of writing: ~14 verification
module docstrings + 65 real presets = ~79 documents) -- see this
package's own docstring / ``ROADMAP.md`` in downstream products for why
a bigger corpus (accumulated real papers, accumulated real run
provenance) is future work, not something fabricated here to look more
complete.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import dataclass
from typing import List

__all__ = ["Document", "build_corpus"]


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


def build_corpus() -> List[Document]:
    """Assemble the real, current corpus -- recomputed fresh every call
    (never a stale cached snapshot), so it always reflects whatever
    modules/presets are actually registered right now."""
    return _verification_module_documents() + _preset_documents()
