"""Grounded LLM answers over a real local
:class:`pinneapple_analysis.retrieval.RetrievalIndex` -- the "local LLM
that knows about the organisation's own code" building block, built for
(and by) ``org-management``'s knowledge-base requirement, but placed here
rather than in that downstream repo for the same reason ``research.py``
and ``guardrail.py`` live in this package: this is the LLM-facing half of
a capability whose *evidence-producing* half (the real corpus + real
cosine-similarity index) already lives in
``pinneapple_analysis.retrieval`` -- see that package's ``corpus.py``
(specifically ``code_repo_documents``, added alongside this module) for
the source-of-truth documents this module is grounded in.

Same anti-hallucination discipline as ``research.py``: the LLM is asked
to synthesise an answer *strictly from the passages a real similarity
search actually retrieved*, never to recall file/module contents from its
own training data (which, for an internal/private org codebase, would not
even be a plausible-looking hallucination -- it would be pure invention,
since the model was never trained on this code at all). Every entry in
:class:`OrgAnswer.context_used` traces back to a real
``Document.source`` (a real, openable file path or URL) returned by the
index's own ``search()``, never invented by this module or the LLM.

"Learning from the org's code over time" (see this repo's ROADMAP.md/
``org-management/README.md`` for the honest scope of that phrase in v0.1)
happens at two independent layers, deliberately not conflated:

1. **Retrieval freshness** (what this module benefits from immediately):
   as ``org-management``'s periodic refresh job re-scans the org's repos
   and re-embeds changed files into the index passed in here, every
   subsequent :func:`answer_with_context` call is grounded in whatever is
   *currently* on disk -- no retraining needed for this layer to "know
   about" a file added yesterday.
2. **Weight fine-tuning** (real, already-built, NOT wired up by this
   module): every call this module makes can be logged to a
   :class:`~pinneapple_llm.conversation_store.ConversationStore` (pass
   ``conversation_store=``), which is exactly the dataset
   :mod:`pinneapple_llm.finetune` already knows how to turn into a LoRA
   fine-tune of a local HF model (see that module's own docstring for the
   real scope/limits of that path, e.g. it does not fine-tune an Ollama
   model directly). This module accumulates that dataset; it does not
   run the fine-tune itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ._dispatch import call_llm

__all__ = ["OrgAnswer", "answer_with_context"]


@dataclass
class OrgAnswer:
    question: str
    context_used: List[Dict[str, Any]] = field(default_factory=list)  # real {source, title, score} entries
    answer: Optional[str] = None  # None if the index returned nothing to ground an answer in


_GROUNDED_SYSTEM_PROMPT = (
    "You are an internal engineering assistant answering questions about a specific "
    "organisation's own codebase. You will be given numbered context passages, each with "
    "a real file/URL source. Answer using ONLY information present in those passages. "
    "If they do not contain enough information to answer, say so plainly instead of "
    "guessing or inventing file names, functions, or behaviour that isn't shown to you. "
    "Cite the passage number(s) you relied on."
)


def answer_with_context(
    question: str,
    retrieval_index: Any,
    *,
    top_k: int = 5,
    provider: str = "ollama",
    model: Optional[str] = "llama3.2:3b",
    api_key: Optional[str] = None,
    conversation_store: Any = None,
    module: str = "org_knowledge",
    llm_call_fn: Optional[Callable[..., str]] = None,
    **provider_kwargs: Any,
) -> OrgAnswer:
    """Answer *question* grounded in *retrieval_index*'s real search
    results (any object exposing a ``search(query, top_k=...) ->
    List[SearchResult]`` method -- i.e. a real
    ``pinneapple_analysis.retrieval.RetrievalIndex``, matching that
    class's own interface).

    Returns ``OrgAnswer(answer=None)`` (never a fabricated "I don't know
    anything" from the LLM) if the index has literally nothing to return
    for this query -- there is no passage to ground an answer in, so no
    LLM call is made at all.

    ``llm_call_fn``, if given, overrides the real
    :func:`pinneapple_llm._dispatch.call_llm` dispatch with a callable of
    shape ``(prompt: str, *, system: str) -> str`` -- the same override
    pattern ``agent_loop.run_agent_loop`` uses, for deterministic testing
    without a live LLM backend. Default provider is ``"ollama"`` (a local
    model, no paid API, no data leaving the machine) since this is meant
    to run continuously over a private/internal corpus.
    """
    results = retrieval_index.search(question, top_k=top_k)
    if not results:
        return OrgAnswer(question=question, context_used=[], answer=None)

    context_block = "\n\n".join(
        f"[{i + 1}] SOURCE: {r.document.source}\nTITLE: {r.document.title}\n{r.document.text}"
        for i, r in enumerate(results)
    )
    prompt = (
        f"QUESTION: {question}\n\n"
        f"CONTEXT PASSAGES (the ONLY information you may use):\n{context_block}\n"
    )

    if llm_call_fn is not None:
        answer_text = llm_call_fn(prompt, system=_GROUNDED_SYSTEM_PROMPT)
    else:
        answer_text = call_llm(
            prompt,
            provider=provider,
            model=model,
            api_key=api_key,
            system=_GROUNDED_SYSTEM_PROMPT,
            module=module,
            conversation_store=conversation_store,
            **provider_kwargs,
        )

    context_used = [
        {"source": r.document.source, "title": r.document.title, "score": r.score} for r in results
    ]
    return OrgAnswer(question=question, context_used=context_used, answer=answer_text)
