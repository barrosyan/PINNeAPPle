"""Tests for ``pinneapple_llm.org_knowledge.answer_with_context`` -- the
grounded-RAG-answer building block for org-management's "local LLM that
knows the org's own code" requirement.

Most tests here use a tiny fake index + the ``llm_call_fn`` override
(same override pattern as ``agent_loop.run_agent_loop``'s own tests) so
they run deterministically with no live LLM backend and no
sentence-transformers dependency. One test additionally exercises a real
local Ollama call end to end, skipped (not failed) if no server is
reachable at 127.0.0.1:11434 -- matching this repo's live-dependency test
convention (see ``tests/pinneapple_analysis/test_retrieval.py``'s arXiv
reachability skip for the same pattern).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest

from pinneapple_analysis.retrieval.corpus import Document
from pinneapple_llm.org_knowledge import OrgAnswer, answer_with_context


@dataclass
class _FakeSearchResult:
    document: Document
    score: float


class _FakeIndex:
    """Minimal stand-in for RetrievalIndex: same ``search()`` interface,
    real (not embedded) documents, deterministic ordering."""

    def __init__(self, documents: List[Document]):
        self._documents = documents

    def search(self, query: str, *, top_k: int = 5):
        return [_FakeSearchResult(document=d, score=1.0) for d in self._documents[:top_k]]


class _EmptyIndex:
    def search(self, query: str, *, top_k: int = 5):
        return []


def test_answer_with_context_returns_none_answer_when_index_is_empty():
    result = answer_with_context("what does this repo do?", _EmptyIndex())
    assert isinstance(result, OrgAnswer)
    assert result.answer is None
    assert result.context_used == []


def test_answer_with_context_uses_llm_call_fn_override_and_grounds_prompt_in_real_docs():
    docs = [
        Document(doc_id="code:demo:core.py#0", source="/repo/core.py", title="demo/core.py",
                  text="def add(a, b):\n    return a + b\n"),
        Document(doc_id="code:demo:util.py#0", source="/repo/util.py", title="demo/util.py",
                  text="CONST = 42\n"),
    ]
    index = _FakeIndex(docs)

    captured = {}

    def fake_llm(prompt: str, *, system: str = "") -> str:
        captured["prompt"] = prompt
        captured["system"] = system
        return "add() returns a + b, per passage [1]."

    result = answer_with_context(
        "what does add() do?", index, top_k=2, llm_call_fn=fake_llm,
    )

    assert result.answer == "add() returns a + b, per passage [1]."
    # the prompt actually contains the real retrieved text, not a placeholder
    assert "def add(a, b):" in captured["prompt"]
    assert "/repo/core.py" in captured["prompt"]
    assert "ONLY" in captured["prompt"] or "only" in captured["prompt"].lower()
    assert len(result.context_used) == 2
    assert result.context_used[0]["source"] == "/repo/core.py"
    assert result.context_used[0]["title"] == "demo/core.py"
    assert result.context_used[0]["score"] == 1.0


def test_answer_with_context_respects_top_k():
    docs = [
        Document(doc_id=f"code:demo:f{i}.py#0", source=f"/repo/f{i}.py", title=f"demo/f{i}.py", text=f"x = {i}\n")
        for i in range(5)
    ]
    index = _FakeIndex(docs)

    def fake_llm(prompt: str, *, system: str = "") -> str:
        return "ok"

    result = answer_with_context("q", index, top_k=2, llm_call_fn=fake_llm)
    assert len(result.context_used) == 2


def _ollama_reachable() -> bool:
    try:
        import requests

        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_reachable(), reason="no local Ollama server reachable at 127.0.0.1:11434")
def test_answer_with_context_real_ollama_call_end_to_end():
    """Real integration test: no llm_call_fn override, no mocked HTTP --
    an actual local Ollama server answers a grounded question. Only
    checks that a real, non-empty answer came back and that the request
    really carried the retrieved context; does not assert on the model's
    exact wording (a real local LLM's phrasing is not something to pin
    down in a test)."""
    docs = [
        Document(
            doc_id="code:demo:core.py#0", source="/repo/core.py", title="demo/core.py",
            text="def multiply_by_three(x):\n    return x * 3\n",
        ),
    ]
    index = _FakeIndex(docs)

    result = answer_with_context(
        "According to the context, what does multiply_by_three do?",
        index, top_k=1, provider="ollama", model="llama3.2:3b",
    )
    assert result.answer is not None
    assert len(result.answer.strip()) > 0
    assert len(result.context_used) == 1
