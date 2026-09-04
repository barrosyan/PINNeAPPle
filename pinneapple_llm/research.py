"""Literature/repository search integrated with an LLM -- built entirely
on PINNeAPPle's own already-existing, real API-backed search
(``pinneapple_tools.hpo_experiments.sources.ArxivSource``/
``GithubSource``, which call the real arXiv and GitHub search APIs), never
an LLM asked to "recall" papers or repositories from training data (which
is exactly the class of hallucination -- plausible-looking but
non-existent or wrong citations -- this module exists to avoid).

The LLM's role here is strictly downstream of the real results: it may be
asked to *summarise or rank the papers/repos actually returned by the real
API calls*, never to add to or invent items in that list. Every citation
in :class:`ResearchReport` traces back to a real ``ArxivPaper``/
``GithubRepo`` object with a real, dereferenceable URL from the API
response itself, not from the LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ._dispatch import call_llm


@dataclass
class ResearchReport:
    query: str
    papers: List[dict] = field(default_factory=list)   # real ArxivPaper.__dict__ entries
    repos: List[dict] = field(default_factory=list)     # real GithubRepo.__dict__ entries
    llm_summary: Optional[str] = None                   # summary of the ABOVE, never a source of new items


def search_literature(
    query: str,
    *,
    k_papers: int = 5,
    k_repos: int = 5,
    summarize: bool = False,
    provider: str = "anthropic",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    conversation_store=None,
) -> ResearchReport:
    """Search arXiv + GitHub for real papers/repositories matching
    ``query``, optionally with an LLM summary of the (real) results.

    Parameters
    ----------
    query : search topic.
    k_papers, k_repos : how many results to fetch from each real source.
    summarize : if True, ask the LLM to write a short summary of the
        results actually returned (grounded -- the prompt includes only
        the real titles/abstracts/URLs fetched above, and the summary is
        clearly a *description* of them, not a new source of citations).
    provider, model, api_key : LLM call config, only used if
        ``summarize=True``.
    """
    from pinneapple_tools.hpo_experiments.sources import ArxivSource
    from pinneapple_tools.hpo_experiments.sources.github_source import GithubSource

    papers = [p.__dict__ for p in ArxivSource().search(query, k=k_papers)]
    try:
        repos = [r.__dict__ for r in GithubSource().search(query, k=k_repos)]
    except RuntimeError:
        # GithubSource needs `requests`; degrade to papers-only rather than
        # hard-failing the whole search over one optional dependency.
        repos = []

    report = ResearchReport(query=query, papers=papers, repos=repos)

    if summarize:
        prompt = (
            f"Summarise the following REAL search results for the query \"{query}\" in a few sentences. "
            "Do NOT mention, cite, or imply the existence of any paper or repository not listed below -- "
            "only summarise what is actually here.\n\n"
            f"PAPERS:\n{papers}\n\nREPOSITORIES:\n{repos}\n"
        )
        if provider == "anthropic":
            report.llm_summary = _call_anthropic(prompt, model, api_key)
        elif provider == "openai":
            report.llm_summary = _call_openai(prompt, model, api_key)
        else:
            raise ValueError(f"unknown provider '{provider}'")

    return report
