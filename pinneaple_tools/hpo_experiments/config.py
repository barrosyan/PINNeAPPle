"""ResearcherConfig for topic, k_papers, k_repos, ranking weights."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class ResearcherConfig:
    """
    Researcher configuration.

    - topic: query topic (free text)
    - k_papers/repos: number of items to retrieve and rank
    - out_dir: where researcher runs will be persisted
    """

    topic: str
    k_papers: int = 10
    k_repos: int = 10

    # Ranking weights (v0.1 uses only heuristics + lexical overlap)
    w_lexical: float = 0.55
    w_signals: float = 0.30
    w_freshness: float = 0.10
    w_quality: float = 0.05

    # GitHub filters
    github_min_stars: int = 20
    github_languages: Optional[Sequence[str]] = None

    # arXiv filters
    arxiv_categories: Optional[Sequence[str]] = None

    # Storage
    out_dir: str = "runs/researcher"
    cache_dir: str = ".cache/pinneaple_researcher"

    # Chunking / KB
    chunk_chars: int = 1800
    chunk_overlap: int = 200
