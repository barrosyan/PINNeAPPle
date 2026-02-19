from __future__ import annotations

import os
from typing import List, Optional

from ..config import ResearcherConfig
from ..models import DiscoveryResult, RankedItem
from ..ranking import lexical_overlap_score, freshness_score, safe_parse_iso8601, sort_ranked
from ..sources.github_source import GithubSource
from ..sources.arxiv_source import ArxivSource

# v0.2: deep-quality (README/tree/key files)
try:
    from ..sources.github_fetcher import GithubFetcher
except Exception:  # pragma: no cover
    GithubFetcher = None


def _repo_quality_from_tree(paths: List[str]) -> dict:
    """
    Light heuristics from repo tree paths.
    """
    lower = [p.lower() for p in paths]

    has_tests = any(p.startswith(("tests/", "test/")) or "/tests/" in p for p in lower)
    has_ci = any(
        p.startswith(".github/workflows/") or "gitlab-ci" in p or "azure-pipelines" in p
        for p in lower
    )
    has_requirements = any(p.endswith("requirements.txt") or p.endswith("environment.yml") for p in lower)
    has_pyproject = any(p.endswith("pyproject.toml") for p in lower)
    has_setup = any(p.endswith("setup.py") for p in lower)
    has_docker = any(p.endswith("dockerfile") or p.endswith("compose.yml") or p.endswith("docker-compose.yml") for p in lower)

    # “entry points”
    has_run = any(p.endswith(("run.py", "train.py", "main.py")) for p in lower)

    score = 0.0
    score += 0.20 if has_tests else 0.0
    score += 0.15 if has_ci else 0.0
    score += 0.20 if (has_requirements or has_pyproject or has_setup) else 0.0
    score += 0.10 if has_docker else 0.0
    score += 0.10 if has_run else 0.0

    # cap 0..1 (remaining room goes to README quality)
    return {
        "has_tests": has_tests,
        "has_ci": has_ci,
        "has_requirements": has_requirements,
        "has_pyproject": has_pyproject,
        "has_setup": has_setup,
        "has_docker": has_docker,
        "has_run": has_run,
        "tree_quality_score": min(0.75, score),
    }


def discover(cfg: ResearcherConfig) -> DiscoveryResult:
    arxiv = ArxivSource()

    token = os.environ.get("GITHUB_TOKEN", None)
    gh = GithubSource(token=token)

    papers_raw = arxiv.search(cfg.topic, k=cfg.k_papers, categories=cfg.arxiv_categories)
    repos_raw = gh.search(
        cfg.topic,
        k=cfg.k_repos,
        min_stars=cfg.github_min_stars,
        languages=cfg.github_languages,
    )

    # ---------- Papers ----------
    papers: List[RankedItem] = []
    for p in papers_raw:
        lex = lexical_overlap_score(cfg.topic, f"{p.title}\n{p.summary}")
        fr = freshness_score(safe_parse_iso8601(p.published), half_life_days=365.0)

        # simple placeholders; paper quality can be improved later by PDF extraction in build_kb
        signals = 0.10
        quality = 0.05

        score = cfg.w_lexical * lex + cfg.w_freshness * fr + cfg.w_signals * signals + cfg.w_quality * quality
        why = [
            f"lexical={lex:.3f}",
            f"freshness={fr:.3f}",
            f"signals={signals:.3f}",
            f"quality={quality:.3f}",
        ]
        papers.append(
            RankedItem(
                type="paper",
                id=p.arxiv_id,
                title=p.title,
                url=p.url,
                score=float(score),
                why=why,
                meta={
                    "published": p.published,
                    "authors": p.authors,
                    "categories": p.categories,
                    "summary": p.summary,
                },
            )
        )

    # ---------- Repos ----------
    fetcher = None
    deep_quality_enabled = (GithubFetcher is not None) and bool(token)
    if deep_quality_enabled:
        fetcher = GithubFetcher(token=token)

    repos: List[RankedItem] = []
    for r in repos_raw:
        lex = lexical_overlap_score(cfg.topic, f"{r.title}\n{r.description}")
        fr = freshness_score(safe_parse_iso8601(r.updated_at), half_life_days=365.0)

        # signals: stars/forks scaled (simple, stable)
        signals = min(1.0, (r.stars / 5000.0) + (r.forks / 5000.0))

        # baseline quality: license + language
        quality = 0.0
        if r.license and r.license != "NOASSERTION":
            quality += 0.35
        if r.language:
            quality += 0.15

        deep_meta = {}
        # deep quality: README + tree + key files
        if fetcher and "/" in r.full_name:
            owner, repo = r.full_name.split("/", 1)
            try:
                default_branch = fetcher.get_default_branch(owner, repo)
                readme = fetcher.get_repo_readme(owner, repo, ref=default_branch)
                tree = fetcher.get_tree_recursive(owner, repo, ref=default_branch)
                paths = [f.path for f in tree]

                tree_q = _repo_quality_from_tree(paths)
                key_files = fetcher.pick_key_files(tree, max_files=30, max_file_size=200_000)

                readme_len = len(readme or "")
                # README quality contribution (0..0.25)
                # 0 at <400 chars, ~0.25 at >=4000 chars
                readme_q = max(0.0, min(0.25, (readme_len - 400) / 3600 * 0.25))

                quality = min(1.0, quality + tree_q["tree_quality_score"] + readme_q)

                deep_meta = {
                    "default_branch": default_branch,
                    "readme_len": readme_len,
                    "key_files": key_files,
                    **tree_q,
                    "readme_quality_score": readme_q,
                    "deep_quality_enabled": True,
                }
            except Exception:
                deep_meta = {"deep_quality_enabled": False, "deep_quality_error": True}

        score = cfg.w_lexical * lex + cfg.w_freshness * fr + cfg.w_signals * signals + cfg.w_quality * quality
        why = [
            f"lexical={lex:.3f}",
            f"freshness={fr:.3f}",
            f"signals(stars/forks)={signals:.3f}",
            f"quality={quality:.3f}" + (" (deep)" if deep_meta.get("deep_quality_enabled") else " (basic)"),
        ]

        meta = {
            "stars": r.stars,
            "forks": r.forks,
            "language": r.language,
            "updated_at": r.updated_at,
            "license": r.license,
            "description": r.description,
            **deep_meta,
        }

        repos.append(
            RankedItem(
                type="repo",
                id=r.full_name,
                title=r.title,
                url=r.url,
                score=float(score),
                why=why,
                meta=meta,
            )
        )

    return DiscoveryResult(
        topic=cfg.topic,
        papers=sort_ranked(papers)[: cfg.k_papers],
        repos=sort_ranked(repos)[: cfg.k_repos],
        meta={
            "note": (
                "v0.2 ranking: lexical overlap + signals + freshness + quality. "
                "Repo quality can use deep signals (README/tree/key files) when GITHUB_TOKEN is set."
            ),
            "deep_quality_enabled": bool(deep_quality_enabled),
        },
    )
