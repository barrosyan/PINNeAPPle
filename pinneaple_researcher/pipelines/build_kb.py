"""KB build pipeline: papers and repos to KBStore."""
from __future__ import annotations

import json
import os
from typing import List, Tuple

from ..config import ResearcherConfig
from ..kb.store import KBStore
from ..models import KBArtifact, KBIndex, DiscoveryResult, RankedItem
from ..sources.github_fetcher import GithubFetcher
from ..sources.arxiv_pdf import arxiv_pdf_url, download_pdf, extract_pdf_text, split_sections


def _paper_to_markdown_with_sections(it: RankedItem, run_dir: str) -> str:
    arxiv_id = it.id
    pdf_url = arxiv_pdf_url(arxiv_id, use_export_subdomain=True)
    pdf_path = os.path.join(run_dir, "papers", arxiv_id.replace("/", "_"), "paper.pdf")

    sections_md = ""
    try:
        download_pdf(pdf_url, pdf_path)
        text = extract_pdf_text(pdf_path)
        sections = split_sections(text)
        # prefer a few sections (keep it small)
        for key in ("abstract", "method", "experiments", "conclusion"):
            if key in sections:
                sections_md += f"\n\n## {key.title()}\n\n{sections[key]}\n"
        sections_md += f"\n\n## FullText (truncated)\n\n{sections.get('full_text','')}\n"
    except Exception as e:
        sections_md = f"\n\n> PDF fetch/extract failed: {e}\n"

    s = it.meta.get("summary", "") or ""
    authors = it.meta.get("authors", []) or []
    cats = it.meta.get("categories", []) or []
    published = it.meta.get("published", "") or ""

    return (
        f"# {it.title}\n\n"
        f"- **arXiv id**: {it.id}\n"
        f"- **URL**: {it.url}\n"
        f"- **PDF**: {pdf_url}\n"
        f"- **Published**: {published}\n"
        f"- **Authors**: {', '.join(authors)}\n"
        f"- **Categories**: {', '.join(cats)}\n\n"
        f"## Summary (API)\n\n{s}\n"
        f"{sections_md}\n"
    )


def _repo_to_markdown_fetched(it: RankedItem, *, token: str | None, ref: str | None) -> str:
    full = it.id  # owner/repo
    if "/" not in full:
        return f"# {it.title}\n\nInvalid repo id: {full}\n"
    owner, repo = full.split("/", 1)

    fetcher = GithubFetcher(token=token)

    # resolve branch if ref not provided
    if not ref:
        try:
            ref = fetcher.get_default_branch(owner, repo)
        except Exception:
            ref = "main"

    readme = ""
    try:
        readme = fetcher.get_repo_readme(owner, repo, ref=ref)
    except Exception as e:
        readme = f"(failed to fetch README: {e})"

    files = []
    try:
        files = fetcher.get_tree_recursive(owner, repo, ref=ref)
    except Exception as e:
        files = []

    key_paths = fetcher.pick_key_files(files, max_files=30, max_file_size=200_000)

    key_files_md = ""
    for p in key_paths:
        try:
            txt = fetcher.get_file_text(owner, repo, p, ref=ref)
            txt = txt[:40_000]
            key_files_md += f"\n\n### {p}\n\n```text\n{txt}\n```\n"
        except Exception as e:
            key_files_md += f"\n\n### {p}\n\n> failed to fetch: {e}\n"

    tree_preview = "\n".join([f.path for f in files[:500]])  # truncated
    meta = it.meta or {}
    d = meta.get("description", "") or ""
    stars = meta.get("stars", 0)
    forks = meta.get("forks", 0)
    lang = meta.get("language", None)
    updated = meta.get("updated_at", "")
    lic = meta.get("license", None)

    return (
        f"# {it.title}\n\n"
        f"- **Repo**: {it.id}\n"
        f"- **URL**: {it.url}\n"
        f"- **Ref**: {ref}\n"
        f"- **Stars/Forks**: {stars}/{forks}\n"
        f"- **Language**: {lang}\n"
        f"- **Updated**: {updated}\n"
        f"- **License**: {lic}\n\n"
        f"## Description\n\n{d}\n\n"
        f"## README\n\n{readme}\n\n"
        f"## Repo tree (first 500 paths)\n\n```text\n{tree_preview}\n```\n"
        f"## Key files (heuristic)\n{key_files_md}\n"
    )


def build_kb(cfg: ResearcherConfig, discovery: DiscoveryResult) -> KBIndex:
    store = KBStore(out_dir=cfg.out_dir)
    run_dir = store.new_run_dir(cfg.topic)

    store.write_manifest(run_dir, topic=cfg.topic, papers=discovery.papers, repos=discovery.repos)

    artifacts: List[KBArtifact] = []

    # Papers: include PDF section extraction
    for it in discovery.papers:
        md = _paper_to_markdown_with_sections(it, run_dir)
        artifacts.append(store.write_artifact_text(run_dir, item=it, text=md))

    # Repos: fetch README + tree + key files
    token = os.environ.get("GITHUB_TOKEN")
    for it in discovery.repos:
        md = _repo_to_markdown_fetched(it, token=token, ref=None)
        artifacts.append(store.write_artifact_text(run_dir, item=it, text=md))

    idx = store.build_index(
        run_dir=run_dir,
        topic=cfg.topic,
        artifacts=artifacts,
        chunk_chars=cfg.chunk_chars,
        chunk_overlap=cfg.chunk_overlap,
    )
    return idx
