from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


@dataclass
class GithubRepo:
    full_name: str
    title: str
    url: str
    description: str
    stars: int
    forks: int
    language: Optional[str]
    updated_at: str
    license: Optional[str]


class GithubSource:
    """
    GitHub Search API wrapper (repos discovery).
    Token optional but recommended (GITHUB_TOKEN).
    """

    def __init__(self, token: Optional[str] = None):
        self.token = token

    def _headers(self) -> Dict[str, str]:
        h = {"Accept": "application/vnd.github+json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def search(
        self,
        topic: str,
        *,
        k: int = 10,
        min_stars: int = 0,
        languages: Optional[Sequence[str]] = None,
        timeout_s: int = 30,
    ) -> List[GithubRepo]:
        if requests is None:
            raise RuntimeError("GithubSource requires 'requests'. Install: pip install -e \".[research]\"")

        q = f"{topic} stars:>={int(min_stars)}"
        if languages:
            # first language only (simple v0.2)
            q += f" language:{languages[0]}"

        url = "https://api.github.com/search/repositories"
        r = requests.get(
            url,
            headers=self._headers(),
            params={"q": q, "sort": "stars", "order": "desc", "per_page": int(k)},
            timeout=timeout_s,
        )
        r.raise_for_status()
        data = r.json()

        out: List[GithubRepo] = []
        for it in data.get("items", []):
            lic = it.get("license") or {}
            out.append(
                GithubRepo(
                    full_name=it["full_name"],
                    title=it["name"],
                    url=it["html_url"],
                    description=it.get("description") or "",
                    stars=int(it.get("stargazers_count", 0)),
                    forks=int(it.get("forks_count", 0)),
                    language=it.get("language"),
                    updated_at=it.get("updated_at", ""),
                    license=(lic.get("spdx_id") if isinstance(lic, dict) else None),
                )
            )

        if languages:
            allowed = set(languages)
            out = [x for x in out if (x.language in allowed)]

        return out
