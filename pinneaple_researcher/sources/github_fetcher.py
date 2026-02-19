from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


@dataclass
class RepoFile:
    path: str
    sha: str
    size: int
    url: str


class GithubFetcher:
    """
    Fetches README, repo tree, and key files using GitHub REST API.

    - README endpoint exists in contents APIs.
    - Recursive listing should use Git Trees API (recursive=1).
    :contentReference[oaicite:3]{index=3}
    """

    def __init__(self, token: Optional[str] = None, api_base: str = "https://api.github.com"):
        if requests is None:
            raise RuntimeError("GithubFetcher requires 'requests'. Install: pip install -e \".[research]\"")
        self.api_base = api_base.rstrip("/")
        self.token = token or os.environ.get("GITHUB_TOKEN")

    def _headers(self, accept: str = "application/vnd.github+json") -> Dict[str, str]:
        h = {"Accept": accept}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def get_default_branch(self, owner: str, repo: str, timeout_s: int = 30) -> str:
        url = f"{self.api_base}/repos/{owner}/{repo}"
        r = requests.get(url, headers=self._headers(), timeout=timeout_s)
        r.raise_for_status()
        return r.json().get("default_branch", "main")

    def get_repo_readme(self, owner: str, repo: str, ref: Optional[str] = None, timeout_s: int = 30) -> str:
        url = f"{self.api_base}/repos/{owner}/{repo}/readme"
        params = {"ref": ref} if ref else None
        r = requests.get(url, headers=self._headers(), params=params, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        content = data.get("content", "") or ""
        enc = data.get("encoding", "base64")
        if enc == "base64" and content:
            return base64.b64decode(content).decode("utf-8", errors="replace")
        # fallback: try raw
        raw_url = data.get("download_url")
        if raw_url:
            rr = requests.get(raw_url, timeout=timeout_s)
            rr.raise_for_status()
            return rr.text
        return ""

    def get_tree_recursive(self, owner: str, repo: str, ref: str, timeout_s: int = 60) -> List[RepoFile]:
        """
        Uses:
          GET /repos/{owner}/{repo}/git/trees/{tree_sha}?recursive=1
        ref can be branch, tag, or commit SHA.
        :contentReference[oaicite:4]{index=4}
        """
        # first: resolve ref to commit SHA
        ref_url = f"{self.api_base}/repos/{owner}/{repo}/commits/{ref}"
        r = requests.get(ref_url, headers=self._headers(), timeout=timeout_s)
        r.raise_for_status()
        commit = r.json()
        tree_sha = commit.get("commit", {}).get("tree", {}).get("sha")
        if not tree_sha:
            return []

        url = f"{self.api_base}/repos/{owner}/{repo}/git/trees/{tree_sha}"
        r2 = requests.get(url, headers=self._headers(), params={"recursive": "1"}, timeout=timeout_s)
        r2.raise_for_status()
        data = r2.json()

        out: List[RepoFile] = []
        for it in data.get("tree", []) or []:
            if it.get("type") != "blob":
                continue
            out.append(
                RepoFile(
                    path=it.get("path", ""),
                    sha=it.get("sha", ""),
                    size=int(it.get("size") or 0),
                    url=it.get("url", ""),
                )
            )
        return out

    def get_file_text(self, owner: str, repo: str, path: str, ref: Optional[str] = None, timeout_s: int = 30) -> str:
        """
        Contents API:
          GET /repos/{owner}/{repo}/contents/{path}
        Prefer raw media type for larger files. :contentReference[oaicite:5]{index=5}
        """
        url = f"{self.api_base}/repos/{owner}/{repo}/contents/{path}"
        params = {"ref": ref} if ref else None

        # try object json first (gets metadata + base64 for <=1MB)
        r = requests.get(url, headers=self._headers("application/vnd.github+json"), params=params, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            enc = data.get("encoding")
            content = data.get("content") or ""
            if enc == "base64" and content:
                return base64.b64decode(content).decode("utf-8", errors="replace")

            # fallback: raw download_url
            dl = data.get("download_url")
            if dl:
                rr = requests.get(dl, timeout=timeout_s)
                rr.raise_for_status()
                return rr.text

        # last fallback: raw media type
        rr = requests.get(url, headers=self._headers("application/vnd.github.raw+json"), params=params, timeout=timeout_s)
        rr.raise_for_status()
        return rr.text

    @staticmethod
    def pick_key_files(
        files: Sequence[RepoFile],
        *,
        max_files: int = 30,
        max_file_size: int = 200_000,
    ) -> List[str]:
        """
        Heuristic: prioritize files that help reproduction.
        """
        priority = [
            "README.md", "readme.md",
            "pyproject.toml", "requirements.txt", "environment.yml", "setup.py",
            "Makefile", "Dockerfile",
            "run.py", "train.py", "main.py",
            "config.yaml", "config.yml",
        ]

        def score(path: str) -> int:
            p = path.lower()
            s = 0
            if any(p.endswith(x.lower()) for x in priority):
                s += 100
            if "examples" in p or "tutorial" in p:
                s += 25
            if p.endswith(".py"):
                s += 10
            if p.endswith(".ipynb"):
                s += 5
            if p.startswith("src/") or p.startswith("pinneaple_") or p.startswith("pinn"):
                s += 8
            if "test" in p:
                s += 2
            return s

        # filter out huge/binary-ish
        candidates = [f for f in files if f.size <= max_file_size and f.path and not f.path.lower().endswith((".png",".jpg",".jpeg",".gif",".pdf",".pt",".pth",".onnx",".zip"))]
        candidates.sort(key=lambda f: score(f.path), reverse=True)

        picked: List[str] = []
        for f in candidates:
            if len(picked) >= max_files:
                break
            picked.append(f.path)
        return picked
