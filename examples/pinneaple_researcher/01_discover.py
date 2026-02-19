from __future__ import annotations

import json
import os

from pinneaple_researcher import ResearcherConfig
from pinneaple_researcher.pipelines.discover import discover

# (recomendado p/ evitar rate limit do GitHub)
# os.environ["GITHUB_TOKEN"] = "..."

topic = "pinn thermal instability virtual sensor"
k_papers = 8
k_repos = 8
min_stars = 50

cfg = ResearcherConfig(
    topic=topic,
    k_papers=k_papers,
    k_repos=k_repos,
    github_min_stars=min_stars,
    # github_languages=["Python"],
    # arxiv_categories=["cs.LG", "math.NA"],
)

res = discover(cfg)
print(json.dumps(res, default=lambda o: o.__dict__, indent=2, ensure_ascii=False))
