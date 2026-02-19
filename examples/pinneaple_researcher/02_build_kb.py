from __future__ import annotations

import argparse
import json

from pinneaple_researcher import ResearcherConfig
from pinneaple_researcher.pipelines.discover import discover
from pinneaple_researcher.pipelines.build_kb import build_kb

topic = "pinn thermal instability virtual sensor"
k_papers = 8
k_repos = 8
min_stars = 50

cfg = ResearcherConfig(topic=topic, k_papers=k_papers, k_repos=k_repos, github_min_stars=min_stars)
disc = discover(cfg)
idx = build_kb(cfg, disc)
print(json.dumps(idx.__dict__, indent=2, ensure_ascii=False))
