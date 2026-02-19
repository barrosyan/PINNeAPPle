from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


ItemType = Literal["paper", "repo"]


@dataclass
class RankedItem:
    type: ItemType
    id: str
    title: str
    url: str
    score: float
    why: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryResult:
    topic: str
    papers: List[RankedItem]
    repos: List[RankedItem]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KBArtifact:
    """
    A persisted artifact (paper/repo) stored inside a KB run folder.
    """
    type: ItemType
    id: str
    title: str
    url: str
    path: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KBIndex:
    """
    A light-weight index pointing to chunked documents (JSONL).
    """
    topic: str
    run_dir: str
    chunks_path: str
    manifest_path: str


@dataclass
class ExtractedProblemSolution:
    """
    Output of ExtractorAgent. Keep it generic but structured enough
    to map into pinneaple_problemdesign later.
    """
    source_type: ItemType
    source_id: str
    title: str

    problem: str
    solution: str

    # optional structured fields
    equations: Optional[str] = None
    data_requirements: Optional[str] = None
    training_recipe: Optional[str] = None
    metrics: Optional[str] = None
    limitations: Optional[str] = None

    extra: Dict[str, Any] = field(default_factory=dict)
