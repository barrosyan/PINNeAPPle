from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
from .schema import ProblemSpec, Gap, Plan, DesignReport


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class DesignState:
    spec: ProblemSpec = field(default_factory=ProblemSpec)
    gaps: List[Gap] = field(default_factory=list)
    plan: Plan = field(default_factory=Plan)
    history: List[Turn] = field(default_factory=list)

    stage: str = "intake"
    done: bool = False

    def add_turn(self, role: str, content: str) -> None:
        self.history.append(Turn(role=role, content=content))

    def unresolved_gaps(self) -> List[Gap]:
        return [g for g in self.gaps if not g.resolved]

    def to_report(self) -> DesignReport:
        return DesignReport(spec=self.spec, gaps=self.gaps, plan=self.plan)
