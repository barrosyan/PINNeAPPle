from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal


StageName = Literal[
    "intake",
    "io_and_time",
    "data",
    "physics",
    "constraints",
    "validation",
    "approach_selection",
    "finalization",
]


@dataclass
class Stage:
    name: StageName
    description: str
    required_fields: List[str]


STAGES_ORDER: List[Stage] = [
    Stage(
        name="intake",
        description="Capture goal, context, and success definition.",
        required_fields=["title", "goal", "task_type", "domain_context"],
    ),
    Stage(
        name="io_and_time",
        description="Define inputs/outputs and time configuration.",
        required_fields=["inputs", "outputs", "frequency", "input_window", "horizon"],
    ),
    Stage(
        name="data",
        description="Data availability, formats, quality, and split policy.",
        required_fields=[
            "data.sources",
            "data.format",
            "data.sampling",
            "data.variables_observed",
            "data.target_variables",
            "data.val_split_policy",
        ],
    ),
    Stage(
        name="physics",
        description="Capture governing equations/constraints if applicable.",
        required_fields=[],
    ),
    Stage(
        name="constraints",
        description="Hardware/runtime/latency/interpretability constraints.",
        required_fields=[],
    ),
    Stage(
        name="validation",
        description="Metrics and acceptance criteria.",
        required_fields=["validation.primary_metrics", "validation.acceptance_criteria"],
    ),
    Stage(
        name="approach_selection",
        description="Select approach (FNO-first by default) with tradeoffs.",
        required_fields=[],
    ),
    Stage(
        name="finalization",
        description="Close remaining blockers and produce report.",
        required_fields=[],
    ),
]
