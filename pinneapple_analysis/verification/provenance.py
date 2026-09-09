"""A real, structured provenance record for one analysis run -- the
concrete implementation of the user's architectural point 17
("Provenance / reproducibility... cada resultado precisa ter um
lineage"). Every field here is something the pipeline actually knows at
run time (not a placeholder) -- problem description, drafted spec,
architecture/training config, package versions, wall-clock timings, and
the verification report -- serialized as one JSON-able record so a run
can be replayed or audited later.

This is intentionally a plain dataclass + ``to_dict()``, not a database
model -- swap the ``save()``/``load()`` stubs for a real Postgres/S3
write before launch; the point of this module is the SHAPE of a
reproducible record, which doesn't change when the storage backend does.
"""
from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProvenanceRecord:
    run_id: str
    created_unix: float = field(default_factory=time.time)

    # -- what was asked --------------------------------------------------
    problem_description: str = ""

    # -- physics formulation (dimensional analysis + LLM draft) ----------
    dimensionless_numbers: Dict[str, Any] = field(default_factory=dict)
    flow_regime: Optional[str] = None
    drafted_preset: Optional[str] = None
    drafted_preset_kwargs: Dict[str, Any] = field(default_factory=dict)
    draft_reasoning: str = ""

    # -- execution --------------------------------------------------------
    architecture: str = ""
    architecture_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    candidate_architectures_tried: List[str] = field(default_factory=list)
    final_training_loss: Optional[float] = None

    # -- verification -------------------------------------------------------
    guardrail_trustworthy: Optional[bool] = None
    guardrail_checks: List[Dict[str, Any]] = field(default_factory=list)
    guardrail_summary: str = ""

    # -- environment (for real reproducibility, not just claimed) --------
    python_version: str = field(default_factory=platform.python_version)
    platform_string: str = field(default_factory=platform.platform)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @staticmethod
    def load(path: str) -> "ProvenanceRecord":
        with open(path, "r") as f:
            data = json.load(f)
        return ProvenanceRecord(**data)
