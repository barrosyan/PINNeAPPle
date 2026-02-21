"""Problem design schema: Assumption, Risk, DataSpec, Gap, ProblemSpec."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime


TaskType = Literal[
    "forecasting",
    "inverse_problem",
    "pde_solution",
    "neural_operator",
    "control",
    "optimization",
    "anomaly_detection",
    "other",
]

RiskLevel = Literal["low", "medium", "high"]
GapSeverity = Literal["blocker", "important", "nice_to_have"]


@dataclass
class Assumption:
    text: str
    confidence: float = 0.5  # 0..1
    needs_confirmation: bool = True


@dataclass
class Risk:
    text: str
    level: RiskLevel = "medium"
    mitigation: str = ""


@dataclass
class DataSpec:
    sources: List[str] = field(default_factory=list)
    format: str = ""
    sampling: str = ""  # e.g. "1Hz", "10min", "irregular"
    variables_observed: List[str] = field(default_factory=list)
    target_variables: List[str] = field(default_factory=list)
    known_quality_issues: List[str] = field(default_factory=list)
    missingness: str = ""
    train_span: str = ""
    val_split_policy: str = ""
    labels_available: bool = True


@dataclass
class PhysicsSpec:
    governing_equations: List[str] = field(default_factory=list)
    boundary_conditions: List[str] = field(default_factory=list)
    initial_conditions: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    parameters_known: List[str] = field(default_factory=list)
    parameters_unknown: List[str] = field(default_factory=list)
    units: Dict[str, str] = field(default_factory=dict)


@dataclass
class GeometrySpec:
    domain: str = ""
    representation: str = ""
    sensors: List[str] = field(default_factory=list)
    coordinate_system: str = ""


@dataclass
class ValidationSpec:
    primary_metrics: List[str] = field(default_factory=list)
    acceptance_criteria: str = ""
    robustness_tests: List[str] = field(default_factory=list)
    ood_scenarios: List[str] = field(default_factory=list)


@dataclass
class DeploymentSpec:
    environment: str = ""
    latency_budget_ms: Optional[int] = None
    update_policy: str = ""
    monitoring: List[str] = field(default_factory=list)


@dataclass
class ConstraintsSpec:
    hardware: str = ""
    max_training_time: str = ""
    interpretability: str = ""
    compliance: List[str] = field(default_factory=list)


@dataclass
class ProblemSpec:
    title: str = ""
    goal: str = ""
    task_type: TaskType = "other"

    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

    horizon: str = ""
    input_window: str = ""
    frequency: str = ""

    domain_context: str = ""

    data: DataSpec = field(default_factory=DataSpec)
    physics: PhysicsSpec = field(default_factory=PhysicsSpec)
    geometry: GeometrySpec = field(default_factory=GeometrySpec)
    validation: ValidationSpec = field(default_factory=ValidationSpec)
    deployment: DeploymentSpec = field(default_factory=DeploymentSpec)
    constraints: ConstraintsSpec = field(default_factory=ConstraintsSpec)

    assumptions: List[Assumption] = field(default_factory=list)
    risks: List[Risk] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Gap:
    id: str
    question: str
    severity: GapSeverity = "important"
    rationale: str = ""
    how_to_obtain: str = ""
    resolved: bool = False


@dataclass
class PlanStep:
    title: str
    why: str
    actions: List[str] = field(default_factory=list)
    pinneaple_modules: List[str] = field(default_factory=list)
    exit_criteria: List[str] = field(default_factory=list)


@dataclass
class Plan:
    recommended_approach: str = ""
    alternatives: List[str] = field(default_factory=list)
    steps: List[PlanStep] = field(default_factory=list)
    go_no_go: List[str] = field(default_factory=list)


@dataclass
class DesignReport:
    spec: ProblemSpec
    gaps: List[Gap]
    plan: Plan
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
