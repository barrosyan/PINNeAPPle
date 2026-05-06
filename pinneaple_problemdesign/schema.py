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
class PinneapleSpec:
    """Concrete Pinneaple API objects derived from the elicited design spec.

    Produced by ``pinneaple_problemdesign.codegen.build_pinneaple_spec`` at
    finalization time. All fields are plain Python dicts/lists so the object
    is JSON-serialisable and renderable in Markdown without requiring the full
    Pinneaple stack at design time.

    Fields
    ------
    pde_kind : str
        Canonical PDE kind accepted by ``pinneaple_pinn.compile_problem`` and
        ``pinneaple_environment.ProblemSpec``.  ``"custom"`` when the PDE
        could not be identified from the elicited physics description.
    pde_confidence : float
        Match confidence score returned by ``identify_pde`` (keyword overlap).
        0 = no match.
    coords : list[str]
        Suggested coordinate names (e.g. ``["x", "t"]``).
    fields : list[str]
        Suggested field names (e.g. ``["u", "v", "p"]``).
    pde_params : dict
        Default PDE parameters for the identified kind.
    environment_kwargs : dict
        Keyword arguments for ``pinneaple_environment.ProblemSpec()``.
    model_name : str
        Model name accepted by ``pp.build_model()``.
    model_kwargs : dict
        Keyword arguments for ``pp.build_model(model_name, **model_kwargs)``.
    train_config_kwargs : dict
        Fields for ``pinneaple_train.TrainConfig``.
    collocation_kwargs : dict
        Keyword arguments for ``CollocationSampler.sample()``.
    pipeline_code : str
        Runnable Python code snippet implementing the end-to-end flow.
    """
    pde_kind: str = "custom"
    pde_confidence: float = 0.0
    coords: List[str] = field(default_factory=list)
    fields: List[str] = field(default_factory=list)
    pde_params: Dict[str, Any] = field(default_factory=dict)

    environment_kwargs: Dict[str, Any] = field(default_factory=dict)

    model_name: str = "VanillaPINN"
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    train_config_kwargs: Dict[str, Any] = field(default_factory=dict)

    collocation_kwargs: Dict[str, Any] = field(default_factory=dict)

    pipeline_code: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pde_kind": self.pde_kind,
            "pde_confidence": self.pde_confidence,
            "coords": self.coords,
            "fields": self.fields,
            "pde_params": self.pde_params,
            "environment_kwargs": self.environment_kwargs,
            "model_name": self.model_name,
            "model_kwargs": self.model_kwargs,
            "train_config_kwargs": self.train_config_kwargs,
            "collocation_kwargs": self.collocation_kwargs,
            "pipeline_code": self.pipeline_code,
        }


@dataclass
class DesignReport:
    spec: ProblemSpec
    gaps: List[Gap]
    plan: Plan
    pinneaple_spec: Optional["PinneapleSpec"] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
