"""Plan builder: FNO-first approach mapping from ProblemSpec and gaps."""
from __future__ import annotations

from typing import List
from ..schema import ProblemSpec, Plan, PlanStep, Gap


def build_plan_fno_first(spec: ProblemSpec, gaps: List[Gap]) -> Plan:
    recommended = (
        "FNO-first baseline (direct multi-horizon forecast), then iterate on data quality, "
        "robustness, and optionally add physics-inspired constraints or hybrid losses."
    )

    alternatives = [
        "Autoregressive 1-step baseline (rollout) for simplicity",
        "Transformer-based time series model for long-range dependencies",
        "Hybrid supervised + constraints (bounds/monotonicity/conservation if applicable)",
        "PINN if PDE residuals + BC/IC are reliable and data is scarce",
    ]

    steps: List[PlanStep] = []

    steps.append(PlanStep(
        title="Consolidate the ProblemSpec and close critical gaps",
        why="Prevents building the wrong pipeline and ensures success is measurable.",
        actions=[
            "Confirm inputs/outputs and units.",
            "Confirm sampling frequency, input window, and forecast horizon.",
            "Confirm temporal validation policy and acceptance criteria.",
            "List key data issues (missingness, drift, outliers).",
        ],
        pinneaple_modules=[],
        exit_criteria=[
            "No 'blocker' gaps remain.",
            "Primary metrics and acceptance criteria are defined.",
        ],
    ))

    steps.append(PlanStep(
        title="Define dataset windowing and temporal splits",
        why="Time series modeling requires leakage-safe splits and consistent windowing.",
        actions=[
            "Implement windowing (input_window, horizon, stride) and scaling/normalization.",
            "Apply temporal split policy (e.g., last 20% time as validation).",
            "Check missingness and distribution shift across splits.",
        ],
        pinneaple_modules=[
            "pinneaple_timeseries (windowed datasets + datamodule)",
        ],
        exit_criteria=[
            "Dataset yields consistent (x, y) shapes for train/val.",
            "Split policy avoids future leakage.",
        ],
    ))

    steps.append(PlanStep(
        title="Train the FNO-first baseline (direct multi-horizon)",
        why="FNO is a strong baseline for operator-like dynamics and can generalize well with sufficient data.",
        actions=[
            "Choose initial FNO config (width, modes, layers) appropriate for hardware.",
            "Train with supervised loss (MSE/MAE) and save best checkpoint.",
            "Compare against naive baselines (persistence, simple AR).",
        ],
        pinneaple_modules=[
            "pinneaple_models.neural_operators (FNO)",
            "pinneaple_train.trainer.Trainer",
            "pinneaple_train.losses (CombinedLoss + SupervisedLoss)",
            "pinneaple_train.metrics.default_metrics",
        ],
        exit_criteria=[
            "Baseline beats persistence on primary metric.",
            "Error by horizon is acceptable or gaps are revisited.",
        ],
    ))

    steps.append(PlanStep(
        title="Validate robustness (stress tests)",
        why="Production failures often come from drift, missingness, and rare extremes.",
        actions=[
            "Evaluate error by horizon (short vs long).",
            "Test synthetic missingness, noise, drift, and extreme scenarios.",
            "Record failures and prioritize mitigations.",
        ],
        pinneaple_modules=[
            "pinneaple_train.metrics (custom TS metrics as needed)",
        ],
        exit_criteria=[
            "Clear list of failure modes and mitigation plan.",
            "Acceptance criteria met OR next iteration decision is justified.",
        ],
    ))

    steps.append(PlanStep(
        title="Iterate: data/features, architecture, and optional physics/hybrid constraints",
        why="Second iteration often yields the biggest gains.",
        actions=[
            "If long-horizon error dominates: consider exogenous features, multi-resolution inputs, or transformer alternative.",
            "If generalization is weak: regularization, augmentation, scaling fixes, or constraints.",
            "If reliable physics exists: add constraint losses (bounds, conservation) as hybrid training.",
        ],
        pinneaple_modules=[
            "pinneaple_train.losses.PhysicsLossHook (when applicable)",
            "pinneaple_models (transformers/recurrent as alternatives)",
        ],
        exit_criteria=[
            "Measured improvement in metric + robustness.",
            "Deployment plan updated (latency, monitoring, update policy).",
        ],
    ))

    return Plan(
        recommended_approach=recommended,
        alternatives=alternatives,
        steps=steps,
        go_no_go=[
            "GO: baseline beats naive and meets acceptance criteria.",
            "NO-GO: data is insufficient/ambiguous (critical gaps), leakage exists, or target definition is unstable.",
            "REVISE: adjust horizon/window/metrics if the real use-case demands it.",
        ],
    )
