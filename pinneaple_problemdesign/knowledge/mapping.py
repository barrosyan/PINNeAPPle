"""Plan builder: dispatches to FNO-first or PINN-first based on task_type."""
from __future__ import annotations

from typing import List
from ..schema import ProblemSpec, Plan, PlanStep, Gap

# Task types that are better served by a PINN-first plan
_PINN_TASK_TYPES = {"pde_solution", "inverse_problem"}


def build_plan(spec: ProblemSpec, gaps: List[Gap]) -> Plan:
    """Dispatch to the appropriate plan builder based on spec.task_type."""
    if spec.task_type in _PINN_TASK_TYPES:
        return build_plan_pinn_first(spec, gaps)
    return build_plan_fno_first(spec, gaps)


def build_plan_pinn_first(spec: ProblemSpec, gaps: List[Gap]) -> Plan:
    """Plan for PDE-solution and inverse-problem tasks via PINNFactory."""
    recommended = (
        "PINN-first: define PDE residuals symbolically via PINNFactory, enforce BCs/ICs "
        "as condition losses, and optionally add supervised data loss."
    )
    alternatives = [
        "Neural operator (FNO/DeepONet) as surrogate if many initial conditions are needed",
        "Hybrid: supervised pre-training + PINN fine-tuning",
        "Classical solver for validation baselines (FEM/FD)",
    ]
    steps = [
        PlanStep(
            title="Define PDE residuals and boundary/initial conditions",
            why="PINN training quality depends entirely on correctly stated physics.",
            actions=[
                "Write PDE residuals as SymPy strings (e.g. 'u_t + u*u_x - nu*u_xx').",
                "List ICs and BCs with their equations and domain definitions.",
                "Identify any inverse parameters to recover.",
            ],
            pinneaple_modules=["pinneaple_pinn.factory (PINNProblemSpec, PINNFactory)"],
            exit_criteria=["PDE + BCs compile without error via PINNFactory."],
        ),
        PlanStep(
            title="Build model and loss function",
            why="The factory compiles symbolic equations into a unified torch loss.",
            actions=[
                "Instantiate VanillaPINN (or NeuralNetwork) with appropriate depth/width.",
                "Call PINNFactory(spec).generate_loss_function().",
                "Verify loss components (pde, conditions, data) are nonzero on a test batch.",
            ],
            pinneaple_modules=[
                "pinneaple_pinn.factory (VanillaPINN, PINNFactory)",
                "pinneaple_train.trainer.Trainer",
            ],
            exit_criteria=["Loss function evaluates without error; PDE residual > 0 before training."],
        ),
        PlanStep(
            title="Train with collocation + condition sampling",
            why="PINNs require careful sampling of collocation and BC points.",
            actions=[
                "Sample interior collocation points and BC/IC boundary points.",
                "Train using Trainer with Adam → L-BFGS schedule for convergence.",
                "Monitor each loss component separately.",
            ],
            pinneaple_modules=[
                "pinneaple_train.trainer.Trainer",
                "pinneaple_train.losses.CombinedLoss",
            ],
            exit_criteria=["PDE residual < 1e-3; BC loss < 1e-4."],
        ),
        PlanStep(
            title="Validate against analytic solution or reference data",
            why="PINN convergence to the wrong solution is a known failure mode.",
            actions=[
                "Compare against analytic solution (if available) or high-fidelity solver.",
                "Check error by region (interior vs boundary).",
                "For inverse problems: verify recovered parameters vs ground truth.",
            ],
            pinneaple_modules=["pinneaple_train.metrics"],
            exit_criteria=["L2 relative error < acceptance threshold defined in spec."],
        ),
    ]
    return Plan(
        recommended_approach=recommended,
        alternatives=alternatives,
        steps=steps,
        go_no_go=[
            "GO: PDE residual and BC losses converge; solution matches reference.",
            "NO-GO: PDE residuals are not physics-faithful (wrong BCs or domain).",
            "REVISE: switch to neural operator surrogate if many PDE solves needed.",
        ],
    )


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
