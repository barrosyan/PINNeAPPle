from __future__ import annotations

from typing import List, Tuple
from ..schema import ProblemSpec, Assumption


def validate_and_suggest(spec: ProblemSpec) -> Tuple[List[str], List[Assumption]]:
    warnings: List[str] = []
    assumptions: List[Assumption] = []

    # Time-series sanity
    if spec.task_type in ("forecasting", "neural_operator"):
        if not spec.frequency:
            warnings.append("Sampling frequency is not defined; windowing/splits may be inconsistent.")
        if not spec.horizon:
            warnings.append("Forecast horizon is not defined; cannot evaluate forecasting properly.")
        if not spec.input_window:
            assumptions.append(
                Assumption(
                    text="Assume input_window = 64 steps as an initial baseline.",
                    confidence=0.35,
                    needs_confirmation=True,
                )
            )

    # Fill outputs from data.target_variables if missing (safe if explicitly provided there)
    if not spec.outputs and spec.data.target_variables:
        spec.outputs = list(spec.data.target_variables)

    # Validation split policy suggestion
    if spec.data.sources and not spec.data.val_split_policy:
        assumptions.append(
            Assumption(
                text="Assume temporal split: last ~20% time as validation.",
                confidence=0.6,
                needs_confirmation=True,
            )
        )

    return warnings, assumptions
