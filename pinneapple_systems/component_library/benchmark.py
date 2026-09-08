"""pinneapple_systems.component_library.benchmark — train every
architecture registered under one ``component_type`` and rank by final
validation loss (real numbers from ``ComponentModel.fit()``, not a
display placeholder).

Kept as a thin loop over ``ComponentModel.fit()``/``.evaluate()`` rather
than routing through ``pinneapple_tools.benchmark_suite.Arena``: Arena
benchmarks *problem presets* end-to-end (it builds and trains its own
model from a ``ProblemSpec``), whereas a registered component here is
already a built, physics-attached ``ComponentModel`` — there is no
per-component ``ProblemSpec`` to hand Arena. No training loop is
duplicated at this layer either way, since ``.fit()`` itself delegates to
the shared ``Trainer``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from .registry import ComponentRegistry


@dataclass
class ComponentBenchmarkResult:
    component_type: str
    ranked: List[Dict[str, Any]] = field(default_factory=list)

    def best(self) -> Optional[Dict[str, Any]]:
        return self.ranked[0] if self.ranked else None


def benchmark_component_type(
    component_type: str,
    train_loader,
    val_loader=None,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    eval_coords: Optional[torch.Tensor] = None,
    eval_targets: Optional[torch.Tensor] = None,
    log_dir: Optional[str] = None,
) -> ComponentBenchmarkResult:
    """Trains every architecture registered for ``component_type`` on the
    same data and ranks them by final validation loss (ascending — lower
    is better). If ``eval_coords``/``eval_targets`` are given, also attaches
    RMSE/MAPE/R² per architecture via ``ComponentModel.evaluate()``.

    ``log_dir`` defaults to a fresh temp directory (never
    ``TrainConfig``'s own default of ``"runs"``, a relative path that
    would otherwise write into — and collide with — whatever the caller's
    current working directory happens to be, e.g. a shared repo checkout).
    """
    import tempfile

    specs = ComponentRegistry.by_type(component_type)
    if not specs:
        raise KeyError(f"No components registered for component_type='{component_type}'")

    resolved_log_dir = log_dir or tempfile.mkdtemp(prefix="pinneapple_component_benchmark_")

    rows: List[Dict[str, Any]] = []
    for spec in specs:
        model = ComponentRegistry.build(spec.name)
        result = model.fit(
            train_loader, val_loader, epochs=epochs, lr=lr,
            run_name=f"bench_{spec.name}", log_dir=resolved_log_dir,
        )
        row: Dict[str, Any] = {"name": spec.name, "final_loss": result["best_val"]}
        if eval_coords is not None and eval_targets is not None:
            row["metrics"] = model.evaluate(eval_coords, eval_targets)
        rows.append(row)

    rows.sort(key=lambda r: r["final_loss"])
    return ComponentBenchmarkResult(component_type=component_type, ranked=rows)
