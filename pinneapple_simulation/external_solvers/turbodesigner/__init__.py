"""TurboDesigner external solver bridge.

Wraps the ``turbodesigner`` package (pip install turbodesigner) to generate
analytical mean-line compressor data for PINN training.

If ``turbodesigner`` is not installed, a built-in analytical fallback is used
automatically — the bridge remains importable at all times.

Public API
----------
TurboDesignerConfig   — design configuration dataclass
run_turbodesigner     — one-call analytical solve → dict of numpy arrays
turbodesigner_to_upd  — solve + package as UPD PhysicalSample
TurboDesignerWorkflow — convenience wrapper for parametric sweeps

Quick start
-----------
>>> from pinneapple_simulation.external_solvers.turbodesigner import (
...     TurboDesignerConfig, TurboDesignerWorkflow
... )
>>> cfg = TurboDesignerConfig(pressure_ratio=3.0, num_stages=5, rpm=10_000)
>>> wf = TurboDesignerWorkflow(cfg)
>>> data = wf.solve()           # single operating point
>>> samples = wf.sweep(         # parametric sweep
...     {"pressure_ratio": [2.0, 3.0, 4.0]},
...     as_upd=True,
... )
"""

from .bridge import (
    TurboDesignerConfig,
    TurboDesignerWorkflow,
    run_turbodesigner,
    turbodesigner_to_upd,
)

__all__ = [
    "TurboDesignerConfig",
    "TurboDesignerWorkflow",
    "run_turbodesigner",
    "turbodesigner_to_upd",
]
