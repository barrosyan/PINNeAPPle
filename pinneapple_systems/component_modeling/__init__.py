"""pinneapple_systems.component_modeling — control, optimization, and
uncertainty-quantification tooling that operates on ANY differentiable
component model, independent of what that component represents physically.

This package deliberately does NOT include a component registry, a base
"component model" class, or a mesh/collocation generator — pinneapple
already has more capable, more mature native equivalents for all three:

- Named, buildable model architectures & introspection:
  ``pinneapple_neural.architectures.registry.ModelRegistry``
- A trainable-model contract (checkpointing, ONNX/TorchScript export, a
  physics-loss adapter): ``pinneapple_neural.architectures.base.BaseModel``
  / ``pinneapple_neural.architectures.pinns.base.PINNBase``
- Collocation-point / mesh generation: ``pinneapple_data.collocation``,
  ``pinneapple_design.geometry``
- Multi-architecture comparison: ``pinneapple_tools.benchmark_suite.Arena``

What IS here is the machinery that was genuinely missing:

- ``control``    — PIDController + a generic closed-loop runner
- ``mpc``        — gradient-based receding-horizon Model Predictive Control
- ``bayesian``   — SWAG: an approximate weight-space Bayesian posterior
- ``ensemble``   — DeepEnsemble epistemic uncertainty
- ``mc_dropout`` — MC-Dropout epistemic uncertainty
- ``physics_residuals`` — generic autograd PDE residuals (incompressible
  flow, heat conduction, linear elasticity, species diffusion)
- ``edge``       — ONNX edge-deployment packaging + a torch-free runtime

Every function/class here works on a plain ``nn.Module`` (or a bare
Python callable, for ``control.run_closed_loop``) supplied by the caller —
none of it references any specific physical component.
"""
from __future__ import annotations

from .bayesian import SWAGApproximation
from .control import PIDController, run_closed_loop
from .edge import EdgeRuntime, export_edge_package
from .ensemble import DeepEnsemble
from .mc_dropout import mc_dropout_uncertainty
from .mpc import run_mpc
from .physics_residuals import (
    heat_conduction_residual,
    incompressible_continuity_residual,
    linear_elasticity_residual,
    species_diffusion_residual,
)

__all__ = [
    "PIDController", "run_closed_loop",
    "run_mpc",
    "SWAGApproximation",
    "DeepEnsemble",
    "mc_dropout_uncertainty",
    "incompressible_continuity_residual",
    "heat_conduction_residual",
    "linear_elasticity_residual",
    "species_diffusion_residual",
    "export_edge_package", "EdgeRuntime",
]
