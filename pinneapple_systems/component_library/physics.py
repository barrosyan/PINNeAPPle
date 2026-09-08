"""pinneapple_systems.component_library.physics — ``Physics``: the thin
attachment layer between a named plant component and the constraint it
must satisfy.

Two ways to build one:

- :meth:`Physics.from_shortcut` wraps one of the generic, closed-form
  residuals already in ``pinneapple_systems.component_modeling
  .physics_residuals`` (incompressible flow, heat conduction, linear
  elasticity, species diffusion) into a ``Trainer``-compatible loss_fn.
  No new physics is implemented here — this only adapts the existing
  ``residual(model, coords, **params) -> Tensor`` contract to the
  ``loss_fn(model, y_hat, batch) -> {"total": Tensor, ...}`` contract
  ``pinneapple_neural.trainer.Trainer`` expects.
- :meth:`Physics.from_preset` wraps a registered PDE preset from
  ``pinneapple_physics.pde_environment`` via the existing symbolic-PDE
  compiler (``pinneapple_physics.pinn_solver.compiler.compile
  .compile_problem``), for a component whose physics is one of PINNeAPPle's
  ~50 named presets rather than one of the four generic shortcuts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import torch

from pinneapple_systems.component_modeling import physics_residuals as _res

LossFn = Callable[[Any, torch.Tensor, Dict[str, Any]], Dict[str, torch.Tensor]]

_SHORTCUTS: Dict[str, Callable[..., torch.Tensor]] = {
    "incompressible_flow": _res.incompressible_continuity_residual,
    "heat_conduction": _res.heat_conduction_residual,
    "linear_elasticity": _res.linear_elasticity_residual,
    "species_diffusion": _res.species_diffusion_residual,
}


def _wrap_residual_as_loss_fn(residual_fn: Callable[..., torch.Tensor], params: Dict[str, Any]) -> LossFn:
    def loss_fn(model: Any, y_hat: torch.Tensor, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        coords = batch.get("x")
        if coords is None:
            coords = batch.get("x_col")
        if coords is None:
            raise KeyError("Physics loss_fn expects batch['x'] or batch['x_col'].")
        r = residual_fn(model, coords, **params)
        return {"total": (r ** 2).mean()}
    return loss_fn


@dataclass
class Physics:
    """A named physical constraint, reduced to a single ``Trainer``-shaped
    loss_fn. ``params`` is kept alongside for introspection (e.g. shown on
    a Toolbox/Component detail view) even though it is already baked into
    ``loss_fn`` via closure."""

    name: str
    loss_fn: LossFn
    params: Dict[str, Any] = field(default_factory=dict)

    def __call__(self, model: Any, y_hat: torch.Tensor, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self.loss_fn(model, y_hat, batch)

    @classmethod
    def from_shortcut(cls, shortcut: str, **params: Any) -> "Physics":
        try:
            residual_fn = _SHORTCUTS[shortcut]
        except KeyError:
            raise KeyError(
                f"Unknown physics shortcut '{shortcut}'. Available: {sorted(_SHORTCUTS)}"
            ) from None
        return cls(name=shortcut, loss_fn=_wrap_residual_as_loss_fn(residual_fn, params), params=dict(params))

    @classmethod
    def from_preset(cls, preset_id: str, *, weights: Optional[Any] = None, **preset_kwargs: Any) -> "Physics":
        from pinneapple_physics.pde_environment.presets.registry import get_preset
        from pinneapple_physics.pinn_solver.compiler.compile import compile_problem

        spec = get_preset(preset_id, **preset_kwargs)
        compiled = compile_problem(spec, weights=weights)
        return cls(name=preset_id, loss_fn=compiled, params=dict(preset_kwargs))

    @staticmethod
    def available_shortcuts() -> list[str]:
        return sorted(_SHORTCUTS)
