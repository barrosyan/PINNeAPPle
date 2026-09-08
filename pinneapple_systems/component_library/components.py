"""pinneapple_systems.component_library.components — six reference plant
components. Deliberately six, not padded to ten: each pairs a genuinely
different (physics shortcut, backbone architecture, component_type)
combination rather than near-duplicating the others.

| component_type   | class                     | physics              | backbone       |
|-------------------|---------------------------|-----------------------|----------------|
| pipe              | PipeVanillaPINN           | incompressible_flow   | vanilla_pinn   |
| heat_exchanger     | HeatExchangerVanillaPINN  | heat_conduction       | vanilla_pinn   |
| tank_shell        | TankShellVanillaPINN      | linear_elasticity     | vanilla_pinn   |
| battery_cell      | BatteryCellVanillaPINN    | species_diffusion     | vanilla_pinn   |
| pump              | PumpModifiedMLP           | incompressible_flow   | modified_mlp   |
| valve             | ValveModifiedMLP          | none (data-driven)    | modified_mlp   |

Importing this module registers all six with ``ComponentRegistry`` (see
``pinneapple_systems.component_library.__init__``, which imports it for
its side effect).
"""
from __future__ import annotations

from typing import Any, Sequence

from .base import ComponentModel
from .physics import Physics
from .registry import register_component


@register_component(
    component_type="pipe",
    name="PipeVanillaPINN",
    physics=Physics.from_shortcut("incompressible_flow"),
    description="Internal pipe flow; mass-conservation-constrained velocity field (u, v, w).",
    tags=["fluid"],
)
class PipeVanillaPINN(ComponentModel):
    def __init__(self, in_dim: int = 3, out_dim: int = 3, hidden: Sequence[int] = (128, 128, 128, 128), **kwargs: Any):
        super().__init__(
            architecture="vanilla_pinn",
            architecture_kwargs=dict(in_dim=in_dim, out_dim=out_dim, hidden=list(hidden)),
            **kwargs,
        )


@register_component(
    component_type="heat_exchanger",
    name="HeatExchangerVanillaPINN",
    physics=Physics.from_shortcut("heat_conduction"),
    description="Steady-state conduction field across a heat-exchanger shell.",
    tags=["thermal"],
)
class HeatExchangerVanillaPINN(ComponentModel):
    def __init__(self, in_dim: int = 2, out_dim: int = 1, hidden: Sequence[int] = (64, 64, 64), **kwargs: Any):
        super().__init__(
            architecture="vanilla_pinn",
            architecture_kwargs=dict(in_dim=in_dim, out_dim=out_dim, hidden=list(hidden)),
            **kwargs,
        )


@register_component(
    component_type="tank_shell",
    name="TankShellVanillaPINN",
    physics=Physics.from_shortcut("linear_elasticity"),
    description="Linear-elastic displacement field (u, v) of a storage-tank shell wall.",
    tags=["structural"],
)
class TankShellVanillaPINN(ComponentModel):
    def __init__(self, in_dim: int = 2, out_dim: int = 2, hidden: Sequence[int] = (64, 64, 64), **kwargs: Any):
        super().__init__(
            architecture="vanilla_pinn",
            architecture_kwargs=dict(in_dim=in_dim, out_dim=out_dim, hidden=list(hidden)),
            **kwargs,
        )


@register_component(
    component_type="battery_cell",
    name="BatteryCellVanillaPINN",
    physics=Physics.from_shortcut("species_diffusion"),
    description="Lithium-ion concentration field inside a cell (Fickian diffusion-reaction proxy — a "
                "deliberately simplified stand-in, not a real electrochemistry stack).",
    tags=["electrochemical"],
)
class BatteryCellVanillaPINN(ComponentModel):
    def __init__(self, in_dim: int = 2, out_dim: int = 1, hidden: Sequence[int] = (64, 64, 64), **kwargs: Any):
        super().__init__(
            architecture="vanilla_pinn",
            architecture_kwargs=dict(in_dim=in_dim, out_dim=out_dim, hidden=list(hidden)),
            **kwargs,
        )


@register_component(
    component_type="pump",
    name="PumpModifiedMLP",
    physics=Physics.from_shortcut("incompressible_flow"),
    description="Internal pump-casing flow field; same physics as PipeVanillaPINN on a different "
                "(Fourier-feature) backbone, to demonstrate two competing architectures for one "
                "component_type.",
    tags=["fluid", "rotating_equipment"],
)
class PumpModifiedMLP(ComponentModel):
    def __init__(self, in_dim: int = 3, out_dim: int = 3, hidden_dim: int = 128, n_layers: int = 6, **kwargs: Any):
        super().__init__(
            architecture="modified_mlp",
            architecture_kwargs=dict(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, n_layers=n_layers),
            **kwargs,
        )


@register_component(
    component_type="valve",
    name="ValveModifiedMLP",
    physics=None,
    description="Pure data-driven valve Cv/flow surrogate — no PDE attached, mirroring a purely "
                "empirical component with no mesh or physics residual at all.",
    tags=["fluid", "data_driven"],
)
class ValveModifiedMLP(ComponentModel):
    def __init__(self, in_dim: int = 2, out_dim: int = 1, hidden_dim: int = 64, n_layers: int = 3, **kwargs: Any):
        super().__init__(
            architecture="modified_mlp",
            architecture_kwargs=dict(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, n_layers=n_layers),
            **kwargs,
        )
