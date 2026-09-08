"""pinneapple_systems.component_library — the piece
``pinneapple_systems.component_modeling``'s own docstring says pinneapple
deliberately did NOT build: a registry of NAMED PLANT COMPONENTS (keyed
by ``component_type``, e.g. "pipe"/"pump"/"valve"), each pairing an
existing ``pinneapple_neural.architectures`` backbone with a ``Physics``
constraint, plus versioned installable ``Toolbox`` bundles of component
types.

Everything else a component needs — control (PID/MPC), uncertainty
quantification (ensemble/SWAG/MC-Dropout), edge export, digital-twin
attachment, generic physics residuals — already exists in
``pinneapple_systems.component_modeling`` and
``pinneapple_systems.digital_twin``; this package composes those, it
does not re-implement them:

    from pinneapple_systems.component_library import ComponentRegistry
    from pinneapple_systems.component_modeling import run_mpc, DeepEnsemble
    from pinneapple_systems.digital_twin import DigitalTwin

    pipe = ComponentRegistry.build("PipeVanillaPINN")
    pipe.fit(train_loader, val_loader, epochs=200)
    twin = DigitalTwin(model=pipe, ...)   # DigitalTwin already accepts any nn.Module
"""
from __future__ import annotations

from .physics import Physics
from .registry import ComponentSpec, ComponentRegistry, register_component
from .base import ComponentModel
from .toolbox import Toolbox, ToolboxRegistry
from .benchmark import ComponentBenchmarkResult, benchmark_component_type
from . import components  # noqa: F401  side effect: registers the 6 reference components

__all__ = [
    "Physics",
    "ComponentSpec",
    "ComponentRegistry",
    "register_component",
    "ComponentModel",
    "Toolbox",
    "ToolboxRegistry",
    "ComponentBenchmarkResult",
    "benchmark_component_type",
]
