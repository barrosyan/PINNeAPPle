"""pinneapple_systems.component_library.registry — ``ComponentRegistry``:
a registry of NAMED PLANT COMPONENTS keyed by ``component_type`` (e.g.
"pipe", "pump", "valve") — several competing architectures may register
under the same ``component_type``, mirroring how
``pinneapple_neural.architectures.registry.ModelRegistry`` lets many
architectures register under one ``family``.

Deliberately layered ON TOP of ``ModelRegistry`` rather than replacing it:
every component class is (or wraps) a ``BaseModel``. What ``ModelRegistry``
has no concept of, and this module adds, is the ``(component_type,
Physics)`` attachment — a component is a named, physically-constrained
*thing you can point at a real pipe/pump/valve*, not just an architecture
family.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from .physics import Physics


@dataclass
class ComponentSpec:
    component_type: str
    name: str
    cls: Type[Any]
    physics: Optional[Physics] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)


class ComponentRegistry:
    """Central registry for named plant components."""

    _REGISTRY: Dict[str, ComponentSpec] = {}

    @classmethod
    def register(
        cls,
        *,
        component_type: str,
        name: str,
        physics: Optional[Physics] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ):
        """Decorator: ``@ComponentRegistry.register(component_type="pipe", name="PipeVanillaPINN")``."""
        def decorator(component_cls: Type[Any]) -> Type[Any]:
            key = name.lower()
            if key in cls._REGISTRY:
                raise KeyError(f"Component '{name}' already registered")
            cls._REGISTRY[key] = ComponentSpec(
                component_type=component_type.lower(),
                name=name,
                cls=component_cls,
                physics=physics,
                description=description,
                tags=list(tags or []),
            )
            return component_cls
        return decorator

    @classmethod
    def by_type(cls, component_type: str) -> List[ComponentSpec]:
        key = component_type.lower()
        return [s for s in cls._REGISTRY.values() if s.component_type == key]

    @classmethod
    def component_types(cls) -> List[str]:
        return sorted({s.component_type for s in cls._REGISTRY.values()})

    @classmethod
    def spec(cls, name: str) -> ComponentSpec:
        key = name.lower()
        if key not in cls._REGISTRY:
            raise KeyError(f"Unknown component '{name}'. Available: {cls.list()}")
        return cls._REGISTRY[key]

    @classmethod
    def list(cls) -> List[str]:
        return sorted(cls._REGISTRY.keys())

    @classmethod
    def build(cls, name: str, **kwargs: Any):
        """Instantiate a registered component, stamping it with the
        bookkeeping (``_component_name``/``_init_kwargs``) that
        ``ComponentModel.save_checkpoint``/``load_checkpoint`` needs to
        reconstruct it later from a checkpoint alone — the same job
        ``pinneapple_hub.hub.from_pretrained`` already does for plain
        ``ModelRegistry`` architectures, done here at the component layer
        instead."""
        from .base import ComponentModel  # local import: avoids a cycle with base.py

        spec = cls.spec(name)
        model = spec.cls(**kwargs)
        if not isinstance(model, ComponentModel):
            raise TypeError(
                f"Registered component class '{spec.cls.__name__}' must subclass ComponentModel"
            )
        model._component_spec = spec
        model._component_name = key = name.lower()
        model._init_kwargs = dict(kwargs)
        if model.physics is None and spec.physics is not None:
            model.physics = spec.physics
        return model


def register_component(
    *,
    component_type: str,
    name: str,
    physics: Optional[Physics] = None,
    description: str = "",
    tags: Optional[List[str]] = None,
):
    """Module-level alias for ``ComponentRegistry.register`` — reads better
    as a bare decorator: ``@register_component(component_type=..., name=...)``."""
    return ComponentRegistry.register(
        component_type=component_type, name=name, physics=physics, description=description, tags=tags
    )
