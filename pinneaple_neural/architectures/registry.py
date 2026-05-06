"""ModelRegistry and ModelSpec for model registration and lookup."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type, List, Any, Optional

from .base import BaseModel
from .instantiate import instantiate


@dataclass
class ModelSpec:
    name: str
    family: str
    cls: Type[BaseModel]
    description: str = ""
    tags: List[str] | None = None
    # --- arena interoperability
    input_kind: str = "pointwise_coords"  # pointwise_coords | grid | graph | sequence
    supports_physics_loss: bool = False
    expects: List[str] | None = None  # keys in batch
    predicts: List[str] | None = None  # physical fields


class ModelRegistry:
    """
    Central registry for all Pinneaple models.
    """

    _REGISTRY: Dict[str, ModelSpec] = {}

    @classmethod
    def register(
        cls,
        *,
        name: str,
        family: str,
        description: str = "",
        tags: List[str] | None = None,
        input_kind: str = "pointwise_coords",
        supports_physics_loss: bool = False,
        expects: Optional[List[str]] = None,
        predicts: Optional[List[str]] = None,
    ):
        """
        Decorator to register a model class.
        """
        def decorator(model_cls: Type[BaseModel]):
            key = name.lower()
            if key in cls._REGISTRY:
                raise KeyError(f"Model '{name}' already registered")

            model_cls.name = name
            model_cls.family = family

            cls._REGISTRY[key] = ModelSpec(
                name=name,
                family=family,
                cls=model_cls,
                description=description,
                tags=tags or [],
                input_kind=str(input_kind or "pointwise_coords"),
                supports_physics_loss=bool(supports_physics_loss),
                expects=list(expects) if expects is not None else None,
                predicts=list(predicts) if predicts is not None else None,
            )
            return model_cls
        return decorator

    @classmethod
    def list(cls, family: str | None = None) -> List[str]:
        if family is None:
            return sorted(cls._REGISTRY.keys())
        return sorted(
            k for k, v in cls._REGISTRY.items()
            if v.family == family
        )

    @classmethod
    def families(cls) -> List[str]:
        return sorted({v.family for v in cls._REGISTRY.values()})

    @classmethod
    def spec(cls, name: str) -> ModelSpec:
        key = name.lower()
        if key not in cls._REGISTRY:
            raise KeyError(f"Unknown model '{name}'. Available: {cls.list()}")
        return cls._REGISTRY[key]

    @classmethod
    def build(cls, name: str, **kwargs) -> BaseModel:
        spec = cls.spec(name)
        model, _report = instantiate(spec.cls, kwargs)
        return model

    @classmethod
    def build_with_report(cls, name: str, **kwargs):
        """Build a model and also return an InstantiateReport (debug/arena logs)."""
        spec = cls.spec(name)
        return instantiate(spec.cls, kwargs)
