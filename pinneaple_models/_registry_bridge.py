from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type
import torch.nn as nn

from .registry import ModelRegistry


def register_family_registry(
    family_registry: Dict[str, Type[nn.Module]],
    *,
    family: str,
    description_getter: Optional[Callable[[str, Type[nn.Module]], str]] = None,
    tags_getter: Optional[Callable[[str, Type[nn.Module]], List[str]]] = None,
    capabilities_getter: Optional[Callable[[str, Type[nn.Module]], Dict[str, Any]]] = None,
) -> None:
    """Registers a dictionary-like family registry into the global ModelRegistry."""
    for name, cls in family_registry.items():
        desc = description_getter(name, cls) if description_getter else ""
        tags = tags_getter(name, cls) if tags_getter else [family]

        caps = capabilities_getter(name, cls) if capabilities_getter else {}
        ModelRegistry.register(
            name=name,
            family=family,
            model_cls=cls,
            description=desc,
            tags=tags,
            input_kind=str(caps.get("input_kind", "pointwise_coords")),
            supports_physics_loss=bool(caps.get("supports_physics_loss", False)),
            expects=list(caps.get("expects", [])),
            predicts=list(caps.get("predicts", [])),
        )
