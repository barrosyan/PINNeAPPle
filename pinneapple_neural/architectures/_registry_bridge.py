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
    seen_classes: set[Type[nn.Module]] = set()

    for name, model_cls in family_registry.items():
        # ✅ skip aliases (same class referenced by multiple keys)
        if model_cls in seen_classes:
            continue
        seen_classes.add(model_cls)

        desc = description_getter(name, model_cls) if description_getter else ""
        tags = tags_getter(name, model_cls) if tags_getter else [family]
        caps = capabilities_getter(name, model_cls) if capabilities_getter else {}

        ModelRegistry.register(
            name=name,  # primeiro nome visto vira o "canônico"
            family=family,
            description=desc,
            tags=tags,
            input_kind=str(caps.get("input_kind", "pointwise_coords")),
            supports_physics_loss=bool(caps.get("supports_physics_loss", False)),
            expects=list(caps.get("expects", [])),
            predicts=list(caps.get("predicts", [])),
        )(model_cls)