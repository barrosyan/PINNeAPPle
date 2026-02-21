from __future__ import annotations
"""Bridge family-local registries into the global ModelRegistry."""

from typing import Dict, Type, Any, Optional, Callable

from .registry import ModelRegistry
from .base import BaseModel


def register_family_registry(
    local_registry: Dict[str, Type[Any]],
    *,
    family: str,
    name_mapper: Optional[Callable[[str, Type[Any]], str]] = None,
    description_getter: Optional[Callable[[Type[Any]], str]] = None,
    tags_getter: Optional[Callable[[Type[Any]], list[str]]] = None,
    skip_aliases: bool = True,
) -> None:
    """
    Bridge a family-local _REGISTRY into the global ModelRegistry.

    local_registry: dict key->cls (your existing pattern)
    family: global family name ("transformers", "pinns", ...)
    name_mapper: maps (key, cls) -> global model name
    skip_aliases: if True, avoids registering multiple keys pointing to the same class
    """
    name_mapper = name_mapper or (lambda key, cls: key)
    description_getter = description_getter or (lambda cls: (getattr(cls, "__doc__", "") or "").strip())
    tags_getter = tags_getter or (lambda cls: [])

    seen_cls: set[int] = set()

    for key, cls in local_registry.items():
        if cls is None:
            continue

        if skip_aliases:
            cid = id(cls)
            if cid in seen_cls:
                continue
            seen_cls.add(cid)

        # Ensure it behaves like a BaseModel (soft check; avoids runtime surprises)
        # We don't enforce issubclass here because some families may use different base classes.
        # But if you want strict behavior, uncomment this:
        # if not issubclass(cls, BaseModel): continue

        global_name = name_mapper(key, cls).strip()
        if not global_name:
            continue

        try:
            ModelRegistry.register(
                name=global_name,
                family=family,
                description=description_getter(cls),
                tags=tags_getter(cls),
            )(cls)  # decorator returns cls
        except KeyError:
            # already registered (e.g. same name from another import path)
            pass
