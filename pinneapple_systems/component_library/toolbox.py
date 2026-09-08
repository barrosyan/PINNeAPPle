"""pinneapple_systems.component_library.toolbox — ``Toolbox``: a named,
versioned, installable bundle of ``component_type``s (e.g. "Fluid
Mechanics Toolbox v1.0" bundling pipe+pump+valve).

Purely an in-repo bookkeeping/dependency-resolution layer, honestly
scoped: there is no external package index or download step here.
``install()``/``uninstall()`` only track an in-memory "installed" set
(mirroring what a real package manager's *resolve* phase would compute)
— they do not fetch or execute any code the component classes don't
already provide via ``ComponentRegistry``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .registry import ComponentRegistry, ComponentSpec


@dataclass
class Toolbox:
    name: str
    version: str
    component_types: List[str]
    description: str = ""
    depends_on: List[str] = field(default_factory=list)  # other toolbox "name@version" keys

    @property
    def key(self) -> str:
        return f"{self.name.lower()}@{self.version}"

    def components(self) -> List[ComponentSpec]:
        out: List[ComponentSpec] = []
        for ct in self.component_types:
            out.extend(ComponentRegistry.by_type(ct))
        return out


class ToolboxRegistry:
    """Registry of :class:`Toolbox` bundles, plus a tiny installed-set."""

    _REGISTRY: Dict[str, Toolbox] = {}
    _INSTALLED: set = set()

    @classmethod
    def register(cls, toolbox: Toolbox) -> None:
        if toolbox.key in cls._REGISTRY:
            raise KeyError(f"Toolbox '{toolbox.key}' already registered")
        cls._REGISTRY[toolbox.key] = toolbox

    @classmethod
    def list(cls) -> List[str]:
        return sorted(cls._REGISTRY.keys())

    @classmethod
    def get(cls, name: str, version: Optional[str] = None) -> Toolbox:
        if version:
            key = f"{name.lower()}@{version}"
            if key not in cls._REGISTRY:
                raise KeyError(f"Unknown toolbox '{key}'. Available: {cls.list()}")
            return cls._REGISTRY[key]
        matches = sorted(k for k in cls._REGISTRY if k.startswith(f"{name.lower()}@"))
        if not matches:
            raise KeyError(f"Unknown toolbox '{name}'. Available: {cls.list()}")
        return cls._REGISTRY[matches[-1]]  # newest by lexicographic version sort

    @classmethod
    def install(cls, name: str, version: Optional[str] = None) -> Toolbox:
        toolbox = cls.get(name, version)
        for dep in toolbox.depends_on:
            cls.install(dep)
        cls._INSTALLED.add(toolbox.key)
        return toolbox

    @classmethod
    def uninstall(cls, name: str, version: Optional[str] = None) -> None:
        toolbox = cls.get(name, version)
        cls._INSTALLED.discard(toolbox.key)

    @classmethod
    def is_installed(cls, name: str, version: Optional[str] = None) -> bool:
        return cls.get(name, version).key in cls._INSTALLED

    @classmethod
    def installed(cls) -> List[str]:
        return sorted(cls._INSTALLED)


def _register_default_toolboxes() -> None:
    """The two toolboxes implied by the six reference components — kept
    here rather than hand-authored by a user, so ``import
    pinneapple_systems.component_library`` has something real to list out
    of the box."""
    try:
        ToolboxRegistry.register(Toolbox(
            name="Fluid Mechanics Toolbox", version="1.0",
            component_types=["pipe", "pump", "valve"],
            description="Internal-flow components: pipe, pump, valve.",
        ))
        ToolboxRegistry.register(Toolbox(
            name="Process Equipment Toolbox", version="1.0",
            component_types=["heat_exchanger", "tank_shell", "battery_cell"],
            description="Thermal, structural, and electrochemical process components.",
        ))
    except KeyError:
        pass  # already registered (e.g. module re-imported)


_register_default_toolboxes()
