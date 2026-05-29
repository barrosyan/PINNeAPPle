"""SynthCatalog factory for building synthetic generators by name from a registry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type, Any

from .pde import PDESynthGenerator
from .distributions import DistributionSynthGenerator
from .curvefit import CurveFitSynthGenerator
from .images import ImageReconstructionSynthGenerator
from .geometry import GeometrySynthGenerator
from .pde_symbolic import SymbolicFDSynthGenerator
from .geometry_cadquery import STLTemplateSynthGenerator, ParametricCadQuerySynthGenerator

_REGISTRY: Dict[str, Type] = {
    "pde": PDESynthGenerator,
    "distributions": DistributionSynthGenerator,
    "curvefit": CurveFitSynthGenerator,
    "images": ImageReconstructionSynthGenerator,
    "geometry": GeometrySynthGenerator,
    "pde_symbolic_fd": SymbolicFDSynthGenerator,
    "stl_template": STLTemplateSynthGenerator, 
    "cadquery_parametric": ParametricCadQuerySynthGenerator
}


@dataclass
class SynthCatalog:
    """
    Factory-style catalog for synthetic data generators.

    This class maintains a registry mapping string identifiers to
    generator classes, enabling dynamic construction of synth generators
    by name.

    Attributes:
        registry: Mapping from string keys to generator classes.
                  If not explicitly provided, it is initialized from
                  the internal `_REGISTRY`.
    """
    registry: Dict[str, Type] = None

    def __post_init__(self):
        """Initialize the catalog registry from the default internal mapping."""
        self.registry = dict(_REGISTRY)

    def list(self):
        """
        List all available synthetic generator keys.

        Returns:
            A sorted list of registered generator names.
        """
        return sorted(self.registry.keys())

    def build(self, name: str, **kwargs) -> Any:
        """
        Instantiate a synthetic generator by name.

        Args:
            name: String key identifying the generator (case-insensitive).
            **kwargs: Keyword arguments forwarded to the generator constructor.

        Returns:
            An instance of the requested synthetic generator.

        Raises:
            KeyError: If the provided name is not found in the registry.
        """
        key = name.lower().strip()
        if key not in self.registry:
            raise KeyError(f"Unknown synth generator '{name}'. Available: {self.list()}")
        return self.registry[key](**kwargs)