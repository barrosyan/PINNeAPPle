"""Base SynthConfig, SynthOutput, and SynthGenerator protocol for synthetic data generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, List


@dataclass
class SynthConfig:
    """
    Configuration container for synthetic data generation.

    Attributes
    ----------
    seed : int
        Random seed for reproducibility.
    device : str
        Target device for tensor allocation (e.g., "cpu", "cuda").
    dtype : str
        Default tensor data type (e.g., "float32", "float64").
    """
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float32"


@dataclass
class SynthOutput:
    """
    Standard output structure for synthetic generators.

    Attributes
    ----------
    samples : List[Any]
        List of generated sample objects (e.g., tensors, datasets,
        PhysicalSample instances, etc.).
    extras : Dict[str, Any]
        Additional metadata or auxiliary outputs produced during generation.
    """
    samples: List[Any]
    extras: Dict[str, Any]


class SynthGenerator(Protocol):
    """
    Protocol defining the interface for synthetic data generators.

    Any concrete synthetic generator must implement a `generate`
    method returning a SynthOutput instance.
    """

    def generate(self, **kwargs) -> SynthOutput:
        """
        Generate synthetic samples.

        Parameters
        ----------
        **kwargs : Any
            Generator-specific configuration parameters.

        Returns
        -------
        SynthOutput
            Generated samples and associated metadata.
        """
        ...