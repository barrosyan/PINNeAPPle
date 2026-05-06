from __future__ import annotations
"""Aggregated ModelCatalog combining all model family catalogs."""
from dataclasses import dataclass, field

from .autoencoders.registry import AutoencoderCatalog
from .pinns.registry import PINNCatalog
from .transformers.registry import TransformerCatalog
from .recurrent.registry import RecurrentCatalog
from .convolutions.registry import ConvolutionCatalog
from .neural_operators.registry import NeuralOperatorCatalog
from .continuous.registry import ContinuousCatalog
from .graphnn.registry import GraphCatalog
from .reservoir_computing.registry import ReservoirCatalog
from .classical_ts.registry import ClassicalTSCatalog
from .rom.registry import ROMCatalog
from .physics_aware.registry import PhysicsAwareCatalog


@dataclass
class ModelCatalog:
    autoencoders:       AutoencoderCatalog    = field(default_factory=AutoencoderCatalog)
    pinns:              PINNCatalog           = field(default_factory=PINNCatalog)
    transformers:       TransformerCatalog    = field(default_factory=TransformerCatalog)
    recurrent:          RecurrentCatalog      = field(default_factory=RecurrentCatalog)
    convolutions:       ConvolutionCatalog    = field(default_factory=ConvolutionCatalog)
    neural_operators:   NeuralOperatorCatalog = field(default_factory=NeuralOperatorCatalog)
    continuous:         ContinuousCatalog     = field(default_factory=ContinuousCatalog)
    graphnn:            GraphCatalog          = field(default_factory=GraphCatalog)
    reservoir_computing: ReservoirCatalog     = field(default_factory=ReservoirCatalog)
    classical_ts:       ClassicalTSCatalog    = field(default_factory=ClassicalTSCatalog)
    rom:                ROMCatalog            = field(default_factory=ROMCatalog)
    physics_aware:      PhysicsAwareCatalog   = field(default_factory=PhysicsAwareCatalog)
