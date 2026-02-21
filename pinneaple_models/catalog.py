from __future__ import annotations
"""Aggregated ModelCatalog combining all model family catalogs."""
from dataclasses import dataclass

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
    autoencoders: AutoencoderCatalog = AutoencoderCatalog()
    pinns: PINNCatalog = PINNCatalog()
    transformers: TransformerCatalog = TransformerCatalog()
    recurrent: RecurrentCatalog = RecurrentCatalog()
    convolutions: ConvolutionCatalog = ConvolutionCatalog()
    neural_operators: NeuralOperatorCatalog = NeuralOperatorCatalog()
    continuous: ContinuousCatalog = ContinuousCatalog()
    graphnn: GraphCatalog = GraphCatalog()
    reservoir_computing: ReservoirCatalog = ReservoirCatalog()
    classical_ts: ClassicalTSCatalog = ClassicalTSCatalog()
    rom: ROMCatalog = ROMCatalog()
    physics_aware: PhysicsAwareCatalog = PhysicsAwareCatalog()