from .base import NeuralOperatorBase, OperatorOutput
from .registry import NeuralOperatorCatalog
from .fno import FourierNeuralOperator, FNO2d, MLPFNOSurrogate

# Noether (Emmi AI) — optional; None if emmiai-noether is not installed
try:
    from .noether_bridge import (
        NoetherUPT,
        NoetherABUPT,
        NoetherTransformer,
        NoetherTransolver,
        NoetherAeroUPT,
        NoetherAeroABUPT,
        NoetherAeroTransformer,
        NoetherAeroTransolver,
        NoetherModelConfig,
    )
except Exception:
    NoetherUPT = None                # type: ignore
    NoetherABUPT = None              # type: ignore
    NoetherTransformer = None        # type: ignore
    NoetherTransolver = None         # type: ignore
    NoetherAeroUPT = None            # type: ignore
    NoetherAeroABUPT = None          # type: ignore
    NoetherAeroTransformer = None    # type: ignore
    NoetherAeroTransolver = None     # type: ignore
    NoetherModelConfig = None        # type: ignore

__all__ = [
    "NeuralOperatorBase",
    "OperatorOutput",
    "NeuralOperatorCatalog",
    "FourierNeuralOperator",
    "FNO2d",
    "MLPFNOSurrogate",
    # Noether
    "NoetherUPT",
    "NoetherABUPT",
    "NoetherTransformer",
    "NoetherTransolver",
    "NoetherAeroUPT",
    "NoetherAeroABUPT",
    "NoetherAeroTransformer",
    "NoetherAeroTransolver",
    "NoetherModelConfig",
]
