from .base import SynthGenerator, SynthOutput, SynthConfig
from .registry import SynthCatalog

from .pde import PDESynthGenerator
from .distributions import DistributionSynthGenerator
from .curvefit import CurveFitSynthGenerator
from .images import ImageReconstructionSynthGenerator
from .geometry import GeometrySynthGenerator

from .sample_adapter import (
    has_pinnego_physical_sample,
    to_physical_sample,
    PhysicalSampleLike,
)

from .pde_symbolic import SymbolicFDSynthGenerator, make_fd_residual_fn
from .geometry_cadquery import ParametricCadQuerySynthGenerator, STLTemplateSynthGenerator


__all__ = [
    "SynthGenerator",
    "SynthOutput",
    "SynthConfig",
    "SynthCatalog",
    "PDESynthGenerator",
    "DistributionSynthGenerator",
    "CurveFitSynthGenerator",
    "ImageReconstructionSynthGenerator",
    "GeometrySynthGenerator",
    "has_pinnego_physical_sample",
    "to_physical_sample",
    "PhysicalSampleLike",
    "SymbolicFDSynthGenerator",
    "make_fd_residual_fn",
    "ParametricCadQuerySynthGenerator",
    "STLTemplateSynthGenerator",
]
