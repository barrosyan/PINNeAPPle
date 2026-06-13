from __future__ import annotations
"""PINN model family — all variants, base classes, and registry."""

from .base import PINNBase, PINNOutput
from .vanilla import VanillaPINN
from .inverse import InversePINN
from .pielm import PIELM
from .pinn_lstm import PINNLSTM
from .pinnsformer import PINNsFormer
from .vpinn import VPINN
from .xpinn import XPINN, SubnetWrapper
from .xtfc import XTFC, XTFCConfig, build_xtfc, tfc_available
from .bpinn import (
    BayesianPINN, BPINNConfig, BayesianLinear,
)
from .physics_diffusion import (
    PhysicsInformedDiffusion, PIDiffConfig,
    MLPDenoiser, VPNoiseScheduler, EDMNoiseScheduler,
    DSMLoss, PDEResidualGuidance, DataConsistencyGuidance, ComposedGuidance,
)
from .registry import PINNCatalog

__all__ = [
    # base
    "PINNBase",
    "PINNOutput",
    # model variants
    "VanillaPINN",
    "InversePINN",
    "PIELM",
    "PINNLSTM",
    "PINNsFormer",
    "VPINN",
    "XPINN",
    "SubnetWrapper",
    "XTFC",
    "XTFCConfig",
    "build_xtfc",
    "tfc_available",
    # Bayesian PINN
    "BayesianPINN",
    "BPINNConfig",
    "BayesianLinear",
    # Physics-Informed Diffusion
    "PhysicsInformedDiffusion",
    "PIDiffConfig",
    "MLPDenoiser",
    "VPNoiseScheduler",
    "EDMNoiseScheduler",
    "DSMLoss",
    "PDEResidualGuidance",
    "DataConsistencyGuidance",
    "ComposedGuidance",
    # registry
    "PINNCatalog",
]
