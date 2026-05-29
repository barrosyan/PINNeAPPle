from .pinn_factory import NeuralNetwork, PINN, PINNFactory, PINNProblemSpec
from .sympy_backend import SympyTorchCompiler, CompiledEquation
from .autodiff import DerivativeComputer, ensure_requires_grad

# Canonical model — preferred over NeuralNetwork/PINN for new code.
from pinneapple_neural.architectures.pinns.vanilla import VanillaPINN  # noqa: F401

__all__ = [
    # Backwards-compatible factory models
    "NeuralNetwork",
    "PINN",
    # Canonical PINN model (preferred)
    "VanillaPINN",
    "PINNFactory",
    "PINNProblemSpec",
    "SympyTorchCompiler",
    "CompiledEquation",
    "DerivativeComputer",
    "ensure_requires_grad",
]
