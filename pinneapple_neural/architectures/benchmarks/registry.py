from __future__ import annotations

from pinneapple_neural.architectures.registry import ModelRegistry
from .generic_pinn_models import GenericMLP, GenericFourierMLP, GenericSIREN, GenericResMLP, GenericLinear

_BENCHMARKS = [
    (GenericMLP,       "bench_mlp",         "Generic MLP baseline (tanh, 4x128).",              False, ["x_data", "y_data"]),
    (GenericFourierMLP,"bench_fourier_mlp",  "Fourier-features MLP — good for high-freq fields.", True,  ["x_col", "x_bc", "x_ic"]),
    (GenericSIREN,     "bench_siren",        "SIREN baseline (sine activations, physics loss).",  True,  ["x_col", "x_bc", "x_ic"]),
    (GenericResMLP,    "bench_res_mlp",      "Residual MLP baseline (GELU, LayerNorm).",          False, ["x_data", "y_data"]),
    (GenericLinear,    "bench_linear",       "Linear baseline.",                                  False, ["x_data", "y_data"]),
]


def register_into_global() -> None:
    """Register benchmark models into the global ModelRegistry.

    Uses the correct decorator-call pattern:
        ModelRegistry.register(**kwargs)(ModelClass)
    """
    for cls, name, desc, phys, expects in _BENCHMARKS:
        ModelRegistry.register(
            name=name,
            family="benchmarks",
            description=desc,
            tags=["benchmark"],
            input_kind="pointwise_coords",
            supports_physics_loss=phys,
            expects=expects,
            predicts=["u"],
        )(cls)
