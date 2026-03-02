
from __future__ import annotations

from pinneaple_models.registry import ModelRegistry
from .generic_pinn_models import GenericMLP, GenericFourierMLP, GenericSIREN, GenericResMLP, GenericLinear


def register_into_global() -> None:
    # 5 benchmark models for Arena sweeps (pointwise coords)
    ModelRegistry.register(
        name="bench_mlp",
        family="benchmarks",
        model_cls=GenericMLP,
        description="Generic MLP baseline (supervised only).",
        tags=["baseline", "mlp"],
        input_kind="pointwise_coords",
        supports_physics_loss=False,
        expects=["x_data", "y_data"],
        predicts=["u"],
    )
    ModelRegistry.register(
        name="bench_fourier_mlp",
        family="benchmarks",
        model_cls=GenericFourierMLP,
        description="Fourier features MLP (good for PINNs / high-freq).",
        tags=["fourier", "mlp"],
        input_kind="pointwise_coords",
        supports_physics_loss=True,
        expects=["x_col", "x_bc", "x_ic", "x_data"],
        predicts=["u"],
    )
    ModelRegistry.register(
        name="bench_siren",
        family="benchmarks",
        model_cls=GenericSIREN,
        description="SIREN baseline (supports physics loss).",
        tags=["siren", "pinn"],
        input_kind="pointwise_coords",
        supports_physics_loss=True,
        expects=["x_col", "x_bc", "x_ic", "x_data"],
        predicts=["u"],
    )
    ModelRegistry.register(
        name="bench_res_mlp",
        family="benchmarks",
        model_cls=GenericResMLP,
        description="Residual MLP baseline (supervised).",
        tags=["residual", "mlp"],
        input_kind="pointwise_coords",
        supports_physics_loss=False,
        expects=["x_data", "y_data"],
        predicts=["u"],
    )
    ModelRegistry.register(
        name="bench_linear",
        family="benchmarks",
        model_cls=GenericLinear,
        description="Linear baseline (supervised).",
        tags=["linear", "baseline"],
        input_kind="pointwise_coords",
        supports_physics_loss=False,
        expects=["x_data", "y_data"],
        predicts=["u"],
    )
