"""Mapping of Pinneaple module capabilities for plan generation."""
PINNEAPLE_CAPABILITIES = {
    "training": [
        "pinneaple_train.trainer.Trainer",
        "pinneaple_train.trainer.TrainConfig",
        "pinneaple_train.losses.CombinedLoss / SupervisedLoss / PhysicsLossHook",
        "pinneaple_train.metrics.default_metrics",
    ],
    "models": [
        "pinneaple_models ModelRegistry (families: transformers, recurrent, neural_operators, pinns, ...)",
        "FNO / Neural Operators for operator learning and spatiotemporal mappings",
        "PINNs for physics-driven learning with PDE residuals",
    ],
    "timeseries": [
        "Windowing, temporal splits, direct multi-horizon vs autoregressive",
        "FNO-first baseline wrapper for forecasting problems",
    ],
}
