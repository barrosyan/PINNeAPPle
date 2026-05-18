"""Mapping of Pinneapple module capabilities for plan generation."""
PINNEAPPLE_CAPABILITIES = {
    "training": [
        "pinneapple_train.trainer.Trainer",
        "pinneapple_train.trainer.TrainConfig",
        "pinneapple_train.losses.CombinedLoss / SupervisedLoss / PhysicsLossHook",
        "pinneapple_train.metrics.default_metrics",
    ],
    "models": [
        "pinneapple_models ModelRegistry (families: transformers, recurrent, neural_operators, pinns, ...)",
        "FNO / Neural Operators for operator learning and spatiotemporal mappings",
        "PINNs for physics-driven learning with PDE residuals",
    ],
    "timeseries": [
        "Windowing, temporal splits, direct multi-horizon vs autoregressive",
        "FNO-first baseline wrapper for forecasting problems",
    ],
}
