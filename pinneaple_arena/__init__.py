"""pinneaple_arena — YAML/JSON-driven multi-model physics benchmark runner.

Features
--------
  * ALL pinneaple_neural models via ModelRegistry.build() (~80+ architectures)
  * pinneaple_physics compiled losses (when physics_preset is set)
  * pinneaple_data datasets (DatasetConfig)
  * pinneaple_analysis inverse problems (InverseConfig.enabled=True)
  * pinneaple_analysis UQ (UQConfig.enabled=True)

Quick start
-----------
    from pinneaple_arena import Arena

    arena = Arena.from_yaml("my_benchmark.yaml")
    arena.run()

    # programmatic:
    from pinneaple_arena import ArenaConfig
    cfg = ArenaConfig.from_dict({
        "problem": {"name": "kovasznay_ns", "params": {"re": 40.0}},
        "models": [
            {"name": "VanillaPINN", "type": "vanilla_pinn",
             "network": {"hidden": [128, 128, 128, 128]},
             "training": {"epochs": 2000, "lr": 5e-4}},
            {"name": "FNO-2D", "type": "fno2d",
             "training": {"epochs": 600, "lr": 1e-3}},
            {"name": "MeshGraphNet", "type": "meshgraphnet",
             "training": {"epochs": 600, "lr": 1e-3}},
        ],
        "output": {"dir": "outputs/", "prefix": "benchmark"},
    })
    Arena(cfg).run()

Dataset mode
------------
    cfg = ArenaConfig.from_dict({
        "problem": {"name": "kovasznay_ns", "params": {}},
        "dataset": {
            "dataset_id": "navier_stokes_2d",
            "input_fields": ["x", "y"],
            "output_fields": ["u", "v"],
            "n_train": 2000,
        },
        "models": [...],
    })

With UQ
-------
    cfg = ArenaConfig.from_dict({
        ...,
        "uq": {"enabled": True, "method": "mc_dropout", "n_samples": 50},
    })

With inverse problem
--------------------
    cfg = ArenaConfig.from_dict({
        ...,
        "inverse": {"enabled": True, "params": ["re"], "n_iters": 1000},
    })
"""
from .config import (
    ArenaConfig,
    ProblemConfig,
    ModelConfig,
    NetworkConfig,
    TrainingConfig,
    OutputConfig,
    InverseConfig,
    UQConfig,
    DatasetConfig,
)
from .arena import Arena
from .problems import get_problem, list_problems, list_problems_by_domain, ArenaProblem, register_problem
from .model_factory import (
    build_model,
    is_pinn_model,
    is_graph_model,
    is_operator_model,
    is_inverse_model,
)
from .trainer import (
    TrainResult,
    train_pinn,
    train_supervised,
    train_graph,
    evaluate_model,
    run_uq,
    run_inverse,
    load_pinneaple_dataset,
)
from .visualizer import (
    plot_field_comparison,
    plot_loss_curves,
    plot_metrics_table,
    plot_streamlines,
    plot_uq,
)

__version__ = "0.2.0"

__all__ = [
    # config
    "ArenaConfig", "ProblemConfig", "ModelConfig",
    "NetworkConfig", "TrainingConfig", "OutputConfig",
    "InverseConfig", "UQConfig", "DatasetConfig",
    # main class
    "Arena",
    # problems
    "ArenaProblem", "get_problem", "register_problem",
    "list_problems", "list_problems_by_domain",
    # model building
    "build_model", "is_pinn_model", "is_graph_model",
    "is_operator_model", "is_inverse_model",
    # training / eval
    "TrainResult", "train_pinn", "train_supervised", "train_graph",
    "evaluate_model", "run_uq", "run_inverse", "load_pinneaple_dataset",
    # visualisation
    "plot_field_comparison", "plot_loss_curves", "plot_metrics_table",
    "plot_streamlines", "plot_uq",
]
