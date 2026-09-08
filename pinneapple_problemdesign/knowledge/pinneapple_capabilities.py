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
    "cfd": [
        "pinneapple_simulation.numerical_solvers.lbm (LBMSolver / LBMSolver3D): "
        "Lattice-Boltzmann D2Q9 (2D) / D3Q19 (3D) BGK solvers with optional "
        "Smagorinsky LES, Zou-He inlet/outlet BCs (2D), and arbitrary solid "
        "obstacles via bounce-back (both)",
        "pinneapple_physics.pde_environment.turbulence_presets: RANS closures "
        "(KOmegaSSTResiduals, SpalartAllmarasResiduals) and WALEResiduals LES as "
        "PDE-residual functions compatible with PINNFactory",
        "pinneapple_tools.visualization.vortex: Q-criterion / lambda2 / vorticity "
        "post-processing and flow-field visualization (2D and 3D)",
        "pinneapple_simulation.external_solvers.openfoam: bridge to run external "
        "OpenFOAM cases, read fields/meshes, and export results into PINNeAPPle",
    ],
    "physics_knowledge": [
        "pinneapple_problemdesign.knowledge.physics_knowledge.PHENOMENON_KNOWLEDGE_BASE: "
        "phenomenon -> governing-equation -> typical-parameters -> assumptions "
        "records sourced from real pinneapple_physics.pde_environment.presets "
        "modules (cfd, engineering, solid_mechanics, astrophysics, ...), each "
        "traceable back to its source preset via preset_module/preset_function",
        "pinneapple_problemdesign.knowledge.physics_knowledge.lookup_phenomenon(name_or_keyword): "
        "keyword search over the knowledge base for problem-design plan generation",
    ],
}
