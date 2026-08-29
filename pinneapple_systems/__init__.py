"""pinneapple_systems — Multi-component physical systems (time series, co-simulation, digital twins).

Sub-modules
-----------
time_series    (was pinneapple_timeseries)
    Time-series forecasting for physical signals: LSTM, GRU, NBeats, TCN,
    TFT, XGBoost, classical decomposition (FFT, HHT), data preparation,
    EDA, backtesting, and visualization.

cosimulation   (was pinneapple_cosim)
    Graph-based differentiable co-simulation engine: compose nodes
    (AnalyticalNode, PINNNode, SymbolicPDENode, TimeSeriesCoSimNode) into
    a CoSimGraph, run with CoSimEngine, train with CoSimTrainer.

digital_twin   (pinneapple_digital_twin — same name, clearer context)
    Digital twin framework: wrap a surrogate model as a live twin, ingest
    real-time sensor streams (MQTT, Kafka, HTTP, file), fuse observations
    with Kalman filters (EKF, EnKF), and detect anomalies.

component_modeling
    Control, optimization, and uncertainty-quantification tooling that
    works on any differentiable component model: PID control, gradient-based
    MPC, SWAG/ensemble/MC-Dropout uncertainty, generic PDE residuals, and
    ONNX edge-deployment packaging.

process_components
    Physical unit-operation models for process/plant digital twins: real-gas
    thermodynamic properties (GERG-2008 via CoolProp), turbomachinery
    (real-gas polytropic compression/expansion + nondimensional similarity
    maps), control valves (IEC 60534-2-1), heat exchangers
    (effectiveness-NTU), and 1D pipe networks (quasi-steady momentum +
    transient continuity).

Integration helpers
-------------------
``forecast(data, horizon, ...)``
    Fit the best auto-selected time-series model and return a forecast.
``cosim_from_models(models_dict, connections)``
    Quick factory that builds a CoSimGraph from a dict of named models.
``build_digital_twin(model, ...)``
    Shortcut for digital_twin.build_digital_twin.

Usage
-----
>>> from pinneapple_systems import build_digital_twin, cosim_from_models, forecast
>>> twin = build_digital_twin(surrogate, field_names=["u", "v", "p"])
>>> graph = cosim_from_models({"fluid": pinn, "thermal": lstm}, [("fluid.T", "thermal.T_in")])
>>> pred = forecast(signal_df, horizon=24, model="lstm")
"""
from __future__ import annotations

# ── sub-modules (new descriptive names) ───────────────────────────────────────
from . import time_series
from . import cosimulation
from . import digital_twin
from . import component_modeling
from . import process_components

# backward-compat aliases
timeseries = time_series
cosim      = cosimulation

# ── time_series re-exports ────────────────────────────────────────────────────
from .time_series import (
    TimeSeriesSpec, ForecastProblemSpec, TSDataModule, TSModelCatalog,
    Split, ExpandingWindowSplitter, RollingWindowSplitter,
    BacktestRunner, BacktestConfig, BacktestResult,
    TSAuditor, AuditReport, AuditSection,
    TSFeatureEngineer, rate_of_change, window_features,
    NaiveForecaster, SeasonalNaiveForecaster, DriftForecaster,
    TimeSeriesImputer, OutlierDetector, TimeSeriesResampler,
    plot_trend, track_setpoint, rolling_statistics, step_response,
    power_spectrum, plot_acf_pacf, stationarity_report,
    changepoint_plot, cross_correlation, rga_matrix,
    ForecastModel,
    XGBoostForecaster, LightGBMForecaster, RandomForestForecaster,
    CatBoostForecaster, GPRForecaster, MLPForecaster,
    RecurrentConfig, LSTMForecaster, GRUForecaster,
    NBeatsConfig, NBeats,
    TCNConfig, TCNForecaster,
    TFTConfig, TFTForecaster,
    FFTForecaster, FFTNNForecaster, HHTNNForecaster,
    ClassicalTuner, NeuralTuner, temporal_split,
    plot_rolling_forecast, plot_forecast_horizon,
    plot_parity, plot_residuals, plot_backtest,
    animate_rolling_forecast,
)

# ── cosimulation re-exports ───────────────────────────────────────────────────
from .cosimulation import (
    CoSimNode, TorchNode, AnalyticalNode, PINNNode, BlackBoxNode,
    PINNeProblemNode, PINNeAPPleModelNode, SymbolicPDENode, TimeSeriesCoSimNode,
    Connection, CoSimGraph,
    Trajectory, TrajectoryRecorder,
    CoSimEngine,
    CoSimTrainer,
    DataLoss, PhysicsLoss, CouplingLoss, CoSimLoss,
)

# ── digital_twin re-exports ───────────────────────────────────────────────────
from .digital_twin import (
    DigitalTwin, DigitalTwinConfig, build_digital_twin,
    SystemState, Observation,
    Sensor, SensorRegistry,
    BaseStream, FileWatchStream, MQTTStream,
    HTTPPollStream, KafkaStream, MockStream,
    ExtendedKalmanFilter, EnsembleKalmanFilter,
    AnomalyEvent, AnomalyMonitor,
    ThresholdDetector, ZScoreDetector, MahalanobisDetector,
)

# ── component_modeling re-exports ─────────────────────────────────────────────
from .component_modeling import (
    PIDController, run_closed_loop,
    run_mpc,
    SWAGApproximation,
    DeepEnsemble,
    mc_dropout_uncertainty,
    incompressible_continuity_residual, heat_conduction_residual,
    linear_elasticity_residual, species_diffusion_residual,
    export_edge_package, EdgeRuntime,
)

# ── process_components re-exports ─────────────────────────────────────────────
from .process_components import (
    BeamResult, solve_beam, rectangular_section_properties, von_mises_stress_rectangular_section,
    StaticNonlinearBeamResult, solve_nonlinear_beam_static,
    ExplicitEquationError, Definition, AnalysisResult, ParameterSpec, CalibrationResult,
    safe_eval, build_definitions, evaluation_order, evaluate, analyze, calibrate,
    GasComposition, GasState, OutOfEnvelopeError, StandardConditions, ValidityEnvelope,
    state_from_PT, state_from_Ph, state_from_Ps, central_difference,
    standard_volumetric_flow_to_mass_flow, mass_flow_to_standard_volumetric_flow,
    PolytropicPathResult, solve_path_from_pressure_ratio, solve_path_from_work,
    MapCoefficients, MapEvaluation, make_map, evaluate_map,
    flow_coefficient, tip_speed_m_s, tip_mach_number,
    polytropic_head_from_psi, required_speed_for_head,
    ValveSpec, ValveFlowResult, installed_cv, effective_cv,
    compressible_mass_flow, incompressible_mass_flow, actuator_response_rhs,
    HeatExchangerSpec, HeatExchangerResult, heat_exchanger_steady_state, heat_exchanger_transient_rhs,
    PipeSpec, PipeState, SteadyProfilePoint, TransientPipe,
    colebrook_white_f, rapid_steady_state_profile,
    Reaction, ReactionNetwork, IntegrationResult, AdvectionDispersionReactionSolver,
    mass_action_rate, arrhenius_rate_constant, quadratic_in_T,
    linear_combination_rate_constant, acid_fraction, base_fraction,
    diprotic_fractions, integrate_network,
    PressureProfile, herschel_bulkley_stress, non_newtonian_effective_viscosity,
    generalized_reynolds_number, metzner_reed_friction_factor,
    non_newtonian_pressure_gradient, integrate_pressure_profile,
    CurvedPathProfile, inclination_at_depth, build_and_hold_profile, circular_arc_tvd_hd,
    ConstrainedRodBucklingResult, RotatingBendingCycleResult,
    lame_hoop_stress_outer, lame_hoop_stress_inner,
    torsional_shear_stress, bending_stress_from_curvature, von_mises_triaxial,
    euler_critical_buckling_load, constrained_rod_buckling_load, classify_buckling_mode,
    beam_column_moment_amplification_factor, rotating_bending_stress_cycle,
    StickSlipResult, stribeck_friction_torque, simulate_torsional_stickslip,
    MinerDamageResult, sn_curve_cycles_to_failure, goodman_equivalent_amplitude,
    goodman_safety_ratio, miners_rule_damage,
)


# ── Integration helpers ────────────────────────────────────────────────────────

def cosim_from_models(models_dict: dict, connections: list) -> "CoSimGraph":
    """Build a CoSimGraph from a dict of named nn.Module instances."""
    graph = CoSimGraph()
    for name, model in models_dict.items():
        node = PINNeAPPleModelNode(name, model,
                                   input_ports=["input"],
                                   output_ports=["output"])
        graph.add_node(node)
    for src, dst in connections:
        graph.connect(src, dst)
    return graph


def forecast(data, horizon: int, *, model: str = "auto", **kwargs):
    """Quick time-series forecast with auto model selection."""
    _MODEL_MAP = {
        "lstm":    LSTMForecaster,
        "gru":     GRUForecaster,
        "nbeats":  NBeats,
        "tft":     TFTForecaster,
        "tcn":     TCNForecaster,
        "xgboost": XGBoostForecaster,
    }
    if model == "auto":
        model = "lstm"
    cls = _MODEL_MAP.get(model)
    if cls is None:
        raise ValueError(f"Unknown model '{model}'. Available: {list(_MODEL_MAP)}")
    forecaster = cls(horizon=horizon, **kwargs)
    forecaster.fit(data)
    return forecaster.predict(horizon)


__all__ = [
    # Sub-modules (new names)
    "time_series", "cosimulation", "digital_twin", "component_modeling", "process_components",
    # Sub-modules (old aliases — backward compat)
    "timeseries", "cosim",
    # Integration
    "cosim_from_models", "forecast",
    # time_series
    "TimeSeriesSpec", "ForecastProblemSpec", "TSDataModule", "TSModelCatalog",
    "Split", "ExpandingWindowSplitter", "RollingWindowSplitter",
    "BacktestRunner", "BacktestConfig", "BacktestResult",
    "TSAuditor", "AuditReport", "AuditSection",
    "TSFeatureEngineer", "rate_of_change", "window_features",
    "NaiveForecaster", "SeasonalNaiveForecaster", "DriftForecaster",
    "TimeSeriesImputer", "OutlierDetector", "TimeSeriesResampler",
    "plot_trend", "track_setpoint", "rolling_statistics", "step_response",
    "power_spectrum", "plot_acf_pacf", "stationarity_report",
    "changepoint_plot", "cross_correlation", "rga_matrix",
    "ForecastModel",
    "XGBoostForecaster", "LightGBMForecaster", "RandomForestForecaster",
    "CatBoostForecaster", "GPRForecaster", "MLPForecaster",
    "RecurrentConfig", "LSTMForecaster", "GRUForecaster",
    "NBeatsConfig", "NBeats", "TCNConfig", "TCNForecaster",
    "TFTConfig", "TFTForecaster",
    "FFTForecaster", "FFTNNForecaster", "HHTNNForecaster",
    "ClassicalTuner", "NeuralTuner", "temporal_split",
    "plot_rolling_forecast", "plot_forecast_horizon",
    "plot_parity", "plot_residuals", "plot_backtest", "animate_rolling_forecast",
    # cosimulation
    "CoSimNode", "TorchNode", "AnalyticalNode", "PINNNode", "BlackBoxNode",
    "PINNeProblemNode", "PINNeAPPleModelNode", "SymbolicPDENode", "TimeSeriesCoSimNode",
    "Connection", "CoSimGraph",
    "Trajectory", "TrajectoryRecorder",
    "CoSimEngine", "CoSimTrainer",
    "DataLoss", "PhysicsLoss", "CouplingLoss", "CoSimLoss",
    # digital_twin
    "DigitalTwin", "DigitalTwinConfig", "build_digital_twin",
    "SystemState", "Observation",
    "Sensor", "SensorRegistry",
    "BaseStream", "FileWatchStream", "MQTTStream",
    "HTTPPollStream", "KafkaStream", "MockStream",
    "ExtendedKalmanFilter", "EnsembleKalmanFilter",
    "AnomalyEvent", "AnomalyMonitor",
    "ThresholdDetector", "ZScoreDetector", "MahalanobisDetector",
    # component_modeling
    "PIDController", "run_closed_loop", "run_mpc",
    "SWAGApproximation", "DeepEnsemble", "mc_dropout_uncertainty",
    "incompressible_continuity_residual", "heat_conduction_residual",
    "linear_elasticity_residual", "species_diffusion_residual",
    "export_edge_package", "EdgeRuntime",
    # process_components
    "BeamResult", "solve_beam", "rectangular_section_properties", "von_mises_stress_rectangular_section",
    "StaticNonlinearBeamResult", "solve_nonlinear_beam_static",
    "ExplicitEquationError", "Definition", "AnalysisResult", "ParameterSpec", "CalibrationResult",
    "safe_eval", "build_definitions", "evaluation_order", "evaluate", "analyze", "calibrate",
    "GasComposition", "GasState", "OutOfEnvelopeError", "StandardConditions", "ValidityEnvelope",
    "state_from_PT", "state_from_Ph", "state_from_Ps", "central_difference",
    "standard_volumetric_flow_to_mass_flow", "mass_flow_to_standard_volumetric_flow",
    "PolytropicPathResult", "solve_path_from_pressure_ratio", "solve_path_from_work",
    "MapCoefficients", "MapEvaluation", "make_map", "evaluate_map",
    "flow_coefficient", "tip_speed_m_s", "tip_mach_number",
    "polytropic_head_from_psi", "required_speed_for_head",
    "ValveSpec", "ValveFlowResult", "installed_cv", "effective_cv",
    "compressible_mass_flow", "incompressible_mass_flow", "actuator_response_rhs",
    "HeatExchangerSpec", "HeatExchangerResult", "heat_exchanger_steady_state", "heat_exchanger_transient_rhs",
    "PipeSpec", "PipeState", "SteadyProfilePoint", "TransientPipe",
    "colebrook_white_f", "rapid_steady_state_profile",
    "Reaction", "ReactionNetwork", "IntegrationResult", "AdvectionDispersionReactionSolver",
    "mass_action_rate", "arrhenius_rate_constant", "quadratic_in_T",
    "linear_combination_rate_constant", "acid_fraction", "base_fraction",
    "diprotic_fractions", "integrate_network",
    "PressureProfile", "herschel_bulkley_stress", "non_newtonian_effective_viscosity",
    "generalized_reynolds_number", "metzner_reed_friction_factor",
    "non_newtonian_pressure_gradient", "integrate_pressure_profile",
    "CurvedPathProfile", "inclination_at_depth", "build_and_hold_profile", "circular_arc_tvd_hd",
    "ConstrainedRodBucklingResult", "RotatingBendingCycleResult",
    "lame_hoop_stress_outer", "lame_hoop_stress_inner",
    "torsional_shear_stress", "bending_stress_from_curvature", "von_mises_triaxial",
    "euler_critical_buckling_load", "constrained_rod_buckling_load", "classify_buckling_mode",
    "beam_column_moment_amplification_factor", "rotating_bending_stress_cycle",
    "StickSlipResult", "stribeck_friction_torque", "simulate_torsional_stickslip",
    "MinerDamageResult", "sn_curve_cycles_to_failure", "goodman_equivalent_amplitude",
    "goodman_safety_ratio", "miners_rule_damage",
]
