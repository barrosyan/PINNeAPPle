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
    "time_series", "cosimulation", "digital_twin",
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
]
