from .spec       import TimeSeriesSpec
from .problem    import ForecastProblemSpec
from .datamodule import TSDataModule
from .registry   import TSModelCatalog

from .validation.splitters import Split, ExpandingWindowSplitter, RollingWindowSplitter
from .validation.backtest  import BacktestRunner, BacktestConfig, BacktestResult

from .audit.tests  import TSAuditor
from .audit.report import AuditReport, AuditSection

from .features.engineering import TSFeatureEngineer, rate_of_change, window_features

from .baselines.naive import NaiveForecaster, SeasonalNaiveForecaster, DriftForecaster

# --- Preparation pipeline ---
from .preparation.imputer  import TimeSeriesImputer
from .preparation.outliers import OutlierDetector
from .preparation.resampler import TimeSeriesResampler

# --- EDA ---
from .eda import (
    plot_trend, track_setpoint, rolling_statistics, step_response,
    power_spectrum, plot_acf_pacf, stationarity_report,
    changepoint_plot, cross_correlation, rga_matrix,
)

# --- Models ---
from .models import (
    ForecastModel,
    XGBoostForecaster, LightGBMForecaster, RandomForestForecaster,
    CatBoostForecaster, GPRForecaster, MLPForecaster,
    RecurrentConfig, LSTMForecaster, GRUForecaster,
    NBeatsConfig, NBeats,
    TCNConfig, TCNForecaster,
    TFTConfig, TFTForecaster,
)

# --- Signal Decomposition ---
from .decomposition import FFTForecaster, FFTNNForecaster, HHTNNForecaster

# --- Tuning ---
from .tuning import ClassicalTuner, NeuralTuner, temporal_split

# --- Forecast Visualization ---
from .viz import (
    plot_rolling_forecast, plot_forecast_horizon,
    plot_parity, plot_residuals, plot_backtest,
    animate_rolling_forecast,
)

__all__ = [
    # Core
    "TimeSeriesSpec", "ForecastProblemSpec", "TSDataModule", "TSModelCatalog",
    # Validation
    "Split", "ExpandingWindowSplitter", "RollingWindowSplitter",
    "BacktestRunner", "BacktestConfig", "BacktestResult",
    # Audit
    "TSAuditor", "AuditReport", "AuditSection",
    # Features
    "TSFeatureEngineer", "rate_of_change", "window_features",
    # Baselines
    "NaiveForecaster", "SeasonalNaiveForecaster", "DriftForecaster",
    # Preparation
    "TimeSeriesImputer", "OutlierDetector", "TimeSeriesResampler",
    # EDA
    "plot_trend", "track_setpoint", "rolling_statistics", "step_response",
    "power_spectrum", "plot_acf_pacf", "stationarity_report",
    "changepoint_plot", "cross_correlation", "rga_matrix",
    # Models — classical
    "ForecastModel",
    "XGBoostForecaster", "LightGBMForecaster", "RandomForestForecaster",
    "CatBoostForecaster", "GPRForecaster", "MLPForecaster",
    # Models — neural
    "RecurrentConfig", "LSTMForecaster", "GRUForecaster",
    "NBeatsConfig", "NBeats",
    "TCNConfig", "TCNForecaster",
    "TFTConfig", "TFTForecaster",
    # Decomposition
    "FFTForecaster", "FFTNNForecaster", "HHTNNForecaster",
    # Tuning
    "ClassicalTuner", "NeuralTuner", "temporal_split",
    # Visualization
    "plot_rolling_forecast", "plot_forecast_horizon",
    "plot_parity", "plot_residuals", "plot_backtest",
    "animate_rolling_forecast",
]
