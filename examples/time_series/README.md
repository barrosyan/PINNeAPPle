# PINNeAPPle Time Series Examples

This folder contains runnable scripts that demonstrate **forecasting workflows** supported by `pinneaple_timeseries`.

## Recommended order

1. **00_quickstart.py**
   - Create a window spec, build deterministic loaders, benchmark strong baselines.

2. **01_FNO.py**
   - Train the default Fourier Neural Operator (FNO) forecaster using `pinneaple_train`.

3. **02_full_pipeline.py**
   - End-to-end: formal problem spec, audit report, splitters, baselines, a deep model, and fold-safe preprocessing.

4. **03_audit_and_features.py**
   - Generate an **audit report** and produce lag/rolling/Fourier features.

5. **04_backtest_baselines.py**
   - Walk-forward backtest baselines (expanding window) to create a deployment-faithful benchmark.

6. **05_quantile_uncertainty.py**
   - Probabilistic forecasts (quantiles) using `QuantileHead` + pinball loss.

7. **06_custom_model_registry.py**
   - Register your own model in `pinneaple_models.registry.ModelRegistry` and build it via `TSModelCatalog`.

## Optional deps

Some audit routines use optional libraries:
- `statsmodels` (ADF/KPSS/ARCH/ACF/PACF)
- `ruptures` (changepoint detection)

The examples are written to degrade gracefully when these are not installed.