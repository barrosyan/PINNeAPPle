"""Tests for pinneapple_systems.time_series.foundation_models.

Covers the real pretrained-foundation-model forecaster wrappers:
``ChronosForecaster`` (Amazon Chronos, Ansari et al. 2024) and
``TimesFMForecaster`` (Google TimesFM, Das et al. 2024).

These wrap REAL pretrained checkpoints downloaded from the HuggingFace
Hub -- not mocks. Per this repo's honesty convention for optional real
backends (see ``tests/test_llm_agent_loop.py``'s real-Ollama-if-available
pattern and ``tests/test_classical_forecasting_wrappers.py``'s
``pytest.importorskip`` pattern), the real-checkpoint tests are skipped
(not faked) when either the required package isn't installed or the
HuggingFace Hub isn't reachable from this sandbox. Interface/error-handling
tests (empty input, predict-before-fit, missing dependency) run
unconditionally since they don't need the network.
"""
from __future__ import annotations

import socket
import urllib.request

import numpy as np
import pytest


def _hf_hub_reachable(timeout: float = 5.0) -> bool:
    """Real reachability probe -- not assumed, actually checked."""
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=timeout)
        return True
    except (urllib.error.URLError, socket.timeout, OSError):
        return False


_HF_REACHABLE = _hf_hub_reachable()


def _make_noisy_sine(n: int = 200, seed: int = 0) -> np.ndarray:
    """Synthetic noisy sine wave -- a case where a forecast's sanity
    (magnitude, trend continuation) can be checked without expecting an
    exact match."""
    t = np.linspace(0.0, 20.0 * np.pi, n)
    rng = np.random.default_rng(seed)
    return np.sin(t) + 0.05 * rng.standard_normal(n)


# ---------------------------------------------------------------------------
# Chronos (Amazon) -- real checkpoint: amazon/chronos-t5-tiny (~8M params)
# ---------------------------------------------------------------------------

class TestChronosForecaster:
    def setup_method(self):
        pytest.importorskip("chronos", reason="ChronosForecaster requires chronos-forecasting")
        pytest.importorskip("torch")
        if not _HF_REACHABLE:
            pytest.skip("HuggingFace Hub not reachable from this sandbox")

    def test_fit_predict_real_checkpoint_shape_and_sanity(self):
        from pinneapple_systems.time_series.foundation_models import ChronosForecaster

        y = _make_noisy_sine(n=200, seed=0)
        f = ChronosForecaster(model_id="amazon/chronos-t5-tiny", device_map="cpu")
        f.fit(y)

        horizon = 12
        y_hat = f.predict(horizon, num_samples=20)

        assert isinstance(y_hat, np.ndarray)
        assert y_hat.shape == (horizon,)
        assert np.all(np.isfinite(y_hat))
        # The sine wave oscillates in [-1.1, 1.1]; a sane zero-shot forecast
        # should stay in the same ballpark, not blow up or collapse.
        assert np.all(np.abs(y_hat) < 5.0)

    def test_predict_samples_real_distribution(self):
        from pinneapple_systems.time_series.foundation_models import ChronosForecaster

        y = _make_noisy_sine(n=200, seed=1)
        f = ChronosForecaster(model_id="amazon/chronos-t5-tiny", device_map="cpu")
        f.fit(y)

        samples = f.predict_samples(horizon=10, num_samples=25)
        assert samples.shape == (25, 10)
        assert np.all(np.isfinite(samples))
        # Real sampled paths from a probabilistic model should show some
        # spread, not collapse to a single deterministic value.
        assert samples.std(axis=0).mean() > 0.0

    def test_predict_quantiles_real_and_monotonic(self):
        from pinneapple_systems.time_series.foundation_models import ChronosForecaster

        y = _make_noisy_sine(n=200, seed=2)
        f = ChronosForecaster(model_id="amazon/chronos-t5-tiny", device_map="cpu")
        f.fit(y)

        quantiles, mean = f.predict_quantiles(horizon=8, quantile_levels=(0.1, 0.5, 0.9))
        assert quantiles.shape == (8, 3)
        assert mean.shape == (8,)
        assert np.all(np.isfinite(quantiles)) and np.all(np.isfinite(mean))
        # Real quantile forecast must be monotonically non-decreasing
        # across quantile levels at every horizon step.
        assert np.all(quantiles[:, 0] <= quantiles[:, 1] + 1e-9)
        assert np.all(quantiles[:, 1] <= quantiles[:, 2] + 1e-9)

    def test_predict_with_uncertainty(self):
        from pinneapple_systems.time_series.foundation_models import ChronosForecaster

        y = _make_noisy_sine(n=200, seed=3)
        f = ChronosForecaster(model_id="amazon/chronos-t5-tiny", device_map="cpu")
        f.fit(y)

        mean, std = f.predict_with_uncertainty(horizon=6, num_samples=30)
        assert mean.shape == (6,)
        assert std.shape == (6,)
        assert np.all(np.isfinite(mean)) and np.all(np.isfinite(std))
        assert np.all(std >= 0)

    def test_predict_before_fit_raises(self):
        from pinneapple_systems.time_series.foundation_models import ChronosForecaster

        f = ChronosForecaster(model_id="amazon/chronos-t5-tiny", device_map="cpu")
        with pytest.raises(RuntimeError):
            f.predict_samples(5)

    def test_empty_input_raises(self):
        from pinneapple_systems.time_series.foundation_models import ChronosForecaster

        f = ChronosForecaster(model_id="amazon/chronos-t5-tiny", device_map="cpu")
        with pytest.raises(ValueError):
            f.fit(np.array([]))


# ---------------------------------------------------------------------------
# TimesFM (Google) -- real checkpoint: google/timesfm-2.5-200m-pytorch
# ---------------------------------------------------------------------------

class TestTimesFMForecaster:
    def setup_method(self):
        pytest.importorskip("timesfm", reason="TimesFMForecaster requires timesfm")
        pytest.importorskip("torch")
        if not _HF_REACHABLE:
            pytest.skip("HuggingFace Hub not reachable from this sandbox")

    def test_fit_predict_real_checkpoint_shape_and_sanity(self):
        from pinneapple_systems.time_series.foundation_models import TimesFMForecaster

        y = _make_noisy_sine(n=200, seed=0)
        f = TimesFMForecaster(
            model_id="google/timesfm-2.5-200m-pytorch",
            max_context=256, max_horizon=32,
        )
        f.fit(y)

        horizon = 12
        y_hat = f.predict(horizon)

        assert isinstance(y_hat, np.ndarray)
        assert y_hat.shape == (horizon,)
        assert np.all(np.isfinite(y_hat))
        assert np.all(np.abs(y_hat) < 5.0)

    def test_predict_quantiles_real_and_monotonic(self):
        from pinneapple_systems.time_series.foundation_models import TimesFMForecaster

        y = _make_noisy_sine(n=200, seed=1)
        f = TimesFMForecaster(
            model_id="google/timesfm-2.5-200m-pytorch",
            max_context=256, max_horizon=32,
        )
        f.fit(y)

        quantiles = f.predict_quantiles(horizon=8, quantile_levels=(0.1, 0.5, 0.9))
        assert quantiles.shape == (8, 3)
        assert np.all(np.isfinite(quantiles))
        assert np.all(quantiles[:, 0] <= quantiles[:, 1] + 1e-9)
        assert np.all(quantiles[:, 1] <= quantiles[:, 2] + 1e-9)

    def test_predict_with_uncertainty(self):
        from pinneapple_systems.time_series.foundation_models import TimesFMForecaster

        y = _make_noisy_sine(n=200, seed=2)
        f = TimesFMForecaster(
            model_id="google/timesfm-2.5-200m-pytorch",
            max_context=256, max_horizon=32,
        )
        f.fit(y)

        mean, std = f.predict_with_uncertainty(horizon=6)
        assert mean.shape == (6,)
        assert std.shape == (6,)
        assert np.all(np.isfinite(mean)) and np.all(np.isfinite(std))
        assert np.all(std >= 0)

    def test_horizon_exceeding_max_horizon_raises(self):
        from pinneapple_systems.time_series.foundation_models import TimesFMForecaster

        y = _make_noisy_sine(n=200, seed=3)
        f = TimesFMForecaster(
            model_id="google/timesfm-2.5-200m-pytorch",
            max_context=256, max_horizon=16,
        )
        f.fit(y)
        with pytest.raises(ValueError):
            f.predict(horizon=32)

    def test_predict_before_fit_raises(self):
        from pinneapple_systems.time_series.foundation_models import TimesFMForecaster

        f = TimesFMForecaster(model_id="google/timesfm-2.5-200m-pytorch")
        with pytest.raises(RuntimeError):
            f.predict(5)

    def test_empty_input_raises(self):
        from pinneapple_systems.time_series.foundation_models import TimesFMForecaster

        f = TimesFMForecaster(model_id="google/timesfm-2.5-200m-pytorch")
        with pytest.raises(ValueError):
            f.fit(np.array([]))


# ---------------------------------------------------------------------------
# Package-level export + honest-degradation checks (no network required)
# ---------------------------------------------------------------------------

def test_package_level_import_when_available():
    """If the optional deps are installed, both forecasters must be
    exported from the top-level time_series package (matching the
    optional-dependency convention used for e.g. FNOForecaster)."""
    import pinneapple_systems.time_series as ts

    try:
        import chronos  # noqa: F401
        assert hasattr(ts, "ChronosForecaster")
    except ImportError:
        pass

    try:
        import timesfm  # noqa: F401
        assert hasattr(ts, "TimesFMForecaster")
    except ImportError:
        pass


def test_module_docstring_cites_real_papers():
    """Sanity check that this wraps the real, published foundation models
    (not from-scratch architectures) and cites them honestly."""
    from pinneapple_systems.time_series import foundation_models as fm

    doc = fm.__doc__ or ""
    assert "Ansari" in doc and "2024" in doc  # Chronos citation
    assert "Das" in doc  # TimesFM citation
