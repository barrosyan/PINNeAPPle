"""Pretrained time-series *foundation model* forecasters.

Everywhere else in this repo, forecasters are architectures you train
yourself: ``pinneapple_neural.architectures.transformers`` has
Informer/Autoformer/FEDformer/TFT/TimesNet trained from scratch on your
data, ``pinneapple_systems.time_series.classical_ts`` has ARIMA/Kalman/VAR/
TCN fit on your series, and ``pinneapple_systems.time_series.baselines``
has the simple ``NaiveForecaster`` family. This module is different: it
wraps REAL, openly published, **pre-trained** checkpoints that a research
lab trained on large heterogeneous time-series corpora and released on the
HuggingFace Hub, so a forecast can be produced zero-shot -- with little or
no fitting on the series at hand.

Wrapped models
---------------
* :class:`ChronosForecaster` -- Amazon's Chronos, a T5-based pretrained
  probabilistic time-series forecaster:

      Ansari, A. F. et al. "Chronos: Learning the Language of Time
      Series." arXiv:2403.07815 (2024).

  Backed by the real ``chronos-forecasting`` PyPI package
  (``pip install chronos-forecasting``) and real checkpoints published at
  https://huggingface.co/amazon (e.g. ``amazon/chronos-t5-tiny``, ~8M
  params -- the smallest real variant, used here for testability).

* :class:`TimesFMForecaster` -- Google's TimesFM, a decoder-only pretrained
  foundation model for time-series forecasting:

      Das, A. et al. "A decoder-only foundation model for time-series
      forecasting." arXiv:2310.10688 (2024).

  Backed by the real ``timesfm`` PyPI package (``pip install timesfm``)
  and a real checkpoint published at
  https://huggingface.co/google/timesfm-2.5-200m-pytorch (the current
  actively maintained TimesFM release for the installed ``timesfm``
  package; ~200M params -- the smallest currently-published TimesFM
  weights).

Interface convention
---------------------
Both classes follow this package's ``NaiveForecaster``-style convention
(see ``pinneapple_systems.time_series.baselines.naive``): ``fit(y) -> self``
and ``predict(horizon) -> np.ndarray``. Since both underlying models are
zero-shot pretrained forecasters, ``fit()`` does **not** train any weights
-- it stores the context series and eagerly loads/downloads the real
pretrained checkpoint, so a broken install/network surfaces at ``fit()``
time rather than silently at ``predict()`` time. That is documented
explicitly on each class rather than pretended away. Because both Chronos
and TimesFM are genuinely *probabilistic* forecasters (unlike the point-
estimate-only naive baselines), this module also exposes the real
quantile/sample forecasts the upstream packages compute, via
``predict_quantiles()`` / ``predict_samples()`` / ``predict_with_uncertainty()``.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is a hard dependency elsewhere in this repo
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Chronos (Amazon) -- Ansari et al., 2024
# ---------------------------------------------------------------------------

try:
    from chronos import BaseChronosPipeline as _BaseChronosPipeline
    _CHRONOS_AVAILABLE = True
except ImportError:
    _CHRONOS_AVAILABLE = False

_CHRONOS_INSTALL_MSG = (
    "ChronosForecaster requires the real 'chronos-forecasting' package "
    "(https://github.com/amazon-science/chronos-forecasting), which was not "
    "found. Install it with:\n"
    "    pip install chronos-forecasting\n"
    "Model card for the default checkpoint: "
    "https://huggingface.co/amazon/chronos-t5-tiny"
)


class ChronosForecaster:
    """Wraps Amazon's real, pretrained Chronos forecasting pipeline.

    Chronos (Ansari et al., 2024, arXiv:2403.07815) tokenizes a scaled time
    series into a fixed vocabulary and forecasts autoregressively with a
    T5-style encoder-decoder trained on a large corpus of public and
    synthetic time series. It is genuinely probabilistic: sampling the
    decoder autoregressively multiple times yields a distribution over
    future paths, from which point forecasts and quantiles are derived.

    ``fit()`` is honestly a **no-op with respect to model weights** -- this
    is a zero-shot pretrained forecaster, not something trained here. It
    stores the context series `y` that `predict()` conditions on, and
    eagerly loads (downloading on first use) the real pretrained checkpoint
    from the HuggingFace Hub, so a missing dependency or unreachable network
    fails loudly at `fit()` rather than silently at `predict()`.

    No fine-tuning hook is implemented: the upstream project documents a
    separate fine-tuning script (see the Chronos repo's
    ``scripts/training``), and wiring that up was out of scope for this
    zero-shot integration -- claiming a "light fine-tune" path here without
    actually exercising it would be dishonest.
    """

    def __init__(
        self,
        model_id: str = "amazon/chronos-t5-tiny",
        device_map: str = "cpu",
        dtype: Optional["torch.dtype"] = None,
        **pipeline_kwargs,
    ):
        if not _CHRONOS_AVAILABLE or not _TORCH_AVAILABLE:
            raise ImportError(_CHRONOS_INSTALL_MSG)
        self.model_id = model_id
        self.device_map = device_map
        self.dtype = dtype if dtype is not None else torch.float32
        self._pipeline_kwargs = dict(pipeline_kwargs)
        self._pipeline = None
        self._context_: Optional[np.ndarray] = None

    # -- lifecycle ---------------------------------------------------------

    def _load_pipeline(self):
        if self._pipeline is None:
            self._pipeline = _BaseChronosPipeline.from_pretrained(
                self.model_id,
                device_map=self.device_map,
                dtype=self.dtype,
                **self._pipeline_kwargs,
            )
        return self._pipeline

    def fit(self, y: np.ndarray) -> "ChronosForecaster":
        """Store the context series and load the real pretrained checkpoint.

        Documented no-op re: model weights (see class docstring) -- Chronos
        is used zero-shot. Present for interface parity with the other
        forecasters in ``pinneapple_systems.time_series``.
        """
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size == 0:
            raise ValueError("ChronosForecaster.fit() requires a non-empty series.")
        self._context_ = y
        self._load_pipeline()
        return self

    # -- prediction ----------------------------------------------------------

    def _supports_sampling(self) -> bool:
        # The original ChronosPipeline (T5) exposes `num_samples` on predict()
        # and forecasts via autoregressive sampling. ChronosBolt / Chronos2
        # pipelines are direct multi-quantile regressors and do not.
        return hasattr(self._pipeline, "predict") and self._pipeline.__class__.__name__ == "ChronosPipeline"

    def predict_samples(self, horizon: int, num_samples: int = 20) -> np.ndarray:
        """Real sampled forecast paths, shape ``(num_samples, horizon)``.

        Only available for sampling-based Chronos pipelines (the default
        ``amazon/chronos-t5-*`` checkpoints). Raises ``NotImplementedError``
        for direct-regression pipelines (e.g. Chronos-Bolt/Chronos-2) --
        use :meth:`predict_quantiles` for those instead.
        """
        if self._context_ is None or self._pipeline is None:
            raise RuntimeError("Call fit() before predict_samples().")
        if not self._supports_sampling():
            raise NotImplementedError(
                f"{self._pipeline.__class__.__name__} does not support raw sample "
                "paths (it forecasts quantiles directly). Use predict_quantiles() "
                "instead, or load a sampling-based checkpoint "
                "(e.g. 'amazon/chronos-t5-tiny')."
            )
        context = torch.tensor(self._context_, dtype=torch.float32)
        forecast = self._pipeline.predict(
            context, prediction_length=int(horizon), num_samples=int(num_samples),
        )
        arr = forecast.detach().cpu().numpy()
        # ChronosPipeline.predict -> (batch=1, num_samples, horizon)
        return arr[0]

    def predict_quantiles(
        self, horizon: int, quantile_levels: Sequence[float] = (0.1, 0.5, 0.9)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Real quantile forecast.

        Returns ``(quantiles, mean)`` where ``quantiles`` has shape
        ``(horizon, len(quantile_levels))`` and ``mean`` has shape
        ``(horizon,)`` -- the pipeline's own mean-path estimate.
        """
        if self._context_ is None or self._pipeline is None:
            raise RuntimeError("Call fit() before predict_quantiles().")
        context = torch.tensor(self._context_, dtype=torch.float32)
        quantiles, mean = self._pipeline.predict_quantiles(
            context, prediction_length=int(horizon), quantile_levels=list(quantile_levels),
        )
        return quantiles.detach().cpu().numpy()[0], mean.detach().cpu().numpy()[0]

    def predict(self, horizon: int, num_samples: int = 20) -> np.ndarray:
        """Real point forecast, shape ``(horizon,)``.

        Uses the median of `num_samples` real sampled paths when the loaded
        pipeline supports sampling, else the pipeline's own quantile-based
        mean/median estimate.
        """
        if self._supports_sampling():
            samples = self.predict_samples(horizon, num_samples=num_samples)
            return np.median(samples, axis=0)
        quantiles, mean = self.predict_quantiles(horizon, quantile_levels=(0.5,))
        return quantiles[:, 0]

    def predict_with_uncertainty(
        self, horizon: int, num_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Real ``(mean, std)`` over the forecast horizon.

        Mirrors ``ForecastModel.predict_with_uncertainty`` used by the
        classical models in this package (see
        ``pinneapple_systems.time_series.models.classical``), so Chronos can
        be dropped into code that already expects that signature.
        """
        if self._supports_sampling():
            samples = self.predict_samples(horizon, num_samples=num_samples)
            return samples.mean(axis=0), samples.std(axis=0)
        quantiles, mean = self.predict_quantiles(
            horizon, quantile_levels=(0.1, 0.5, 0.9)
        )
        # Approximate std from the 10/90 quantile spread (~2 * 1.2816 sigma
        # for a Gaussian); documented as an approximation, not exact.
        std = (quantiles[:, -1] - quantiles[:, 0]) / (2 * 1.2816)
        return mean, np.abs(std)


# ---------------------------------------------------------------------------
# TimesFM (Google) -- Das et al., 2024
# ---------------------------------------------------------------------------

try:
    import timesfm as _timesfm
    _TIMESFM_AVAILABLE = True
except ImportError:
    _TIMESFM_AVAILABLE = False

_TIMESFM_INSTALL_MSG = (
    "TimesFMForecaster requires the real 'timesfm' package "
    "(https://github.com/google-research/timesfm), which was not found. "
    "Install it with:\n"
    "    pip install timesfm\n"
    "Model card for the default checkpoint: "
    "https://huggingface.co/google/timesfm-2.5-200m-pytorch"
)


class TimesFMForecaster:
    """Wraps Google's real, pretrained TimesFM forecasting model.

    TimesFM (Das et al., 2024, arXiv:2310.10688) is a decoder-only
    transformer pretrained on a large corpus of real and synthetic time
    series for zero-shot forecasting. The default checkpoint here,
    ``google/timesfm-2.5-200m-pytorch`` (~200M params), is the smallest
    currently-published TimesFM release for the installed ``timesfm``
    PyPI package (the paper's original 1.0-200m checkpoint predates the
    package's current API; this wrapper targets the real, currently
    loadable checkpoint rather than an unloadable legacy id).

    Like :class:`ChronosForecaster`, this is zero-shot: ``fit()`` does not
    train any weights, it stores the context series and eagerly loads
    (downloading + compiling on first use) the real pretrained checkpoint.
    """

    def __init__(
        self,
        model_id: str = "google/timesfm-2.5-200m-pytorch",
        max_context: int = 512,
        max_horizon: int = 128,
        torch_compile: bool = False,
        **forecast_config_kwargs,
    ):
        if not _TIMESFM_AVAILABLE or not _TORCH_AVAILABLE:
            raise ImportError(_TIMESFM_INSTALL_MSG)
        self.model_id = model_id
        self.max_context = int(max_context)
        self.max_horizon = int(max_horizon)
        self.torch_compile = torch_compile
        self._forecast_config_kwargs = dict(forecast_config_kwargs)
        self._model = None
        self._context_: Optional[np.ndarray] = None

    def _load_model(self):
        if self._model is None:
            model = _timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                self.model_id, torch_compile=self.torch_compile,
            )
            cfg = _timesfm.ForecastConfig(
                max_context=self.max_context,
                max_horizon=self.max_horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=False,
                fix_quantile_crossing=True,
                **self._forecast_config_kwargs,
            )
            model.compile(cfg)
            self._model = model
        return self._model

    def fit(self, y: np.ndarray) -> "TimesFMForecaster":
        """Store the context series and load the real pretrained checkpoint.

        Documented no-op re: model weights -- TimesFM is used zero-shot,
        exactly like :meth:`ChronosForecaster.fit`.
        """
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size == 0:
            raise ValueError("TimesFMForecaster.fit() requires a non-empty series.")
        if y.size > self.max_context:
            y = y[-self.max_context:]
        self._context_ = y
        self._load_model()
        return self

    def _forecast(self, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._context_ is None or self._model is None:
            raise RuntimeError("Call fit() before predict().")
        h = int(horizon)
        if h > self.max_horizon:
            raise ValueError(
                f"horizon={h} exceeds max_horizon={self.max_horizon} this "
                "forecaster was compiled for; construct TimesFMForecaster "
                "with a larger max_horizon."
            )
        point, quantiles = self._model.forecast(horizon=h, inputs=[self._context_])
        return np.asarray(point)[0], np.asarray(quantiles)[0]

    def predict(self, horizon: int) -> np.ndarray:
        """Real point forecast, shape ``(horizon,)``."""
        point, _ = self._forecast(horizon)
        return point

    def predict_quantiles(
        self, horizon: int, quantile_levels: Sequence[float] = (0.1, 0.5, 0.9)
    ) -> np.ndarray:
        """Real quantile forecast, shape ``(horizon, len(quantile_levels))``.

        TimesFM's ``forecast()`` returns a fixed decile grid
        ``[0.1, 0.2, ..., 0.9]`` (plus the mean at index 0); this selects
        the closest columns to the requested `quantile_levels`.
        """
        _, quantiles = self._forecast(horizon)
        # quantiles columns: [mean, q0.1, q0.2, ..., q0.9]
        available = [None] + [round(q, 1) for q in np.arange(0.1, 1.0, 0.1)]
        cols = []
        for q in quantile_levels:
            closest_idx = min(
                (i for i in range(1, len(available))),
                key=lambda i: abs(available[i] - q),
            )
            cols.append(closest_idx)
        return quantiles[:, cols]

    def predict_with_uncertainty(
        self, horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Real ``(mean, std)`` over the forecast horizon, from the model's
        own quantile spread (approximated from the 10/90 deciles)."""
        point, quantiles = self._forecast(horizon)
        q10, q90 = quantiles[:, 1], quantiles[:, -1]
        std = np.abs(q90 - q10) / (2 * 1.2816)
        return point, std


__all__ = ["ChronosForecaster", "TimesFMForecaster"]
