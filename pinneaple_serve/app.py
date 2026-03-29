"""FastAPI application factory for the pinneaple inference server."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np

from .config import ServerConfig

logger = logging.getLogger("pinneaple_serve")

# ---------------------------------------------------------------------------
# Lazy imports — module must survive without FastAPI installed
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn

    _TORCH_OK = True
except ModuleNotFoundError:  # pragma: no cover
    _TORCH_OK = False

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, field_validator, model_validator

    _FASTAPI_OK = True
except ModuleNotFoundError:
    _FASTAPI_OK = False

    # Stubs so the rest of the module can be parsed without FastAPI
    class BaseModel:  # type: ignore[no-redef]
        pass

    def field_validator(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def _dec(fn: Any) -> Any:
            return fn
        return _dec

    def model_validator(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def _dec(fn: Any) -> Any:
            return fn
        return _dec


def _require_fastapi() -> None:
    if not _FASTAPI_OK:
        raise ImportError(
            "pinneaple_serve requires FastAPI. "
            "Install it with:  pip install 'pinneaple[serve]'  or  pip install fastapi uvicorn"
        )


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Single predict request: a dict mapping coord name → list of floats."""

    coords: Dict[str, List[float]]

    @field_validator("coords")
    @classmethod
    def coords_non_empty(cls, v: Dict[str, List[float]]) -> Dict[str, List[float]]:
        if not v:
            raise ValueError("'coords' must not be empty.")
        lengths = {k: len(lst) for k, lst in v.items()}
        unique = set(lengths.values())
        if len(unique) != 1:
            raise ValueError(
                f"All coordinate arrays must have the same length. Got: {lengths}"
            )
        return v


class PredictResponse(BaseModel):
    fields: Dict[str, List[float]]
    n_points: int
    elapsed_ms: float


class BatchPredictRequest(BaseModel):
    """List of individual coordinate dicts (one per sub-request)."""

    requests: List[Dict[str, List[float]]]

    @model_validator(mode="after")
    def requests_non_empty(self) -> "BatchPredictRequest":
        if not self.requests:
            raise ValueError("'requests' list must not be empty.")
        return self


class BatchPredictResponse(BaseModel):
    results: List[Dict[str, List[float]]]
    total_points: int
    elapsed_ms: float


class SensorUpdateRequest(BaseModel):
    """Digital-twin sensor update."""

    sensor_id: str
    field: str
    value: float
    coords: Dict[str, float]
    timestamp: Optional[float] = None


class SensorUpdateResponse(BaseModel):
    accepted: bool
    sensor_id: str
    timestamp: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_tensor(model_output: Any) -> "torch.Tensor":
    """Pull a plain Tensor out of various model output containers."""
    if _TORCH_OK and isinstance(model_output, torch.Tensor):
        return model_output
    for attr in ("y", "pred", "out", "logits"):
        if hasattr(model_output, attr):
            val = getattr(model_output, attr)
            if _TORCH_OK and isinstance(val, torch.Tensor):
                return val
    raise TypeError(
        f"Cannot extract a Tensor from model output of type {type(model_output).__name__}. "
        "Ensure the model returns a Tensor or a namedtuple/dataclass with a 'y' attribute."
    )


def _coords_to_numpy(coords: Dict[str, List[float]]) -> np.ndarray:
    """Stack coord lists into an (N, D) float32 array, columns ordered by dict insertion."""
    arrays = [np.asarray(v, dtype=np.float32) for v in coords.values()]
    return np.stack(arrays, axis=1)  # (N, D)


def _batched_forward(
    model: Any,
    x: np.ndarray,
    *,
    device: "torch.device",
    batch_size: int,
    use_amp: bool,
) -> np.ndarray:
    """OOM-safe batched inference returning (N, F) float32 numpy array."""
    results: list[np.ndarray] = []
    n = x.shape[0]
    for start in range(0, n, batch_size):
        chunk = x[start : start + batch_size]
        x_t = torch.from_numpy(chunk).to(device)
        with torch.no_grad():
            if use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda"):
                    out = model(x_t)
            else:
                out = model(x_t)
        t = _extract_tensor(out).detach().cpu()
        results.append(t.numpy())
    return np.concatenate(results, axis=0)  # (N, F)


def _output_to_field_dict(
    y: np.ndarray,
    field_names: List[str],
) -> Dict[str, List[float]]:
    """Convert (N, F) array → {field_name: [float, …]}."""
    out: Dict[str, List[float]] = {}
    n_fields = y.shape[1] if y.ndim > 1 else 1
    for i, name in enumerate(field_names):
        if i >= n_fields:
            break
        col = y[:, i] if y.ndim > 1 else y
        out[name] = col.tolist()
    return out


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    model: Any,
    field_names: List[str],
    coord_names: List[str],
    config: ServerConfig,
) -> "FastAPI":
    """Build and return the FastAPI application.

    Parameters
    ----------
    model :
        A trained ``torch.nn.Module`` (or any callable with a compatible
        forward signature).
    field_names :
        Names of the output fields produced by the model
        (e.g. ``["u", "v", "p"]``).
    coord_names :
        Names of the input coordinate dimensions
        (e.g. ``["x", "y", "t"]``).
    config :
        :class:`ServerConfig` controlling host, port, batch size, etc.

    Returns
    -------
    FastAPI
        A fully configured application ready to be run with uvicorn.
    """
    _require_fastapi()

    # Move model to target device once at startup
    device = torch.device(config.device)
    if _TORCH_OK:
        model = model.to(device)
        model.eval()

    # Rolling prediction history
    history: deque[Dict[str, Any]] = deque(maxlen=config.history_max_len)

    # Sensor state for digital twin
    sensor_state: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Count parameters
    # ------------------------------------------------------------------ #
    def _param_count() -> int:
        try:
            return sum(p.numel() for p in model.parameters())
        except Exception:
            return -1

    # ------------------------------------------------------------------ #
    # Lifespan (startup / shutdown logging)
    # ------------------------------------------------------------------ #
    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[valid-type]
        logger.info(
            "pinneaple_serve starting — model=%s device=%s fields=%s coords=%s",
            config.model_name,
            config.device,
            field_names,
            coord_names,
        )
        yield
        logger.info("pinneaple_serve shutting down.")

    app = FastAPI(
        title="pinneaple model server",
        description=(
            "REST API inference server for trained pinneaple physics-informed models. "
            f"Model: {config.model_name}"
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------ #
    # CORS
    # ------------------------------------------------------------------ #
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ #
    # Optional logging middleware
    # ------------------------------------------------------------------ #
    if config.log_requests:
        try:
            from .middleware import RequestLoggingMiddleware
            app.add_middleware(RequestLoggingMiddleware)
        except Exception as exc:
            logger.warning("Could not attach logging middleware: %s", exc)

    # ------------------------------------------------------------------ #
    # Helpers (async wrappers around blocking inference)
    # ------------------------------------------------------------------ #
    async def _run_inference(x: np.ndarray) -> np.ndarray:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: _batched_forward(
                model,
                x,
                device=device,
                batch_size=config.batch_size,
                use_amp=config.use_amp,
            ),
        )

    # ------------------------------------------------------------------ #
    # Endpoints
    # ------------------------------------------------------------------ #

    @app.get("/health", tags=["meta"])
    async def health() -> Dict[str, Any]:
        """Liveness probe — always returns 200 when the server is up."""
        return {
            "status": "ok",
            "model": config.model_name,
            "device": config.device,
            "field_names": field_names,
            "coord_names": coord_names,
        }

    @app.get("/info", tags=["meta"])
    async def info() -> Dict[str, Any]:
        """Detailed model and server information."""
        return {
            "model_name": config.model_name,
            "param_count": _param_count(),
            "field_names": field_names,
            "coord_names": coord_names,
            "server": {
                "host": config.host,
                "port": config.port,
                "device": config.device,
                "batch_size": config.batch_size,
                "use_amp": config.use_amp,
                "max_input_size": config.max_input_size,
                "enable_digital_twin": config.enable_digital_twin,
                "workers": config.workers,
            },
        }

    @app.post("/predict", response_model=PredictResponse, tags=["inference"])
    async def predict(req: PredictRequest) -> PredictResponse:
        """Run model inference on a set of coordinate points.

        The ``coords`` dict maps each coordinate name to a list of floats.
        All lists must have the same length.  The response ``fields`` dict
        maps each field name to a list of predicted values.
        """
        n_pts = len(next(iter(req.coords.values())))
        if n_pts > config.max_input_size:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Request contains {n_pts} points which exceeds "
                    f"max_input_size={config.max_input_size}."
                ),
            )

        x = _coords_to_numpy(req.coords)
        t0 = time.perf_counter()
        try:
            y = await _run_inference(x)
        except Exception as exc:
            logger.exception("Model inference failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"Model inference error: {exc}") from exc

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        fields_out = _output_to_field_dict(y, field_names)

        record: Dict[str, Any] = {
            "timestamp": time.time(),
            "n_points": n_pts,
            "elapsed_ms": round(elapsed_ms, 2),
            "coords_keys": list(req.coords.keys()),
            "fields_keys": list(fields_out.keys()),
        }
        history.append(record)

        return PredictResponse(fields=fields_out, n_points=n_pts, elapsed_ms=round(elapsed_ms, 2))

    @app.post("/predict_batch", response_model=BatchPredictResponse, tags=["inference"])
    async def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
        """Batched inference over multiple independent coordinate sets.

        Accepts a list of coordinate dicts (same schema as ``/predict``).
        All sub-requests are concatenated into a single forward pass
        (chunked by ``batch_size``) then split back into individual results.
        """
        # Validate sizes and build index map
        offsets: list[int] = [0]
        for sub in req.requests:
            lengths = {k: len(v) for k, v in sub.items()}
            unique = set(lengths.values())
            if len(unique) != 1:
                raise HTTPException(
                    status_code=422,
                    detail=f"All coordinate arrays must have the same length. Got: {lengths}",
                )
            n = next(iter(unique))
            offsets.append(offsets[-1] + n)

        total = offsets[-1]
        if total > config.max_input_size:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Batch total of {total} points exceeds "
                    f"max_input_size={config.max_input_size}."
                ),
            )

        # Concatenate all sub-requests
        chunks = [_coords_to_numpy(sub) for sub in req.requests]
        x_all = np.concatenate(chunks, axis=0)

        t0 = time.perf_counter()
        try:
            y_all = await _run_inference(x_all)
        except Exception as exc:
            logger.exception("Batch inference failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"Model inference error: {exc}") from exc

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Split back
        results: list[Dict[str, List[float]]] = []
        for i, sub in enumerate(req.requests):
            y_sub = y_all[offsets[i] : offsets[i + 1]]
            results.append(_output_to_field_dict(y_sub, field_names))

        history.append(
            {
                "timestamp": time.time(),
                "n_points": total,
                "n_sub_requests": len(req.requests),
                "elapsed_ms": round(elapsed_ms, 2),
                "endpoint": "predict_batch",
            }
        )

        return BatchPredictResponse(
            results=results,
            total_points=total,
            elapsed_ms=round(elapsed_ms, 2),
        )

    @app.post("/update", response_model=SensorUpdateResponse, tags=["digital_twin"])
    async def update(req: SensorUpdateRequest) -> SensorUpdateResponse:
        """Ingest a sensor reading for digital-twin state fusion.

        Only available when ``ServerConfig.enable_digital_twin=True``.
        """
        if not config.enable_digital_twin:
            raise HTTPException(
                status_code=403,
                detail="Digital twin endpoints are disabled. Set enable_digital_twin=True in ServerConfig.",
            )
        ts = req.timestamp or time.time()
        sensor_state[req.sensor_id] = {
            "field": req.field,
            "value": req.value,
            "coords": req.coords,
            "timestamp": ts,
        }
        logger.info(
            "Sensor update: id=%s field=%s value=%s coords=%s",
            req.sensor_id,
            req.field,
            req.value,
            req.coords,
        )
        return SensorUpdateResponse(accepted=True, sensor_id=req.sensor_id, timestamp=ts)

    @app.get("/history", tags=["meta"])
    async def get_history(last: int = 50) -> Dict[str, Any]:
        """Return the last *N* prediction records (default 50)."""
        last = min(last, len(history))
        records = list(history)[-last:]
        return {"count": len(records), "history": records}

    @app.post("/reset_history", tags=["meta"])
    async def reset_history() -> Dict[str, str]:
        """Clear the in-memory prediction history."""
        history.clear()
        sensor_state.clear()
        return {"status": "cleared"}

    return app
