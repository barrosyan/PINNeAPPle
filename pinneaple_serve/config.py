"""ServerConfig dataclass for pinneaple_serve."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ServerConfig:
    """Configuration for the pinneaple model inference server.

    Attributes
    ----------
    host : str
        Interface to bind (default ``"0.0.0.0"`` — all interfaces).
    port : int
        TCP port the server listens on.
    device : str
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``, …).
    batch_size : int
        Maximum points per forward pass on the GPU/CPU.
    use_amp : bool
        Enable automatic mixed precision (requires CUDA).
    cors_origins : list of str
        Allowed CORS origins. Default ``["*"]`` permits all origins.
    max_input_size : int
        Hard cap on the number of coordinate points accepted per request.
    enable_digital_twin : bool
        Expose the ``/update`` sensor-ingestion endpoint.
    model_name : str
        Human-readable model name returned by ``/health`` and ``/info``.
    log_requests : bool
        Attach request/response logging middleware.
    workers : int
        Number of uvicorn worker processes.
    history_max_len : int
        Maximum number of prediction records kept in the rolling history.
    """

    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "cpu"
    batch_size: int = 1024
    use_amp: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_input_size: int = 100_000
    enable_digital_twin: bool = False
    model_name: str = "pinneaple_model"
    log_requests: bool = True
    workers: int = 1
    history_max_len: int = 1000
