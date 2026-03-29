from __future__ import annotations
"""pinneaple_serve — REST API inference server for pinneaple models.

Exposes any trained pinneaple model over HTTP so downstream systems
(MATLAB, C++, dashboards, etc.) can query predictions without Python.

Endpoints
---------
GET  /health          → status, model name, device, field/coord names
GET  /info            → parameter count, full server config
POST /predict         → {"coords": {"x": [...], "y": [...]}} → {"fields": {...}}
POST /predict_batch   → list of coord dicts → batched predictions
POST /update          → digital twin sensor observation ingestion
GET  /history         → recent prediction log
POST /reset_history   → clear prediction history

Quick start
-----------
>>> from pinneaple_serve import ModelServer
>>> server = ModelServer(model, field_names=["u", "v", "p"], coord_names=["x", "y"])
>>> server.serve()          # blocks; visit http://localhost:8000/docs

Or from the command line::

    pinneaple-serve --model path/to/model.pt --port 8000

Requirements
------------
FastAPI and uvicorn are optional dependencies::

    pip install pinneaple[serve]   # or:  pip install fastapi uvicorn pydantic
"""

try:
    from .server import ModelServer
    from .config import ServerConfig
    from .app import create_app
    _SERVE_AVAILABLE = True
except ImportError:
    _SERVE_AVAILABLE = False

    class _Stub:  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "pinneaple_serve requires FastAPI and uvicorn.\n"
                "Install with:  pip install fastapi uvicorn pydantic"
            )

    ModelServer = _Stub  # type: ignore[misc,assignment]
    ServerConfig = _Stub  # type: ignore[misc,assignment]

    def create_app(*args, **kwargs):
        raise ImportError(
            "pinneaple_serve requires FastAPI and uvicorn.\n"
            "Install with:  pip install fastapi uvicorn pydantic"
        )


def serve(model, field_names, coord_names, *, port: int = 8000, device: str = "cpu", **kwargs):
    """Convenience one-liner to serve a model.

    Parameters
    ----------
    model       : trained nn.Module (PINNBase or any callable)
    field_names : list of output field names, e.g. ["u", "v", "p"]
    coord_names : list of input coord names, e.g. ["x", "y"]
    port        : HTTP port (default 8000)
    device      : inference device (default "cpu")
    """
    if not _SERVE_AVAILABLE:
        raise ImportError(
            "pinneaple_serve requires FastAPI and uvicorn.\n"
            "Install with:  pip install fastapi uvicorn pydantic"
        )
    cfg = ServerConfig(port=port, device=device, **kwargs)
    server = ModelServer(model, field_names=field_names, coord_names=coord_names, config=cfg)
    server.serve()


__all__ = [
    "ModelServer",
    "ServerConfig",
    "create_app",
    "serve",
]
