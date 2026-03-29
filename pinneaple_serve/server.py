"""ModelServer — high-level wrapper around the FastAPI inference application."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import ServerConfig

logger = logging.getLogger("pinneaple_serve")

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn

    _TORCH_OK = True
except ModuleNotFoundError:  # pragma: no cover
    _TORCH_OK = False


def _require_torch() -> None:
    if not _TORCH_OK:
        raise ImportError("pinneaple_serve requires PyTorch. Install torch first.")


def _require_fastapi() -> None:
    try:
        import fastapi  # noqa: F401
    except ModuleNotFoundError:
        raise ImportError(
            "pinneaple_serve requires FastAPI. "
            "Install it with:  pip install 'pinneaple[serve]'  or  pip install fastapi uvicorn"
        )


def _require_uvicorn() -> None:
    try:
        import uvicorn  # noqa: F401
    except ModuleNotFoundError:
        raise ImportError(
            "pinneaple_serve requires uvicorn. "
            "Install it with:  pip install uvicorn"
        )


# ---------------------------------------------------------------------------
# ModelServer
# ---------------------------------------------------------------------------

class ModelServer:
    """Wraps a trained pinneaple model and exposes it as an HTTP inference server.

    Parameters
    ----------
    model :
        A trained ``torch.nn.Module``.
    field_names :
        Names of the predicted output fields (e.g. ``["u", "v", "p"]``).
    coord_names :
        Names of the input coordinate dimensions (e.g. ``["x", "y", "t"]``).
    config :
        Optional :class:`~pinneaple_serve.ServerConfig`.  A default config is
        used when ``None``.

    Examples
    --------
    >>> server = ModelServer(my_model, field_names=["u"], coord_names=["x", "t"])
    >>> server.serve()  # blocks until Ctrl-C

    Or embed in another ASGI framework:

    >>> app = server.app
    """

    def __init__(
        self,
        model: Any,
        field_names: List[str],
        coord_names: List[str],
        config: Optional[ServerConfig] = None,
    ) -> None:
        _require_torch()
        self._model = model
        self._field_names: List[str] = list(field_names)
        self._coord_names: List[str] = list(coord_names)
        self._config: ServerConfig = config or ServerConfig()
        self._app: Any = None  # lazily built
        self._server_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def load_from_checkpoint(
        cls,
        path: str | Path,
        field_names: List[str],
        coord_names: List[str],
        config: Optional[ServerConfig] = None,
        *,
        map_location: Optional[str] = None,
        model_class: Optional[Any] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "ModelServer":
        """Load a ``ModelServer`` from a saved checkpoint file.

        The checkpoint may be:

        * A plain ``torch.save``'d ``state_dict`` — in this case you must
          supply *model_class* and *model_kwargs* so the architecture can be
          reconstructed.
        * A full saved model (``torch.save(model, path)``) — the class/kwargs
          are not required.
        * A dict with keys ``"model_state_dict"`` and optionally
          ``"model_class"`` / ``"model_kwargs"`` (as produced by
          pinneaple_train checkpointing).

        Parameters
        ----------
        path :
            Path to the checkpoint file (``.pt`` / ``.pth`` / ``.ckpt``).
        field_names :
            Names of the output fields.
        coord_names :
            Names of the input coordinate dimensions.
        config :
            Optional server configuration.
        map_location :
            Passed directly to ``torch.load`` (e.g. ``"cpu"``).
        model_class :
            ``nn.Module`` subclass to instantiate when loading a state dict.
        model_kwargs :
            Keyword arguments forwarded to *model_class*.
        """
        _require_torch()
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ml = map_location or (config.device if config else "cpu")
        raw = torch.load(str(path), map_location=ml, weights_only=False)

        if isinstance(raw, dict) and "model_state_dict" in raw:
            # pinneaple_train-style checkpoint dict
            sd = raw["model_state_dict"]
            mc = model_class or raw.get("model_class")
            mk = model_kwargs or raw.get("model_kwargs", {})
            if mc is None:
                raise ValueError(
                    "Checkpoint contains a state_dict but 'model_class' was not provided "
                    "and is not embedded in the checkpoint. Pass model_class=YourModel."
                )
            model = mc(**mk)
            model.load_state_dict(sd)
        elif isinstance(raw, dict) and not isinstance(raw, nn.Module):
            # Bare state dict — needs model_class
            if model_class is None:
                raise ValueError(
                    "The checkpoint appears to be a raw state_dict. "
                    "Provide model_class= and optionally model_kwargs= to reconstruct the model."
                )
            model = model_class(**(model_kwargs or {}))
            model.load_state_dict(raw)
        else:
            # Full saved model
            model = raw

        logger.info("Loaded model from checkpoint: %s", path)
        return cls(model, field_names, coord_names, config)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def app(self) -> Any:
        """The FastAPI application (lazily constructed on first access)."""
        if self._app is None:
            _require_fastapi()
            from .app import create_app
            self._app = create_app(
                self._model,
                self._field_names,
                self._coord_names,
                self._config,
            )
        return self._app

    @property
    def config(self) -> ServerConfig:
        return self._config

    @property
    def field_names(self) -> List[str]:
        return list(self._field_names)

    @property
    def coord_names(self) -> List[str]:
        return list(self._coord_names)

    # ------------------------------------------------------------------
    # Direct (no-HTTP) inference
    # ------------------------------------------------------------------

    def predict(self, coords: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference directly in Python without going through HTTP.

        Parameters
        ----------
        coords :
            Dict mapping coordinate name → list/array of floats.
            All arrays must have the same length.

        Returns
        -------
        dict
            ``{"fields": {field_name: [float, …], …}, "n_points": int}``
        """
        _require_torch()
        from .app import _coords_to_numpy, _batched_forward, _output_to_field_dict

        # Coerce inputs
        coords_np: Dict[str, list] = {
            k: (v.tolist() if hasattr(v, "tolist") else list(v))
            for k, v in coords.items()
        }

        lengths = {k: len(v) for k, v in coords_np.items()}
        unique = set(lengths.values())
        if len(unique) != 1:
            raise ValueError(
                f"All coordinate arrays must have the same length. Got: {lengths}"
            )

        n_pts = next(iter(unique))
        if n_pts > self._config.max_input_size:
            raise ValueError(
                f"coords contain {n_pts} points which exceeds "
                f"max_input_size={self._config.max_input_size}."
            )

        x = _coords_to_numpy(coords_np)
        device = torch.device(self._config.device)
        model = self._model.to(device)
        model.eval()

        y = _batched_forward(
            model,
            x,
            device=device,
            batch_size=self._config.batch_size,
            use_amp=self._config.use_amp,
        )
        fields = _output_to_field_dict(y, self._field_names)
        return {"fields": fields, "n_points": n_pts}

    # ------------------------------------------------------------------
    # Serve
    # ------------------------------------------------------------------

    def serve(self, block: bool = True) -> None:
        """Start the uvicorn HTTP server.

        Parameters
        ----------
        block :
            If ``True`` (default) the call blocks until the server is stopped
            (e.g. via Ctrl-C).  If ``False`` the server is started in a
            background daemon thread and control is returned immediately.
        """
        _require_uvicorn()
        import uvicorn

        cfg = uvicorn.Config(
            app=self.app,
            host=self._config.host,
            port=self._config.port,
            workers=1,  # multiple workers require passing an import string
            log_level="info",
        )
        server = uvicorn.Server(cfg)

        if block:
            server.run()
        else:
            t = threading.Thread(target=server.run, daemon=True, name="pinneaple-serve")
            t.start()
            self._server_thread = t
            logger.info(
                "Server started in background thread on %s:%s",
                self._config.host,
                self._config.port,
            )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ModelServer(model={type(self._model).__name__!r}, "
            f"fields={self._field_names}, coords={self._coord_names}, "
            f"device={self._config.device!r}, port={self._config.port})"
        )
