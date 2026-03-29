"""Command-line interface for the pinneaple inference server.

Entry point: ``pinneaple-serve``

Usage
-----
    pinneaple-serve --model path/to/checkpoint.pt \\
                    --field-names u v p \\
                    --coord-names x y \\
                    --port 8000 \\
                    --device cpu
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("pinneaple_serve")

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

_BANNER = r"""
  ____  _                             _
 |  _ \(_)_ __  _ __   ___  __ _ _ __ | | ___
 | |_) | | '_ \| '_ \ / _ \/ _` | '_ \| |/ _ \
 |  __/| | | | | | | |  __/ (_| | |_) | |  __/
 |_|   |_|_| |_|_| |_|\___|\__,_| .__/|_|\___|
                                 |_|
   ┌─ serve ──────────────────────────────────────┐
   │  Physics-Informed Model Inference Server      │
   └───────────────────────────────────────────────┘
"""


def _print_banner(config: "ServerConfig") -> None:  # type: ignore[name-defined]
    print(_BANNER)
    print(f"  Model   : {config.model_name}")
    print(f"  Device  : {config.device}")
    print(f"  Host    : {config.host}")
    print(f"  Port    : {config.port}")
    print()
    print(f"  Endpoints:")
    base = f"http://{config.host if config.host != '0.0.0.0' else 'localhost'}:{config.port}"
    endpoints = [
        ("GET ", "/health",        "liveness probe"),
        ("GET ", "/info",          "model info"),
        ("POST", "/predict",       "single-request inference"),
        ("POST", "/predict_batch", "batched inference"),
        ("GET ", "/history",       "recent prediction log"),
        ("POST", "/reset_history", "clear history"),
        ("GET ", "/docs",          "OpenAPI / Swagger UI"),
    ]
    if config.enable_digital_twin:
        endpoints.insert(5, ("POST", "/update", "sensor ingestion (digital twin)"))
    for method, path, desc in endpoints:
        print(f"    {method}  {base}{path:<22}  {desc}")
    print()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pinneaple-serve",
        description="Serve a trained pinneaple model over REST/HTTP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument(
        "--model",
        required=True,
        metavar="PATH",
        help="Path to the saved model or checkpoint file (.pt / .pth / .ckpt).",
    )

    # Field / coord names (optional but strongly recommended)
    p.add_argument(
        "--field-names",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Output field names (e.g. --field-names u v p). "
             "Defaults to field_0, field_1, … if omitted.",
    )
    p.add_argument(
        "--coord-names",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Input coordinate names (e.g. --coord-names x y t). "
             "Defaults to coord_0, coord_1, … if omitted.",
    )

    # Server settings
    p.add_argument("--host", default="0.0.0.0", help="Interface to bind.")
    p.add_argument("--port", type=int, default=8000, help="TCP port.")
    p.add_argument(
        "--device",
        default="cpu",
        help="Torch device (e.g. cpu, cuda, cuda:0).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        dest="batch_size",
        help="Maximum points per forward pass.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn worker processes (>1 requires a module path, not a file path).",
    )
    p.add_argument(
        "--model-name",
        default=None,
        dest="model_name",
        metavar="NAME",
        help="Human-readable model name shown in /health. Defaults to the checkpoint filename.",
    )
    p.add_argument(
        "--use-amp",
        action="store_true",
        dest="use_amp",
        help="Enable automatic mixed precision (CUDA only).",
    )
    p.add_argument(
        "--enable-digital-twin",
        action="store_true",
        dest="enable_digital_twin",
        help="Expose the /update sensor ingestion endpoint.",
    )
    p.add_argument(
        "--no-log-requests",
        action="store_false",
        dest="log_requests",
        help="Disable per-request access logging.",
    )
    p.add_argument(
        "--max-input-size",
        type=int,
        default=100_000,
        dest="max_input_size",
        help="Hard cap on coordinate points per request.",
    )
    p.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        dest="log_level",
        help="Root log level.",
    )
    return p


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Entry point called by the ``pinneaple-serve`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Configure root logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Validate checkpoint path early
    ckpt_path = Path(args.model)
    if not ckpt_path.exists():
        parser.error(f"Checkpoint not found: {ckpt_path}")

    # Resolve model name
    model_name = args.model_name or ckpt_path.stem

    # Build config
    try:
        from .config import ServerConfig
    except ImportError:
        from pinneaple_serve.config import ServerConfig  # type: ignore[no-redef]

    config = ServerConfig(
        host=args.host,
        port=args.port,
        device=args.device,
        batch_size=args.batch_size,
        use_amp=args.use_amp,
        enable_digital_twin=args.enable_digital_twin,
        model_name=model_name,
        log_requests=args.log_requests,
        workers=args.workers,
        max_input_size=args.max_input_size,
    )

    # Field / coord names — introspect from checkpoint if available
    field_names: list[str] = args.field_names or []
    coord_names: list[str] = args.coord_names or []

    # Load server
    try:
        from .server import ModelServer
    except ImportError:
        from pinneaple_serve.server import ModelServer  # type: ignore[no-redef]

    logger.info("Loading model from %s …", ckpt_path)
    try:
        server = ModelServer.load_from_checkpoint(
            ckpt_path,
            field_names=field_names or ["output"],
            coord_names=coord_names or ["x"],
            config=config,
        )
    except Exception as exc:
        logger.error("Failed to load checkpoint: %s", exc)
        sys.exit(1)

    # If field/coord names were not specified, try to introspect them from
    # the loaded model (pinneaple models often carry metadata attributes)
    if not args.field_names:
        model = server._model
        for attr in ("field_names", "output_names", "fields"):
            if hasattr(model, attr):
                names = getattr(model, attr)
                if isinstance(names, (list, tuple)) and names:
                    server._field_names = list(names)
                    logger.info("Introspected field_names from model: %s", server._field_names)
                    break
    if not args.coord_names:
        model = server._model
        for attr in ("coord_names", "input_names", "coords"):
            if hasattr(model, attr):
                names = getattr(model, attr)
                if isinstance(names, (list, tuple)) and names:
                    server._coord_names = list(names)
                    logger.info("Introspected coord_names from model: %s", server._coord_names)
                    break

    _print_banner(config)
    logger.info("Starting uvicorn…")

    try:
        server.serve(block=True)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
