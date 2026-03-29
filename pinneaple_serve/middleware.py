"""Request/response logging middleware and simple in-process rate limiter."""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Callable

logger = logging.getLogger("pinneaple_serve")

# ---------------------------------------------------------------------------
# Lazy FastAPI import — module must survive without it installed
# ---------------------------------------------------------------------------
try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response, JSONResponse

    _STARLETTE_OK = True
except ModuleNotFoundError:
    _STARLETTE_OK = False


def _require_starlette() -> None:
    if not _STARLETTE_OK:
        raise ImportError(
            "pinneaple_serve requires FastAPI/Starlette. "
            "Install it with:  pip install 'pinneaple[serve]'  or  pip install fastapi uvicorn"
        )


# ---------------------------------------------------------------------------
# Logging middleware
# ---------------------------------------------------------------------------

class RequestLoggingMiddleware:  # type: ignore[misc]
    """ASGI middleware that logs every request with timing information.

    Compatible with Starlette's ``BaseHTTPMiddleware`` interface when
    FastAPI is available; falls back to a no-op descriptor otherwise so
    the module can still be imported.
    """

    def __new__(cls, app: object, **kwargs: object) -> "RequestLoggingMiddleware":  # type: ignore[misc]
        _require_starlette()
        instance = super().__new__(cls)
        return instance  # type: ignore[return-value]

    def __init__(self, app: object) -> None:
        # Starlette BaseHTTPMiddleware sets self.app
        super().__init__(app)  # type: ignore[call-arg]

    async def dispatch(self, request: "Request", call_next: Callable) -> "Response":  # type: ignore[override]
        start = time.perf_counter()
        method = request.method
        path = request.url.path
        client = getattr(request.client, "host", "unknown")

        try:
            response: Response = await call_next(request)
        except Exception as exc:  # pragma: no cover
            logger.error("Unhandled error for %s %s from %s: %s", method, path, client, exc)
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        status = response.status_code
        logger.info(
            "%s %s %s %.1f ms [%s]",
            method,
            path,
            status,
            elapsed_ms,
            client,
        )
        return response


# Patch base class after defining the class, so import-time failures are deferred
if _STARLETTE_OK:
    RequestLoggingMiddleware = type(  # type: ignore[misc]
        "RequestLoggingMiddleware",
        (BaseHTTPMiddleware,),
        dict(RequestLoggingMiddleware.__dict__),
    )


# ---------------------------------------------------------------------------
# Rate-limiting middleware (sliding window, in-process)
# ---------------------------------------------------------------------------

class RateLimitMiddleware:  # type: ignore[misc]
    """Simple per-IP sliding-window rate limiter.

    Parameters
    ----------
    app :
        The ASGI application to wrap.
    max_requests : int
        Maximum number of requests allowed within ``window_seconds``.
    window_seconds : float
        Length of the sliding window in seconds.
    """

    _windows: dict[str, deque[float]] = defaultdict(deque)

    def __new__(cls, app: object, **kwargs: object) -> "RateLimitMiddleware":  # type: ignore[misc]
        _require_starlette()
        instance = super().__new__(cls)
        return instance  # type: ignore[return-value]

    def __init__(self, app: object, max_requests: int = 200, window_seconds: float = 60.0) -> None:
        super().__init__(app)  # type: ignore[call-arg]
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._windows: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: "Request", call_next: Callable) -> "Response":  # type: ignore[override]
        client_ip: str = getattr(request.client, "host", "unknown")
        now = time.monotonic()
        window = self._windows[client_ip]

        # Evict timestamps outside the window
        cutoff = now - self.window_seconds
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= self.max_requests:
            logger.warning("Rate limit exceeded for %s", client_ip)
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"Rate limit exceeded: max {self.max_requests} requests "
                        f"per {self.window_seconds:.0f}s"
                    )
                },
            )

        window.append(now)
        return await call_next(request)


if _STARLETTE_OK:
    RateLimitMiddleware = type(  # type: ignore[misc]
        "RateLimitMiddleware",
        (BaseHTTPMiddleware,),
        dict(RateLimitMiddleware.__dict__),
    )


__all__ = ["RequestLoggingMiddleware", "RateLimitMiddleware"]
