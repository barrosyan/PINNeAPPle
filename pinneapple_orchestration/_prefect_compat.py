"""pinneapple_orchestration._prefect_compat — a thin compatibility shim.

Uses real Prefect decorators when the optional ``orchestration`` extra
(``pip install "pinneapple[orchestration]"``, i.e. ``prefect``) is
installed, and falls back to no-op passthrough decorators otherwise, so
every flow/task function in this package stays a plain, directly
callable, directly testable Python function either way. Mirrors this
repo's existing optional-dependency pattern (guarded imports in
``pinneapple_llm/__init__.py``, the ``cad``/``process``/``fenics`` extras
in ``pyproject.toml``): the package is fully usable and testable without
the extra, and gains real orchestration (retries, scheduling, a UI,
deployments) once it is installed.
"""
from __future__ import annotations

try:
    from prefect import flow as flow, task as task  # noqa: F401
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False

    def _passthrough(*d_args, **d_kwargs):
        # Supports both bare `@flow` / `@task` and `@flow(name=...)` /
        # `@task(retries=...)` call shapes, ignoring any Prefect-specific
        # kwargs (there is no real Prefect engine to hand them to).
        if len(d_args) == 1 and callable(d_args[0]) and not d_kwargs:
            return d_args[0]

        def decorator(fn):
            return fn

        return decorator

    flow = _passthrough
    task = _passthrough
