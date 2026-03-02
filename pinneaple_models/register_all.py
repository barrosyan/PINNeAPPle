from __future__ import annotations

def register_all() -> None:
    """Register all model families into the global ModelRegistry.

    This function should be safe to call multiple times.
    """
    # NOTE: Keep your existing registrations here (if any). This patch only adds neural operators.
    try:
        # If your project already has other register calls, keep them.
        pass
    except Exception:
        pass

    # Neural Operators family registry
    try:
        from pinneaple_models.neural_operators.registry import register_into_global
        register_into_global()
    except Exception:
        # Do not hard-fail if operators aren't available in minimal envs
        pass
