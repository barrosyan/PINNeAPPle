from __future__ import annotations
"""Register all model families into the global ModelRegistry."""

_REGISTERED = False


def register_all() -> None:
    """Populate ModelRegistry with every family. Idempotent — safe to call multiple times."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    _FAMILIES = [
        ("pinneaple_models.autoencoders.registry",        "register_into_global"),
        ("pinneaple_models.classical_ts.registry",        "register_into_global"),
        ("pinneaple_models.continuous.registry",          "register_into_global"),
        ("pinneaple_models.convolutions.registry",        "register_into_global"),
        ("pinneaple_models.graphnn.registry",             "register_into_global"),
        ("pinneaple_models.neural_operators.registry",    "register_into_global"),
        ("pinneaple_models.physics_aware.registry",       "register_into_global"),
        ("pinneaple_models.pinns.registry",               "register_into_global"),
        ("pinneaple_models.recurrent.registry",           "register_into_global"),
        ("pinneaple_models.reservoir_computing.registry", "register_into_global"),
        ("pinneaple_models.rom.registry",                 "register_into_global"),
        ("pinneaple_models.transformers.registry",        "register_into_global"),
        ("pinneaple_models.benchmarks.registry",          "register_into_global"),
        ("pinneaple_models.group_b_registry",             "register_into_global"),
    ]

    for module_path, fn_name in _FAMILIES:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            getattr(mod, fn_name)()
        except Exception:
            pass  # skip families with import errors or duplicate-key conflicts
