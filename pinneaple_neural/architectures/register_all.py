from __future__ import annotations
"""Register all model families into the global ModelRegistry."""

import importlib
import logging

logger = logging.getLogger(__name__)

_REGISTERED = False


def register_all() -> None:
    """Populate ModelRegistry with every family. Idempotent — safe to call multiple times."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    _FAMILIES = [
        ("pinneaple_neural.architectures.autoencoders.registry",        "register_into_global"),
        ("pinneaple_neural.architectures.classical_ts.registry",        "register_into_global"),
        ("pinneaple_neural.architectures.continuous.registry",          "register_into_global"),
        ("pinneaple_neural.architectures.convolutions.registry",        "register_into_global"),
        ("pinneaple_neural.architectures.graphnn.registry",             "register_into_global"),
        ("pinneaple_neural.architectures.neural_operators.registry",    "register_into_global"),
        ("pinneaple_neural.architectures.physics_aware.registry",       "register_into_global"),
        ("pinneaple_neural.architectures.pinns.registry",               "register_into_global"),
        ("pinneaple_neural.architectures.recurrent.registry",           "register_into_global"),
        ("pinneaple_neural.architectures.reservoir_computing.registry", "register_into_global"),
        ("pinneaple_neural.architectures.rom.registry",                 "register_into_global"),
        ("pinneaple_neural.architectures.transformers.registry",        "register_into_global"),
        ("pinneaple_neural.architectures.benchmarks.registry",          "register_into_global"),
        ("pinneaple_neural.architectures.group_b_registry",             "register_into_global"),
    ]

    for module_path, fn_name in _FAMILIES:
        try:
            mod = importlib.import_module(module_path)
            getattr(mod, fn_name)()
        except ImportError as exc:
            logger.debug("pinneaple_models: skipping %s (missing dependency: %s)", module_path, exc)
        except KeyError as exc:
            logger.warning("pinneaple_models: registry collision in %s — %s", module_path, exc)
