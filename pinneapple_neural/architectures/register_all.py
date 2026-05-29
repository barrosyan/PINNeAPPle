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
        ("pinneapple_neural.architectures.autoencoders.registry",        "register_into_global"),
        ("pinneapple_neural.architectures.classical_ts.registry",        "register_into_global"),
        ("pinneapple_neural.architectures.continuous.registry",          "register_into_global"),
        ("pinneapple_neural.architectures.convolutions.registry",        "register_into_global"),
        ("pinneapple_neural.architectures.graphnn.registry",             "register_into_global"),
        ("pinneapple_neural.architectures.neural_operators.registry",    "register_into_global"),
        ("pinneapple_neural.architectures.physics_aware.registry",       "register_into_global"),
        ("pinneapple_neural.architectures.pinns.registry",               "register_into_global"),
        ("pinneapple_neural.architectures.recurrent.registry",           "register_into_global"),
        ("pinneapple_neural.architectures.reservoir_computing.registry", "register_into_global"),
        ("pinneapple_neural.architectures.rom.registry",                 "register_into_global"),
        ("pinneapple_neural.architectures.transformers.registry",        "register_into_global"),
        ("pinneapple_neural.architectures.benchmarks.registry",          "register_into_global"),
        ("pinneapple_neural.architectures.group_b_registry",             "register_into_global"),
    ]

    for module_path, fn_name in _FAMILIES:
        try:
            mod = importlib.import_module(module_path)
            getattr(mod, fn_name)()
        except ImportError as exc:
            logger.debug("pinneapple_models: skipping %s (missing dependency: %s)", module_path, exc)
        except KeyError as exc:
            logger.warning("pinneapple_models: registry collision in %s — %s", module_path, exc)
