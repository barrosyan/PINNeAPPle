"""
pinneaple.

This package exposes the project's modular subpackages (pinneaple_*)
under the `pinneaple.*` namespace for a stable public API and docs.
"""

import importlib
import pkgutil

__all__ = []

for mod in pkgutil.iter_modules():
    if mod.name.startswith("pinneaple_"):
        m = importlib.import_module(mod.name)
        globals()[mod.name] = m
        __all__.append(mod.name)