"""``pinneapple_models.registry`` compatibility submodule -- same gap
pattern as ``pinneapple_train``/``pinneapple_solvers`` (found via
``tests/pinneapple_models/test_registry.py`` failing to collect): the
package's own ``__init__.py`` only re-exported ``ModelRegistry`` at the
flat top level, not as this submodule path.
"""
from pinneapple_neural.architectures.registry import ModelRegistry

__all__ = ["ModelRegistry"]
