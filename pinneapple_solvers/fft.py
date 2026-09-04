"""``pinneapple_solvers.fft`` compatibility submodule -- see ``__init__.py``
for the two independent bugs found here (a nonexistent ``FFTProcessor``
name silently swallowed by a bare ``except``, and this submodule path not
existing at all despite ``tests/pinneapple_solvers/test_fft.py`` using
it).
"""
from pinneapple_simulation.numerical_solvers.fft import FFTSolver

__all__ = ["FFTSolver"]
