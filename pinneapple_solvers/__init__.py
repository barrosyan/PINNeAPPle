"""pinneapple_solvers — compatibility shim.

Re-exports from pinneapple_simulation.numerical_solvers so that legacy code
importing ``from pinneapple_solvers.*`` continues to work.

All new code should import directly from
``pinneapple_simulation.numerical_solvers``.
"""
from pinneapple_simulation.numerical_solvers.base     import SolverBase, SolverOutput
from pinneapple_simulation.numerical_solvers.registry import SolverRegistry
from pinneapple_simulation.numerical_solvers.fdm      import FDMSolver

try:
    from pinneapple_simulation.numerical_solvers.fem import FEMSolver
except Exception:
    FEMSolver = None  # type: ignore

try:
    from pinneapple_simulation.numerical_solvers.fvm import FVMSolver
except Exception:
    FVMSolver = None  # type: ignore

try:
    from pinneapple_simulation.numerical_solvers.lbm import LBMSolver
except Exception:
    LBMSolver = None  # type: ignore

try:
    from pinneapple_simulation.numerical_solvers.spectral import SpectralSolver
except Exception:
    SpectralSolver = None  # type: ignore

try:
    from pinneapple_simulation.numerical_solvers.sph import SPHSolver
except Exception:
    SPHSolver = None  # type: ignore

try:
    # Was `import FFTProcessor` -- no such name exists anywhere in
    # pinneapple_simulation.numerical_solvers.fft (only `FFTSolver` does),
    # so this always raised and was silently swallowed by the `except`,
    # leaving `FFTProcessor` permanently `None` with no error ever
    # surfaced. Found via tests/pinneapple_solvers/test_fft.py failing to
    # collect (`from pinneapple_solvers.fft import FFTSolver`, a second,
    # independent gap: `pinneapple_solvers.fft` wasn't a real importable
    # submodule either -- see fft.py alongside this file).
    from pinneapple_simulation.numerical_solvers.fft import FFTSolver
except Exception:
    FFTSolver = None  # type: ignore

# Convenience: solver registry instance
registry = SolverRegistry

__all__ = [
    "SolverBase", "SolverOutput", "SolverRegistry",
    "FDMSolver", "FEMSolver", "FVMSolver",
    "LBMSolver", "SpectralSolver", "SPHSolver",
    "FFTSolver",
    "registry",
]
