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
    from pinneapple_simulation.numerical_solvers.fft import FFTProcessor
except Exception:
    FFTProcessor = None  # type: ignore

# Convenience: solver registry instance
registry = SolverRegistry

__all__ = [
    "SolverBase", "SolverOutput", "SolverRegistry",
    "FDMSolver", "FEMSolver", "FVMSolver",
    "LBMSolver", "SpectralSolver", "SPHSolver",
    "FFTProcessor",
    "registry",
]
