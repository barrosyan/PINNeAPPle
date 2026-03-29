from .base import SolverBase, SolverOutput
from .registry import SolverRegistry, SolverSpec, register_all, SolverCatalog
from .problem_runner import generate_pinn_dataset

# Optional bridges (graceful import — not installed in all environments)
try:
    from .openfoam_bridge import OpenFOAMBridge, generate_case, run_openfoam, extract_fields, openfoam_to_dataset
    _OPENFOAM_AVAILABLE = True
except Exception:
    _OPENFOAM_AVAILABLE = False

try:
    from .fenics_bridge import FEnicsBridge
    _FENICS_AVAILABLE = True
except Exception:
    _FENICS_AVAILABLE = False


def openfoam_available() -> bool:
    """Return True if the OpenFOAM bridge is importable."""
    return _OPENFOAM_AVAILABLE


def fenics_available() -> bool:
    """Return True if the FEniCS bridge is importable."""
    return _FENICS_AVAILABLE


__all__ = [
    "SolverBase",
    "SolverOutput",
    "SolverRegistry",
    "SolverSpec",
    "register_all",
    "SolverCatalog",
    "generate_pinn_dataset",
    "openfoam_available",
    "fenics_available",
]

if _OPENFOAM_AVAILABLE:
    __all__ += ["OpenFOAMBridge", "generate_case", "run_openfoam", "extract_fields", "openfoam_to_dataset"]

if _FENICS_AVAILABLE:
    __all__ += ["FEnicsBridge"]
