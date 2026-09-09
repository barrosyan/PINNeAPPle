"""pinneapple_tools.sandbox — running untrusted, user-supplied solver
scripts out-of-process with best-effort resource limits.

See ``custom_solver_runner`` for the (important, honest) limits of what
this actually protects against -- it is process-level resource limiting,
not a real sandbox.
"""
from .custom_solver_runner import run_custom_solver

__all__ = ["run_custom_solver"]
