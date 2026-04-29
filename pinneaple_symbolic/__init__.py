"""pinneaple_symbolic — Symbolic PDE compiler for PINNeAPPle.

Converts SymPy PDE residual expressions into callable PyTorch autograd functions
and provides hard / soft boundary condition implementations.

Quick start::

    import sympy as sp
    from pinneaple_symbolic import SymbolicPDE, HardBC

    x, y = sp.symbols("x y")
    u = sp.Function("u")

    # Poisson: u_xx + u_yy + 2*pi^2 * sin(pi*x)*sin(pi*y) = 0
    expr = (
        u(x, y).diff(x, 2)
        + u(x, y).diff(y, 2)
        + 2 * sp.pi**2 * sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
    )
    pde = SymbolicPDE(expr, coord_syms=[x, y], field_syms=[u])

    # Hard BC: u = 0 on unit-square boundary via distance-function ansatz
    phi = lambda pts: pts[:,0:1]*(1-pts[:,0:1])*pts[:,1:2]*(1-pts[:,1:2])
    import torch
    hard_bc = HardBC(phi, lambda pts: torch.zeros(pts.shape[0], 1, device=pts.device))
    wrapped_model = hard_bc.wrap_model(base_model)

    residual_fn = pde.to_residual_fn(wrapped_model)
    res = residual_fn(x_col)   # (N, 1) PDE residual tensor
"""

from .compiler import SymbolicPDE, auto_residual, pde_from_sympy
from .bc import DirichletBC, HardBC, NeumannBC, PeriodicBC

__all__ = [
    # Compiler
    "SymbolicPDE",
    "pde_from_sympy",
    "auto_residual",
    # Boundary conditions
    "HardBC",
    "PeriodicBC",
    "DirichletBC",
    "NeumannBC",
]
