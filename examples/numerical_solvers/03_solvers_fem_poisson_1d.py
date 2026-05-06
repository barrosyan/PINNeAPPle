"""
pinneaple_solvers: FEMSolver demo (1D Poisson with Dirichlet BCs)

Problem
-------
Solve on x in [0, 1]:
  -u''(x) = f(x),   u(0)=u(1)=0

Choose:
  u*(x) = sin(pi x)  =>  f(x) = pi^2 sin(pi x)

What this shows
--------------
- How FEMSolver's API expects an assemble_fn(mesh, params) -> (K, f)
- How to apply Dirichlet BCs via apply_bcs_fn(K, f, params) -> (K2, f2)
- How to solve with "direct" or MVP "cg".

Run
---
python examples/pinneaple_solvers/03_solvers_fem_poisson_1d.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import torch

from pinneaple_solvers.fem import FEMSolver


@dataclass
class LineMesh1D:
    """Minimal 1D mesh: nodes (N,) and elements (E,2) connectivity."""
    nodes: torch.Tensor  # (N,)
    elements: torch.Tensor  # (E,2) long


def assemble_poisson_1d(mesh: LineMesh1D, params: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Linear FEM assembly for -u''=f on [0,1] with P1 elements."""
    x = mesh.nodes
    elems = mesh.elements
    N = x.numel()

    K = torch.zeros((N, N), dtype=x.dtype)
    f = torch.zeros((N,), dtype=x.dtype)

    # source f(x)
    def rhs(xp: torch.Tensor) -> torch.Tensor:
        return (math.pi ** 2) * torch.sin(math.pi * xp)

    for e in range(elems.shape[0]):
        i = int(elems[e, 0])
        j = int(elems[e, 1])
        xi, xj = x[i], x[j]
        h = (xj - xi).abs().clamp_min(1e-12)

        # element stiffness for P1: (1/h) * [[ 1, -1],[-1, 1]]
        ke = (1.0 / h) * torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=x.dtype)

        # element load using midpoint rule: fe ~= f(xm)*h/2*[1,1]
        xm = 0.5 * (xi + xj)
        fe = rhs(xm) * (h / 2.0) * torch.tensor([1.0, 1.0], dtype=x.dtype)

        # scatter
        K[i, i] += ke[0, 0]
        K[i, j] += ke[0, 1]
        K[j, i] += ke[1, 0]
        K[j, j] += ke[1, 1]
        f[i] += fe[0]
        f[j] += fe[1]

    return K, f


def apply_dirichlet(mesh: LineMesh1D, K: torch.Tensor, f: torch.Tensor, params: Dict[str, Any]):
    """Strong Dirichlet BCs at x=0 and x=1 (nodes 0 and N-1)."""
    N = mesh.nodes.numel()
    bc_nodes = params.get("bc_nodes", [0, N - 1])
    bc_values = params.get("bc_values", [0.0, 0.0])

    K2 = K.clone()
    f2 = f.clone()

    for n, val in zip(bc_nodes, bc_values):
        n = int(n)
        K2[n, :] = 0.0
        K2[:, n] = 0.0
        K2[n, n] = 1.0
        f2[n] = float(val)
    return K2, f2


def main():
    torch.manual_seed(0)
    dtype = torch.float64

    N = 200
    nodes = torch.linspace(0.0, 1.0, N, dtype=dtype)
    elements = torch.stack([torch.arange(N - 1), torch.arange(1, N)], dim=1).to(torch.long)
    mesh = LineMesh1D(nodes=nodes, elements=elements)

    params = {"bc_nodes": [0, N - 1], "bc_values": [0.0, 0.0]}

    solver = FEMSolver(
        assemble_fn=lambda m, p: assemble_poisson_1d(m, p),
        apply_bcs_fn=lambda K, f, p: apply_dirichlet(mesh, K, f, p),
        solver="cg",  # try "direct" too
        tol=1e-12,
        max_iter=5000,
    )

    out = solver(mesh=mesh, params=params)
    u = out.result

    u_true = torch.sin(math.pi * nodes)
    err_l2 = torch.sqrt(torch.mean((u - u_true) ** 2)).item()

    print("--- FEM Poisson 1D")
    print(f"N nodes: {N}")
    print(f"solver: {out.extras['solver']}")
    print(f"L2 error (mean RMSE): {err_l2:.6e}")


if __name__ == "__main__":
    main()