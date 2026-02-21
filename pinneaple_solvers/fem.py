"""Finite Element Method solver scaffold with assemble/solve API."""
from __future__ import annotations
from typing import Dict, Optional, Callable, Any

import torch

from .base import SolverBase, SolverOutput


class FEMSolver(SolverBase):
    """
    FEM Solver (scaffold MVP).

    Design:
      - assemble(K, f) from mesh + coefficients (provided by user)
      - apply BCs
      - solve linear system K u = f (or nonlinear via iteration later)

    This MVP focuses on API shape & linear solve.
    """
    def __init__(
        self,
        *,
        assemble_fn: Callable[[Any, Dict[str, Any]], tuple],
        apply_bcs_fn: Optional[Callable[[torch.Tensor, torch.Tensor, Dict[str, Any]], tuple]] = None,
        solver: str = "cg",  # "cg" | "direct"
        tol: float = 1e-8,
        max_iter: int = 2000,
    ):
        super().__init__()
        self.assemble_fn = assemble_fn
        self.apply_bcs_fn = apply_bcs_fn
        self.solver = str(solver).lower().strip()
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    def _solve(self, K: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        # K: (n,n), f:(n,) or (n,m)
        if self.solver == "direct":
            return torch.linalg.solve(K, f)

        # Conjugate Gradient (SPD) MVP for vector RHS
        x = torch.zeros_like(f)
        r = f - K @ x
        p = r.clone()
        rsold = (r * r).sum()

        for _ in range(self.max_iter):
            Kp = K @ p
            alpha = rsold / ((p * Kp).sum().clamp_min(1e-12))
            x = x + alpha * p
            r = r - alpha * Kp
            rsnew = (r * r).sum()
            if torch.sqrt(rsnew).item() < self.tol:
                break
            p = r + (rsnew / rsold.clamp_min(1e-12)) * p
            rsold = rsnew
        return x

    def forward(self, *, mesh: Any, params: Dict[str, Any]) -> SolverOutput:
        """
        mesh: pinneaple_geom mesh object (or trimesh/meshio wrapper)
        params: PDE coefficients, source terms, BC definitions, etc.

        assemble_fn(mesh, params) -> (K, f)
        apply_bcs_fn(K, f, params) -> (K2, f2) optional
        """
        K, f = self.assemble_fn(mesh, params)
        if self.apply_bcs_fn is not None:
            K, f = self.apply_bcs_fn(K, f, params)

        u = self._solve(K, f)

        return SolverOutput(
            result=u,
            losses={"total": torch.tensor(0.0, device=u.device)},
            extras={"K": K, "f": f, "solver": self.solver},
        )
