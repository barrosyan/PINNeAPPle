"""Meshfree solvers: RBF collocation and Moving Least Squares (MLS)."""
from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry


# ---------------------------------------------------------------------------
# RBF kernels
# ---------------------------------------------------------------------------

def _rbf_multiquadric(r: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(1.0 + (eps * r) ** 2)


def _rbf_inverse_multiquadric(r: torch.Tensor, eps: float) -> torch.Tensor:
    return 1.0 / torch.sqrt(1.0 + (eps * r) ** 2)


def _rbf_gaussian(r: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.exp(-(eps * r) ** 2)


def _rbf_thin_plate(r: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.where(r > 0, r ** 2 * torch.log(r.clamp_min(1e-14)), torch.zeros_like(r))


_RBF_KERNELS = {
    "multiquadric": _rbf_multiquadric,
    "imq": _rbf_inverse_multiquadric,
    "gaussian": _rbf_gaussian,
    "thin_plate": _rbf_thin_plate,
}


# ---------------------------------------------------------------------------
# Distance matrix
# ---------------------------------------------------------------------------

def _pairwise_dist(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """(N,d),(M,d) → (N,M) Euclidean distances."""
    diff = X.unsqueeze(1) - Y.unsqueeze(0)          # (N,M,d)
    return torch.norm(diff, dim=-1)                   # (N,M)


# ---------------------------------------------------------------------------
# RBF Laplacian for Poisson (-Δu = f)
# ---------------------------------------------------------------------------

def _rbf_laplacian_gaussian(centres: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Analytical Laplacian of φ(r) = exp(-(eps·r)²) w.r.t. evaluation point x.
    Returns (N,N) matrix L where L[i,j] = Δ_x φ(||x_i - c_j||) at x_i.
    """
    N, d = centres.shape
    D = centres.unsqueeze(1) - centres.unsqueeze(0)   # (N,N,d)
    r2 = (D ** 2).sum(-1)                             # (N,N)
    e2 = eps ** 2
    # Δ exp(-e²r²) = exp(-e²r²) * 2e²(2e²r² - d)
    phi = torch.exp(-e2 * r2)
    lap = phi * 2 * e2 * (2 * e2 * r2 - d)
    return lap


def _rbf_laplacian_mq(centres: torch.Tensor, eps: float) -> torch.Tensor:
    """Analytical Laplacian of multiquadric φ = √(1+(eps·r)²)."""
    N, d = centres.shape
    r2 = ((centres.unsqueeze(1) - centres.unsqueeze(0)) ** 2).sum(-1)
    e2 = eps ** 2
    phi = torch.sqrt(1.0 + e2 * r2)
    # Δ_x φ(||x-c||) = e²(d + (d-2)*e²r²) / φ³
    lap = e2 * (d + (d - 2) * e2 * r2) / phi ** 3
    return lap


def _rbf_laplacian_thin_plate(centres, eps=None):
    """Analytical Laplacian of thin-plate spline φ(r) = r² log(r).

    In d dimensions: Δφ = d·log(r²) + (d+2). Diagonal (r=0) set to 0.
    """
    import numpy as np
    N, d = centres.shape
    diff = centres[:, None, :] - centres[None, :, :]
    r2 = (diff ** 2).sum(-1)
    r2_safe = np.where(r2 < 1e-24, 1e-24, r2)
    lap = d * np.log(r2_safe) + (d + 2)
    np.fill_diagonal(lap, 0.0)
    return lap


def _rbf_laplacian_imq(centres, eps: float):
    """Analytical Laplacian of inverse multiquadric φ(r) = (1+ε²r²)^(-1/2).

    Δφ = ε²(1+ε²r²)^(-5/2) · (-d + ε²r²(3-d))
    """
    import numpy as np
    N, d = centres.shape
    diff = centres[:, None, :] - centres[None, :, :]
    r2 = (diff ** 2).sum(-1)
    denom = (1.0 + eps ** 2 * r2) ** 2.5
    lap = eps ** 2 * (-d + eps ** 2 * r2 * (3 - d)) / denom
    return lap


# ---------------------------------------------------------------------------
# RBF Collocation Solver
# ---------------------------------------------------------------------------

@SolverRegistry.register(
    name="rbf_collocation",
    family="meshfree",
    description="RBF collocation (strong form) for Poisson/Laplace/Helmholtz on scattered nodes.",
    tags=["meshfree", "rbf", "collocation", "agnostic"],
)
class RBFCollocationSolver(SolverBase):
    """
    Meshfree RBF collocation solver (Kansa method).

    Solves  -coeff·Δu + k²·u = source  at interior nodes,
    with Dirichlet BCs at boundary nodes.

    Supported PDE kinds:
      poisson / laplace / helmholtz

    Parameters
    ----------
    rbf       : kernel name ("gaussian" | "multiquadric" | "imq" | "thin_plate")
    eps       : shape parameter
    coeff     : diffusion coefficient (overridden by spec.pde.params.coeff)
    reg       : Tikhonov regularisation for ill-conditioned A
    """

    def __init__(
        self,
        rbf: str = "gaussian",
        eps: float = 3.0,
        coeff: float = 1.0,
        reg: float = 1e-8,
    ):
        super().__init__()
        self.rbf_name = rbf
        self.eps = float(eps)
        self.coeff = float(coeff)
        self.reg = float(reg)

    @classmethod
    def from_problem_spec(cls, spec, **kwargs) -> "RBFCollocationSolver":
        params = dict(spec.pde.params)
        return cls(
            eps=float(params.get("eps", 3.0)),
            coeff=float(params.get("coeff", 1.0)),
            **kwargs,
        )

    # ------------------------------------------------------------------
    def _phi(self, r: torch.Tensor) -> torch.Tensor:
        return _RBF_KERNELS[self.rbf_name](r, self.eps)

    def _lap_matrix(self, centres: torch.Tensor) -> torch.Tensor:
        if self.rbf_name == "gaussian":
            return _rbf_laplacian_gaussian(centres, self.eps)
        if self.rbf_name == "multiquadric":
            return _rbf_laplacian_mq(centres, self.eps)
        if self.rbf_name == "thin_plate":
            import numpy as np
            lap_np = _rbf_laplacian_thin_plate(centres.detach().cpu().numpy())
            return torch.tensor(lap_np, dtype=centres.dtype, device=centres.device)
        if self.rbf_name == "imq":
            import numpy as np
            lap_np = _rbf_laplacian_imq(centres.detach().cpu().numpy(), self.eps)
            return torch.tensor(lap_np, dtype=centres.dtype, device=centres.device)
        raise NotImplementedError(
            f"Analytical Laplacian not implemented for rbf={self.rbf_name!r}. "
            "Use 'gaussian', 'multiquadric', 'thin_plate', or 'imq'."
        )

    # ------------------------------------------------------------------
    def solve_from_spec(self, spec, interior_pts: torch.Tensor, boundary_pts: torch.Tensor) -> SolverOutput:
        """
        interior_pts : (Ni, d) collocation nodes inside domain
        boundary_pts : (Nb, d) boundary nodes with Dirichlet condition

        Returns solution values at [interior_pts; boundary_pts].
        """
        params = dict(spec.pde.params)
        coeff  = float(params.get("coeff",  self.coeff))
        source = float(params.get("source", 0.0))
        k2     = float(params.get("k",      0.0)) ** 2

        all_pts = torch.cat([interior_pts, boundary_pts], dim=0)   # (N,d)
        N  = all_pts.shape[0]
        Ni = interior_pts.shape[0]
        Nb = boundary_pts.shape[0]

        # Collocation matrix A
        A = torch.zeros(N, N, device=all_pts.device)

        # Interior rows: -coeff·Δφ + k²·φ  at interior centres
        Lap = self._lap_matrix(all_pts)   # (N,N): Lap[i,j] = Δ_x φ(x_i, c_j)
        Phi = self._phi(_pairwise_dist(all_pts, all_pts))

        A[:Ni, :] = -coeff * Lap[:Ni, :] + k2 * Phi[:Ni, :]

        # Boundary rows: φ (interpolation condition)
        A[Ni:, :] = Phi[Ni:, :]

        # RHS
        rhs = torch.zeros(N, device=all_pts.device)
        rhs[:Ni] = source

        # Dirichlet BC values
        from pinneaple_environment.conditions import DirichletBC
        for cond in spec.conditions:
            if isinstance(cond, DirichletBC):
                val_fn = cond.value_fn
                vals = (val_fn(boundary_pts) if callable(val_fn)
                        else torch.full((Nb,), float(val_fn), device=all_pts.device))
                rhs[Ni:] = vals
                break

        # Regularised solve
        Areg = A + self.reg * torch.eye(N, device=A.device)
        coeffs = torch.linalg.solve(Areg, rhs)

        # Evaluate solution at all nodes
        u = Phi @ coeffs
        residual = torch.norm(A @ coeffs - rhs) / torch.norm(rhs).clamp_min(1e-12)

        return SolverOutput(
            result=u,
            losses={"residual": residual},
            extras={
                "coeffs": coeffs,
                "all_pts": all_pts,
                "A": A,
                "rbf": self.rbf_name,
                "eps": self.eps,
            },
        )

    def forward(self, *, spec=None, interior_pts: torch.Tensor = None,
                boundary_pts: torch.Tensor = None, **kwargs) -> SolverOutput:
        if spec is not None and interior_pts is not None:
            return self.solve_from_spec(spec, interior_pts, boundary_pts)
        raise ValueError("Call forward(spec=..., interior_pts=..., boundary_pts=...)")


# ---------------------------------------------------------------------------
# Moving Least Squares (MLS) approximation
# ---------------------------------------------------------------------------

class MLSApproximation:
    """
    Meshfree Moving Least Squares approximation.

    Builds a local polynomial basis fit at each query point
    weighted by a compactly-supported weight function.

    Usage
    -----
    mls = MLSApproximation(degree=1, support_radius=0.2)
    u_query = mls.evaluate(query_pts, source_pts, source_vals)
    grad_u  = mls.gradient(query_pts, source_pts, source_vals)
    """

    def __init__(
        self,
        degree: int = 1,
        support_radius: float = 0.2,
        weight: str = "quartic",
    ):
        self.degree = int(degree)
        self.h = float(support_radius)
        self.weight_name = weight

    def _w(self, r_norm: torch.Tensor) -> torch.Tensor:
        """Normalised weight w(r/h); zero outside [0,1]."""
        s = r_norm / self.h
        if self.weight_name == "quartic":
            w = (1 - s ** 2) ** 2
        elif self.weight_name == "cubic":
            w = torch.where(s < 0.5,
                            2.0 / 3.0 - 4 * s ** 2 + 4 * s ** 3,
                            4.0 / 3.0 * (1 - s) ** 3)
        elif self.weight_name == "gaussian":
            w = torch.exp(-9 * s ** 2)
        else:
            w = (1 - s) ** 4 * (4 * s + 1)   # Wendland C2
        return torch.where(s <= 1.0, w, torch.zeros_like(w))

    def _basis(self, pts: torch.Tensor, centre: torch.Tensor) -> torch.Tensor:
        """
        Polynomial basis matrix p(x - x_c) for pts (N,d).
        degree=1: [1, dx, dy(, dz)]
        degree=2: [1, dx, dy, dx², dxy, dy²]  (2-D)
        Returns (N, m).
        """
        d = pts.shape[-1]
        dx = pts - centre.unsqueeze(0)    # (N,d)
        cols = [torch.ones(pts.shape[0], 1, device=pts.device)]
        cols.append(dx)                    # linear terms
        if self.degree >= 2:
            # Quadratic terms
            for i in range(d):
                for j in range(i, d):
                    cols.append((dx[:, i] * dx[:, j]).unsqueeze(1))
        return torch.cat(cols, dim=1)      # (N, m)

    def evaluate(
        self,
        query_pts: torch.Tensor,
        source_pts: torch.Tensor,
        source_vals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate MLS approximation at query_pts.

        Parameters
        ----------
        query_pts  : (Q, d)
        source_pts : (N, d)
        source_vals: (N,)

        Returns
        -------
        u_q : (Q,)
        """
        Q = query_pts.shape[0]
        u_q = torch.zeros(Q, device=query_pts.device)

        for q in range(Q):
            xq = query_pts[q]
            r  = torch.norm(source_pts - xq, dim=1)           # (N,)
            W  = torch.diag(self._w(r))                         # (N,N)
            P  = self._basis(source_pts, xq)                   # (N,m)
            m  = P.shape[1]
            A  = P.T @ W @ P                                    # (m,m)
            A  = A + 1e-10 * torch.eye(m, device=A.device)
            b  = P.T @ W @ source_vals                          # (m,)
            try:
                c = torch.linalg.solve(A, b)                   # (m,)
            except Exception:
                c = torch.linalg.lstsq(P, source_vals.unsqueeze(1)).solution[:, 0]
            # Basis at query point: p(0) = [1, 0, ...]
            p0 = self._basis(xq.unsqueeze(0), xq)[0]           # (m,)
            u_q[q] = p0 @ c

        return u_q

    def gradient(
        self,
        query_pts: torch.Tensor,
        source_pts: torch.Tensor,
        source_vals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gradient of MLS approximation via autograd.
        Returns (Q, d).
        """
        qp = query_pts.detach().requires_grad_(True)
        # Evaluate with autograd-capable path (scalar loop)
        grads = []
        for q in range(qp.shape[0]):
            xq = qp[q:q+1]
            r  = torch.norm(source_pts - xq, dim=1)
            W  = torch.diag(self._w(r))
            P  = self._basis(source_pts, xq[0])
            m  = P.shape[1]
            A  = P.T @ W @ P + 1e-10 * torch.eye(m, device=P.device)
            b  = P.T @ W @ source_vals
            c  = torch.linalg.solve(A, b)
            p0 = self._basis(xq, xq[0])  # (1,m)
            u  = (p0 @ c).squeeze()
            g  = torch.autograd.grad(u, xq, create_graph=False)[0]
            grads.append(g.squeeze())
        return torch.stack(grads, dim=0)   # (Q, d)
