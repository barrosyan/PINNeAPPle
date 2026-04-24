"""Finite Element Method — problem-agnostic P1/Q1 solver."""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import torch

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def _make_structured_mesh(nx: int, ny: int, bounds: Dict[str, Tuple[float, float]]):
    """Build a regular (nx x ny) quad mesh; returns nodes (N,2) and elem (E,4)."""
    x0, x1 = bounds.get("x", (0.0, 1.0))
    y0, y1 = bounds.get("y", (0.0, 1.0))
    xs = torch.linspace(x0, x1, nx + 1)
    ys = torch.linspace(y0, y1, ny + 1)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    nodes = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)  # (N,2)

    elems = []
    for j in range(ny):
        for i in range(nx):
            n0 = i * (ny + 1) + j
            n1 = (i + 1) * (ny + 1) + j
            n2 = (i + 1) * (ny + 1) + (j + 1)
            n3 = i * (ny + 1) + (j + 1)
            elems.append([n0, n1, n2, n3])
    elems = torch.tensor(elems, dtype=torch.long)  # (E,4)
    return nodes, elems


# ---------------------------------------------------------------------------
# Bilinear Q1 shape functions on [-1,1]^2
# ---------------------------------------------------------------------------

_GQ = 1.0 / 3.0 ** 0.5  # Gauss point coord
_QP = torch.tensor([[-_GQ, -_GQ], [_GQ, -_GQ], [_GQ, _GQ], [-_GQ, _GQ]])  # (4,2)
_QW = torch.ones(4)


def _q1_shape(xi: float, eta: float) -> torch.Tensor:
    """Q1 shape functions at (xi, eta) in [-1,1]^2."""
    return 0.25 * torch.tensor([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta),
    ])


def _q1_grad(xi: float, eta: float) -> torch.Tensor:
    """Returns (2,4) matrix of dN/d(xi,eta)."""
    dxi = 0.25 * torch.tensor([
        -(1 - eta), (1 - eta), (1 + eta), -(1 + eta),
    ])
    deta = 0.25 * torch.tensor([
        -(1 - xi), -(1 + xi), (1 + xi), (1 - xi),
    ])
    return torch.stack([dxi, deta], dim=0)  # (2,4)


def _elem_stiffness_scalar(xy: torch.Tensor, coeff: float = 1.0) -> torch.Tensor:
    """
    Q1 element stiffness K_e (4,4) for -coeff * Δu = f.
    xy: (4,2) node coordinates.
    """
    Ke = torch.zeros(4, 4, device=xy.device)
    qp = _QP.to(xy.device)
    for q in range(4):
        xi, eta = qp[q, 0].item(), qp[q, 1].item()
        dN = _q1_grad(xi, eta).to(xy.device)   # (2,4)
        J = dN @ xy                              # (2,2)
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        Jinv = torch.zeros(2, 2, device=xy.device)
        Jinv[0, 0] = J[1, 1]; Jinv[0, 1] = -J[0, 1]
        Jinv[1, 0] = -J[1, 0]; Jinv[1, 1] = J[0, 0]
        Jinv = Jinv / detJ
        B = Jinv @ dN  # (2,4)  physical gradients
        Ke = Ke + coeff * detJ * (B.T @ B)
    return Ke


def _elem_mass(xy: torch.Tensor) -> torch.Tensor:
    """Q1 element mass M_e (4,4)."""
    Me = torch.zeros(4, 4, device=xy.device)
    qp = _QP.to(xy.device)
    for q in range(4):
        xi, eta = qp[q, 0].item(), qp[q, 1].item()
        N = _q1_shape(xi, eta).to(xy.device)   # (4,)
        dN = _q1_grad(xi, eta).to(xy.device)
        J = dN @ xy
        detJ = (J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]).abs()
        Me = Me + detJ * torch.outer(N, N)
    return Me


# ---------------------------------------------------------------------------
# Assembler (Poisson / Laplace / Helmholtz)
# ---------------------------------------------------------------------------

def _assemble_poisson(
    nodes: torch.Tensor,
    elems: torch.Tensor,
    coeff: float,
    k2: float,
    source: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assemble K (N,N) and f (N,) for (coeff*Δ + k²)u = source."""
    N = nodes.shape[0]
    K = torch.zeros(N, N, device=nodes.device)
    f = torch.zeros(N, device=nodes.device)

    for e in range(elems.shape[0]):
        idx = elems[e]           # (4,)
        xy = nodes[idx]          # (4,2)
        Ke = _elem_stiffness_scalar(xy, coeff)
        if k2 != 0.0:
            Me = _elem_mass(xy)
            Ke = Ke - k2 * Me
        # element load: f_e = source * M_e @ [1,1,1,1]
        fe = source * _elem_mass(xy) @ torch.ones(4, device=nodes.device)
        for i in range(4):
            for j in range(4):
                K[idx[i], idx[j]] += Ke[i, j]
            f[idx[i]] += fe[i]
    return K, f


# ---------------------------------------------------------------------------
# Axisymmetric elasticity assembler  (ur, uz unknowns, r-weighted)
# ---------------------------------------------------------------------------

def _assemble_axisym_elasticity(
    nodes: torch.Tensor,
    elems: torch.Tensor,
    E: float,
    nu: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Plane-strain axisymmetric elasticity stiffness (interleaved DOF ordering).
    DOF: [ur0, uz0, ur1, uz1, ...] — 2N total.
    Returns K (2N,2N), f zeros (2N,).
    """
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    # Voigt 4-component strain: [err, ezz, etheta, erz]
    # C matrix (4x4 plane-strain axisymmetric)
    C = torch.tensor([
        [lam + 2 * mu, lam, lam, 0.0],
        [lam, lam + 2 * mu, lam, 0.0],
        [lam, lam, lam + 2 * mu, 0.0],
        [0.0, 0.0, 0.0, mu],
    ])
    N_nodes = nodes.shape[0]
    ndof = 2 * N_nodes
    K = torch.zeros(ndof, ndof, device=nodes.device)

    qp = _QP.to(nodes.device)
    for e in range(elems.shape[0]):
        idx = elems[e]           # (4,)
        xy = nodes[idx]          # (4,2)  col0=r, col1=z
        Ke = torch.zeros(8, 8, device=nodes.device)
        for q in range(4):
            xi, eta = qp[q, 0].item(), qp[q, 1].item()
            N = _q1_shape(xi, eta).to(nodes.device)   # (4,)
            dN = _q1_grad(xi, eta).to(nodes.device)   # (2,4)
            J = dN @ xy  # (2,2)
            detJ = (J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]).abs()
            Jinv = torch.zeros(2, 2, device=nodes.device)
            Jinv[0, 0] = J[1, 1]; Jinv[0, 1] = -J[0, 1]
            Jinv[1, 0] = -J[1, 0]; Jinv[1, 1] = J[0, 0]
            Jinv = Jinv / (J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
            dNxy = Jinv @ dN  # (2,4): dN/dr, dN/dz
            r = (N * xy[:, 0]).sum().clamp_min(1e-12)
            # B matrix (4,8): strains from [ur,uz] DOFs
            B = torch.zeros(4, 8, device=nodes.device)
            for a in range(4):
                col_r = 2 * a
                col_z = 2 * a + 1
                B[0, col_r] = dNxy[0, a]          # err = dur/dr
                B[1, col_z] = dNxy[1, a]          # ezz = duz/dz
                B[2, col_r] = N[a] / r             # etheta = ur/r
                B[3, col_r] = dNxy[1, a]           # erz (r-part)
                B[3, col_z] = dNxy[0, a]           # erz (z-part)
            Ke = Ke + r * detJ * (B.T @ C.to(nodes.device) @ B)
        # scatter into global
        for i in range(4):
            for j in range(4):
                for di in range(2):
                    for dj in range(2):
                        K[2 * idx[i] + di, 2 * idx[j] + dj] += Ke[2 * i + di, 2 * j + dj]

    f = torch.zeros(ndof, device=nodes.device)
    return K, f


# ---------------------------------------------------------------------------
# BC application
# ---------------------------------------------------------------------------

def _apply_dirichlet(K: torch.Tensor, f: torch.Tensor, dofs: torch.Tensor, vals: torch.Tensor):
    """Row/column elimination for Dirichlet DOFs."""
    for k, d in enumerate(dofs.tolist()):
        d = int(d)
        f -= K[:, d] * vals[k]
        K[d, :] = 0.0
        K[:, d] = 0.0
        K[d, d] = 1.0
        f[d] = vals[k]
    return K, f


def _boundary_dofs_rect(nodes: torch.Tensor, bounds: Dict[str, Tuple[float, float]], tol: float = 1e-9):
    """Return dict of edge → node indices for rectangular domain."""
    x0, x1 = bounds.get("x", (0.0, 1.0))
    y0, y1 = bounds.get("y", (0.0, 1.0))
    x = nodes[:, 0]; y = nodes[:, 1]
    return {
        "left":   torch.where(torch.abs(x - x0) < tol)[0],
        "right":  torch.where(torch.abs(x - x1) < tol)[0],
        "bottom": torch.where(torch.abs(y - y0) < tol)[0],
        "top":    torch.where(torch.abs(y - y1) < tol)[0],
    }


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

@SolverRegistry.register(
    name="fem",
    family="pde",
    description="Finite Element Method — problem-agnostic Q1 elements for Poisson/Helmholtz and axisymmetric elasticity.",
    tags=["fem", "pde", "agnostic", "elliptic", "elasticity"],
)
class FEMSolver(SolverBase):
    """
    Problem-agnostic FEM solver.

    Supported PDE kinds (spec.pde.kind):
      poisson / laplace            — -coeff·Δu = source  (coeff default 1)
      helmholtz                    — -Δu + k²u = source  (k from params.k)
      axisymmetric_linear_elasticity — r-weighted weak form (E, nu from params)

    Params consumed from PDETermSpec.params:
      coeff  — diffusion coefficient (poisson, default 1.0)
      source — uniform RHS scalar     (poisson/helmholtz, default 0.0)
      k      — wavenumber             (helmholtz)
      E, nu  — Young's modulus, Poisson's ratio (elasticity)

    Conditions from ProblemSpec.conditions:
      DirichletBC with .selector as tag string ("left","right","bottom","top")
      or callable selector(nodes) -> bool mask

    Linear system solved by CG or direct (torch.linalg.solve).
    """

    def __init__(
        self,
        nx: int = 32,
        ny: int = 32,
        solver: str = "cg",
        tol: float = 1e-8,
        max_iter: int = 4000,
    ):
        super().__init__()
        self.nx = int(nx)
        self.ny = int(ny)
        self.solver_name = str(solver).lower().strip()
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    @classmethod
    def from_problem_spec(cls, spec, nx: int = 32, ny: int = 32, **kwargs) -> "FEMSolver":
        return cls(nx=nx, ny=ny, **kwargs)

    # ------------------------------------------------------------------
    def _linear_solve(self, K: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        if self.solver_name == "direct":
            return torch.linalg.solve(K, f)
        # CG
        x = torch.zeros_like(f)
        r = f - K @ x
        p = r.clone()
        rsold = (r * r).sum()
        for _ in range(self.max_iter):
            Kp = K @ p
            alpha = rsold / (p * Kp).sum().clamp_min(1e-14)
            x = x + alpha * p
            r = r - alpha * Kp
            rsnew = (r * r).sum()
            if torch.sqrt(rsnew).item() < self.tol:
                break
            p = r + (rsnew / rsold.clamp_min(1e-14)) * p
            rsold = rsnew
        return x

    # ------------------------------------------------------------------
    def solve_from_spec(self, spec) -> SolverOutput:
        kind = spec.pde.kind.lower().replace("-", "_").replace(" ", "_")
        bounds = dict(spec.domain_bounds)
        params = dict(spec.pde.params)
        conditions = spec.conditions

        nodes, elems = _make_structured_mesh(self.nx, self.ny, bounds)
        edge_map = _boundary_dofs_rect(nodes, bounds)

        if kind in ("axisymmetric_linear_elasticity", "linear_elasticity_axisymmetric"):
            K, f = _assemble_axisym_elasticity(
                nodes, elems,
                E=float(params.get("E", 200e9)),
                nu=float(params.get("nu", 0.3)),
            )
        else:
            coeff  = float(params.get("coeff",  1.0))
            source = float(params.get("source", 0.0))
            k2     = float(params.get("k",      0.0)) ** 2
            K, f = _assemble_poisson(nodes, elems, coeff, k2, source)

        # Apply boundary conditions
        from pinneaple_environment.conditions import DirichletBC
        for cond in conditions:
            if not isinstance(cond, DirichletBC):
                continue
            sel = cond.selector
            if isinstance(sel, str) and sel in edge_map:
                dofs = edge_map[sel]
            elif callable(sel):
                mask = sel(nodes)
                dofs = torch.where(mask)[0]
            else:
                continue
            val_fn = cond.value_fn
            vals = val_fn(nodes[dofs]) if callable(val_fn) else torch.full((len(dofs),), float(val_fn))
            K, f = _apply_dirichlet(K, f, dofs, vals.to(nodes.device))

        u = self._linear_solve(K, f)

        residual = torch.norm(K @ u - f) / torch.norm(f).clamp_min(1e-12)
        return SolverOutput(
            result=u.reshape(self.nx + 1, self.ny + 1),
            losses={"residual": residual},
            extras={
                "nodes": nodes,
                "elems": elems,
                "K": K,
                "f_rhs": f,
                "kind": kind,
                "solver": self.solver_name,
            },
        )

    # ------------------------------------------------------------------
    def forward(self, *, spec=None, mesh: Any = None, params: Dict[str, Any] = None) -> SolverOutput:
        """
        Two call styles:
          forward(spec=problem_spec)                       # problem-agnostic
          forward(mesh=my_mesh, params={"K": K, "f": f})  # legacy manual assembly
        """
        if spec is not None:
            return self.solve_from_spec(spec)

        # Legacy: user provides pre-assembled K, f inside params
        K = params["K"]
        f = params["f"]
        if params.get("apply_bcs_fn") is not None:
            K, f = params["apply_bcs_fn"](K, f, params)
        u = self._linear_solve(K, f)
        return SolverOutput(
            result=u,
            losses={"total": torch.tensor(0.0, device=u.device)},
            extras={"K": K, "f": f, "solver": self.solver_name},
        )
