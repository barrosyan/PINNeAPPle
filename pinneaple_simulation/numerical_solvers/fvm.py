"""Finite Volume Method — problem-agnostic cell-centered structured solver."""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import torch

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry


# ---------------------------------------------------------------------------
# Structured 2-D cell-centred grid
# ---------------------------------------------------------------------------

def _make_fvm_grid(
    nx: int, ny: int,
    bounds: Dict[str, Tuple[float, float]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    Returns:
      xc (nx,ny), yc (nx,ny) — cell centres
      dx, dy — cell widths
    """
    x0, x1 = bounds.get("x", (0.0, 1.0))
    y0, y1 = bounds.get("y", (0.0, 1.0))
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny
    xs = torch.linspace(x0 + 0.5 * dx, x1 - 0.5 * dx, nx, device=device)
    ys = torch.linspace(y0 + 0.5 * dy, y1 - 0.5 * dy, ny, device=device)
    xc, yc = torch.meshgrid(xs, ys, indexing="ij")
    return xc, yc, dx, dy


# ---------------------------------------------------------------------------
# FVM kernels
# ---------------------------------------------------------------------------

def _fvm_diffusion_2d(
    u: torch.Tensor,
    dx: float,
    dy: float,
    alpha: float,
    source: float,
    dt: float,
    bc_val: float,
) -> torch.Tensor:
    """
    Explicit finite volume for  du/dt = alpha * Δu + source.
    Ghost-cell Dirichlet at all four walls (value = bc_val).
    u: (nx, ny)
    """
    ug = torch.zeros(u.shape[0] + 2, u.shape[1] + 2, device=u.device)
    ug[1:-1, 1:-1] = u
    # Dirichlet ghosts
    ug[0, :] = 2 * bc_val - ug[1, :]
    ug[-1, :] = 2 * bc_val - ug[-2, :]
    ug[:, 0] = 2 * bc_val - ug[:, 1]
    ug[:, -1] = 2 * bc_val - ug[:, -2]

    lap = (
        (ug[2:, 1:-1] - 2 * ug[1:-1, 1:-1] + ug[:-2, 1:-1]) / dx ** 2
        + (ug[1:-1, 2:] - 2 * ug[1:-1, 1:-1] + ug[1:-1, :-2]) / dy ** 2
    )
    return u + dt * (alpha * lap + source)


def _fvm_convdiff_2d(
    u: torch.Tensor,
    dx: float,
    dy: float,
    vx: float,
    vy: float,
    nu: float,
    source: float,
    dt: float,
) -> torch.Tensor:
    """
    Explicit FVM for  du/dt + vx*du/dx + vy*du/dy = nu*Δu + source.
    Upwind advection + central diffusion.
    Periodic BCs for advection; zero-gradient for diffusion.
    """
    # Pad for upwind (wrap for advection, replicate for diffusion)
    ux = torch.nn.functional.pad(u.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate")[0, 0]

    # Upwind x
    if vx > 0:
        adv_x = vx * (ux[1:-1, 1:-1] - ux[:-2, 1:-1]) / dx
    else:
        adv_x = vx * (ux[2:, 1:-1] - ux[1:-1, 1:-1]) / dx

    # Upwind y
    if vy > 0:
        adv_y = vy * (ux[1:-1, 1:-1] - ux[1:-1, :-2]) / dy
    else:
        adv_y = vy * (ux[1:-1, 2:] - ux[1:-1, 1:-1]) / dy

    # Central diffusion
    diff = nu * (
        (ux[2:, 1:-1] - 2 * ux[1:-1, 1:-1] + ux[:-2, 1:-1]) / dx ** 2
        + (ux[1:-1, 2:] - 2 * ux[1:-1, 1:-1] + ux[1:-1, :-2]) / dy ** 2
    )

    return u + dt * (-adv_x - adv_y + diff + source)


def _fvm_burgers_2d(
    u: torch.Tensor,
    dx: float,
    dy: float,
    nu: float,
    dt: float,
) -> torch.Tensor:
    """
    Explicit FVM for viscous Burgers: du/dt + u*du/dx = nu*d²u/dx² (2-D, isotropic).
    """
    ux = torch.nn.functional.pad(u.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate")[0, 0]
    u_c = ux[1:-1, 1:-1]

    # Self-advection upwind
    adv_x = torch.where(
        u_c > 0,
        u_c * (u_c - ux[:-2, 1:-1]) / dx,
        u_c * (ux[2:, 1:-1] - u_c) / dx,
    )
    adv_y = torch.where(
        u_c > 0,
        u_c * (u_c - ux[1:-1, :-2]) / dy,
        u_c * (ux[1:-1, 2:] - u_c) / dy,
    )
    diff = nu * (
        (ux[2:, 1:-1] - 2 * u_c + ux[:-2, 1:-1]) / dx ** 2
        + (ux[1:-1, 2:] - 2 * u_c + ux[1:-1, :-2]) / dy ** 2
    )
    return u + dt * (-adv_x - adv_y + diff)


def _fvm_euler_explicit(
    u0: torch.Tensor,
    dx: float,
    dy: float,
    params: Dict[str, Any],
    nt: int,
    dt: float,
    kind: str,
) -> torch.Tensor:
    """Run explicit time loop; returns trajectory (nt+1, nx, ny)."""
    u = u0.clone()
    traj = [u]
    alpha  = float(params.get("alpha",  1.0))
    source = float(params.get("source", 0.0))
    bc_val = float(params.get("bc_val", 0.0))
    vx     = float(params.get("vx",     1.0))
    vy     = float(params.get("vy",     0.0))
    nu     = float(params.get("nu",     0.01))

    for _ in range(nt):
        if kind in ("heat", "diffusion"):
            u = _fvm_diffusion_2d(u, dx, dy, alpha, source, dt, bc_val)
        elif kind in ("convection_diffusion", "advection"):
            u = _fvm_convdiff_2d(u, dx, dy, vx, vy, nu, source, dt)
        elif kind == "burgers":
            u = _fvm_burgers_2d(u, dx, dy, nu, dt)
        else:
            # Default: pure diffusion
            u = _fvm_diffusion_2d(u, dx, dy, alpha, source, dt, bc_val)
        traj.append(u.clone())
    return torch.stack(traj, dim=0)  # (nt+1, nx, ny)


# ---------------------------------------------------------------------------
# CFL helpers
# ---------------------------------------------------------------------------

def _cfl_diffusion(dx: float, dy: float, alpha: float, safety: float = 0.4) -> float:
    return safety * min(dx, dy) ** 2 / (2.0 * alpha + 1e-14)


def _cfl_convection(dx: float, dy: float, vx: float, vy: float, nu: float, safety: float = 0.4) -> float:
    v_max = max(abs(vx), abs(vy), 1e-12)
    dt_adv  = safety * min(dx, dy) / v_max
    dt_diff = safety * min(dx, dy) ** 2 / (2.0 * nu + 1e-14)
    return min(dt_adv, dt_diff)


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

@SolverRegistry.register(
    name="fvm",
    family="pde",
    description="Finite Volume Method — problem-agnostic cell-centred 2-D solver with CFL-stable explicit time stepping.",
    tags=["fvm", "pde", "agnostic", "diffusion", "convection"],
)
class FVMSolver(SolverBase):
    """
    Problem-agnostic FVM solver on a structured cell-centred 2-D grid.

    Supported PDE kinds (spec.pde.kind):
      heat / diffusion           — du/dt = alpha·Δu + source
      convection_diffusion / advection — du/dt + v·∇u = nu·Δu + source
      burgers                    — du/dt + u·∇u = nu·Δu

    Params consumed from PDETermSpec.params:
      alpha  — diffusivity      (heat/diffusion, default 1.0)
      nu     — kinematic viscosity (convection/burgers, default 0.01)
      vx, vy — advection velocity (convection, default 1.0 / 0.0)
      source — uniform RHS scalar (default 0.0)
      bc_val — Dirichlet wall value (default 0.0)
      t_end  — total simulation time (overrides nt)

    If domain_bounds contains a "t" key it is used for t_end.
    Initial condition defaults to zero unless spec.solver_spec["u0"] provided.
    """

    def __init__(
        self,
        nx: int = 64,
        ny: int = 64,
        nt: int = 200,
        dt: Optional[float] = None,
        safety: float = 0.4,
    ):
        super().__init__()
        self.nx = int(nx)
        self.ny = int(ny)
        self.nt = int(nt)
        self.dt = dt          # None → CFL auto
        self.safety = float(safety)

    @classmethod
    def from_problem_spec(cls, spec, nx: int = 64, ny: int = 64, nt: int = 200, **kwargs) -> "FVMSolver":
        return cls(nx=nx, ny=ny, nt=nt, **kwargs)

    # ------------------------------------------------------------------
    def solve_from_spec(self, spec) -> SolverOutput:
        kind   = spec.pde.kind.lower().replace("-", "_").replace(" ", "_")
        bounds = dict(spec.domain_bounds)
        params = dict(spec.pde.params)
        device = torch.device("cpu")

        xc, yc, dx, dy = _make_fvm_grid(self.nx, self.ny, bounds, device)

        # Initial condition
        if "u0" in spec.solver_spec:
            u0_val = spec.solver_spec["u0"]
            if callable(u0_val):
                u0 = u0_val(xc, yc)
            else:
                u0 = torch.full((self.nx, self.ny), float(u0_val), device=device)
        else:
            u0 = torch.zeros(self.nx, self.ny, device=device)

        # Time step
        alpha = float(params.get("alpha", 1.0))
        nu    = float(params.get("nu",    0.01))
        vx    = float(params.get("vx",    1.0))
        vy    = float(params.get("vy",    0.0))

        if self.dt is None:
            if kind in ("heat", "diffusion"):
                dt = _cfl_diffusion(dx, dy, alpha, self.safety)
            else:
                dt = _cfl_convection(dx, dy, vx, vy, nu, self.safety)
        else:
            dt = float(self.dt)

        # Override nt from t_end if given
        t_end = params.get("t_end", None) or bounds.get("t", (None, None))[1]
        nt = int(t_end / dt) if t_end is not None else self.nt

        traj = _fvm_euler_explicit(u0, dx, dy, params, nt, dt, kind)

        t_vals = torch.linspace(0.0, dt * nt, nt + 1, device=device)
        return SolverOutput(
            result=traj,          # (nt+1, nx, ny)
            losses={"total": torch.tensor(0.0, device=device)},
            extras={
                "xc": xc, "yc": yc,
                "dx": dx, "dy": dy,
                "dt": dt, "nt": nt,
                "t": t_vals,
                "kind": kind,
            },
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        *,
        spec=None,
        mesh: Any = None,
        u0: Optional[torch.Tensor] = None,
        steps: int = 200,
        dt: float = 1e-3,
        topology: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        flux_fn=None,
        source_fn=None,
    ) -> SolverOutput:
        """
        Two call styles:
          forward(spec=problem_spec)                            # problem-agnostic
          forward(mesh=..., u0=..., steps=..., dt=..., ...)    # legacy callback-based
        """
        if spec is not None:
            return self.solve_from_spec(spec)

        # Legacy: user-supplied flux/source callbacks (old API)
        assert u0 is not None and topology is not None
        assert flux_fn is not None, "flux_fn required in legacy mode"
        u = u0.clone()
        faces = topology["faces"]
        areas = topology["areas"]
        vols  = topology["volumes"]
        traj  = [u]

        for _ in range(int(steps)):
            dudt = torch.zeros_like(u)
            for f_idx in range(faces.shape[0]):
                i = int(faces[f_idx, 0].item())
                j = int(faces[f_idx, 1].item())
                ui = u[i]
                uj = u[j] if j >= 0 else ui
                face_info = {k: topology[k][f_idx] for k in topology if k not in {"faces", "volumes"}}
                flux = flux_fn(ui, uj, face_info, params or {})
                Ai = areas[f_idx]
                dudt[i] -= (Ai * flux) / vols[i]
                if j >= 0:
                    dudt[j] += (Ai * flux) / vols[j]
            if source_fn is not None:
                for c in range(u.shape[0]):
                    dudt[c] += source_fn(u[c], {"volume": vols[c]}, params or {})
            u = u + float(dt) * dudt
            traj.append(u.clone())

        out = torch.stack(traj, dim=0)
        return SolverOutput(
            result=out,
            losses={"total": torch.tensor(0.0, device=u.device)},
            extras={"dt": dt, "steps": steps},
        )
