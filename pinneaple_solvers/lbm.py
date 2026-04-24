"""
Lattice Boltzmann Method — D2Q9 BGK + D3Q19 BGK.

Features
--------
D2Q9 (LBMSolver):
  - BGK collision with optional Smagorinsky LES (Hou 1994)
  - Zou-He velocity inlet (left wall) / pressure outlet (right wall)
  - Full-way bounce-back for arbitrary solid obstacles + top/bottom no-slip walls
  - Reynolds-number → omega derivation in lattice units
  - Trajectory output (rho, ux, uy) at configurable intervals
  - from_problem_spec() / solve_from_spec() integration

D3Q19 (LBMSolver3D):
  - BGK, periodic BCs, trajectory output (rho, ux, uy, uz)

References
----------
Zou & He (1997) — Zou-He boundary conditions
Hou et al. (1994) — Smagorinsky LES-LBM
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry


# ===========================================================================
# D2Q9 constants (module-level singletons, moved to device lazily)
# ===========================================================================

# Velocity vectors   0  1  2   3   4   5   6   7   8
_C2Q9_X = [          0, 1, 0, -1,  0,  1, -1, -1,  1]
_C2Q9_Y = [          0, 0, 1,  0, -1,  1,  1, -1, -1]
# Opposite direction indices
_OPP_D2Q9 = [0, 3, 4, 1, 2, 7, 8, 5, 6]
# Weights
_W_D2Q9 = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]


def _d2q9_tensors(device):
    cx  = torch.tensor(_C2Q9_X, dtype=torch.float32, device=device)
    cy  = torch.tensor(_C2Q9_Y, dtype=torch.float32, device=device)
    w   = torch.tensor(_W_D2Q9, dtype=torch.float32, device=device)
    opp = torch.tensor(_OPP_D2Q9, dtype=torch.long, device=device)
    return cx, cy, w, opp


# ===========================================================================
# Core D2Q9 functions
# ===========================================================================

def _macroscopic_2d(f, cx, cy):
    """Compute rho, ux, uy from distribution f (9, nx, ny)."""
    rho = f.sum(0)
    ux  = (f * cx.view(9, 1, 1)).sum(0) / rho.clamp(min=1e-10)
    uy  = (f * cy.view(9, 1, 1)).sum(0) / rho.clamp(min=1e-10)
    return rho, ux, uy


def _equilibrium_2d(rho, ux, uy, cx, cy, w):
    """Maxwell-Boltzmann equilibrium distribution (9, nx, ny)."""
    cu  = cx.view(9,1,1)*ux + cy.view(9,1,1)*uy   # (9,nx,ny)
    u2  = ux**2 + uy**2
    feq = w.view(9,1,1) * rho * (1.0 + 3.0*cu + 4.5*cu**2 - 1.5*u2)
    return feq


def _stream_2d(f, cx, cy):
    """Streaming step: shift each distribution by its velocity vector."""
    f_out = torch.empty_like(f)
    for i in range(9):
        dx = int(cx[i].item())
        dy = int(cy[i].item())
        f_out[i] = torch.roll(f[i], shifts=(dx, dy), dims=(0, 1))
    return f_out


def _zou_he_inlet(f, u_in, u_y_in=0.0):
    """
    Zou-He velocity BC at left wall (x=0).
    Prescribes ux=u_in, uy=u_y_in on the x=0 slice.
    Modifies f in-place and returns rho at inlet.
    """
    # known directions: 0, 2, 3, 4, 6, 7
    # unknown (came from outside): 1 (E), 5 (NE), 8 (SE)
    rho0 = (f[0, 0, :] + f[2, 0, :] + f[4, 0, :] +
            2.0*(f[3, 0, :] + f[6, 0, :] + f[7, 0, :])) / (1.0 - u_in)
    f[1, 0, :] = f[3, 0, :] + (2.0/3.0)*rho0*u_in
    f[5, 0, :] = f[7, 0, :] + (1.0/6.0)*rho0*u_in - 0.5*(f[2, 0, :] - f[4, 0, :]) + (2.0/3.0)*rho0*u_y_in
    f[8, 0, :] = f[6, 0, :] + (1.0/6.0)*rho0*u_in + 0.5*(f[2, 0, :] - f[4, 0, :]) - (2.0/3.0)*rho0*u_y_in
    return rho0


def _zou_he_outlet(f, rho_out=1.0):
    """
    Zou-He pressure BC at right wall (x=nx-1).
    Prescribes rho=rho_out; extrapolates u_x.

    Derivation (Zou & He 1997):
      rho = sum_i f_i  →  rho*u_x = f[1]+f[5]+f[8] - f[3]-f[6]-f[7]
      Known: f[0],f[1],f[2],f[4],f[5],f[8]  →  S = f[0]+f[2]+f[4] + 2*(f[1]+f[5]+f[8])
      u_x = S/rho_out - 1
    """
    # unknown (came from outside): 3 (W), 6 (NW), 7 (SW)
    u_x = (f[0, -1, :] + f[2, -1, :] + f[4, -1, :] +
           2.0*(f[1, -1, :] + f[5, -1, :] + f[8, -1, :])) / rho_out - 1.0
    f[3, -1, :] = f[1, -1, :] - (2.0/3.0)*rho_out*u_x
    f[7, -1, :] = f[5, -1, :] - (1.0/6.0)*rho_out*u_x + 0.5*(f[2, -1, :] - f[4, -1, :])
    f[6, -1, :] = f[8, -1, :] - (1.0/6.0)*rho_out*u_x - 0.5*(f[2, -1, :] - f[4, -1, :])


def _bounce_back_2d(f_post, f_streamed, solid, opp):
    """
    Full-way bounce-back at solid nodes.
    At each solid cell: incoming streamed population = pre-streaming opposite.
    """
    for i in range(9):
        f_streamed[i][solid] = f_post[opp[i]][solid]
    return f_streamed


def _smagorinsky_omega(f, feq, rho, omega0, Cs, cx, cy):
    """
    Hou (1994) LES-LBM: locally adjusts omega using Smagorinsky model.
    Returns omega field (nx, ny).
    """
    cs2   = 1.0 / 3.0
    tau0  = 1.0 / omega0
    f_neq = f - feq

    Pi_xx = (f_neq * cx.view(9,1,1)**2).sum(0)
    Pi_xy = (f_neq * cx.view(9,1,1) * cy.view(9,1,1)).sum(0)
    Pi_yy = (f_neq * cy.view(9,1,1)**2).sum(0)

    # |Π^neq|_F = sqrt(Pxx^2 + 2*Pxy^2 + Pyy^2)
    Pi_norm = torch.sqrt(Pi_xx**2 + 2.0*Pi_xy**2 + Pi_yy**2)

    # Effective tau from Hou (1994): tau_eff = tau0/2 + sqrt((tau0/2)^2 + Cs^2*sqrt(2)*|S|)
    # |S| estimated from non-equilibrium: |S| ≈ Pi_norm / (2*rho*cs2*tau0)
    S_mag  = Pi_norm / (2.0 * rho.clamp(1e-8) * cs2 * tau0)
    tau_eff = 0.5*tau0 + torch.sqrt((0.5*tau0)**2 + Cs**2 * 1.4142 * S_mag)
    return 1.0 / tau_eff.clamp(min=0.5001)   # stability: tau > 0.5


def _lbm_step_2d(
    f: torch.Tensor,
    omega: float,
    solid: Optional[torch.Tensor],
    cx: torch.Tensor,
    cy: torch.Tensor,
    w: torch.Tensor,
    opp: torch.Tensor,
    u_in: Optional[float] = None,
    rho_out: Optional[float] = None,
    Cs: float = 0.0,
    top_bottom_bounce: bool = True,
) -> torch.Tensor:
    """
    One complete D2Q9 BGK-LBM step.

    Order: collision → streaming → bounce-back → Zou-He BCs
    """
    # --- Macroscopic ---
    rho, ux, uy = _macroscopic_2d(f, cx, cy)
    feq = _equilibrium_2d(rho, ux, uy, cx, cy, w)

    # --- Collision (BGK, optionally LES) ---
    if Cs > 0.0:
        omega_field = _smagorinsky_omega(f, feq, rho, omega, Cs, cx, cy)
        f_post = f - omega_field * (f - feq)
    else:
        f_post = f - omega * (f - feq)

    # --- Streaming ---
    f_new = _stream_2d(f_post, cx, cy)

    # --- Top/bottom no-slip (y=0 and y=ny-1) via bounce-back ---
    if top_bottom_bounce:
        # Bottom wall y=0: dirs with cy>0 bounce back
        for i, dcy in enumerate(_C2Q9_Y):
            if dcy > 0:
                f_new[i, :, 0] = f_post[_OPP_D2Q9[i], :, 0]
        # Top wall y=ny-1: dirs with cy<0 bounce back
        for i, dcy in enumerate(_C2Q9_Y):
            if dcy < 0:
                f_new[i, :, -1] = f_post[_OPP_D2Q9[i], :, -1]

    # --- Obstacle bounce-back ---
    if solid is not None:
        f_new = _bounce_back_2d(f_post, f_new, solid, opp)

    # --- Zou-He BCs ---
    if u_in is not None:
        _zou_he_inlet(f_new, u_in)
    if rho_out is not None:
        _zou_he_outlet(f_new, rho_out)

    return f_new


# ===========================================================================
# D3Q19 constants
# ===========================================================================

_C3Q19 = [
    [0, 0, 0],
    [1, 0, 0], [-1,  0,  0], [0,  1,  0], [0, -1,  0], [0,  0,  1], [0,  0, -1],
    [1, 1, 0], [-1,  1,  0], [1, -1,  0], [-1,-1,  0],
    [1, 0, 1], [-1,  0,  1], [1,  0, -1], [-1, 0, -1],
    [0, 1, 1], [0,  -1,  1], [0,  1, -1], [0, -1, -1],
]
_W3Q19 = [
    1/3,
    1/18, 1/18, 1/18, 1/18, 1/18, 1/18,
    1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36,
]
_OPP_D3Q19 = [0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15]


def _d3q19_tensors(device):
    c = torch.tensor(_C3Q19, dtype=torch.float32, device=device)
    w = torch.tensor(_W3Q19, dtype=torch.float32, device=device)
    opp = torch.tensor(_OPP_D3Q19, dtype=torch.long, device=device)
    return c, w, opp


# ===========================================================================
# LBMSolver (D2Q9, full-featured)
# ===========================================================================

@SolverRegistry.register(
    name="lbm",
    family="pde",
    description="Lattice Boltzmann D2Q9 BGK with Zou-He BCs, bounce-back obstacles, optional Smagorinsky LES.",
    tags=["lbm", "fluids", "navier_stokes", "cfd"],
)
class LBMSolver(SolverBase):
    """
    D2Q9 Lattice Boltzmann solver.

    Lattice units: dx=dy=dt=1, cs²=1/3.
    Physical Re is mapped to omega via:  nu = u_in * L / Re,  omega = 1/(3*nu + 0.5)

    Parameters
    ----------
    nx, ny       : grid size
    Re           : Reynolds number (sets omega)
    u_in         : inlet velocity (lattice units, typically 0.01–0.1)
    rho_out      : outlet density (default 1.0)
    obstacle_mask: bool tensor (nx, ny), True = solid cell
    Cs           : Smagorinsky constant (0 = pure BGK; 0.1–0.18 for turbulence)
    """

    def __init__(
        self,
        nx:           int = 128,
        ny:           int = 64,
        Re:           float = 100.0,
        u_in:         float = 0.05,
        rho_out:      float = 1.0,
        obstacle_mask: Optional[torch.Tensor] = None,
        Cs:           float = 0.0,
    ):
        super().__init__()
        self.nx      = nx
        self.ny      = ny
        self.Re      = Re
        self.u_in    = u_in
        self.rho_out = rho_out
        self.Cs      = Cs

        # Viscosity from Reynolds number (characteristic length = ny-2 for channel)
        L     = float(ny - 2)
        nu    = u_in * L / Re
        tau   = 3.0 * nu + 0.5
        if tau <= 0.5:
            raise ValueError(
                f"tau={tau:.4f} ≤ 0.5 → unstable. Reduce u_in or increase Re. "
                f"Hint: try u_in=0.05, Re<{int(u_in*(ny-2)/0.01):d}."
            )
        # Warn if Ma > 0.3 (compressibility errors become significant)
        Ma = u_in * (3.0 ** 0.5)
        if Ma > 0.3:
            import warnings
            warnings.warn(f"Ma={Ma:.3f} > 0.3 — LBM compressibility errors will be significant. Reduce u_in.")
        # Warn if omega close to 2 (near stability limit)
        if tau < 0.6:
            import warnings
            warnings.warn(f"tau={tau:.4f} (omega={1/tau:.3f}) is close to the stability limit. "
                          f"Consider using Smagorinsky LES (Cs>0) or increasing Re/ny.")
        self.omega = float(1.0 / tau)

        if obstacle_mask is not None:
            self.register_buffer("solid", obstacle_mask.bool())
        else:
            self.solid = None

    # ------------------------------------------------------------------
    @classmethod
    def from_problem_spec(cls, spec) -> "LBMSolver":
        """Build from a ProblemSpec (pde.kind ∈ {'lbm','navier_stokes','channel_flow'})."""
        p  = spec.pde.params if hasattr(spec.pde, "params") else {}
        db = spec.domain_bounds if hasattr(spec, "domain_bounds") else {}
        nx = int(p.get("nx", 128));  ny = int(p.get("ny", 64))
        return cls(
            nx=nx, ny=ny,
            Re=float(p.get("Re", 100.0)),
            u_in=float(p.get("u_in", 0.05)),
            rho_out=float(p.get("rho_out", 1.0)),
            Cs=float(p.get("Cs", 0.0)),
        )

    # ------------------------------------------------------------------
    def _init_f(self, device: torch.device) -> torch.Tensor:
        """Equilibrium initialisation with rho=1, u=(u_in, 0) everywhere."""
        cx, cy, w, _ = _d2q9_tensors(device)
        rho = torch.ones(self.nx, self.ny, device=device)
        ux  = torch.full((self.nx, self.ny), self.u_in, device=device)
        uy  = torch.zeros(self.nx, self.ny, device=device)
        return _equilibrium_2d(rho, ux, uy, cx, cy, w)

    # ------------------------------------------------------------------
    def solve_from_spec(self, spec, steps: int = 5000, save_every: int = 500) -> SolverOutput:
        """Run from a ProblemSpec and return SolverOutput with flow fields."""
        p = spec.pde.params if hasattr(spec.pde, "params") else {}
        return self.forward(
            steps=int(p.get("steps", steps)),
            save_every=int(p.get("save_every", save_every)),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        f0:          Optional[torch.Tensor] = None,
        *,
        steps:       int = 5000,
        save_every:  int = 500,
    ) -> SolverOutput:
        """
        Run the LBM simulation.

        Parameters
        ----------
        f0         : initial distribution (9, nx, ny); None → equilibrium init
        steps      : total timesteps
        save_every : save macroscopic fields every N steps

        Returns
        -------
        SolverOutput
          result          : final f (9, nx, ny)
          extras['rho']   : final density (nx, ny)
          extras['ux']    : final x-velocity (nx, ny)
          extras['uy']    : final y-velocity (nx, ny)
          extras['trajectory_rho/ux/uy'] : list of saved snapshots
        """
        dev = next(self.parameters(), torch.tensor(0.0)).device if len(list(self.parameters())) else torch.device("cpu")
        # Infer device from solid mask if available, else CPU
        if self.solid is not None:
            dev = self.solid.device
        else:
            dev = torch.device("cpu")

        cx, cy, w, opp = _d2q9_tensors(dev)
        solid = self.solid.to(dev) if self.solid is not None else None

        f = f0.to(dev) if f0 is not None else self._init_f(dev)

        traj_rho, traj_ux, traj_uy = [], [], []

        for step in range(steps):
            f = _lbm_step_2d(
                f, self.omega, solid, cx, cy, w, opp,
                u_in=self.u_in, rho_out=self.rho_out, Cs=self.Cs,
            )
            if (step + 1) % save_every == 0:
                rho, ux, uy = _macroscopic_2d(f, cx, cy)
                traj_rho.append(rho.cpu())
                traj_ux.append(ux.cpu())
                traj_uy.append(uy.cpu())

        rho, ux, uy = _macroscopic_2d(f, cx, cy)
        return SolverOutput(
            result=f,
            losses={},
            extras={
                "rho": rho.cpu(),
                "ux":  ux.cpu(),
                "uy":  uy.cpu(),
                "vel_mag": torch.sqrt(ux**2 + uy**2).cpu(),
                "trajectory_rho": traj_rho,
                "trajectory_ux":  traj_ux,
                "trajectory_uy":  traj_uy,
                "omega": self.omega,
                "Re":    self.Re,
            },
        )


# ===========================================================================
# Obstacle factory helpers
# ===========================================================================

def cylinder_mask(nx: int, ny: int, cx: float, cy: float, r: float) -> torch.Tensor:
    """Create a circular obstacle mask (True = solid). cx, cy, r in lattice units."""
    X = torch.arange(nx).float().view(-1, 1).expand(nx, ny)
    Y = torch.arange(ny).float().view(1, -1).expand(nx, ny)
    return ((X - cx)**2 + (Y - cy)**2) <= r**2


def rectangle_mask(nx: int, ny: int, x0: int, x1: int, y0: int, y1: int) -> torch.Tensor:
    """Rectangular obstacle mask."""
    mask = torch.zeros(nx, ny, dtype=torch.bool)
    mask[x0:x1, y0:y1] = True
    return mask


def airfoil_naca_mask(nx: int, ny: int, chord: int, aoa_deg: float = 0.0,
                      naca: str = "0012") -> torch.Tensor:
    """
    Approximate NACA 4-digit airfoil mask centred in the domain.
    chord: chord length in lattice units.
    """
    import math
    t   = int(naca[-2:]) / 100.0  # max thickness as fraction of chord
    cx0 = nx // 4;  cy0 = ny // 2
    alpha = math.radians(aoa_deg)
    X = torch.arange(nx).float().view(-1, 1).expand(nx, ny)
    Y = torch.arange(ny).float().view(1, -1).expand(nx, ny)
    # Rotate about leading edge
    dX = X - cx0;  dY = Y - cy0
    xr =  dX*math.cos(alpha) + dY*math.sin(alpha)
    yr = -dX*math.sin(alpha) + dY*math.cos(alpha)
    xn = xr / chord  # normalised [0, 1]
    # NACA thickness profile
    yt = 5*t*(0.2969*xn**0.5 - 0.1260*xn - 0.3516*xn**2 + 0.2843*xn**3 - 0.1015*xn**4)
    yt_grid = yt * chord
    mask = (xn >= 0) & (xn <= 1) & (torch.abs(yr) <= yt_grid)
    return mask


# ===========================================================================
# LBMSolver3D (D3Q19, BGK, periodic BCs)
# ===========================================================================

@SolverRegistry.register(
    name="lbm_3d",
    family="pde",
    description="Lattice Boltzmann D3Q19 BGK with periodic BCs.",
    tags=["lbm", "fluids", "3d"],
)
class LBMSolver3D(SolverBase):
    """
    D3Q19 Lattice Boltzmann solver with periodic boundary conditions.

    Parameters
    ----------
    nx, ny, nz  : grid dimensions
    Re          : Reynolds number
    u_in        : mean inlet velocity
    """

    def __init__(
        self,
        nx:   int   = 32,
        ny:   int   = 32,
        nz:   int   = 32,
        Re:   float = 100.0,
        u_in: float = 0.05,
    ):
        super().__init__()
        self.nx = nx; self.ny = ny; self.nz = nz
        L    = float(nx)
        nu   = u_in * L / Re
        tau  = 3.0 * nu + 0.5
        self.omega = float(1.0 / max(tau, 0.5001))
        self.u_in  = u_in

    def _init_f3(self, device):
        c, w, _ = _d3q19_tensors(device)
        Q = 19
        rho = torch.ones(self.nx, self.ny, self.nz, device=device)
        ux  = torch.full_like(rho, self.u_in)
        uy  = torch.zeros_like(rho)
        uz  = torch.zeros_like(rho)
        # feq: Q-directions × nx × ny × nz
        cu  = (c[:, 0].view(Q,1,1,1)*ux + c[:,1].view(Q,1,1,1)*uy +
               c[:,2].view(Q,1,1,1)*uz)
        u2  = ux**2 + uy**2 + uz**2
        feq = w.view(Q,1,1,1) * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)
        return feq

    def forward(
        self,
        f0:         Optional[torch.Tensor] = None,
        *,
        steps:      int = 1000,
        save_every: int = 100,
    ) -> SolverOutput:
        dev = torch.device("cpu")
        c, w, opp = _d3q19_tensors(dev)
        Q = 19
        f = f0.to(dev) if f0 is not None else self._init_f3(dev)

        traj_ux, traj_uy, traj_uz = [], [], []
        cx, cy, cz = c[:,0], c[:,1], c[:,2]

        for step in range(steps):
            # Macroscopic
            rho = f.sum(0)
            ux  = (f * cx.view(Q,1,1,1)).sum(0) / rho.clamp(1e-10)
            uy  = (f * cy.view(Q,1,1,1)).sum(0) / rho.clamp(1e-10)
            uz  = (f * cz.view(Q,1,1,1)).sum(0) / rho.clamp(1e-10)

            # Equilibrium
            cu  = cx.view(Q,1,1,1)*ux + cy.view(Q,1,1,1)*uy + cz.view(Q,1,1,1)*uz
            u2  = ux**2 + uy**2 + uz**2
            feq = w.view(Q,1,1,1) * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)

            # Collision
            f = f - self.omega * (f - feq)

            # Streaming (periodic)
            f_new = torch.empty_like(f)
            for i in range(Q):
                dx, dy, dz = int(cx[i].item()), int(cy[i].item()), int(cz[i].item())
                f_new[i] = torch.roll(f[i], shifts=(dx, dy, dz), dims=(0, 1, 2))
            f = f_new

            if (step + 1) % save_every == 0:
                traj_ux.append(ux.cpu())
                traj_uy.append(uy.cpu())
                traj_uz.append(uz.cpu())

        rho = f.sum(0)
        ux  = (f * cx.view(Q,1,1,1)).sum(0) / rho.clamp(1e-10)
        uy  = (f * cy.view(Q,1,1,1)).sum(0) / rho.clamp(1e-10)
        uz  = (f * cz.view(Q,1,1,1)).sum(0) / rho.clamp(1e-10)

        return SolverOutput(
            result=f,
            losses={},
            extras={
                "rho": rho.cpu(), "ux": ux.cpu(), "uy": uy.cpu(), "uz": uz.cpu(),
                "vel_mag": torch.sqrt(ux**2 + uy**2 + uz**2).cpu(),
                "trajectory_ux": traj_ux,
                "trajectory_uy": traj_uy,
                "trajectory_uz": traj_uz,
            },
        )
