"""Physics trajectory simulator.

:class:`PhysicsSimulator` generates temporal field trajectories for a given
:class:`~pinneaple_worldmodel.scenario.PhysicsScenario`.  It tries each
available backend in priority order:

1. **pinneaple_solvers** (``fdm``, ``lbm``, ``fem``, ``sph``) — uses the
   registered solver for the scenario's PDE kind.
2. **builtin** — pure-PyTorch finite-difference solvers for ``heat``,
   ``burgers``, ``wave``, ``advection``, ``ns2d``.  Always available, no
   extra dependencies.

The output of each run is a :class:`TrajectoryData` containing the full
field sequence ``(T+1, C, *grid_shape)`` and the parameter dict used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .scenario import PhysicsScenario


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryData:
    """One physics trajectory: sequence of field snapshots + metadata.

    Attributes
    ----------
    states : Tensor ``(T+1, C, *grid_shape)``
        All snapshots including the initial condition at t=0.
    params : dict
        PDE parameter values used for this trajectory.
    scenario_name : str
    metadata : dict
        Solver info, wall-clock time, etc.
    """
    states: torch.Tensor
    params: Dict[str, float]
    scenario_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return self.states.shape[0] - 1

    @property
    def n_fields(self) -> int:
        return self.states.shape[1]

    def transitions(self, horizon: int = 1) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Return all ``(state_t, state_{t+horizon})`` pairs."""
        out = []
        T = self.states.shape[0]
        for t in range(T - horizon):
            out.append((self.states[t], self.states[t + horizon]))
        return out


# ---------------------------------------------------------------------------
# PhysicsSimulator
# ---------------------------------------------------------------------------

class PhysicsSimulator:
    """Generate physics trajectories from a :class:`PhysicsScenario`.

    Parameters
    ----------
    scenario : PhysicsScenario
    device : str — compute device for builtin solvers.
    verbose : bool — log solver details.
    """

    def __init__(
        self,
        scenario: PhysicsScenario,
        *,
        device: str = "cpu",
        verbose: bool = False,
    ) -> None:
        self.scenario = scenario
        self.device = torch.device(device)
        self.verbose = verbose
        self._solver_fn = self._resolve_solver()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_trajectory(
        self,
        params: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
    ) -> TrajectoryData:
        """Generate one trajectory with the given (or randomly sampled) params.

        Parameters
        ----------
        params : dict or None
            PDE parameter overrides.  Missing keys are sampled from
            ``scenario.param_ranges``.
        seed : optional RNG seed for reproducibility.

        Returns
        -------
        TrajectoryData
        """
        if seed is not None:
            torch.manual_seed(seed)

        resolved_params = self._sample_params(params)
        states = self._solver_fn(resolved_params, seed)
        return TrajectoryData(
            states=states,
            params=resolved_params,
            scenario_name=self.scenario.name,
            metadata={"solver": self.scenario.solver, "device": str(self.device)},
        )

    def generate_batch(
        self,
        n_samples: int,
        *,
        base_seed: int = 0,
    ) -> List[TrajectoryData]:
        """Generate *n_samples* trajectories with random parameters.

        Parameters
        ----------
        n_samples : int
        base_seed : int — each trajectory uses seed ``base_seed + i``.

        Returns
        -------
        list of TrajectoryData
        """
        trajectories = []
        for i in range(n_samples):
            traj = self.generate_trajectory(seed=base_seed + i)
            trajectories.append(traj)
            if self.verbose and (i + 1) % max(1, n_samples // 10) == 0:
                print(f"  [{self.scenario.name}] generated {i + 1}/{n_samples}")
        return trajectories

    # ------------------------------------------------------------------
    # Solver resolution
    # ------------------------------------------------------------------

    def _resolve_solver(self):
        """Return the solver callable for this scenario's solver type."""
        kind = self.scenario.solver
        pde = self.scenario.pde_kind

        if kind != "builtin":
            fn = self._try_pinneaple_solver(kind, pde)
            if fn is not None:
                return fn
            if self.verbose:
                print(f"  [WorldModel] pinneaple_solvers.{kind} unavailable, "
                      f"falling back to builtin for '{pde}'.")

        return self._builtin_solver(pde)

    def _try_pinneaple_solver(self, kind: str, pde: str):
        """Try to build a solver using pinneaple_solvers. Returns None on failure."""
        try:
            if kind == "fdm":
                return self._make_fdm_solver(pde)
            if kind == "lbm":
                return self._make_lbm_solver(pde)
            if kind == "fem":
                return self._make_fem_solver(pde)
        except Exception:
            pass
        return None

    def _make_fdm_solver(self, pde: str):
        """Build an FDM-backed solver callable."""
        from pinneaple_simulation.numerical_solvers.fdm import FDMSolver  # type: ignore

        sc = self.scenario

        def _run(params: Dict[str, float], seed: Optional[int]) -> torch.Tensor:
            if seed is not None:
                torch.manual_seed(seed)
            cfg = {
                "pde_kind": pde,
                "grid_shape": sc.grid_shape,
                "dt": sc.dt,
                "n_steps": sc.n_steps,
                "domain_bounds": sc.domain_bounds,
                "bc_type": sc.bc_type,
                **params,
            }
            solver = FDMSolver(cfg)
            u0 = _make_ic(sc, seed)
            result = solver.solve(u0)
            # SolverOutput.result shape depends on solver; normalise to (T+1, C, *grid)
            return _normalise_trajectory(result.result, sc)

        return _run

    def _make_lbm_solver(self, pde: str):
        """Build an LBM-backed solver callable."""
        from pinneaple_simulation.numerical_solvers.lbm import LBMSolver  # type: ignore

        sc = self.scenario

        def _run(params: Dict[str, float], seed: Optional[int]) -> torch.Tensor:
            if seed is not None:
                torch.manual_seed(seed)
            Re = params.get("Re", 400.0)
            nu = 1.0 / Re if Re > 0 else 0.01
            solver = LBMSolver(grid_shape=sc.grid_shape, nu=nu,
                               n_steps=sc.n_steps, bc_type=sc.bc_type)
            result = solver.solve()
            return _normalise_trajectory(result.result, sc)

        return _run

    def _make_fem_solver(self, pde: str):
        """Build an FEM-backed solver callable."""
        from pinneaple_simulation.numerical_solvers.fem import FEMSolver  # type: ignore

        sc = self.scenario

        def _run(params: Dict[str, float], seed: Optional[int]) -> torch.Tensor:
            if seed is not None:
                torch.manual_seed(seed)
            solver = FEMSolver(grid_shape=sc.grid_shape, params=params,
                               n_steps=sc.n_steps, pde_kind=pde)
            result = solver.solve(_make_ic(sc, seed))
            return _normalise_trajectory(result.result, sc)

        return _run

    # ------------------------------------------------------------------
    # Builtin pure-PyTorch solvers
    # ------------------------------------------------------------------

    def _builtin_solver(self, pde: str):
        """Return the builtin solver for the given PDE kind."""
        _map = {
            "heat":      self._solve_heat,
            "burgers":   self._solve_burgers,
            "wave":      self._solve_wave,
            "advection": self._solve_advection,
            "ns2d":      self._solve_ns2d,
            "elasticity": self._solve_heat,   # placeholder — same diffusion structure
        }
        fn = _map.get(pde)
        if fn is None:
            raise ValueError(
                f"No builtin solver for PDE kind '{pde}'. "
                f"Choose from: {sorted(_map)}"
            )
        return fn

    # -- Heat equation: ∂T/∂t = α∇²T ----------------------------------------

    def _solve_heat(
        self, params: Dict[str, float], seed: Optional[int]
    ) -> torch.Tensor:
        sc = self.scenario
        alpha = params.get("alpha", 0.01)
        dev = self.device
        u = _make_ic(sc, seed).to(dev)  # (C, *grid)
        dt = sc.dt
        states = [u.cpu()]

        for _ in range(sc.n_steps):
            lap = _laplacian(u, sc.bc_type)
            u = u + dt * alpha * lap
            states.append(u.cpu())

        return torch.stack(states, dim=0)  # (T+1, C, *grid)

    # -- Viscous Burgers: ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x² ----------------------

    def _solve_burgers(
        self, params: Dict[str, float], seed: Optional[int]
    ) -> torch.Tensor:
        sc = self.scenario
        nu = params.get("nu", 0.01)
        dev = self.device
        u = _make_ic(sc, seed).to(dev)  # (1, Nx)
        dt = sc.dt
        dx = (sc.domain_bounds[0][1] - sc.domain_bounds[0][0]) / sc.grid_shape[0]
        states = [u.cpu()]

        for _ in range(sc.n_steps):
            # Upwind advection + central diffusion
            u_right = torch.roll(u, -1, dims=-1)
            u_left  = torch.roll(u,  1, dims=-1)
            adv = torch.where(u >= 0,
                              (u - u_left) / dx,
                              (u_right - u) / dx)
            diff = (u_right - 2 * u + u_left) / (dx ** 2)
            u = u + dt * (-u * adv + nu * diff)
            if sc.bc_type == "dirichlet_zero":
                u[..., 0] = 0.0
                u[..., -1] = 0.0
            states.append(u.cpu())

        return torch.stack(states, dim=0)

    # -- Wave equation: ∂²u/∂t² = c²∇²u (Leapfrog) --------------------------

    def _solve_wave(
        self, params: Dict[str, float], seed: Optional[int]
    ) -> torch.Tensor:
        sc = self.scenario
        c = params.get("c", 1.0)
        dev = self.device
        u = _make_ic(sc, seed).to(dev)
        # zero initial velocity
        u_prev = u.clone()
        dt = sc.dt
        states = [u.cpu()]

        for _ in range(sc.n_steps):
            lap = _laplacian(u, sc.bc_type)
            u_next = 2 * u - u_prev + (c * dt) ** 2 * lap
            u_prev, u = u, u_next
            if sc.bc_type == "dirichlet_zero":
                u = _apply_dirichlet_zero(u)
            states.append(u.cpu())

        return torch.stack(states, dim=0)

    # -- Linear advection: ∂φ/∂t + v·∇φ = 0 (upwind) -------------------------

    def _solve_advection(
        self, params: Dict[str, float], seed: Optional[int]
    ) -> torch.Tensor:
        sc = self.scenario
        vx = params.get("vx", 0.5)
        vy = params.get("vy", 0.5) if sc.spatial_dim >= 2 else 0.0
        dev = self.device
        u = _make_ic(sc, seed).to(dev)
        dt = sc.dt
        Lx = sc.domain_bounds[0][1] - sc.domain_bounds[0][0]
        dx = Lx / sc.grid_shape[0]
        states = [u.cpu()]

        for _ in range(sc.n_steps):
            if vx >= 0:
                flux_x = vx * (u - torch.roll(u, 1, dims=-1)) / dx
            else:
                flux_x = vx * (torch.roll(u, -1, dims=-1) - u) / dx

            if sc.spatial_dim >= 2:
                Ly = sc.domain_bounds[1][1] - sc.domain_bounds[1][0]
                dy = Ly / sc.grid_shape[1]
                if vy >= 0:
                    flux_y = vy * (u - torch.roll(u, 1, dims=-2)) / dy
                else:
                    flux_y = vy * (torch.roll(u, -1, dims=-2) - u) / dy
                u = u - dt * (flux_x + flux_y)
            else:
                u = u - dt * flux_x

            states.append(u.cpu())

        return torch.stack(states, dim=0)

    # -- Simplified 2D Navier-Stokes (vorticity-stream formulation) -----------

    def _solve_ns2d(
        self, params: Dict[str, float], seed: Optional[int]
    ) -> torch.Tensor:
        """2D incompressible NS via vorticity-stream function.

        Returns fields (u, v, ω) stacked as (T+1, 3, Nx, Ny).
        """
        sc = self.scenario
        Re = params.get("Re", 400.0)
        nu = 1.0 / max(Re, 1.0)
        dev = self.device
        Nx, Ny = sc.grid_shape

        # Random smooth initial vorticity
        if seed is not None:
            torch.manual_seed(seed)
        omega = _smooth_random_field((1, Nx, Ny), dev)
        dt = sc.dt
        states = []

        for _ in range(sc.n_steps + 1):
            # Compute velocity from vorticity via stream function (simplified)
            psi = _solve_poisson_fft(omega[0])    # stream function
            u =  _deriv_y(psi).unsqueeze(0)        # u =  ∂ψ/∂y
            v = -_deriv_x(psi).unsqueeze(0)        # v = -∂ψ/∂x

            states.append(torch.stack([u[0], v[0], omega[0]], dim=0).cpu())

            # Advect vorticity + diffusion
            adv = u * _deriv_x(omega[0]).unsqueeze(0) + v * _deriv_y(omega[0]).unsqueeze(0)
            lap = _laplacian(omega, "periodic")
            omega = omega + dt * (-adv + nu * lap)

        return torch.stack(states, dim=0)  # (T+1, 3, Nx, Ny)

    # ------------------------------------------------------------------
    # Parameter sampling
    # ------------------------------------------------------------------

    def _sample_params(self, overrides: Optional[Dict[str, float]]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, (lo, hi) in self.scenario.param_ranges.items():
            out[k] = float(torch.empty(1).uniform_(lo, hi).item())
        if overrides:
            out.update(overrides)
        return out


# ---------------------------------------------------------------------------
# Grid / IC helpers
# ---------------------------------------------------------------------------

def _make_ic(scenario: PhysicsScenario, seed: Optional[int]) -> torch.Tensor:
    """Generate an initial condition field ``(C, *grid_shape)``."""
    if seed is not None:
        torch.manual_seed(seed)

    C = scenario.n_fields
    shape = (C, *scenario.grid_shape)
    kind = scenario.ic_type

    if kind == "random":
        return torch.randn(shape) * 0.5

    if kind == "random_smooth":
        return _smooth_random_field(shape, torch.device("cpu"))

    if kind == "gaussian":
        return _gaussian_bump(shape)

    if kind == "sine":
        return _sine_ic(shape)

    if kind == "step":
        return _step_ic(shape)

    return torch.zeros(shape)


def _smooth_random_field(shape: Tuple, device: torch.device) -> torch.Tensor:
    """Gaussian-filtered random field for smooth ICs."""
    u = torch.randn(shape, device=device)
    # Smooth via FFT: multiply by Gaussian in frequency domain
    for dim in range(1, u.ndim):
        N = u.shape[dim]
        freqs = torch.fft.rfftfreq(N, device=device)
        gauss = torch.exp(-8 * (freqs * math.pi) ** 2)
        idx = [None] * u.ndim
        idx[dim] = slice(None)
        u_fft = torch.fft.rfft(u, dim=dim)
        u_fft = u_fft * gauss[idx]
        u = torch.fft.irfft(u_fft, n=N, dim=dim)
    return u


def _gaussian_bump(shape: Tuple) -> torch.Tensor:
    u = torch.zeros(shape)
    C = shape[0]
    for c in range(C):
        cx = torch.rand(1).item()
        cy = torch.rand(1).item() if len(shape) > 2 else 0.5
        sigma = 0.05 + 0.1 * torch.rand(1).item()
        if len(shape) == 2:  # 1D: (C, Nx)
            x = torch.linspace(0, 1, shape[1])
            u[c] = torch.exp(-((x - cx) ** 2) / (2 * sigma ** 2))
        else:  # 2D: (C, Nx, Ny)
            x = torch.linspace(0, 1, shape[1]).unsqueeze(1)
            y = torch.linspace(0, 1, shape[2]).unsqueeze(0)
            u[c] = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    return u


def _sine_ic(shape: Tuple) -> torch.Tensor:
    u = torch.zeros(shape)
    k = 1 + int(3 * torch.rand(1).item())
    x = torch.linspace(0, 2 * math.pi, shape[-1])
    base = torch.sin(k * x)
    for c in range(shape[0]):
        if len(shape) == 2:
            u[c] = base
        else:
            y = torch.linspace(0, 2 * math.pi, shape[-2])
            u[c] = base.unsqueeze(0) * torch.sin(k * y).unsqueeze(1)
    return u


def _step_ic(shape: Tuple) -> torch.Tensor:
    u = torch.zeros(shape)
    cut = 0.2 + 0.6 * torch.rand(1).item()
    idx = int(cut * shape[-1])
    if len(shape) == 2:
        u[:, :idx] = 1.0
    else:
        u[:, :idx, :] = 1.0
    return u


# ---------------------------------------------------------------------------
# Differential operators (periodic / Dirichlet)
# ---------------------------------------------------------------------------

def _laplacian(u: torch.Tensor, bc: str) -> torch.Tensor:
    """2nd-order Laplacian of u (any spatial dim > 0, after channel dim)."""
    lap = torch.zeros_like(u)
    for dim in range(1, u.ndim):
        N = u.shape[dim]
        if bc == "periodic":
            lap += torch.roll(u, 1, dim) + torch.roll(u, -1, dim) - 2 * u
        else:
            # Interior only; boundaries remain 0
            sl_fwd = [slice(None)] * u.ndim
            sl_bwd = [slice(None)] * u.ndim
            sl_mid = [slice(None)] * u.ndim
            sl_fwd[dim] = slice(2, N)
            sl_bwd[dim] = slice(0, N - 2)
            sl_mid[dim] = slice(1, N - 1)
            lap[sl_mid] += (u[sl_fwd] + u[sl_bwd] - 2 * u[sl_mid])
    return lap


def _apply_dirichlet_zero(u: torch.Tensor) -> torch.Tensor:
    u[..., 0]  = 0.0
    u[..., -1] = 0.0
    if u.ndim > 2:
        u[:, 0, :]  = 0.0
        u[:, -1, :] = 0.0
    return u


def _deriv_x(f: torch.Tensor) -> torch.Tensor:
    """Central difference ∂f/∂x along last axis (periodic)."""
    return (torch.roll(f, -1, -1) - torch.roll(f, 1, -1)) * 0.5


def _deriv_y(f: torch.Tensor) -> torch.Tensor:
    """Central difference ∂f/∂y along second-to-last axis (periodic)."""
    return (torch.roll(f, -1, -2) - torch.roll(f, 1, -2)) * 0.5


def _solve_poisson_fft(rhs: torch.Tensor) -> torch.Tensor:
    """Solve ∇²ψ = rhs periodically via spectral method."""
    Nx, Ny = rhs.shape[-2], rhs.shape[-1]
    rhs_hat = torch.fft.rfft2(rhs)
    kx = torch.fft.fftfreq(Nx, device=rhs.device).unsqueeze(1) * Nx
    ky = torch.fft.rfftfreq(Ny, device=rhs.device).unsqueeze(0) * Ny
    k2 = kx ** 2 + ky ** 2
    k2[0, 0] = 1.0  # avoid div-by-zero; DC mode = 0
    psi_hat = rhs_hat / (-k2.to(rhs_hat.dtype))
    psi_hat[..., 0, 0] = 0.0
    return torch.fft.irfft2(psi_hat, s=(Nx, Ny))


def _normalise_trajectory(raw: Any, sc: PhysicsScenario) -> torch.Tensor:
    """Best-effort conversion of solver output to (T+1, C, *grid) tensor."""
    import numpy as np
    if isinstance(raw, torch.Tensor):
        t = raw
    elif isinstance(raw, np.ndarray):
        t = torch.from_numpy(raw).float()
    else:
        t = torch.zeros(sc.n_steps + 1, sc.n_fields, *sc.grid_shape)

    # Add channel dim if missing
    if t.ndim == len(sc.grid_shape) + 1:
        t = t.unsqueeze(1)
    return t.float()
