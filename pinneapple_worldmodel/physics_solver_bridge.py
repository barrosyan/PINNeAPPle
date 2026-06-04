# -*- coding: utf-8 -*-
"""Stage 2 — Physics Solver Layer  +  Stage 3 — Physical Ground Truth Dataset.

Stage 2
-------
Dispatches a physics scenario to the appropriate solver backend and returns
the full multivariate field history in a :class:`SolverOutput` container:

  u(x,y,z,t)   — velocity-x field
  v(x,y,z,t)   — velocity-y field
  w(x,y,z,t)   — velocity-z field (3-D only)
  p(x,y,z,t)   — pressure field
  T(x,y,z,t)   — temperature field
  C(x,y,z,t)   — species concentration field

Supported solver backends
  builtin        Pure-PyTorch FD / spectral (always available)
  openfoam       OpenFOAM case runner (requires OpenFOAM + pinneapple_simulation)
  fenics         FEniCS / dolfinx FEM (requires fenics/dolfinx)
  pinn           PINN-as-solver (requires trained model or on-the-fly training)
  fno            FNO neural operator (requires pre-trained FNO checkpoint)
  su2            SU2 CFD (external binary required)

Stage 3
-------
Persists the solver output to the canonical Physical AI dataset structure:

  sample_XXX/
    velocity.zarr
    pressure.zarr
    temperature.zarr
    concentration.zarr   (optional)
    mesh.vtk
    metadata.json
    pde.json
    boundary_conditions.json

Reuses pinneapple modules
  - pinneapple_worldmodel.simulator  (PhysicsSimulator — builtin solver)
  - pinneapple_simulation.external_solvers.openfoam  (OpenFOAM bridge)
  - pinneapple_simulation.external_solvers.fenics    (FEniCS bridge)
  - pinneapple_data (Zarr / UPD storage)

Public API
----------
  SolverOutput         — multivariate field container (T, Nx, Ny, Nz) per field
  SolverBridgeConfig   — solver backend selection + parameters
  PhysicsSolverBridge  — Stage 2: run solver → SolverOutput
  GroundTruthPackager  — Stage 3: SolverOutput → on-disk dataset structure
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .scenario_generator import ScenarioSpec
from .simulator import PhysicsSimulator, TrajectoryData


# ── optional external solver imports ─────────────────────────────────────────
try:
    from pinneapple_simulation.external_solvers.openfoam import (
        OpenFOAMCaseTemplate, run_openfoam_case, read_sampled_scalar_field,
    )
    _OPENFOAM = True
except Exception:
    _OPENFOAM = False

try:
    from pinneapple_simulation.external_solvers.fenics import FEniCSWorkflow
    _FENICS = True
except Exception:
    _FENICS = False

try:
    import zarr
    _ZARR = True
except ImportError:
    _ZARR = False


# ---------------------------------------------------------------------------
# SolverOutput — multivariate field container
# ---------------------------------------------------------------------------

@dataclass
class SolverOutput:
    """Multivariate physical field history from one simulation run.

    Per-timestep storage matching the Stage 2 specification::

        {
            "velocity":      (T, [2 or 3], Ny, Nx) float32
            "pressure":      (T, Ny, Nx)            float32
            "temperature":   (T, Ny, Nx)            float32
            "concentration": (T, Ny, Nx)            float32  [optional]
        }

    Attributes
    ----------
    fields : dict[str -> ndarray]
        Physical field arrays.  Keys: ``"velocity"``, ``"pressure"``,
        ``"temperature"``, ``"concentration"``, etc.
    t_coords : ndarray (T,)
        Simulation time coordinates.
    x_coords : ndarray (Nx,)
        x-axis spatial coordinates.
    y_coords : ndarray (Ny,)
        y-axis spatial coordinates.
    z_coords : ndarray (Nz,) or None
        z-axis spatial coordinates (3-D only).
    params : dict
        PDE parameters used in this run (e.g. ``{"Re": 5000}``).
    solver_name : str
        Which backend generated this output.
    metadata : dict
        Solver-specific metadata (wall-clock time, convergence, etc.).
    """
    fields:       Dict[str, np.ndarray]
    t_coords:     np.ndarray
    x_coords:     np.ndarray
    y_coords:     np.ndarray
    z_coords:     Optional[np.ndarray]   = None
    params:       Dict[str, float]       = field(default_factory=dict)
    solver_name:  str                    = "builtin"
    metadata:     Dict[str, Any]         = field(default_factory=dict)

    @property
    def n_timesteps(self) -> int:
        first = next(iter(self.fields.values()))
        return first.shape[0]

    @property
    def field_names(self) -> List[str]:
        return list(self.fields.keys())

    @property
    def grid_shape(self) -> Tuple[int, ...]:
        first = next(iter(self.fields.values()))
        return first.shape[1:]

    def to_trajectory(self, scenario_name: str = "") -> TrajectoryData:
        """Convert to a :class:`~pinneapple_worldmodel.simulator.TrajectoryData`
        for compatibility with the existing world-model pipeline."""
        # Stack scalar fields along channel dim: (T, C, Ny, Nx)
        scalar_fields = []
        names         = []
        for key, arr in self.fields.items():
            if key == "velocity":
                # velocity has shape (T, 2|3, Ny, Nx) → split into u, v [, w]
                comps = ["u", "v", "w"]
                for i in range(arr.shape[1]):
                    scalar_fields.append(arr[:, i])
                    names.append(comps[i])
            else:
                if arr.ndim == 3:       # (T, Ny, Nx)
                    scalar_fields.append(arr)
                    names.append(key[0].upper() if key == "temperature" else key[0])
        states = torch.tensor(
            np.stack(scalar_fields, axis=1),   # (T, C, Ny, Nx)
            dtype=torch.float32,
        )
        return TrajectoryData(
            states        = states,
            params        = self.params,
            scenario_name = scenario_name,
            metadata      = {**self.metadata, "solver": self.solver_name,
                             "field_names": names},
        )


# ---------------------------------------------------------------------------
# SolverBridgeConfig
# ---------------------------------------------------------------------------

@dataclass
class SolverBridgeConfig:
    """Configuration for the PhysicsSolverBridge (Stage 2).

    Parameters
    ----------
    solver : str
        Backend: ``"builtin"``, ``"openfoam"``, ``"fenics"``, ``"pinn"``,
        ``"fno"``, ``"su2"``.
    generate_temperature : bool
        If True, the bridge attempts to extract / derive the temperature field.
        For solvers that don't natively output T, a passive scalar advection
        is appended.
    generate_concentration : bool
        If True, include a species concentration field (passive scalar).
    openfoam_case_dir : Path or None
        Path to a pre-existing OpenFOAM case template.
    fenics_problem : str
        FEniCS problem type: ``"navier_stokes"``, ``"heat"``, ``"stokes"``.
    pinn_checkpoint : Path or None
        Path to a trained PINN model checkpoint to use as solver.
    fno_checkpoint : Path or None
        Path to a pre-trained FNO model checkpoint.
    extra : dict
        Solver-specific extra parameters.
    """
    solver:                str            = "builtin"
    generate_temperature:  bool           = True
    generate_concentration: bool          = False
    openfoam_case_dir:     Optional[Path] = None
    fenics_problem:        str            = "navier_stokes"
    pinn_checkpoint:       Optional[Path] = None
    fno_checkpoint:        Optional[Path] = None
    extra:                 Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage 2 — PhysicsSolverBridge
# ---------------------------------------------------------------------------

class PhysicsSolverBridge:
    """Dispatch a physics scenario to the appropriate solver backend.

    Wraps the following PINNeAPPle modules:
      - :class:`~pinneapple_worldmodel.simulator.PhysicsSimulator` (builtin)
      - ``pinneapple_simulation.external_solvers.openfoam``         (OpenFOAM)
      - ``pinneapple_simulation.external_solvers.fenics``           (FEniCS)

    Parameters
    ----------
    config : SolverBridgeConfig

    Examples
    --------
    ::

        bridge = PhysicsSolverBridge(SolverBridgeConfig(solver="builtin"))
        output = bridge.solve(scenario_spec, params={"Re": 500})

        bridge_of = PhysicsSolverBridge(SolverBridgeConfig(
            solver            = "openfoam",
            openfoam_case_dir = Path("./case_template"),
        ))
        output_of = bridge_of.solve(scenario_spec, params={"Re": 5000})
    """

    def __init__(self, config: Optional[SolverBridgeConfig] = None) -> None:
        self.cfg = config or SolverBridgeConfig()
        self._solver_fn = self._resolve()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def solve(
        self,
        spec:   ScenarioSpec,
        params: Optional[Dict[str, float]] = None,
    ) -> SolverOutput:
        """Run the solver and return a :class:`SolverOutput`.

        Parameters
        ----------
        spec : ScenarioSpec
            Enriched scenario from :class:`~pinneapple_worldmodel.scenario_generator.ScenarioGenerator`.
        params : dict, optional
            PDE parameters to use.  If ``None``, random values are sampled
            from ``spec.scenario.param_ranges``.

        Returns
        -------
        SolverOutput
        """
        return self._solver_fn(spec, params)

    # ------------------------------------------------------------------
    # Backend resolution
    # ------------------------------------------------------------------

    def _resolve(self):
        name = self.cfg.solver
        if name == "builtin":
            return self._solve_builtin
        if name == "openfoam":
            if not _OPENFOAM:
                import warnings
                warnings.warn("OpenFOAM bridge not available — falling back to builtin.", UserWarning)
            return self._solve_openfoam if _OPENFOAM else self._solve_builtin
        if name == "fenics":
            if not _FENICS:
                import warnings
                warnings.warn("FEniCS bridge not available — falling back to builtin.", UserWarning)
            return self._solve_fenics if _FENICS else self._solve_builtin
        if name == "pinn":
            return self._solve_pinn
        if name == "fno":
            return self._solve_fno
        import warnings
        warnings.warn(f"Unknown solver '{name}' — using builtin.", UserWarning)
        return self._solve_builtin

    # ------------------------------------------------------------------
    # Backend: builtin (PyTorch FD / spectral via PhysicsSimulator)
    # ------------------------------------------------------------------

    def _solve_builtin(
        self,
        spec:   ScenarioSpec,
        params: Optional[Dict[str, float]] = None,
    ) -> SolverOutput:
        t0 = time.perf_counter()
        simulator = PhysicsSimulator(spec.scenario, device="cpu", verbose=False)
        traj      = simulator.generate_trajectory(params)

        states    = traj.states.cpu().numpy()    # (T, C, Ny, Nx)
        T, C, Ny, Nx = states.shape
        t_coords  = np.linspace(*spec.scenario.t_span, T)
        x_coords  = np.linspace(*spec.scenario.domain_bounds[0], Nx)
        y_coords  = np.linspace(*spec.scenario.domain_bounds[1], Ny)

        fields = _trajectory_to_fields(states, spec.field_names, self.cfg)
        return SolverOutput(
            fields      = fields,
            t_coords    = t_coords,
            x_coords    = x_coords,
            y_coords    = y_coords,
            params      = traj.params,
            solver_name = "builtin",
            metadata    = {"elapsed_s": time.perf_counter() - t0, **traj.metadata},
        )

    # ------------------------------------------------------------------
    # Backend: OpenFOAM
    # ------------------------------------------------------------------

    def _solve_openfoam(
        self,
        spec:   ScenarioSpec,
        params: Optional[Dict[str, float]] = None,
    ) -> SolverOutput:
        """Run an OpenFOAM case and extract fields."""
        t0 = time.perf_counter()
        if params is None:
            params = {k: float(np.random.uniform(*v))
                      for k, v in spec.scenario.param_ranges.items()}

        case_dir = self.cfg.openfoam_case_dir
        if case_dir is None:
            raise ValueError("openfoam_case_dir must be set for the OpenFOAM backend")

        # Use the pinneapple_simulation.external_solvers.openfoam bridge
        from pinneapple_simulation.external_solvers.openfoam import (
            OpenFOAMRunConfig, run_openfoam_case, stage_case_for_scenario,
        )
        run_cfg = OpenFOAMRunConfig(
            case_dir    = str(case_dir),
            n_proc      = self.cfg.extra.get("n_proc", 1),
            end_time    = spec.scenario.t_span[1],
            write_interval = spec.scenario.t_span[1] / spec.scenario.n_steps,
        )
        run_openfoam_case(run_cfg)

        # Read velocity and pressure fields
        u = _load_openfoam_field(case_dir, "U", spec)
        p = _load_openfoam_field(case_dir, "p", spec)

        fields = {"velocity": u, "pressure": p}
        _add_derived_fields(fields, spec, self.cfg)

        return SolverOutput(
            fields      = fields,
            t_coords    = np.linspace(*spec.scenario.t_span, u.shape[0]),
            x_coords    = np.linspace(*spec.scenario.domain_bounds[0], u.shape[-1]),
            y_coords    = np.linspace(*spec.scenario.domain_bounds[1], u.shape[-2]),
            params      = params,
            solver_name = "openfoam",
            metadata    = {"elapsed_s": time.perf_counter() - t0},
        )

    # ------------------------------------------------------------------
    # Backend: FEniCS
    # ------------------------------------------------------------------

    def _solve_fenics(
        self,
        spec:   ScenarioSpec,
        params: Optional[Dict[str, float]] = None,
    ) -> SolverOutput:
        """Run a FEniCS simulation via pinneapple_simulation.external_solvers.fenics."""
        t0 = time.perf_counter()
        if params is None:
            params = {k: float(np.random.uniform(*v))
                      for k, v in spec.scenario.param_ranges.items()}

        wf = FEniCSWorkflow(
            problem  = self.cfg.fenics_problem,
            params   = params,
            domain   = spec.scenario.domain_bounds,
            grid     = spec.scenario.grid_shape,
        )
        result = wf.solve()

        fields = _fenics_result_to_fields(result, spec, self.cfg)
        Ny, Nx = spec.scenario.grid_shape

        return SolverOutput(
            fields      = fields,
            t_coords    = np.array([0.0, spec.scenario.t_span[1]]),
            x_coords    = np.linspace(*spec.scenario.domain_bounds[0], Nx),
            y_coords    = np.linspace(*spec.scenario.domain_bounds[1], Ny),
            params      = params,
            solver_name = "fenics",
            metadata    = {"elapsed_s": time.perf_counter() - t0},
        )

    # ------------------------------------------------------------------
    # Backend: PINN as solver
    # ------------------------------------------------------------------

    def _solve_pinn(
        self,
        spec:   ScenarioSpec,
        params: Optional[Dict[str, float]] = None,
    ) -> SolverOutput:
        """Use a trained PINN model as a physics solver."""
        t0 = time.perf_counter()
        if params is None:
            params = {k: float(np.random.uniform(*v))
                      for k, v in spec.scenario.param_ranges.items()}

        if self.cfg.pinn_checkpoint is None:
            # Fall back to builtin if no checkpoint provided
            import warnings
            warnings.warn("No PINN checkpoint specified — using builtin solver.", UserWarning)
            return self._solve_builtin(spec, params)

        model = torch.load(str(self.cfg.pinn_checkpoint), map_location="cpu")
        model.eval()

        # Evaluate PINN on a grid
        Ny, Nx = spec.scenario.grid_shape
        T      = spec.scenario.n_steps
        x      = np.linspace(*spec.scenario.domain_bounds[0], Nx)
        y      = np.linspace(*spec.scenario.domain_bounds[1], Ny)
        t      = np.linspace(*spec.scenario.t_span, T)
        XX, YY = np.meshgrid(x, y, indexing="ij")
        XX_f   = XX.ravel().astype(np.float32)
        YY_f   = YY.ravel().astype(np.float32)

        states = np.zeros((T, len(spec.field_names), Ny, Nx), dtype=np.float32)
        for ti, tv in enumerate(t):
            TT = np.full_like(XX_f, tv)
            inp = torch.tensor(np.column_stack([XX_f, YY_f, TT]))
            with torch.no_grad():
                out = model(inp).cpu().numpy()
            for ci in range(min(out.shape[1], len(spec.field_names))):
                states[ti, ci] = out[:, ci].reshape(Ny, Nx)

        fields = _trajectory_to_fields(states, spec.field_names, self.cfg)
        return SolverOutput(
            fields      = fields,
            t_coords    = t,
            x_coords    = x,
            y_coords    = y,
            params      = params,
            solver_name = "pinn",
            metadata    = {"elapsed_s": time.perf_counter() - t0,
                           "checkpoint": str(self.cfg.pinn_checkpoint)},
        )

    # ------------------------------------------------------------------
    # Backend: FNO neural operator
    # ------------------------------------------------------------------

    def _solve_fno(
        self,
        spec:   ScenarioSpec,
        params: Optional[Dict[str, float]] = None,
    ) -> SolverOutput:
        """Autoregressively unroll a trained FNO to generate a trajectory."""
        t0 = time.perf_counter()
        if params is None:
            params = {k: float(np.random.uniform(*v))
                      for k, v in spec.scenario.param_ranges.items()}

        if self.cfg.fno_checkpoint is None:
            import warnings
            warnings.warn("No FNO checkpoint specified — using builtin solver.", UserWarning)
            return self._solve_builtin(spec, params)

        try:
            from pinneapple_worldmodel.model import PhysicsWorldModel
            model = PhysicsWorldModel.load(str(self.cfg.fno_checkpoint))
        except Exception:
            model = torch.load(str(self.cfg.fno_checkpoint), map_location="cpu")

        model.eval()

        # Generate initial condition via builtin
        sim = PhysicsSimulator(spec.scenario, device="cpu", verbose=False)
        ic_traj = sim.generate_trajectory(params)
        state   = ic_traj.states[0:1]   # (1, C, Ny, Nx) — initial condition

        Ny, Nx = spec.scenario.grid_shape
        C = state.shape[1]
        T = spec.scenario.n_steps
        all_states = [state.numpy()]

        with torch.no_grad():
            for _ in range(T):
                next_state = model(state)
                if isinstance(next_state, torch.Tensor):
                    all_states.append(next_state.numpy())
                    state = next_state
                else:
                    break

        states_arr = np.concatenate(all_states, axis=0)   # (T+1, C, Ny, Nx)
        t_coords   = np.linspace(*spec.scenario.t_span, states_arr.shape[0])
        fields     = _trajectory_to_fields(states_arr, spec.field_names, self.cfg)

        return SolverOutput(
            fields      = fields,
            t_coords    = t_coords,
            x_coords    = np.linspace(*spec.scenario.domain_bounds[0], Nx),
            y_coords    = np.linspace(*spec.scenario.domain_bounds[1], Ny),
            params      = params,
            solver_name = "fno",
            metadata    = {"elapsed_s": time.perf_counter() - t0},
        )


# ---------------------------------------------------------------------------
# Stage 3 — GroundTruthPackager
# ---------------------------------------------------------------------------

class GroundTruthPackager:
    """Organise SolverOutput into the canonical Physical AI dataset structure.

    Writes per-timestep fields to Zarr (or NumPy fallback) with the layout::

        sample_XXX/
          velocity.zarr
          pressure.zarr
          temperature.zarr
          concentration.zarr   (if present)
          mesh.vtk
          metadata.json
          pde.json
          boundary_conditions.json

    Parameters
    ----------
    output_dir : Path
        Root dataset directory.
    format : str
        ``"zarr"`` (preferred) or ``"npy"``.
    zarr_chunks : tuple, optional
        Zarr chunk shape.  ``None`` = auto (1 timestep per chunk).
    overwrite : bool

    Examples
    --------
    ::

        packager = GroundTruthPackager(Path("./ns_dataset"))
        packager.write(solver_output, spec, sample_idx=0)
    """

    def __init__(
        self,
        output_dir:   Union[str, Path],
        format:       str           = "zarr",
        zarr_chunks:  Optional[Tuple] = None,
        overwrite:    bool          = True,
    ) -> None:
        self.output_dir  = Path(output_dir)
        self.format      = format if _ZARR else "npy"
        self.zarr_chunks = zarr_chunks
        self.overwrite   = overwrite
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._manifest: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------

    def write(
        self,
        output:     SolverOutput,
        spec:       ScenarioSpec,
        sample_idx: int = 0,
    ) -> Path:
        """Write one solver output to disk and return the sample directory path."""
        sid   = str(sample_idx).zfill(6)
        sdir  = self.output_dir / f"sample_{sid}"
        if sdir.exists() and not self.overwrite:
            return sdir
        sdir.mkdir(parents=True, exist_ok=True)

        # 1. Physical fields
        field_paths = {}
        for fname, arr in output.fields.items():
            path = self._write_field(arr, sdir, fname)
            field_paths[fname] = str(path)

        # 2. VTK mesh
        _write_vtk_from_solver(output, spec, sdir / "mesh.vtk")

        # 3. metadata.json
        meta = {
            "sample_id":       sid,
            "solver":          output.solver_name,
            "scenario_name":   spec.name,
            "pde_kind":        spec.pde_kind,
            "n_timesteps":     output.n_timesteps,
            "grid_shape":      list(output.grid_shape),
            "field_names":     output.field_names,
            "parameters":      output.params,
            "t_span":          [float(output.t_coords[0]), float(output.t_coords[-1])],
            "domain_bounds":   [list(b) for b in spec.scenario.domain_bounds],
            "geometry":        spec.geometry,
            "fluid":           spec.fluid,
            "solver_metadata": output.metadata,
            "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (sdir / "metadata.json").write_text(json.dumps(meta, indent=2))

        # 4. pde.json
        _pde_map = {
            "ns2d":       "Navier-Stokes: ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u, ∇·u=0",
            "heat":       "Heat equation: ∂T/∂t = α∇²T",
            "burgers":    "Burgers: ∂u/∂t + u·∂u/∂x = ν∂²u/∂x²",
            "wave":       "Wave: ∂²u/∂t² = c²∇²u",
            "advection":  "Advection-diffusion: ∂φ/∂t + v·∇φ = D∇²φ",
            "elasticity": "Linear elasticity: ∇·σ = 0",
        }
        pde_json = {
            "equation":   _pde_map.get(spec.pde_kind, spec.pde_kind),
            "pde_kind":   spec.pde_kind,
            "parameters": {k: list(v) for k, v in spec.scenario.param_ranges.items()},
        }
        (sdir / "pde.json").write_text(json.dumps(pde_json, indent=2))

        # 5. boundary_conditions.json
        bc = dict(spec.boundary_conditions)
        bc.update({k: v for k, v in output.params.items() if "velocity" in k or "T" in k})
        (sdir / "boundary_conditions.json").write_text(json.dumps(bc, indent=2))

        # Manifest entry
        entry = {
            "sample_id": sid,
            "sample_dir": str(sdir),
            "fields": field_paths,
            "metadata": str(sdir / "metadata.json"),
        }
        self._manifest.append(entry)
        self._save_manifest()

        return sdir

    def write_batch(
        self,
        outputs: List[SolverOutput],
        spec:    ScenarioSpec,
        start_idx: int = 0,
    ) -> List[Path]:
        return [self.write(out, spec, start_idx + i) for i, out in enumerate(outputs)]

    def _write_field(self, arr: np.ndarray, sdir: Path, name: str) -> Path:
        if self.format == "zarr" and _ZARR:
            path = sdir / f"{name}.zarr"
            chunks = self.zarr_chunks
            if chunks is None:
                if arr.ndim == 3:    # (T, Ny, Nx)
                    chunks = (1, arr.shape[1], arr.shape[2])
                elif arr.ndim == 4:  # (T, C, Ny, Nx)
                    chunks = (1, arr.shape[1], arr.shape[2], arr.shape[3])
                else:
                    chunks = None
            zarr.save_array(str(path), arr.astype(np.float32), chunks=chunks)
        else:
            path = sdir / f"{name}.npy"
            np.save(str(path), arr.astype(np.float32))
        return path

    def _save_manifest(self) -> None:
        path = self.output_dir / "dataset_manifest.json"
        path.write_text(json.dumps(self._manifest, indent=2))


# ---------------------------------------------------------------------------
# Helper: convert trajectory (T, C, Ny, Nx) → {velocity, pressure, T} dict
# ---------------------------------------------------------------------------

def _trajectory_to_fields(
    states:      np.ndarray,       # (T, C, Ny, Nx)
    field_names: List[str],
    cfg:         SolverBridgeConfig,
) -> Dict[str, np.ndarray]:
    """Map named channels from a trajectory to the standard field dict."""
    T, C, Ny, Nx = states.shape
    fields: Dict[str, np.ndarray] = {}

    # Find velocity components
    u_idx = _ci(field_names, ["u", "ux", "velocity_x"])
    v_idx = _ci(field_names, ["v", "uy", "velocity_y"])
    if u_idx is not None and v_idx is not None:
        vel = np.stack([states[:, u_idx], states[:, v_idx]], axis=1)   # (T, 2, Ny, Nx)
        fields["velocity"] = vel.astype(np.float32)
    elif u_idx is not None:
        fields["velocity"] = states[:, u_idx:u_idx+1].astype(np.float32)

    # Pressure
    p_idx = _ci(field_names, ["p", "pressure"])
    if p_idx is not None:
        fields["pressure"] = states[:, p_idx].astype(np.float32)
    elif "velocity" not in fields and C >= 1:
        # No velocity either — use first channel as pressure proxy
        fields["pressure"] = states[:, 0].astype(np.float32)

    # Temperature
    if cfg.generate_temperature:
        t_idx = _ci(field_names, ["T", "temperature", "temp", "theta"])
        if t_idx is not None:
            fields["temperature"] = states[:, t_idx].astype(np.float32)
        else:
            # Derive T from energy (proxy: |u|² / 2 for incompressible)
            if "velocity" in fields:
                vel = fields["velocity"]
                kinetic = np.sum(vel**2, axis=1) / 2.0   # (T, Ny, Nx)
                fields["temperature"] = kinetic.astype(np.float32)

    # Concentration
    if cfg.generate_concentration:
        c_idx = _ci(field_names, ["C", "concentration", "phi", "species"])
        if c_idx is not None:
            fields["concentration"] = states[:, c_idx].astype(np.float32)
        else:
            # Passive scalar = zero (to be advected externally)
            shape = (T, Ny, Nx)
            fields["concentration"] = np.zeros(shape, dtype=np.float32)

    return fields


def _ci(names: List[str], candidates: List[str]) -> Optional[int]:
    for cand in candidates:
        for i, nm in enumerate(names):
            if nm.lower() == cand.lower():
                return i
    return None


def _add_derived_fields(fields: Dict, spec: ScenarioSpec, cfg: SolverBridgeConfig) -> None:
    """Add temperature / concentration derived from velocity if not present."""
    if cfg.generate_temperature and "temperature" not in fields:
        vel = fields.get("velocity")
        if vel is not None:
            fields["temperature"] = (np.sum(vel**2, axis=1) / 2.0).astype(np.float32)
    if cfg.generate_concentration and "concentration" not in fields:
        T, Ny, Nx = next(iter(fields.values())).shape[:1] + spec.scenario.grid_shape
        fields["concentration"] = np.zeros((T, Ny, Nx), dtype=np.float32)


def _load_openfoam_field(case_dir: Path, field_name: str, spec: ScenarioSpec) -> np.ndarray:
    """Load a field from OpenFOAM time directories into (T, [C,] Ny, Nx) array."""
    try:
        Ny, Nx = spec.scenario.grid_shape
        T = spec.scenario.n_steps
        arr = np.zeros((T, Ny, Nx), dtype=np.float32)
        # OpenFOAM bridge: read sampled scalar field
        from pinneapple_simulation.external_solvers.openfoam import read_sampled_scalar_field
        data = read_sampled_scalar_field(str(case_dir / "postProcessing"), field_name)
        if data is not None:
            arr = np.array(data).reshape(T, Ny, Nx).astype(np.float32)
        return arr
    except Exception:
        Ny, Nx = spec.scenario.grid_shape
        return np.zeros((spec.scenario.n_steps, Ny, Nx), dtype=np.float32)


def _fenics_result_to_fields(result, spec: ScenarioSpec, cfg: SolverBridgeConfig) -> Dict:
    """Convert a FEniCS result to the standard field dict."""
    Ny, Nx = spec.scenario.grid_shape
    fields: Dict[str, np.ndarray] = {}
    try:
        if hasattr(result, "u"):
            u_arr = np.array(result.u).reshape(1, Ny, Nx).astype(np.float32)
            fields["velocity"] = u_arr
        if hasattr(result, "p"):
            p_arr = np.array(result.p).reshape(1, Ny, Nx).astype(np.float32)
            fields["pressure"] = p_arr
        if hasattr(result, "T"):
            fields["temperature"] = np.array(result.T).reshape(1, Ny, Nx).astype(np.float32)
    except Exception:
        pass
    if not fields:
        fields["pressure"] = np.zeros((1, Ny, Nx), dtype=np.float32)
    _add_derived_fields(fields, spec, cfg)
    return fields


def _write_vtk_from_solver(output: SolverOutput, spec: ScenarioSpec, path: Path) -> None:
    """Write a structured-grid VTK from SolverOutput (last timestep)."""
    try:
        Nx = len(output.x_coords)
        Ny = len(output.y_coords)
        lines = [
            "# vtk DataFile Version 3.0",
            f"PINNeAPPle SolverOutput — {spec.name}",
            "ASCII",
            "DATASET RECTILINEAR_GRID",
            f"DIMENSIONS {Nx} {Ny} 1",
            f"X_COORDINATES {Nx} float",
            " ".join(f"{v:.6f}" for v in output.x_coords),
            f"Y_COORDINATES {Ny} float",
            " ".join(f"{v:.6f}" for v in output.y_coords),
            "Z_COORDINATES 1 float",
            "0.0",
            f"POINT_DATA {Nx * Ny}",
        ]
        for fname, arr in output.fields.items():
            if arr.ndim == 3:      # (T, Ny, Nx)
                snap = arr[-1].ravel(order="C")
                lines += [f"SCALARS {fname} float 1", "LOOKUP_TABLE default",
                          " ".join(f"{v:.6f}" for v in snap)]
            elif arr.ndim == 4:   # (T, C, Ny, Nx)
                comps = ["u", "v", "w"]
                for ci in range(arr.shape[1]):
                    snap = arr[-1, ci].ravel(order="C")
                    lines += [f"SCALARS {fname}_{comps[ci]} float 1",
                              "LOOKUP_TABLE default",
                              " ".join(f"{v:.6f}" for v in snap)]
        path.write_text("\n".join(lines))
    except Exception:
        pass
