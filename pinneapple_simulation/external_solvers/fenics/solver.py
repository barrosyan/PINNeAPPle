"""FEniCS integration — workflow layer on top of pinneapple_solvers.fenics_bridge.

The low-level bridge (pinneapple_solvers.fenics_bridge.FEniCSBridge) implements
the actual FEM assembly and solve. This module adds:

  FEniCSConfig     — structured configuration dataclass
  solve_and_package — one-call solve + UPD packaging
  dof_to_upd       — convert raw DOF vector + mesh coords to PhysicalSample
  FEniCSWorkflow   — convenience wrapper for repeated solves (e.g. parametric sweeps)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class FEniCSConfig:
    """Configuration for a FEniCS solve.

    Parameters
    ----------
    pde : PDE identifier understood by FEniCSBridge
        e.g. "heat_equation_steady", "linear_elasticity_plane_stress"
    domain : domain specification dict accepted by FEniCSBridge
        e.g. {"type": "rectangle", "x": [0, 1], "y": [0, 1], "nx": 32, "ny": 32}
    bcs : list of boundary condition specs accepted by FEniCSBridge
    params : PDE parameter values (conductivity k, Young's modulus E, etc.)
    solver_opts : additional solver options forwarded to FEniCSBridge
    """
    pde: str
    domain: Dict[str, Any]
    bcs: List[Dict[str, Any]]
    params: Dict[str, float] = field(default_factory=dict)
    solver_opts: Dict[str, Any] = field(default_factory=dict)


def solve_and_package(config: FEniCSConfig):
    """Run a FEniCS solve and return a UPD PhysicalSample.

    Returns
    -------
    PhysicalSample with fields = DOF arrays, coords = mesh node positions
    """
    try:
        from pinneapple_simulation.numerical_solvers.fenics_bridge import FEniCSBridge
    except ImportError:
        raise ImportError(
            "pinneapple_solvers.fenics_bridge requires FEniCS. "
            "Install dolfinx or legacy FEniCS, then reinstall pinneapple_solvers."
        )

    bridge = FEniCSBridge(
        pde=config.pde,
        domain=config.domain,
        bcs=config.bcs,
        **config.solver_opts,
    )
    solver_output = bridge.forward(config.params)
    return _output_to_upd(solver_output, config)


def _output_to_upd(solver_output: Any, config: FEniCSConfig):
    """Convert a FEniCSBridge SolverOutput to a UPD PhysicalSample."""
    import torch
    from pinneapple_data.physical_sample import PhysicalSample

    fields: Dict[str, "torch.Tensor"] = {}
    coords: Dict[str, np.ndarray] = {}

    if hasattr(solver_output, "fields"):
        for k, v in solver_output.fields.items():
            fields[k] = torch.as_tensor(np.asarray(v), dtype=torch.float32)

    mesh_src = getattr(solver_output, "coords", None) or getattr(solver_output, "mesh", None)
    if mesh_src is not None:
        xy = None
        if hasattr(mesh_src, "coordinates"):
            xy = np.asarray(mesh_src.coordinates())
        elif isinstance(mesh_src, np.ndarray):
            xy = mesh_src
        if xy is not None:
            coord_names = ["x", "y", "z"]
            for i, name in enumerate(coord_names[: xy.shape[1]]):
                coords[name] = xy[:, i]

    return PhysicalSample(
        fields=fields,
        coords=coords,
        meta={
            "upd": {"version": "0.1", "source": "fenics"},
            "provenance": {"pde": config.pde},
            "units": {},
        },
    )


def dof_to_upd(
    dof_vector: np.ndarray,
    mesh_coords: np.ndarray,
    field_name: str = "u",
):
    """Package a raw FEniCS DOF vector and mesh coordinates as a UPD PhysicalSample.

    Parameters
    ----------
    dof_vector : (N,) or (N, F) array of DOF values
    mesh_coords : (N, d) array of node coordinates
    field_name : name for the field in the PhysicalSample

    Returns
    -------
    PhysicalSample
    """
    import torch
    from pinneapple_data.physical_sample import PhysicalSample

    coord_names = ["x", "y", "z"]
    coords = {
        coord_names[i]: mesh_coords[:, i]
        for i in range(min(mesh_coords.shape[1], 3))
    }

    return PhysicalSample(
        fields={field_name: torch.as_tensor(np.asarray(dof_vector), dtype=torch.float32)},
        coords=coords,
        meta={
            "upd": {"version": "0.1", "source": "fenics"},
            "provenance": {},
            "units": {},
        },
    )


class FEniCSWorkflow:
    """Convenience wrapper for repeated FEniCS solves over a parameter space.

    Useful for generating training datasets for surrogate models or PINNs.

    Examples
    --------
    >>> base_cfg = FEniCSConfig(pde="heat_equation_steady", domain=..., bcs=...)
    >>> wf = FEniCSWorkflow(base_cfg)
    >>> samples = wf.sweep({"k": [0.5, 1.0, 2.0, 5.0]})
    """

    def __init__(self, base_config: FEniCSConfig) -> None:
        self.base_config = base_config

    def solve(self, param_override: Optional[Dict[str, float]] = None):
        """Run a single solve with optional parameter override."""
        cfg = self.base_config
        if param_override:
            cfg = FEniCSConfig(
                pde=cfg.pde,
                domain=cfg.domain,
                bcs=cfg.bcs,
                params={**cfg.params, **param_override},
                solver_opts=cfg.solver_opts,
            )
        return solve_and_package(cfg)

    def sweep(self, param_grid: Dict[str, List[float]]) -> List[Any]:
        """Solve for each combination in a parameter grid.

        Parameters
        ----------
        param_grid : dict mapping parameter name → list of values.
            All combinations are evaluated (Cartesian product).

        Returns
        -------
        list of PhysicalSample, one per parameter combination
        """
        from itertools import product

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        results = []
        for combo in product(*values):
            overrides = dict(zip(keys, combo))
            results.append(self.solve(overrides))
        return results
