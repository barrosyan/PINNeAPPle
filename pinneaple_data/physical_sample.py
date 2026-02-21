"""PhysicalSample dataclass and training bridge for UPD-aligned physical data."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, Iterable, Mapping

import xarray as xr


@dataclass
class PhysicalSample:
    """
    Unified Physical Sample (UPD-aligned) used across Pinneaple.

    This structure serves as a standardized container for physical data
    used in modeling, simulation, and PINN training workflows.

    Attributes
    ----------
    state : xr.Dataset or Dict[str, Any]
        Core physical state representation. Preferably an xarray.Dataset
        for structured grid data, or a dict-like structure for other formats.
    geometry : Optional[Any]
        Optional geometry asset describing spatial structure.
    schema : Dict[str, Any]
        Governing equations, boundary/initial conditions, forcing terms,
        unit policies, and related metadata.
    domain : Dict[str, Any]
        Domain interpretation metadata (grid, mesh, graph, points, etc.).
    provenance : Dict[str, Any]
        Lineage metadata including identifiers, source, tiling,
        time span, and other traceability information.
    extras : Dict[str, Any]
        Extensible container for additional artifacts such as
        feature caches, mesh labels, SDFs, etc.
    """
    state: Union[xr.Dataset, Dict[str, Any]]
    geometry: Optional[Any] = None
    schema: Dict[str, Any] = field(default_factory=dict)
    domain: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    # -------------------------
    # Domain helpers
    # -------------------------
    def domain_type(self) -> str:
        """
        Infer the domain type of this sample.

        Returns
        -------
        str
            Lowercase domain type string. Defaults to "grid"
            if state is an xarray.Dataset and no explicit type is set.
        """
        t = (self.domain or {}).get("type")
        if t:
            return str(t).lower()
        return "grid" if isinstance(self.state, xr.Dataset) else "unknown"

    def is_grid(self) -> bool:
        """
        Check if the sample represents structured grid data.

        Returns
        -------
        bool
            True if domain type is "grid".
        """
        return self.domain_type() == "grid"

    def is_mesh(self) -> bool:
        """
        Check if the sample represents mesh-based data.

        Returns
        -------
        bool
            True if domain type is "mesh".
        """
        return self.domain_type() == "mesh"

    def is_graph(self) -> bool:
        """
        Check if the sample represents graph-based data.

        Returns
        -------
        bool
            True if domain type is "graph".
        """
        return self.domain_type() == "graph"

    # -------------------------
    # Introspection
    # -------------------------
    def list_variables(self) -> list[str]:
        """
        List available physical variables in the state.

        Returns
        -------
        list[str]
            Variable names extracted from state.
        """
        if isinstance(self.state, xr.Dataset):
            return list(self.state.data_vars)
        if isinstance(self.state, dict):
            return [str(k) for k in self.state.keys()]
        return []

    def summary(self) -> Dict[str, Any]:
        """
        Generate a structured summary of this PhysicalSample.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing domain type, geometry presence,
            schema keys, provenance keys, extras keys,
            and state structure information.
        """
        out = {
            "domain_type": self.domain_type(),
            "has_geometry": self.geometry is not None,
            "schema_keys": sorted(list(self.schema.keys())),
            "provenance_keys": sorted(list(self.provenance.keys())),
            "extras_keys": sorted(list(self.extras.keys())),
        }
        if isinstance(self.state, xr.Dataset):
            out["state"] = {
                "coords": list(self.state.coords),
                "dims": {k: int(v) for k, v in self.state.sizes.items()},
                "vars": list(self.state.data_vars),
            }
        else:
            out["state"] = {"type": type(self.state).__name__, "vars": self.list_variables()}
        return out

    # -------------------------
    # Training bridge
    # -------------------------
    def to_train_dict(
        self,
        *,
        x_vars: Iterable[str],
        y_vars: Optional[Iterable[str]] = None,
        coords: Optional[Iterable[str]] = None,
        time_dim: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convert this PhysicalSample into a canonical training dictionary.

        Output format:
            {
                "x": <stacked input array/tensor>,
                "y": <stacked output array/tensor?>,
                "coords": {...},
                "schema": {...},
                "domain": {...},
                "provenance": {...},
                "geometry": ...
            }

        Parameters
        ----------
        x_vars : Iterable[str]
            Names of variables to be stacked as model inputs.
        y_vars : Optional[Iterable[str]]
            Names of variables to be stacked as supervised targets.
        coords : Optional[Iterable[str]]
            Coordinate names to include in the output dictionary.
        time_dim : Optional[str]
            If provided and present in variables, ensures this dimension
            is transposed to the first axis before stacking.

        Returns
        -------
        Dict[str, Any]
            Training-ready dictionary with stacked arrays or tensors.

        Notes
        -----
        - For xr.Dataset states, variables are stacked along the last dimension.
        - For dict states, assumes variables are already array/tensor-like
          and stackable via numpy or torch.
        """
        x_vars = list(x_vars)
        y_vars = list(y_vars) if y_vars is not None else []
        coords = list(coords) if coords is not None else []

        def _stack_from_xr(ds: xr.Dataset, vars_: list[str]):
            """
            Stack variables from an xarray.Dataset along the last dimension.
            """
            arrs = []
            for v in vars_:
                da = ds[v]
                if time_dim and time_dim in da.dims:
                    da = da.transpose(time_dim, ...)
                arrs.append(da.data)
            import numpy as np
            return np.stack(arrs, axis=-1)

        def _stack_from_dict(d: Dict[str, Any], vars_: list[str]):
            """
            Stack variables from a dictionary-based state.
            """
            import numpy as np
            arrs = [d[v] for v in vars_]
            if hasattr(arrs[0], "shape") and "torch" in str(type(arrs[0])):
                import torch
                return torch.stack(arrs, dim=-1)
            return np.stack(arrs, axis=-1)

        if isinstance(self.state, xr.Dataset):
            x = _stack_from_xr(self.state, x_vars)
            y = _stack_from_xr(self.state, y_vars) if y_vars else None
            c = {k: self.state.coords[k].data for k in coords if k in self.state.coords}
        else:
            assert isinstance(self.state, dict), "state must be xr.Dataset or dict"
            x = _stack_from_dict(self.state, x_vars)
            y = _stack_from_dict(self.state, y_vars) if y_vars else None
            c = {}

        out = {
            "x": x,
            "coords": c,
            "schema": self.schema,
            "domain": self.domain,
            "provenance": self.provenance,
            "geometry": self.geometry,
        }
        if y is not None:
            out["y"] = y
        return out