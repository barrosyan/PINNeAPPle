"""TurboDesigner integration — analytical mean-line solver as a PINN data source.

The low-level bridge wraps the ``turbodesigner`` Python package
(pip install turbodesigner) and exposes:

  TurboDesignerConfig  — structured configuration dataclass (mirrors the
                         TurboDesigner JSON design format)
  run_turbodesigner    — one-call analytical solve, returns field arrays
  turbodesigner_to_upd — packages solve results as a UPD PhysicalSample
  TurboDesignerWorkflow — convenience wrapper for parametric sweeps

The workflow is:
  1. Define a design with ``TurboDesignerConfig``.
  2. Call ``run_turbodesigner(cfg)`` → dict of numpy arrays (station fields).
  3. Feed the arrays as ``DataConstraint`` into a PINN training run so the
     network learns residual corrections on top of the analytical solution.

Example
-------
>>> from pinneapple_simulation.external_solvers.turbodesigner import (
...     TurboDesignerConfig, run_turbodesigner, TurboDesignerWorkflow
... )
>>> cfg = TurboDesignerConfig(
...     pressure_ratio=3.0,
...     num_stages=5,
...     mass_flow_rate=4.37,
...     rpm=10_000,
... )
>>> data = run_turbodesigner(cfg)
>>> print(data["T_t_K"].shape)   # (num_stations,)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration dataclass — mirrors TurboDesigner JSON "definition" block
# ---------------------------------------------------------------------------

@dataclass
class TurboDesignerConfig:
    """Parameters for one TurboDesigner axial compressor solve.

    All fields map directly to the ``definition`` block of a TurboDesigner
    JSON design file (see turbodesigner CLI ``design schema``).

    Parameters
    ----------
    pressure_ratio          : overall total-to-total pressure ratio
    num_stages              : number of compressor stages
    mass_flow_rate          : kg/s
    rpm                     : shaft speed (rev/min)
    inlet_total_pressure    : Pa
    inlet_total_temperature : K
    isentropic_efficiency   : stage isentropic efficiency (0–1)
    hub_to_tip_ratio        : inlet hub-to-tip radius ratio
    axial_velocity          : mean axial velocity (m/s)
    gamma                   : specific heat ratio (default 1.4 for air)
    gas_constant            : J/(kg·K) (default 287 for air)
    num_streams             : spanwise stream count for free-vortex analysis
    aspect_ratio            : {"rotor": ..., "stator": ...}
    stage_reaction          : list of per-stage reaction values (len = num_stages)
    extra_params            : any additional definition keys forwarded as-is
    """
    pressure_ratio: float = 3.0
    num_stages: int = 5
    mass_flow_rate: float = 4.37
    rpm: float = 10_000.0
    inlet_total_pressure: float = 101_325.0
    inlet_total_temperature: float = 288.15
    isentropic_efficiency: float = 0.878
    hub_to_tip_ratio: float = 0.5
    axial_velocity: float = 136.0
    gamma: float = 1.4
    gas_constant: float = 287.0
    num_streams: int = 9
    aspect_ratio: Dict[str, float] = field(default_factory=lambda: {"rotor": 3.0, "stator": 3.25})
    stage_reaction: Optional[List[float]] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_turbodesigner_dict(self) -> Dict[str, Any]:
        """Serialise to the TurboDesigner JSON ``definition`` schema."""
        reaction = self.stage_reaction or [0.5] * self.num_stages
        defn: Dict[str, Any] = {
            "gamma": self.gamma,
            "axial_velocity": self.axial_velocity,
            "rpm": self.rpm,
            "gas_constant": self.gas_constant,
            "mass_flow_rate": self.mass_flow_rate,
            "pressure_ratio": self.pressure_ratio,
            "inlet_total_pressure": self.inlet_total_pressure,
            "inlet_total_temperature": self.inlet_total_temperature,
            "isentropic_efficiency": self.isentropic_efficiency,
            "num_stages": self.num_stages,
            "stage_temperature_rise": "equal",
            "stage_reaction": reaction,
            "inlet_blockage": 0.0,
            "outlet_blockage": 0.0,
            "hub_to_tip_ratio": self.hub_to_tip_ratio,
            "num_streams": self.num_streams,
            "aspect_ratio": self.aspect_ratio,
            "spacing_to_chord": {"rotor": 1.0, "stator": 1.0},
            "max_thickness_to_chord": {"rotor": 0.1, "stator": 0.1},
            "row_gap_to_chord": 0.25,
            "stage_gap_to_chord": 0.5,
        }
        defn.update(self.extra_params)
        return {
            "machine_type": "axial",
            "configuration": "compressor",
            "definition": defn,
        }


# ---------------------------------------------------------------------------
# Analytical fallback (no turbodesigner installed)
# ---------------------------------------------------------------------------

def _analytical_meanline(cfg: TurboDesignerConfig) -> Dict[str, np.ndarray]:
    """Pure-Python mean-line calculation used when turbodesigner is not installed.

    Computes stage-by-stage total temperature, total pressure, static density
    and axial velocity along the normalised streamwise coordinate s ∈ [0, 1].
    Returns one station per stage boundary (num_stages + 1 stations).
    """
    gamma = cfg.gamma
    R = cfg.gas_constant
    eta = cfg.isentropic_efficiency
    c_p = gamma * R / (gamma - 1.0)

    # Per-stage temperature rise (equal split)
    T_t0 = cfg.inlet_total_temperature
    p_t0 = cfg.inlet_total_pressure
    pr = cfg.pressure_ratio
    n = cfg.num_stages

    T_t_out = T_t0 * (1.0 + (pr ** ((gamma - 1.0) / gamma) - 1.0) / eta)
    delta_T = (T_t_out - T_t0) / n

    T_stations = np.array([T_t0 + i * delta_T for i in range(n + 1)], dtype=np.float64)
    # Isentropic pressure from temperature ratio
    p_stations = p_t0 * ((T_stations / T_t0) * eta + (1 - eta)) ** (gamma / (gamma - 1.0))
    # Mass-flux conservation (rho_static * u = const, approximate constant-area
    # annulus) uses *static* density, not stagnation density -- at the Mach
    # numbers typical of axial compressors the two differ by several percent.
    # T_static = T_t - u^2/(2 c_p) depends on u itself, so iterate to a
    # self-consistent (rho_static, u) pair from a stagnation-density seed.
    rho_t_stations = p_stations / (R * T_stations)
    rho_stations = rho_t_stations
    u_stations = cfg.axial_velocity * (rho_t_stations[0] / rho_t_stations)
    for _ in range(6):
        T_static = T_stations - u_stations ** 2 / (2.0 * c_p)
        p_static = p_stations * (T_static / T_stations) ** (gamma / (gamma - 1.0))
        rho_stations = p_static / (R * T_static)
        u_stations = cfg.axial_velocity * (rho_stations[0] / rho_stations)
    s_stations = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)

    return {
        "s": s_stations,
        "T_t_K": T_stations,
        "p_t_Pa": p_stations,
        "rho_kg_m3": rho_stations,
        "u_axial_m_s": u_stations,
        "c_theta_m_s": np.zeros(n + 1, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Main solve entry point
# ---------------------------------------------------------------------------

def run_turbodesigner(
    cfg: TurboDesignerConfig,
    *,
    fallback_if_missing: bool = True,
) -> Dict[str, np.ndarray]:
    """Run an axial compressor mean-line analysis with TurboDesigner.

    Attempts to use the ``turbodesigner`` package for a full mean-line +
    free-vortex solve.  If the package is not installed and
    ``fallback_if_missing=True``, a simplified analytical calculation is
    performed instead (suitable for quick prototyping).

    Parameters
    ----------
    cfg : TurboDesignerConfig
    fallback_if_missing : bool
        When True, fall back to a pure-Python approximation if turbodesigner
        is not importable.  When False, raise ImportError instead.

    Returns
    -------
    dict with keys:
      s             : normalised streamwise coordinate, shape (S,)
      T_t_K         : total temperature at each station, shape (S,)
      p_t_Pa        : total pressure at each station, shape (S,)
      rho_kg_m3     : density at each station, shape (S,)
      u_axial_m_s   : axial velocity at each station, shape (S,)
      c_theta_m_s   : tangential velocity at each station, shape (S,)
      stage_work_J_kg : work input per stage, shape (num_stages,)  [if available]
    """
    try:
        from turbodesigner.turbomachinery import Turbomachinery
    except ImportError:
        if not fallback_if_missing:
            raise ImportError(
                "turbodesigner is not installed. "
                "Run: pip install turbodesigner"
            )
        return _analytical_meanline(cfg)

    design = cfg.to_turbodesigner_dict()
    machine = Turbomachinery.from_dict(design)

    # --- extract per-station data from TurboDesigner objects ---
    stations: list[Dict[str, float]] = []

    # Inlet flow station
    inlet_fs = machine.stages[0].rotor.inlet_flow_station
    stations.append({
        "T_t": inlet_fs.total_temperature,
        "p_t": inlet_fs.total_pressure,
        "rho": inlet_fs.density,
        "u": inlet_fs.axial_velocity,
        "c_theta": getattr(inlet_fs, "tangential_velocity", 0.0),
    })

    for stage in machine.stages:
        # Rotor exit / stator exit = stage exit
        fs = stage.stator.exit_flow_station
        stations.append({
            "T_t": fs.total_temperature,
            "p_t": fs.total_pressure,
            "rho": fs.density,
            "u": fs.axial_velocity,
            "c_theta": getattr(fs, "tangential_velocity", 0.0),
        })

    n = len(stations)
    s = np.linspace(0.0, 1.0, n, dtype=np.float64)
    T_t = np.array([st["T_t"] for st in stations], dtype=np.float64)
    p_t = np.array([st["p_t"] for st in stations], dtype=np.float64)
    rho = np.array([st["rho"] for st in stations], dtype=np.float64)
    u = np.array([st["u"] for st in stations], dtype=np.float64)
    c_th = np.array([st["c_theta"] for st in stations], dtype=np.float64)

    out: Dict[str, np.ndarray] = {
        "s": s,
        "T_t_K": T_t,
        "p_t_Pa": p_t,
        "rho_kg_m3": rho,
        "u_axial_m_s": u,
        "c_theta_m_s": c_th,
    }

    # Stage work if accessible
    try:
        c_p = cfg.gamma * cfg.gas_constant / (cfg.gamma - 1.0)
        out["stage_work_J_kg"] = np.array(
            [c_p * (machine.stages[i].stator.exit_flow_station.total_temperature
                    - machine.stages[i].rotor.inlet_flow_station.total_temperature)
             for i in range(len(machine.stages))],
            dtype=np.float64,
        )
    except Exception:
        pass

    return out


# ---------------------------------------------------------------------------
# UPD packaging
# ---------------------------------------------------------------------------

def turbodesigner_to_upd(cfg: TurboDesignerConfig, **run_kwargs):
    """Run TurboDesigner and package results as a UPD PhysicalSample.

    Returns
    -------
    PhysicalSample with:
      - coords : {"s": array}
      - fields : {"T_t": tensor, "p_t": tensor, "rho": tensor,
                  "u": tensor, "c_theta": tensor}
    """
    import torch
    from pinneapple_data.physical_sample import PhysicalSample

    data = run_turbodesigner(cfg, **run_kwargs)

    fields = {
        "T_t": torch.as_tensor(data["T_t_K"], dtype=torch.float32),
        "p_t": torch.as_tensor(data["p_t_Pa"], dtype=torch.float32),
        "rho": torch.as_tensor(data["rho_kg_m3"], dtype=torch.float32),
        "u": torch.as_tensor(data["u_axial_m_s"], dtype=torch.float32),
        "c_theta": torch.as_tensor(data["c_theta_m_s"], dtype=torch.float32),
    }

    return PhysicalSample(
        state=fields,
        domain={"type": "grid", "coords": {"s": data["s"]}},
        provenance={
            "version": "0.1",
            "source": "turbodesigner",
            "num_stages": cfg.num_stages,
            "pressure_ratio": cfg.pressure_ratio,
            "rpm": cfg.rpm,
        },
        schema={
            "units": {
                "T_t": "K",
                "p_t": "Pa",
                "rho": "kg/m3",
                "u": "m/s",
                "c_theta": "m/s",
            },
        },
    )


# ---------------------------------------------------------------------------
# Convenience workflow for parametric sweeps
# ---------------------------------------------------------------------------

class TurboDesignerWorkflow:
    """Convenience wrapper for repeated TurboDesigner solves over a parameter space.

    Useful for generating PINN training datasets that cover a range of
    operating conditions (off-design, multi-point training).

    Examples
    --------
    >>> wf = TurboDesignerWorkflow(TurboDesignerConfig(num_stages=5))
    >>> samples = wf.sweep({"pressure_ratio": [2.0, 3.0, 4.0, 5.0]})
    >>> samples = wf.sweep({"rpm": [8000, 10000, 12000], "pressure_ratio": [2.5, 3.0]})
    """

    def __init__(self, base_config: TurboDesignerConfig) -> None:
        self.base_config = base_config

    def solve(self, param_override: Optional[Dict[str, Any]] = None, **run_kwargs) -> Dict[str, np.ndarray]:
        """Run a single TurboDesigner solve with optional parameter override.

        Parameters
        ----------
        param_override : dict mapping TurboDesignerConfig field names to new values
        run_kwargs     : forwarded to ``run_turbodesigner``
        """
        if not param_override:
            return run_turbodesigner(self.base_config, **run_kwargs)

        import dataclasses
        cfg = dataclasses.replace(self.base_config, **param_override)
        return run_turbodesigner(cfg, **run_kwargs)

    def solve_upd(self, param_override: Optional[Dict[str, Any]] = None):
        """Run a single solve and return a UPD PhysicalSample."""
        import dataclasses
        cfg = self.base_config
        if param_override:
            cfg = dataclasses.replace(cfg, **param_override)
        return turbodesigner_to_upd(cfg)

    def sweep(
        self,
        param_grid: Dict[str, List[Any]],
        as_upd: bool = False,
        **run_kwargs,
    ) -> List[Any]:
        """Solve for each combination in a parameter grid.

        Parameters
        ----------
        param_grid : dict mapping config field name → list of values.
            All combinations are evaluated (Cartesian product).
        as_upd : bool
            If True, return PhysicalSample objects; otherwise return raw dicts.

        Returns
        -------
        List of results (dicts or PhysicalSamples), one per parameter combination.
        """
        from itertools import product as _product

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        results: List[Any] = []
        for combo in _product(*values):
            overrides = dict(zip(keys, combo))
            if as_upd:
                results.append(self.solve_upd(overrides))
            else:
                results.append(self.solve(overrides, **run_kwargs))
        return results
