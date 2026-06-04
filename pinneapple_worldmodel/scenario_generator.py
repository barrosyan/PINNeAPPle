# -*- coding: utf-8 -*-
"""Stage 1 — Scenario Generator.

Parses a user prompt (JSON dict or natural-language string) into a fully
specified :class:`~pinneapple_worldmodel.scenario.PhysicsScenario` plus the
geometry and boundary-condition metadata required by the downstream stages.

Supported physical domains
--------------------------
  fluid_dynamics    pipe, cylinder, cavity, channel, backward_step
  heat_transfer     plate, fin, wall, pcb
  mass_transfer     mixing, reactor, diffusion, absorption
  combustion        premixed, diffusion_flame, ignition
  hvac              duct, room, plenum
  reservoir         porous, two_phase
  structural        beam, plate, shell
  electromagnetics  waveguide, coil
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .scenario import PhysicsScenario, BUILTIN_SCENARIOS


# ---------------------------------------------------------------------------
# ScenarioSpec — enriched scenario with geometry & BC metadata
# ---------------------------------------------------------------------------

@dataclass
class ScenarioSpec:
    """Extended scenario specification produced by the generator.

    Wraps :class:`PhysicsScenario` and adds geometry/BC metadata needed by
    the renderer and packager (stages 4–8).
    """
    scenario:    PhysicsScenario
    domain:      str                        # e.g. "pipe", "cylinder", "cavity"
    geometry:    Dict[str, Any]             # geometry parameters (length, diameter, …)
    fluid:       Dict[str, Any]             # fluid properties (density, viscosity, …)
    boundary_conditions: Dict[str, Any]    # inlet/outlet/wall BCs
    initial_conditions:  Dict[str, Any]    # IC type + parameters
    sensor_config: Dict[str, Any]           # suggested camera / sensor setup
    extra: Dict[str, Any] = field(default_factory=dict)

    # Convenience pass-throughs
    @property
    def name(self) -> str: return self.scenario.name
    @property
    def pde_kind(self) -> str: return self.scenario.pde_kind
    @property
    def field_names(self): return self.scenario.field_names


# ---------------------------------------------------------------------------
# ScenarioGenerator
# ---------------------------------------------------------------------------

class ScenarioGenerator:
    """Convert a user prompt or JSON dict into a :class:`ScenarioSpec`.

    Parameters
    ----------
    default_grid_2d : (Nx, Ny)
        Default 2-D grid when the prompt doesn't specify one.
    default_n_steps : int
        Default number of timesteps.
    device : str
        Compute device passed through to the PhysicsSimulator.

    Examples
    --------
    From JSON dict::

        gen = ScenarioGenerator()
        spec = gen.from_dict({
            "domain": "fluid_dynamics",
            "geometry": "pipe",
            "length": 10,
            "diameter": 0.5,
            "fluid": "water",
        })

    From prompt string::

        spec = gen.from_prompt("simulate turbulent pipe flow with water at Re=5000")

    From a built-in name::

        spec = gen.from_name("ns2d_cavity")
    """

    def __init__(
        self,
        default_grid_2d: Tuple[int, int] = (64, 64),
        default_n_steps:  int = 32,
        device:           str = "cpu",
    ) -> None:
        self.default_grid_2d = default_grid_2d
        self.default_n_steps  = default_n_steps
        self.device           = device

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def from_dict(self, spec_dict: Dict[str, Any]) -> ScenarioSpec:
        """Build a ScenarioSpec from a JSON-like dict."""
        return _dict_to_spec(spec_dict, self.default_grid_2d, self.default_n_steps)

    def from_json(self, json_str: str) -> ScenarioSpec:
        """Build a ScenarioSpec from a JSON string."""
        return self.from_dict(json.loads(json_str))

    def from_prompt(self, prompt: str) -> ScenarioSpec:
        """Build a ScenarioSpec from a natural-language prompt.

        Parses keywords to identify the physics domain, geometry, and
        parameters.  Falls back to a default NS2D cavity scenario if the
        prompt is too ambiguous.
        """
        d = _prompt_to_dict(prompt)
        return self.from_dict(d)

    def from_name(self, name: str) -> ScenarioSpec:
        """Return a ScenarioSpec for a built-in scenario by name."""
        if name not in BUILTIN_SCENARIOS:
            raise KeyError(f"Unknown built-in scenario '{name}'. "
                           f"Available: {sorted(BUILTIN_SCENARIOS)}")
        scenario = BUILTIN_SCENARIOS[name]
        return ScenarioSpec(
            scenario  = scenario,
            domain    = _infer_domain_from_pde(scenario.pde_kind),
            geometry  = {"type": "unit_square"},
            fluid     = {},
            boundary_conditions = {"type": scenario.bc_type},
            initial_conditions  = {"type": scenario.ic_type},
            sensor_config = _default_sensor_config(),
        )

    def from_any(self, source) -> ScenarioSpec:
        """Accept dict, JSON string, prompt string, or built-in name."""
        if isinstance(source, dict):
            return self.from_dict(source)
        if isinstance(source, str):
            try:
                return self.from_json(source)
            except (json.JSONDecodeError, ValueError):
                pass
            if source in BUILTIN_SCENARIOS:
                return self.from_name(source)
            return self.from_prompt(source)
        if isinstance(source, PhysicsScenario):
            return ScenarioSpec(
                scenario=source,
                domain=_infer_domain_from_pde(source.pde_kind),
                geometry={}, fluid={},
                boundary_conditions={"type": source.bc_type},
                initial_conditions={"type": source.ic_type},
                sensor_config=_default_sensor_config(),
            )
        raise TypeError(f"Cannot parse scenario from {type(source)}")


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------

_FLUID_DB: Dict[str, Dict[str, float]] = {
    "water":    {"rho": 997.0,  "mu": 8.9e-4, "Pr": 6.9,   "cp": 4182.0},
    "air":      {"rho": 1.225,  "mu": 1.81e-5, "Pr": 0.71, "cp": 1005.0},
    "oil":      {"rho": 870.0,  "mu": 0.05,    "Pr": 300.0, "cp": 1900.0},
    "hydrogen": {"rho": 0.0899, "mu": 8.8e-6,  "Pr": 0.69,  "cp": 14310.0},
    "co2":      {"rho": 1.977,  "mu": 1.47e-5, "Pr": 0.77,  "cp": 844.0},
}

_DOMAIN_TO_PDE: Dict[str, str] = {
    "pipe":         "ns2d",
    "cylinder":     "ns2d",
    "cavity":       "ns2d",
    "channel":      "ns2d",
    "backward_step":"ns2d",
    "plate":        "heat",
    "fin":          "heat",
    "wall":         "heat",
    "mixing":       "advection",
    "reactor":      "advection",
    "diffusion":    "heat",
    "combustion":   "ns2d",
    "beam":         "elasticity",
    "waveguide":    "wave",
}

_DOMAIN_TO_BC: Dict[str, Dict[str, Any]] = {
    "pipe":     {"inlet": "velocity_inlet", "outlet": "pressure_outlet", "walls": "no_slip"},
    "cylinder": {"farfield": "freestream", "body": "no_slip"},
    "cavity":   {"lid": "moving_wall", "walls": "no_slip"},
    "channel":  {"inlet": "velocity_inlet", "outlet": "pressure_outlet", "walls": "no_slip"},
    "plate":    {"bottom": "fixed_temperature", "top": "convective"},
}


def _dict_to_spec(
    d: Dict[str, Any],
    default_grid: Tuple[int, int],
    default_n_steps: int,
) -> ScenarioSpec:
    """Core builder: dict → ScenarioSpec."""
    top_domain = str(d.get("domain", "fluid_dynamics")).lower()
    geom_key   = str(d.get("geometry", "pipe")).lower()

    pde_kind = _DOMAIN_TO_PDE.get(geom_key, "ns2d")

    # Geometry
    geom: Dict[str, Any] = {"type": geom_key}
    for key in ("length", "diameter", "width", "height", "radius",
                "chord", "span", "thickness", "resolution"):
        if key in d:
            geom[key] = float(d[key])

    # Fluid
    fluid_name = str(d.get("fluid", "air")).lower()
    fluid = dict(_FLUID_DB.get(fluid_name, _FLUID_DB["air"]))
    fluid["name"] = fluid_name

    # Reynolds number / physical parameters
    Re   = float(d.get("Re", d.get("reynolds", 1000.0)))
    Lref = geom.get("diameter") or geom.get("length") or 1.0
    mu   = fluid["mu"]
    rho  = fluid["rho"]
    U    = Re * mu / (rho * Lref)
    param_ranges: Dict[str, Tuple[float, float]] = {"Re": (Re * 0.5, Re * 2.0)}
    if pde_kind == "heat":
        alpha = mu / (rho * fluid.get("cp", 1000.0))
        param_ranges = {"alpha": (alpha * 0.5, alpha * 2.0)}

    # Grid
    Nx = int(d.get("Nx", default_grid[0]))
    Ny = int(d.get("Ny", default_grid[1]))
    grid_shape = (Nx, Ny)

    # Time
    n_steps = int(d.get("n_steps", default_n_steps))
    t_end   = float(d.get("t_end", Lref / max(U, 1e-6) * 5.0))
    t_end   = max(t_end, 1.0)

    # Boundary conditions
    bc_dict = dict(_DOMAIN_TO_BC.get(geom_key, {"type": "periodic"}))
    if "inlet_velocity" in d:
        bc_dict["inlet_velocity"] = float(d["inlet_velocity"])
    if "T_wall" in d:
        bc_dict["T_wall"] = float(d["T_wall"])
    bc_type = "dirichlet_zero" if geom_key in ("pipe", "channel", "backward_step") else "periodic"

    # IC
    ic_dict = {"type": str(d.get("ic_type", "random_smooth"))}

    # Name
    name = d.get("name") or f"{geom_key}_{pde_kind}_{int(Re)}"

    scenario = PhysicsScenario(
        name          = name,
        pde_kind      = pde_kind,
        grid_shape    = grid_shape,
        t_span        = (0.0, t_end),
        n_steps       = n_steps,
        domain_bounds = ((0.0, Lref), (0.0, Lref * Ny / Nx)),
        param_ranges  = param_ranges,
        ic_type       = ic_dict["type"],
        bc_type       = bc_type,
        solver        = str(d.get("solver", "builtin")),
        description   = d.get("description", f"{geom_key} {pde_kind} Re={Re:.0f}"),
        tags          = [top_domain, geom_key, pde_kind, "2d"],
    )

    sensor = d.get("sensor_config") or _default_sensor_config()
    if isinstance(sensor, dict):
        sensor_cfg = sensor
    else:
        sensor_cfg = _default_sensor_config()

    return ScenarioSpec(
        scenario            = scenario,
        domain              = top_domain,
        geometry            = geom,
        fluid               = fluid,
        boundary_conditions = bc_dict,
        initial_conditions  = ic_dict,
        sensor_config       = sensor_cfg,
    )


def _prompt_to_dict(prompt: str) -> Dict[str, Any]:
    """Heuristic keyword extractor: prompt → spec dict."""
    p = prompt.lower()

    # Geometry detection
    geom = "cavity"
    for keyword, key in [
        ("pipe", "pipe"), ("cylinder", "cylinder"), ("channel", "channel"),
        ("backward step", "backward_step"), ("cavity", "cavity"),
        ("heat transfer", "plate"), ("combustion", "combustion"),
        ("mixing", "mixing"), ("reactor", "reactor"),
        ("beam", "beam"), ("waveguide", "waveguide"),
    ]:
        if keyword in p:
            geom = key
            break

    # Fluid detection
    fluid = "air"
    for f in ("water", "oil", "hydrogen", "co2", "air"):
        if f in p:
            fluid = f
            break

    # Reynolds number
    re_match = re.search(r"re[=\s]*([0-9]+(?:\.[0-9]+)?)", p)
    Re = float(re_match.group(1)) if re_match else 1000.0

    # Domain
    domain = "fluid_dynamics"
    if any(k in p for k in ("heat", "temperature", "thermal")):
        domain = "heat_transfer"
    elif any(k in p for k in ("combust", "flame", "ignit")):
        domain = "combustion"
    elif any(k in p for k in ("mixing", "concentration", "species")):
        domain = "mass_transfer"

    return {
        "domain":   domain,
        "geometry": geom,
        "fluid":    fluid,
        "Re":       Re,
    }


def _infer_domain_from_pde(pde_kind: str) -> str:
    return {
        "ns2d":       "fluid_dynamics",
        "heat":       "heat_transfer",
        "burgers":    "fluid_dynamics",
        "wave":       "structural",
        "advection":  "mass_transfer",
        "elasticity": "structural",
    }.get(pde_kind, "fluid_dynamics")


def _default_sensor_config() -> Dict[str, Any]:
    return {
        "camera_position": [0.5, 0.5, 2.0],
        "fov":             60.0,
        "fps":             24,
        "sensors":         ["rgb", "thermal", "depth"],
        "resolution":      [256, 256],
    }
