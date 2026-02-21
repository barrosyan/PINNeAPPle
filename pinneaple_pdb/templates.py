"""Physical schema templates for common datasets (e.g. atmosphere reanalysis)."""

from typing import Dict, Any


def schema_templates() -> Dict[str, Dict[str, Any]]:
    """Return mapping of template_id -> PhysicalSchema-compatible dict."""
    return {
        "atmosphere_reanalysis_v1": {
            "physical_system": "atmosphere",
            "governing_equations": {
                "name": "primitive_equations_hydrostatic",
                "constraints": ["mass_continuity", "momentum_rotating_frame", "thermodynamics", "moisture"],
                "assumptions": ["hydrostatic", "ideal_gas", "thin_shell"],
            },
            "ics": {"type": "windowed_time_series"},
            "bcs": {"type": "global_sphere_periodic_lon"},
            "forcings": {"notes": "implicit in reanalysis/model system"},
            "units_policy": {"require_units": True, "convert_to": "SI"},
            "regime_tags": [],
            "validity": {"notes": "Assimilated/model state estimate; not direct observation."},
            "version": "upd/v1",
        }
    }
