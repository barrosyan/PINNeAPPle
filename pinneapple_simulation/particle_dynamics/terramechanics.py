"""Terramechanics parameter dataclasses for Bekker-Wong wheel-soil interaction."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SoilParams:
    """Bekker-Wong soil parameters.

    Defaults correspond to GRC-1 lunar regolith simulant (compacted).
    """
    c: float = 1_400.0       # cohesion [Pa]
    phi_deg: float = 30.0    # internal friction angle [deg]
    K: float = 0.018         # shear deformation modulus [m]
    k_c: float = 1_370.0     # Bekker cohesion modulus [N/m^(n+1)]
    k_phi: float = 814_000.0 # Bekker friction modulus [N/m^(n+2)]
    n: float = 1.0           # sinkage exponent [-]
    rho: float = 1_650.0     # bulk density [kg/m^3]
    g: float = 1.62          # gravitational acceleration [m/s^2] (lunar default)
    a0: float = 0.40         # contact angle coefficient a0 (Wong 1978)
    a1: float = 0.15         # contact angle coefficient a1 (Wong 1978)

    @property
    def phi_rad(self) -> float:
        return math.radians(self.phi_deg)

    @property
    def tan_phi(self) -> float:
        return math.tan(self.phi_rad)


@dataclass
class WheelParams:
    """Rigid wheel geometry parameters."""
    R: float = 0.125         # wheel radius [m]
    b: float = 0.060         # wheel width [m]
    n_wheels: int = 6        # number of driven wheels
    mass_rover: float = 40.0 # total rover mass [kg]

    def weight_per_wheel(self, g: float) -> float:
        """Normal load carried by one wheel [N]."""
        return self.mass_rover * g / self.n_wheels


__all__ = ["SoilParams", "WheelParams"]
