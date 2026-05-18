"""Bekker-Wong numerical solver for rigid-wheel / deformable-soil interaction.

References
----------
- Bekker, M.G. (1969). Introduction to Terrain-Vehicle Systems. U Michigan Press.
- Wong, J.Y. (1978). Theory of Ground Vehicles. Wiley.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry
from ..particle_dynamics.terramechanics import SoilParams, WheelParams

try:
    from scipy.integrate import quad as _quad
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


@SolverRegistry.register(
    name="bekker_wong",
    family="terramechanics",
    description="Bekker-Wong rigid-wheel / deformable-soil force model",
    tags=["rover", "lunar", "soil", "terramechanics"],
)
class BekkerWongSolver(SolverBase):
    """Numerical Bekker-Wong solver.

    Computes drawbar pull (F_x), normal load (F_z), and driving torque (M_y)
    for a rigid wheel rolling on deformable soil via scipy numerical integration.

    Parameters
    ----------
    soil : SoilParams
    wheel : WheelParams
    """

    def __init__(
        self,
        soil: Optional[SoilParams] = None,
        wheel: Optional[WheelParams] = None,
    ):
        super().__init__()
        self.soil = soil or SoilParams()
        self.wheel = wheel or WheelParams()

    # ------------------------------------------------------------------
    # Contact angle geometry
    # ------------------------------------------------------------------

    def _contact_angles(self, z: float) -> Tuple[float, float, float]:
        """Compute entry, mid, and exit contact angles for sinkage z [m]."""
        R = self.wheel.R
        s = self.soil
        theta_f = math.acos(1.0 - z / R)
        theta_m = (s.a0 + s.a1 * 0.0) * theta_f   # evaluated at s=0 baseline
        theta_r = -theta_f / 3.0
        return theta_f, theta_m, theta_r

    def _contact_angles_slip(
        self, z: float, slip: float
    ) -> Tuple[float, float, float]:
        """Contact angles accounting for slip ratio."""
        R = self.wheel.R
        s = self.soil
        theta_f = math.acos(max(-1.0, min(1.0, 1.0 - z / R)))
        theta_m = (s.a0 + s.a1 * slip) * theta_f
        theta_r = -theta_f / 3.0
        return theta_f, theta_m, theta_r

    # ------------------------------------------------------------------
    # Stress distributions
    # ------------------------------------------------------------------

    def sigma(self, theta: float, z: float, slip: float) -> float:
        """Normal stress at contact angle theta [Pa] using Bekker pressure-sinkage."""
        s = self.soil
        R = self.wheel.R
        theta_f, theta_m, theta_r = self._contact_angles_slip(z, slip)
        ksn = s.k_c / self.wheel.b + s.k_phi

        if theta >= theta_m:
            h = R * max(math.cos(theta) - math.cos(theta_f), 0.0)
        else:
            denom = max(theta_m - theta_r, 1e-9)
            ratio = (theta_f - theta_m) * (theta - theta_r) / denom
            h = R * max(math.cos(theta_f - ratio) - math.cos(theta_f), 0.0)

        return ksn * (h ** s.n)

    def tau(self, theta: float, z: float, slip: float) -> float:
        """Shear stress at contact angle theta [Pa] — Mohr-Coulomb + Wong displacement."""
        s = self.soil
        R = self.wheel.R
        theta_f, _, _ = self._contact_angles_slip(z, slip)
        sig = self.sigma(theta, z, slip)
        j = R * ((theta_f - theta) - (1.0 - slip) * (math.sin(theta_f) - math.sin(theta)))
        return (s.c + sig * s.tan_phi) * (1.0 - math.exp(-j / s.K))

    # ------------------------------------------------------------------
    # Force integration
    # ------------------------------------------------------------------

    def forces(
        self, slip: float, z: float
    ) -> Tuple[float, float, float]:
        """Compute (F_x, F_z, M_y) via numerical integration over contact patch.

        Parameters
        ----------
        slip : slip ratio [-], 0 = free rolling, 1 = locked wheel
        z : sinkage [m]

        Returns
        -------
        F_x : drawbar pull [N]
        F_z : normal load [N]
        M_y : driving torque [N·m]
        """
        if not _SCIPY_AVAILABLE:
            raise ImportError("scipy is required for BekkerWongSolver.forces()")

        R = self.wheel.R
        b = self.wheel.b
        theta_f, _, theta_r = self._contact_angles_slip(z, slip)

        def integrand_fx(theta: float) -> float:
            return self.tau(theta, z, slip) * math.cos(theta) - self.sigma(theta, z, slip) * math.sin(theta)

        def integrand_fz(theta: float) -> float:
            return self.sigma(theta, z, slip) * math.cos(theta) + self.tau(theta, z, slip) * math.sin(theta)

        def integrand_my(theta: float) -> float:
            return self.tau(theta, z, slip)

        Fx, _ = _quad(integrand_fx, theta_r, theta_f, limit=80)
        Fz, _ = _quad(integrand_fz, theta_r, theta_f, limit=80)
        My, _ = _quad(integrand_my, theta_r, theta_f, limit=80)

        return R * b * Fx, R * b * Fz, R * R * b * My

    def sinkage_from_load(
        self,
        W: float,
        slip: float,
        z_init: float = 0.01,
        tol: float = 0.5,
        max_iter: int = 50,
    ) -> float:
        """Find sinkage z such that F_z ≈ W using secant iteration."""
        z0, z1 = z_init, z_init * 1.1
        for _ in range(max_iter):
            f0 = self.forces(slip, z0)[1] - W
            f1 = self.forces(slip, z1)[1] - W
            if abs(f1 - f0) < 1e-12:
                break
            z2 = z1 - f1 * (z1 - z0) / (f1 - f0)
            z2 = max(1e-4, min(z2, 0.20))
            if abs(z2 - z1) < tol * 1e-3:
                return z2
            z0, z1 = z1, z2
        return z1

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------

    def generate_dataset(
        self,
        n_slip: int = 40,
        n_sink: int = 30,
        slip_range: Tuple[float, float] = (0.0, 0.75),
        sink_range: Tuple[float, float] = (0.002, 0.058),
        n_lhs: int = 500,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate (slip, sinkage) → (F_x, F_z, M_y) dataset.

        Returns
        -------
        X : array of shape (N, 2) — [slip, sinkage_m]
        Y : array of shape (N, 3) — [F_x, F_z, M_y]
        """
        rng = np.random.default_rng(seed)

        s_grid = np.linspace(*slip_range, n_slip)
        z_grid = np.linspace(*sink_range, n_sink)
        ss, zz = np.meshgrid(s_grid, z_grid)
        grid_pts = np.stack([ss.ravel(), zz.ravel()], axis=1)

        # Latin hypercube sampling
        lhs = np.column_stack([
            rng.uniform(*slip_range, n_lhs),
            rng.uniform(*sink_range, n_lhs),
        ])

        pts = np.vstack([grid_pts, lhs])
        Y_list = []
        for slip, z in pts:
            try:
                fx, fz, my = self.forces(float(slip), float(z))
            except Exception:
                fx, fz, my = 0.0, 0.0, 0.0
            Y_list.append([fx, fz, my])

        return pts.astype(np.float32), np.array(Y_list, dtype=np.float32)

    # ------------------------------------------------------------------
    # SolverBase interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> SolverOutput:
        """Evaluate forces for a batch of (slip, sinkage) inputs.

        Parameters
        ----------
        x : Tensor of shape (N, 2) — columns [slip, sinkage_m]

        Returns
        -------
        SolverOutput with result of shape (N, 3) — [F_x, F_z, M_y]
        """
        results = []
        for row in x.detach().cpu().numpy():
            fx, fz, my = self.forces(float(row[0]), float(row[1]))
            results.append([fx, fz, my])
        result = torch.tensor(results, dtype=x.dtype, device=x.device)
        return SolverOutput(result=result, losses={}, extras={})


__all__ = ["BekkerWongSolver"]
