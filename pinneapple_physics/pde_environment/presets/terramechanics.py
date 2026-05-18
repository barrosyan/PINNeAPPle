"""Terramechanics physics residuals and preset for Bekker-Wong PINN surrogate.

Usage
-----
from pinneapple_physics.pde_environment.presets.terramechanics import (
    TerramechanicsResiduals,
    bekker_wong_surrogate_2d,
)

residuals = TerramechanicsResiduals()
r_dict = residuals(model, norm_x, norm_y)
loss = sum(v.mean() for v in r_dict.values())
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .registry import register_preset
from ..spec import PDETermSpec, ProblemSpec


class TerramechanicsResiduals:
    """Four physics residuals for the Bekker-Wong PINN surrogate.

    R1 — Zero-drawbar at zero slip: F_x(s=0, z) = 0
    R2 — Mohr-Coulomb traction limit: F_x ≤ c·A + F_z·tan(phi)  (soft, ReLU²)
    R3 — Monotonicity: dF_x/ds ≥ 0 for s ∈ [0, 0.4]            (autograd)
    R4 — Torque coupling: M_y ≥ R · F_x                         (thermodynamic)

    Parameters
    ----------
    c_Pa : soil cohesion [Pa]
    phi_deg : internal friction angle [degrees]
    R_m : wheel radius [m]
    b_m : wheel width [m]
    n_phys : number of collocation points per residual
    R_factor : scale factor converting normalised M_y vs F_x for R4
    """

    def __init__(
        self,
        c_Pa: float = 1_400.0,
        phi_deg: float = 30.0,
        R_m: float = 0.125,
        b_m: float = 0.060,
        n_phys: int = 256,
        R_factor: float = 0.06,
    ):
        self.c = c_Pa
        self.tan_phi = math.tan(math.radians(phi_deg))
        self.R = R_m
        self.b = b_m
        self.n_phys = n_phys
        self.R_factor = R_factor

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        model: nn.Module,
        norm_x: Any,
        norm_y: Any,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all four physics residuals.

        Parameters
        ----------
        model : PINN model, maps normalised (s, z) -> normalised (Fx, Fz, My)
        norm_x : normaliser for inputs  (must have transform_torch / inverse)
        norm_y : normaliser for outputs (must have transform_torch / inverse)
        device : target device (defaults to first model parameter device)

        Returns
        -------
        dict with keys "r1", "r2", "r3", "r4" — each a scalar loss tensor
        """
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        n = self.n_phys

        # ------------------------------------------------------------------
        # R1: F_x(s=0) = 0
        # ------------------------------------------------------------------
        z_r1 = torch.rand(n, 1, device=device) * 0.056 + 0.002
        s_r1 = torch.zeros(n, 1, device=device)
        x_r1_raw = torch.cat([s_r1, z_r1], dim=1)
        x_r1_n = norm_x.transform_torch(x_r1_raw)
        pred_r1 = model(x_r1_n)
        fx_r1 = norm_y.inverse_torch(pred_r1)[:, 0:1]
        r1 = (fx_r1 ** 2).mean()

        # ------------------------------------------------------------------
        # R2: Mohr-Coulomb traction limit (soft, one-sided)
        # ------------------------------------------------------------------
        s_r2 = torch.rand(n, 1, device=device) * 0.75
        z_r2 = torch.rand(n, 1, device=device) * 0.056 + 0.002
        x_r2_raw = torch.cat([s_r2, z_r2], dim=1)
        x_r2_n = norm_x.transform_torch(x_r2_raw)
        pred_r2 = model(x_r2_n)
        phy_r2 = norm_y.inverse_torch(pred_r2)
        fx_r2 = phy_r2[:, 0:1]
        fz_r2 = phy_r2[:, 1:2]
        A = self.b * self.R * math.pi
        limit = self.c * A + fz_r2.detach() * self.tan_phi
        violation = torch.nn.functional.relu(fx_r2 - limit)
        r2 = (violation ** 2).mean()

        # ------------------------------------------------------------------
        # R3: dF_x/ds >= 0 for s in [0, 0.4]
        # ------------------------------------------------------------------
        s_r3_raw = torch.rand(n, 1, device=device) * 0.4
        s_r3 = s_r3_raw.detach().requires_grad_(True)
        z_r3 = torch.rand(n, 1, device=device) * 0.056 + 0.002
        x_r3_raw = torch.cat([s_r3, z_r3], dim=1)
        x_r3_n = norm_x.transform_torch(x_r3_raw)
        pred_r3 = model(x_r3_n)
        fx_r3_n = pred_r3[:, 0:1]
        dfx_ds = torch.autograd.grad(
            fx_r3_n, s_r3,
            grad_outputs=torch.ones_like(fx_r3_n),
            create_graph=True,
            retain_graph=True,
        )[0]
        violation_r3 = torch.nn.functional.relu(-dfx_ds)
        r3 = (violation_r3 ** 2).mean()

        # ------------------------------------------------------------------
        # R4: M_y >= R * F_x (torque coupling)
        # ------------------------------------------------------------------
        s_r4 = torch.rand(n, 1, device=device) * 0.75
        z_r4 = torch.rand(n, 1, device=device) * 0.056 + 0.002
        x_r4_raw = torch.cat([s_r4, z_r4], dim=1)
        x_r4_n = norm_x.transform_torch(x_r4_raw)
        pred_r4 = model(x_r4_n)
        fx_r4 = pred_r4[:, 0:1]
        my_r4 = pred_r4[:, 2:3]
        violation_r4 = torch.nn.functional.relu(self.R_factor * fx_r4 - my_r4)
        r4 = (violation_r4 ** 2).mean()

        return {"r1": r1, "r2": r2, "r3": r3, "r4": r4}


# ---------------------------------------------------------------------------
# Preset factory
# ---------------------------------------------------------------------------

@register_preset("bekker_wong_surrogate_2d")
def bekker_wong_surrogate_2d(
    c_Pa: float = 1_400.0,
    phi_deg: float = 30.0,
    R_m: float = 0.125,
    b_m: float = 0.060,
) -> ProblemSpec:
    """Return a ProblemSpec for the 2-D Bekker-Wong PINN surrogate.

    Inputs : (slip_ratio, sinkage_m)
    Outputs: (F_x, F_z, M_y)
    """
    pde = PDETermSpec(
        kind="bekker_wong_terramechanics",
        fields=("Fx", "Fz", "My"),
        coords=("slip", "sinkage"),
        params={
            "c_Pa": float(c_Pa),
            "phi_deg": float(phi_deg),
            "R_m": float(R_m),
            "b_m": float(b_m),
        },
        meta={
            "description": "Bekker-Wong rigid-wheel / deformable-soil surrogate",
            "physics_constraints": [
                "R1: Fx(s=0) = 0",
                "R2: Fx <= c*A + Fz*tan(phi)",
                "R3: dFx/ds >= 0 for s in [0, 0.4]",
                "R4: My >= R*Fx",
            ],
        },
    )
    return ProblemSpec(
        name="bekker_wong_surrogate_2d",
        dim=2,
        coords=("slip", "sinkage"),
        fields=("Fx", "Fz", "My"),
        pde=pde,
        domain_bounds={"slip": (0.0, 0.75), "sinkage": (0.002, 0.058)},
        meta={"description": "Bekker-Wong rigid-wheel terramechanics surrogate for rover simulation"},
    )


__all__ = ["TerramechanicsResiduals", "bekker_wong_surrogate_2d"]
