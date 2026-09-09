"""pinneapple_analysis.inspection.eddy_current -- probe-scan synthetic
signal generation and physics-informed training for eddy-current NDE, on
top of the *existing* axisymmetric complex-Helmholtz FDM solver in
``pinneapple_simulation.numerical_solvers.eddy_current_fdm``.

This module does NOT reimplement the Helmholtz solve -- it calls
``solve_axisymmetric_eddy_current`` / ``annular_current_source`` /
``eddy_current_density`` directly, exactly per that module's own
documented functional API, and adds two things that module doesn't
have: (1) a synthetic probe-SCAN signal (a coil response as a function
of lateral scan position, not just a single field solve) and (2) a PINN
training path whose physics loss reuses the same governing equation.

Probe-scan approximation (read before using for anything beyond a demo)
------------------------------------------------------------------------
``eddy_current_fdm`` solves the AXISYMMETRIC (2D, rotationally
symmetric about the z-axis) magnetostatic/eddy-current equation. A real
eddy-current probe scan moving laterally across a small localized flaw
(e.g. a crack) is NOT axisymmetric -- it is a fully 3D problem. Rather
than build a 3D solver (out of scope here), this module keeps the
existing axisymmetric solver and reinterprets the scan as follows: a
small disk-shaped conductivity anomaly ("defect") is placed inside the
conductor at a radius ``coil_r_center + scan_position`` for each
requested lateral scan offset, and the coil-region eddy-current-density
response is recorded relative to a defect-free baseline solve. This is
an axisymmetric SURROGATE for a lateral B-scan over a radially
symmetric flaw (e.g. a corrosion pit or annular groove directly under
the probe's sweep line) -- it is NOT a literal 3D moving-probe-over-a-
crack simulation, and is documented here as an explicit, honest
simplification, not a validated instrument model.

Governing equation (reused from eddy_current_fdm.py's own docstring)
------------------------------------------------------------------------
    d2A/dr2 + (1/r) dA/dr - A/r2 + d2A/dz2 - i*omega*mu*sigma*A = -mu*J

The PINN physics loss below enforces this equation (with J=0, evaluated
in the homogeneous conductor bulk away from the coil and the defect,
where the FDM ground truth already satisfies it exactly) on a PINN with
2 real output channels (Re(A), Im(A)) via autograd -- the complex
Helmholtz operator split into its real/imaginary parts:

    L[A_r] + omega*mu*sigma*A_i = 0
    L[A_i] - omega*mu*sigma*A_r = 0

where L[.] = d2(.)/dr2 + (1/r)d(.)/dr - (.)/r2 + d2(.)/dz2. Because this
homogeneous-bulk equation is linear and source-free, it is invariant
under any constant rescaling of A -- so training against a numerically
rescaled target (done below purely for gradient conditioning) does not
change what the residual being driven to zero means physically.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from pinneapple_simulation.numerical_solvers.eddy_current_fdm import (
    MU0,
    annular_current_source,
    eddy_current_density,
    solve_axisymmetric_eddy_current,
)

from .base import InspectionPINN, PhysicsLossFn, train_generic_inspection_pinn


def generate_eddy_current_synthetic(
    scan_positions: Optional[np.ndarray] = None,
    *,
    nr: int = 24,
    nz: int = 40,
    r_bounds: tuple = (1e-3, 0.05),
    z_bounds: tuple = (-0.03, 0.03),
    omega: float = 2.0 * np.pi * 100e3,
    sigma_bulk: float = 5.0e6,   # S/m, illustrative (steel-like order of magnitude)
    mu_r: float = 1.0,
    coil_r_center: float = 0.02,
    coil_z_center: float = 0.012,
    coil_width_r: float = 0.004,
    coil_width_z: float = 0.004,
    coil_turns: float = 100.0,
    coil_current: float = 1.0,
    defect_radius: float = 0.003,
    defect_z_center: float = -0.006,
    defect_sigma_frac: float = 0.2,
    n_supervised: int = 400,
    n_collocation: int = 200,
    seed: Optional[int] = 0,
) -> Dict[str, Any]:
    """Generate a synthetic eddy-current probe-scan dataset.

    Physically: a lumped coil (via `annular_current_source`, the same
    lumped-coil source term eddy_current_fdm.py's own docstring
    describes) sits above a conductive half-space (z < 0 is the
    conductor, z > 0 is air, sigma=0). The coil is driven at angular
    frequency `omega`. `solve_axisymmetric_eddy_current` gives the
    baseline vector-potential field A(r, z). For each requested lateral
    `scan_position`, a small disk-shaped conductivity defect
    (`sigma *= defect_sigma_frac` inside a disk of `defect_radius`) is
    placed at radius ``coil_r_center + scan_position`` (see module
    docstring for why this is an axisymmetric surrogate for a lateral
    scan, not a literal 3D one) and the field is re-solved; the
    DIFFERENTIAL coil-region eddy-current density (defect minus
    baseline) is the synthetic probe-scan "signal" -- exactly the
    differential impedance-plane signal a real eddy-current instrument
    displays for a flaw indication.

    Returns a dict with:
      r, z            -- SAME 1D grids passed to solve_axisymmetric_eddy_current
      A_baseline      -- (len(r), len(z)) complex, SAME shape/dtype convention
                          as solve_axisymmetric_eddy_current's own return value
      scan_positions  -- (n_scan,) lateral offsets (m)
      scan_signal     -- (n_scan,) complex differential probe signal
      x, y            -- InspectionPINN supervised training pairs: (r, z)
                          coordinates and the (rescaled) [Re(A), Im(A)] target,
                          subsampled from the baseline field
      x_col           -- (n_collocation, 2) points in the homogeneous
                          conductor bulk (away from coil and defect), used
                          for the physics residual
      a_scale, omega, mu, sigma_bulk -- parameters needed to rebuild/interpret
                          the physics residual and undo the target rescaling
    """
    rng = np.random.default_rng(seed)
    if scan_positions is None:
        scan_positions = np.linspace(-0.01, 0.01, 9)
    scan_positions = np.asarray(scan_positions, dtype=np.float64)

    r = np.linspace(r_bounds[0], r_bounds[1], nr)
    z = np.linspace(z_bounds[0], z_bounds[1], nz)
    RR, ZZ = np.meshgrid(r, z, indexing="ij")

    mu = MU0 * mu_r * np.ones((nr, nz))
    sigma_baseline = np.zeros((nr, nz))
    sigma_baseline[ZZ < 0.0] = sigma_bulk

    J_source = annular_current_source(
        r, z, coil_r_center, coil_z_center, coil_width_z, coil_width_r, coil_turns, coil_current,
    )

    A_baseline = solve_axisymmetric_eddy_current(r, z, omega, mu, sigma_baseline, J_source)
    J_baseline = eddy_current_density(A_baseline, omega, sigma_baseline)

    # Pickup region: the classic skin-depth argument (delta =
    # sqrt(2/(omega*mu*sigma))) says induced eddy currents are
    # concentrated within a few skin depths of the surface directly
    # beneath the coil -- this is where a real probe's impedance change
    # is actually sensitive, NOT the (non-conductive, sigma=0) air gap
    # where the coil itself sits.
    skin_depth = float(np.sqrt(2.0 / (omega * MU0 * mu_r * sigma_bulk)))
    pickup_mask = (
        (ZZ < 0.0)
        & (ZZ >= -3.0 * skin_depth)
        & (np.abs(RR - coil_r_center) <= coil_width_r)
    )
    if not pickup_mask.any():
        pickup_mask = np.zeros_like(RR, dtype=bool)
        pickup_mask[np.argmin(np.abs(r - coil_r_center)), np.argmin(np.abs(z - (-skin_depth)))] = True

    scan_signal = np.zeros(len(scan_positions), dtype=complex)
    for k, s in enumerate(scan_positions):
        r_defect = coil_r_center + float(s)
        sigma_defect = sigma_baseline.copy()
        mask = ((RR - r_defect) ** 2 + (ZZ - defect_z_center) ** 2) <= defect_radius ** 2
        mask &= ZZ < 0.0
        sigma_defect[mask] *= defect_sigma_frac

        A_defect = solve_axisymmetric_eddy_current(r, z, omega, mu, sigma_defect, J_source)
        J_defect = eddy_current_density(A_defect, omega, sigma_defect)

        scan_signal[k] = np.mean(J_defect[pickup_mask] - J_baseline[pickup_mask])

    # --- Supervised training pairs: PINN learns to reproduce the baseline field ---
    a_scale = float(np.max(np.abs(A_baseline)))
    if a_scale <= 0.0:
        a_scale = 1.0
    n_sup = min(n_supervised, nr * nz)
    flat_idx = rng.choice(nr * nz, size=n_sup, replace=False)
    ri = flat_idx // nz
    zi = flat_idx % nz
    x_sup = np.stack([r[ri], z[zi]], axis=1).astype(np.float32)
    A_sup = A_baseline[ri, zi] / a_scale
    y_sup = np.stack([A_sup.real, A_sup.imag], axis=1).astype(np.float32)

    # --- Collocation points: homogeneous conductor bulk, away from coil/defect ---
    r_margin = 0.2 * (r_bounds[1] - r_bounds[0])
    r_col = rng.uniform(r_bounds[0] + r_margin, r_bounds[1] - r_margin, size=n_collocation)
    z_col = rng.uniform(z_bounds[0] * 0.8, -0.002, size=n_collocation)
    x_col = np.stack([r_col, z_col], axis=1).astype(np.float32)

    return {
        "r": r,
        "z": z,
        "A_baseline": A_baseline,
        "scan_positions": scan_positions,
        "scan_signal": scan_signal,
        "x": x_sup,
        "y": y_sup,
        "x_col": x_col,
        "a_scale": a_scale,
        "omega": float(omega),
        "mu": float(MU0 * mu_r),
        "sigma_bulk": float(sigma_bulk),
    }


def make_eddy_current_physics_loss(omega: float, mu: float, sigma: float) -> PhysicsLossFn:
    """Build a `(model, batch) -> (loss, comps)` physics-loss callable
    (the ``PhysicsLossHook`` contract) enforcing the real/imaginary split
    of the axisymmetric complex-Helmholtz operator from
    ``eddy_current_fdm.build_axisymmetric_helmholtz_system``, with J=0
    (valid in the homogeneous conductor bulk collocation region -- see
    `generate_eddy_current_synthetic`)."""

    def physics_loss_fn(model: torch.nn.Module, batch: Dict[str, Any]):
        x_col = batch["x_col"]
        if not x_col.requires_grad:
            x_col = x_col.clone().requires_grad_(True)

        out = model.predict(x_col) if hasattr(model, "predict") else model(x_col)
        A_r = out[:, 0:1]
        A_i = out[:, 1:2]
        r = x_col[:, 0:1]

        gAr = torch.autograd.grad(A_r, x_col, grad_outputs=torch.ones_like(A_r), create_graph=True, retain_graph=True)[0]
        gAi = torch.autograd.grad(A_i, x_col, grad_outputs=torch.ones_like(A_i), create_graph=True, retain_graph=True)[0]
        dAr_dr, dAr_dz = gAr[:, 0:1], gAr[:, 1:2]
        dAi_dr, dAi_dz = gAi[:, 0:1], gAi[:, 1:2]

        d2Ar_dr2 = torch.autograd.grad(dAr_dr, x_col, grad_outputs=torch.ones_like(dAr_dr), create_graph=True, retain_graph=True)[0][:, 0:1]
        d2Ar_dz2 = torch.autograd.grad(dAr_dz, x_col, grad_outputs=torch.ones_like(dAr_dz), create_graph=True, retain_graph=True)[0][:, 1:2]
        d2Ai_dr2 = torch.autograd.grad(dAi_dr, x_col, grad_outputs=torch.ones_like(dAi_dr), create_graph=True, retain_graph=True)[0][:, 0:1]
        d2Ai_dz2 = torch.autograd.grad(dAi_dz, x_col, grad_outputs=torch.ones_like(dAi_dz), create_graph=True, retain_graph=True)[0][:, 1:2]

        r_safe = torch.clamp(r, min=1e-4)
        L_Ar = d2Ar_dr2 + dAr_dr / r_safe - A_r / r_safe ** 2 + d2Ar_dz2
        L_Ai = d2Ai_dr2 + dAi_dr / r_safe - A_i / r_safe ** 2 + d2Ai_dz2

        res_real = L_Ar + omega * mu * sigma * A_i
        res_imag = L_Ai - omega * mu * sigma * A_r
        residual = torch.cat([res_real, res_imag], dim=1)

        loss = torch.mean(residual ** 2)
        comps = {"residual_rmse": float(torch.sqrt(torch.mean(residual.detach() ** 2)).item())}
        return loss, comps

    return physics_loss_fn


def train_eddy_current(
    synthetic_data: Optional[Dict[str, Any]] = None,
    *,
    hidden=(64, 64, 64),
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cpu",
    seed: Optional[int] = 0,
    log_dir: str = "runs/inspection_eddy_current",
    run_name: str = "eddy_current_pinn",
    w_supervised: float = 1.0,
    w_physics: float = 1.0,
    **synthetic_kwargs: Any,
) -> Dict[str, Any]:
    """Generate (if not supplied) eddy-current synthetic data, build an
    `InspectionPINN` and train it via `train_generic_inspection_pinn`
    with the Helmholtz-residual physics loss above."""
    if synthetic_data is None:
        synthetic_data = generate_eddy_current_synthetic(seed=seed, **synthetic_kwargs)

    model = InspectionPINN(in_dim=2, out_dim=2, hidden=list(hidden), modality="eddy_current")
    physics_fn = make_eddy_current_physics_loss(
        omega=synthetic_data["omega"], mu=synthetic_data["mu"], sigma=synthetic_data["sigma_bulk"],
    )
    train_out = train_generic_inspection_pinn(
        model, synthetic_data, physics_fn,
        epochs=epochs, lr=lr, batch_size=batch_size, device=device, seed=seed,
        log_dir=log_dir, run_name=run_name, w_supervised=w_supervised, w_physics=w_physics,
    )
    return {"model": model, "train_out": train_out, "synthetic_data": synthetic_data}
