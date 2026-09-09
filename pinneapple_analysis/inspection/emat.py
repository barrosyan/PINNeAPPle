"""pinneapple_analysis.inspection.emat -- a deliberately simplified
Electromagnetic Acoustic Transducer (EMAT) synthetic signal generator
and physics-informed training path.

What a real EMAT does (for context, not what this module solves)
-------------------------------------------------------------------
An EMAT couples an RF coil's induced eddy currents with a static bias
magnetic field: the resulting Lorentz body force ``F = J x B0`` launches
an elastic wave directly in a conductive test-piece, with no couplant
(unlike a piezoelectric transducer). A first-principles EMAT model is a
COUPLED electromagnetic + elastodynamic problem (see Hirao & Ogi,
"EMATs for Science and Industry: Noncontacting Ultrasonic
Measurements", Springer, 2003, and Ogi, J. Appl. Phys. 82, 1997, for the
Lorentz-force EMAT generation mechanism) -- a full 2D/3D solve of that
kind is well beyond this module's scope (it would essentially require
coupling ``eddy_current_fdm.py``'s Helmholtz solve to a 2D elastic-wave
solver, a substantial undertaking of its own).

What this module actually does (explicit simplification)
-----------------------------------------------------------
The electromagnetic generation step is reduced to a SCALAR forcing
term ``F(x, t) = J0 * envelope(t) * footprint(x)`` (a burst-modulated,
spatially-localized force under the coil footprint -- footprint(x) is
a Gaussian standing in for the coil's real current-density profile,
and envelope(t) is a tone-burst standing in for the Lorentz-force
time-history), which drives a 1D scalar elastic wave equation

    rho * u_tt = G(x) * u_xx + F(x, t)

with a local shear-modulus reduction ``G(x)`` modeling a defect
(thinning/void), the same "local property change reflects/attenuates a
wave packet" NDE signature used throughout this package. Solved via the
same explicit FDTD scheme as ``pinneapple_analysis.inspection.acoustic``.
This is honestly a 1D scalar surrogate, not a validated EMAT model --
its purpose is to demonstrate the "EM source term + elastic-wave
response" PINN training pattern the plan calls for, not to predict a
real instrument's output.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from .base import InspectionPINN, PhysicsLossFn, train_generic_inspection_pinn


def _tone_burst_envelope(t: np.ndarray, t0: float, f0: float, n_cycles: float) -> np.ndarray:
    """Hann-windowed tone burst centered at t0, carrier frequency f0."""
    dur = n_cycles / f0
    tau = t - t0
    window = np.where(np.abs(tau) <= dur / 2, 0.5 * (1 + np.cos(2 * np.pi * tau / dur)), 0.0)
    return window * np.sin(2 * np.pi * f0 * tau)


def _fdtd_forced_wave_1d(x: np.ndarray, c: np.ndarray, dt: float, nt: int, F: np.ndarray):
    """Explicit FDTD solve of u_tt = c(x)^2 u_xx + F(x,t)/rho_ref (F
    already includes the 1/rho factor), free-free ends, quiescent
    initial condition. F has shape (nt, nx)."""
    nx = len(x)
    dx = x[1] - x[0]
    u = np.zeros(nx)
    u_prev = np.zeros(nx)
    history = np.zeros((nt, nx))
    r2 = (c * dt / dx) ** 2
    for n in range(1, nt):
        u_new = np.empty(nx)
        u_new[1:-1] = (
            2 * u[1:-1] - u_prev[1:-1]
            + r2[1:-1] * (u[2:] - 2 * u[1:-1] + u[:-2])
            + dt ** 2 * F[n, 1:-1]
        )
        u_new[0] = 2 * u[0] - u_prev[0] + r2[0] * 2.0 * (u[1] - u[0]) + dt ** 2 * F[n, 0]
        u_new[-1] = 2 * u[-1] - u_prev[-1] + r2[-1] * 2.0 * (u[-2] - u[-1]) + dt ** 2 * F[n, -1]
        u_prev, u = u, u_new
        history[n] = u
    return history


def generate_emat_synthetic(
    *,
    L: float = 0.3,
    rho: float = 7800.0,   # kg/m^3, steel-like
    G0: float = 8.0e10,    # Pa, shear-modulus-like stiffness (steel-like order of magnitude)
    n_points: int = 60,
    duration: float = 0.00006,
    coil_position: float = 0.05,   # fraction of L -- EM source (transducer) footprint center
    coil_footprint: float = 0.02,  # fraction of L
    burst_freq: float = 1.0e6,     # Hz, typical EMAT/ultrasonic burst frequency
    n_cycles: float = 3.0,
    J0: float = 5.0e9,             # illustrative forcing amplitude (N/m^3-equivalent)
    receiver_position: float = 0.85,  # fraction of L (pitch-catch receiver)
    defect_position: float = 0.5,     # fraction of L
    defect_width: float = 0.02,       # fraction of L
    defect_severity: float = 0.4,     # fractional stiffness loss
    n_supervised: int = 400,
    n_collocation: int = 200,
    seed: Optional[int] = 0,
) -> Dict[str, Any]:
    """Generate a synthetic EMAT pitch-catch signal: a Lorentz-force-like
    tone burst launches an elastic wave from the coil footprint; a defect
    (local stiffness drop) partially reflects/attenuates it before it
    reaches the receiver.

    Returns a dict with:
      x_grid, t_grid    -- 1D grids
      c_field           -- (nx,) wave-speed map (sqrt(G(x)/rho))
      F_field           -- (nt, nx) EM forcing term used to drive the wave
      u_field           -- (nt, nx) displacement history
      receiver_trace    -- u(t) at the receiver
      x, y, x_col       -- InspectionPINN training data (see other modules)
      c_scale, u_scale  -- normalization info
    """
    rng = np.random.default_rng(seed)

    x = np.linspace(0.0, L, n_points)
    dx = x[1] - x[0]
    c0 = float(np.sqrt(G0 / rho))
    dt = 0.3 * dx / c0
    nt = max(16, int(round(duration / dt)))
    t = np.arange(nt) * dt

    coil_x0 = coil_position * L
    footprint_sigma = 0.5 * coil_footprint * L
    footprint = np.exp(-0.5 * ((x - coil_x0) / max(footprint_sigma, 1e-9)) ** 2)

    t0_burst = 3.0 * n_cycles / burst_freq / 2.0
    envelope_t = _tone_burst_envelope(t, t0_burst, burst_freq, n_cycles)
    F_field = (J0 / rho) * np.outer(envelope_t, footprint)

    defect_x0 = defect_position * L
    defect_halfwidth = 0.5 * defect_width * L
    in_defect = np.abs(x - defect_x0) <= defect_halfwidth
    G_field = G0 * np.ones(n_points)
    G_field[in_defect] = G0 * max(1e-3, 1.0 - defect_severity)
    c_field = np.sqrt(G_field / rho)

    u_field = _fdtd_forced_wave_1d(x, c_field, dt, nt, F_field)

    receiver_idx = int(round(receiver_position * (n_points - 1)))
    receiver_trace = u_field[:, receiver_idx]

    u_scale = float(np.max(np.abs(u_field)))
    if u_scale <= 0:
        u_scale = 1.0

    n_sup = min(n_supervised, nt * n_points)
    flat_idx = rng.choice(nt * n_points, size=n_sup, replace=False)
    ti, xi = flat_idx // n_points, flat_idx % n_points
    x_sup = np.stack([x[xi], t[ti]], axis=1).astype(np.float32)
    y_sup = (u_field[ti, xi] / u_scale).astype(np.float32)[:, None]

    x_col = np.stack(
        [rng.uniform(x[1], x[-2], n_collocation), rng.uniform(t[1], t[-2], n_collocation)], axis=1,
    ).astype(np.float32)

    return {
        "x_grid": x,
        "t_grid": t,
        "c_field": c_field,
        "F_field": F_field,
        "u_field": u_field,
        "receiver_trace": receiver_trace,
        "x": x_sup,
        "y": y_sup,
        "x_col": x_col,
        "u_scale": u_scale,
        "rho": float(rho),
        "footprint": footprint,
        "envelope_t": envelope_t,
        "coil_x0": float(coil_x0),
        "footprint_sigma": float(footprint_sigma),
        "burst_freq": float(burst_freq),
        "n_cycles": float(n_cycles),
        "t0_burst": float(t0_burst),
        "J0_over_rho": float(J0 / rho),
    }


def make_emat_physics_loss(
    x_grid: np.ndarray,
    c_field: np.ndarray,
    coil_x0: float,
    footprint_sigma: float,
    burst_freq: float,
    n_cycles: float,
    t0_burst: float,
    J0_over_rho: float,
) -> PhysicsLossFn:
    """Build a `(model, batch) -> (loss, comps)` physics loss enforcing
    the FORCED wave equation ``u_tt - c(x)^2 u_xx - F(x,t) = 0``, with
    ``c(x)`` looked up from a fixed material map (as in the `acoustic`
    module) and ``F(x,t)`` evaluated in CLOSED FORM (it is an explicit
    function of x,t used to generate the ground truth, so no
    interpolation table is needed for it)."""
    def c_of_x(x_np: np.ndarray) -> np.ndarray:
        return np.interp(x_np, x_grid, c_field)

    def F_of_xt(x_np: np.ndarray, t_np: np.ndarray) -> np.ndarray:
        footprint = np.exp(-0.5 * ((x_np - coil_x0) / max(footprint_sigma, 1e-9)) ** 2)
        dur = n_cycles / burst_freq
        tau = t_np - t0_burst
        window = np.where(np.abs(tau) <= dur / 2, 0.5 * (1 + np.cos(2 * np.pi * tau / dur)), 0.0)
        envelope = window * np.sin(2 * np.pi * burst_freq * tau)
        return J0_over_rho * envelope * footprint

    def physics_loss_fn(model: torch.nn.Module, batch: Dict[str, Any]):
        x_col = batch["x_col"]
        if not x_col.requires_grad:
            x_col = x_col.clone().requires_grad_(True)

        u = model.predict(x_col) if hasattr(model, "predict") else model(x_col)
        u = u[:, 0:1]

        g = torch.autograd.grad(u, x_col, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        du_dx, du_dt = g[:, 0:1], g[:, 1:2]
        d2u_dx2 = torch.autograd.grad(du_dx, x_col, grad_outputs=torch.ones_like(du_dx), create_graph=True, retain_graph=True)[0][:, 0:1]
        d2u_dt2 = torch.autograd.grad(du_dt, x_col, grad_outputs=torch.ones_like(du_dt), create_graph=True, retain_graph=True)[0][:, 1:2]

        x_np = x_col[:, 0:1].detach().cpu().numpy()
        t_np = x_col[:, 1:2].detach().cpu().numpy()
        c_vals = c_of_x(x_np)
        F_vals = F_of_xt(x_np, t_np)
        c_t = torch.as_tensor(c_vals, dtype=u.dtype, device=u.device).reshape(-1, 1)
        F_t = torch.as_tensor(F_vals, dtype=u.dtype, device=u.device).reshape(-1, 1)

        residual = d2u_dt2 - (c_t ** 2) * d2u_dx2 - F_t
        loss = torch.mean(residual ** 2)
        comps = {"residual_rmse": float(torch.sqrt(torch.mean(residual.detach() ** 2)).item())}
        return loss, comps

    return physics_loss_fn


def train_emat(
    synthetic_data: Optional[Dict[str, Any]] = None,
    *,
    hidden=(64, 64, 64),
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cpu",
    seed: Optional[int] = 0,
    log_dir: str = "runs/inspection_emat",
    run_name: str = "emat_pinn",
    w_supervised: float = 1.0,
    w_physics: float = 1.0,
    **synthetic_kwargs: Any,
) -> Dict[str, Any]:
    if synthetic_data is None:
        synthetic_data = generate_emat_synthetic(seed=seed, **synthetic_kwargs)

    model = InspectionPINN(in_dim=2, out_dim=1, hidden=list(hidden), modality="emat")
    physics_fn = make_emat_physics_loss(
        synthetic_data["x_grid"], synthetic_data["c_field"],
        synthetic_data["coil_x0"], synthetic_data["footprint_sigma"],
        synthetic_data["burst_freq"], synthetic_data["n_cycles"],
        synthetic_data["t0_burst"], synthetic_data["J0_over_rho"],
    )
    train_out = train_generic_inspection_pinn(
        model, synthetic_data, physics_fn,
        epochs=epochs, lr=lr, batch_size=batch_size, device=device, seed=seed,
        log_dir=log_dir, run_name=run_name, w_supervised=w_supervised, w_physics=w_physics,
    )
    return {"model": model, "train_out": train_out, "synthetic_data": synthetic_data}
