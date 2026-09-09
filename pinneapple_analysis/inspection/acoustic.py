"""pinneapple_analysis.inspection.acoustic -- acoustic/vibration
condition-spectrum synthetic generation (resonance/impact-echo style
NDE) and physics-informed training.

Physical picture
-----------------
A slender rod/bar of length L, longitudinal wave speed c0, is struck
once (a simulated hammer tap near one end) and its free-free vibration
u(x, t) is propagated forward with an explicit finite-difference
time-domain (FDTD) scheme for the 1D wave equation

    u_tt = c(x)^2 * u_xx

A defect (e.g. a local wall-thinning or a partial crack) is modeled as
a LOCAL REDUCTION in wave speed, ``c(x) = c0 * sqrt(1 - severity)``
inside a small region -- since the longitudinal wave speed of a rod is
``c = sqrt(E/rho)``, reducing the effective local stiffness (as a crack
or thinning would) lowers c there. A sensor at a fixed location records
u(t); its FFT is the "condition spectrum". Because the defect locally
slows the wave, the resonant (modal) frequencies of the whole bar shift
DOWN relative to the defect-free baseline -- this is the classic
resonance/impact-echo defect signature (see e.g. Cawley & Adams, "The
location of defects in structures from measurements of natural
frequencies", J. Strain Analysis, 1979, for the general principle that
a local stiffness loss produces a mode-shape-dependent natural-frequency
shift; the FDTD wave-equation approach used here reproduces this
directly from first principles rather than via their closed-form
sensitivity formula).

Free-free boundary conditions are approximated with reflecting
(Neumann, du/dx=0) ends via simple ghost-node mirroring -- the standard
free-free elastic-bar boundary condition.

Simplification note: 1D scalar wave equation only (no cross-sectional/
shear/bending coupling, no material damping beyond none -- the bar
rings undamped for the simulated duration); a real impact-echo signal
also decays from radiation and internal friction, deliberately omitted
here to keep the ground-truth PDE this module's PINN needs to learn a
single, clean equation.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from .base import InspectionPINN, PhysicsLossFn, train_generic_inspection_pinn


def _fdtd_wave_1d(x: np.ndarray, c: np.ndarray, dt: float, nt: int, excite_index: int, excite_width_pts: int):
    """Explicit FDTD solve of u_tt = c(x)^2 u_xx with free-free (Neumann)
    ends, starting from a Gaussian displacement bump (simulated tap) at
    `excite_index` with zero initial velocity. Returns the full (nt, nx)
    displacement history."""
    nx = len(x)
    dx = x[1] - x[0]
    u = np.zeros(nx)
    idx = np.arange(nx)
    u += np.exp(-0.5 * ((idx - excite_index) / max(1.0, excite_width_pts)) ** 2)
    u_prev = u.copy()  # zero initial velocity -> first Taylor step equals u

    history = np.zeros((nt, nx))
    history[0] = u
    r2 = (c * dt / dx) ** 2
    for n in range(1, nt):
        u_new = np.empty(nx)
        u_new[1:-1] = 2 * u[1:-1] - u_prev[1:-1] + r2[1:-1] * (u[2:] - 2 * u[1:-1] + u[:-2])
        # free-free (Neumann) ends via mirrored ghost points
        u_new[0] = 2 * u[0] - u_prev[0] + r2[0] * 2.0 * (u[1] - u[0])
        u_new[-1] = 2 * u[-1] - u_prev[-1] + r2[-1] * 2.0 * (u[-2] - u[-1])
        u_prev, u = u, u_new
        history[n] = u
    return history


def generate_acoustic_synthetic(
    *,
    L: float = 1.0,
    c0: float = 5000.0,
    n_points: int = 60,
    duration: float = 0.0015,
    defect_position: float = 0.6,   # fraction of L
    defect_width: float = 0.05,     # fraction of L
    defect_severity: float = 0.35,  # fractional stiffness/wave-speed loss
    sensor_position: float = 0.08,  # fraction of L
    excite_position: float = 0.02,  # fraction of L
    n_supervised: int = 400,
    n_collocation: int = 200,
    seed: Optional[int] = 0,
) -> Dict[str, Any]:
    """Generate a synthetic impact-echo / resonance condition-spectrum
    dataset via 1D FDTD. Returns baseline (no-defect) and defect-case
    time traces and spectra, plus InspectionPINN training data for the
    defect-case displacement field u(x, t).

    Returns a dict with:
      x_grid, t_grid       -- 1D spatial/time grids
      c_field              -- (nx,) wave-speed map used for the defect case
      u_field              -- (nt, nx) defect-case displacement history
      freqs, spectrum_defect, spectrum_baseline -- FFTs of the sensor trace
      x, y                 -- InspectionPINN (x,t) -> u supervised pairs
      x_col                -- physics collocation points
      c_scale, u_scale     -- normalization constants
    """
    rng = np.random.default_rng(seed)

    x = np.linspace(0.0, L, n_points)
    dx = x[1] - x[0]
    dt = 0.4 * dx / c0
    nt = max(8, int(round(duration / dt)))
    t = np.arange(nt) * dt

    defect_x0 = defect_position * L
    defect_halfwidth = 0.5 * defect_width * L
    in_defect = np.abs(x - defect_x0) <= defect_halfwidth

    c_baseline = c0 * np.ones(n_points)
    c_defect = c_baseline.copy()
    c_defect[in_defect] = c0 * np.sqrt(max(1e-3, 1.0 - defect_severity))

    excite_idx = int(round(excite_position * (n_points - 1)))
    sensor_idx = int(round(sensor_position * (n_points - 1)))
    excite_width_pts = max(1.0, 0.01 * n_points)

    u_baseline = _fdtd_wave_1d(x, c_baseline, dt, nt, excite_idx, excite_width_pts)
    u_defect = _fdtd_wave_1d(x, c_defect, dt, nt, excite_idx, excite_width_pts)

    sensor_trace_baseline = u_baseline[:, sensor_idx]
    sensor_trace_defect = u_defect[:, sensor_idx]

    freqs = np.fft.rfftfreq(nt, d=dt)
    spectrum_baseline = np.abs(np.fft.rfft(sensor_trace_baseline))
    spectrum_defect = np.abs(np.fft.rfft(sensor_trace_defect))

    u_scale = float(np.max(np.abs(u_defect)))
    if u_scale <= 0:
        u_scale = 1.0

    n_sup = min(n_supervised, nt * n_points)
    flat_idx = rng.choice(nt * n_points, size=n_sup, replace=False)
    ti, xi = flat_idx // n_points, flat_idx % n_points
    x_sup = np.stack([x[xi], t[ti]], axis=1).astype(np.float32)
    y_sup = (u_defect[ti, xi] / u_scale).astype(np.float32)[:, None]

    x_col = np.stack(
        [rng.uniform(x[1], x[-2], n_collocation), rng.uniform(t[1], t[-2], n_collocation)], axis=1,
    ).astype(np.float32)

    return {
        "x_grid": x,
        "t_grid": t,
        "c_field": c_defect,
        "u_field": u_defect,
        "freqs": freqs,
        "spectrum_baseline": spectrum_baseline,
        "spectrum_defect": spectrum_defect,
        "sensor_trace_baseline": sensor_trace_baseline,
        "sensor_trace_defect": sensor_trace_defect,
        "x": x_sup,
        "y": y_sup,
        "x_col": x_col,
        "u_scale": u_scale,
        "c0": float(c0),
    }


def make_acoustic_physics_loss(x_grid: np.ndarray, c_field: np.ndarray) -> PhysicsLossFn:
    """Build a `(model, batch) -> (loss, comps)` physics loss enforcing
    the 1D wave equation ``u_tt - c(x)^2 * u_xx = 0`` via autograd, with
    ``c(x)`` looked up from the (fixed, precomputed) FDTD wave-speed map
    -- the same "known material map" pattern as the eddy-current and MFL
    modules' physics losses."""
    def c_of_x(x_np: np.ndarray) -> np.ndarray:
        return np.interp(x_np, x_grid, c_field)

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

        c_vals = c_of_x(x_col[:, 0:1].detach().cpu().numpy())
        c_t = torch.as_tensor(c_vals, dtype=u.dtype, device=u.device).reshape(-1, 1)

        residual = d2u_dt2 - (c_t ** 2) * d2u_dx2
        loss = torch.mean(residual ** 2)
        comps = {"residual_rmse": float(torch.sqrt(torch.mean(residual.detach() ** 2)).item())}
        return loss, comps

    return physics_loss_fn


def train_acoustic(
    synthetic_data: Optional[Dict[str, Any]] = None,
    *,
    hidden=(64, 64, 64),
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cpu",
    seed: Optional[int] = 0,
    log_dir: str = "runs/inspection_acoustic",
    run_name: str = "acoustic_pinn",
    w_supervised: float = 1.0,
    w_physics: float = 1.0,
    **synthetic_kwargs: Any,
) -> Dict[str, Any]:
    if synthetic_data is None:
        synthetic_data = generate_acoustic_synthetic(seed=seed, **synthetic_kwargs)

    model = InspectionPINN(in_dim=2, out_dim=1, hidden=list(hidden), modality="acoustic")
    physics_fn = make_acoustic_physics_loss(synthetic_data["x_grid"], synthetic_data["c_field"])
    train_out = train_generic_inspection_pinn(
        model, synthetic_data, physics_fn,
        epochs=epochs, lr=lr, batch_size=batch_size, device=device, seed=seed,
        log_dir=log_dir, run_name=run_name, w_supervised=w_supervised, w_physics=w_physics,
    )
    return {"model": model, "train_out": train_out, "synthetic_data": synthetic_data}
