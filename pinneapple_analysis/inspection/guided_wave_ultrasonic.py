"""pinneapple_analysis.inspection.guided_wave_ultrasonic -- 1D guided-wave
(Lamb/torsional-wave style) dispersion + defect-reflection synthetic
signal generation, plus physics-informed training.

Physical picture (simplified -- see caveat)
---------------------------------------------
A real guided wave in a plate or pipe (Lamb waves, SH waves, torsional
waves in a pipe, ...) is DISPERSIVE: its phase/group velocity depends on
frequency, governed by the Rayleigh-Lamb frequency equation, a
transcendental system requiring root-finding across branches (see Rose,
"Ultrasonic Guided Waves in Solid Media", Cambridge University Press,
2014). Solving that exactly is out of scope here.

CAVEAT / simplification: this module instead uses the classic
Klein-Gordon-type dispersive wave equation

    u_tt = c0^2 * u_xx - beta(x) * u

whose plane-wave dispersion relation is ``omega^2 = c0^2 k^2 + beta``,
i.e. a wave that only propagates above the cutoff frequency
``sqrt(beta)/(2*pi)`` and is increasingly dispersive near it -- the same
QUALITATIVE signature (a cutoff frequency, group velocity below phase
velocity) as the lowest-order guided-wave mode of a real waveguide, but
NOT a fit to any specific mode's exact Rayleigh-Lamb dispersion curve.

A defect (e.g. a wall-loss region) is modeled as a local change in
``beta(x)`` (an effective local impedance/cutoff change), producing a
partial reflection and transmission of an incident tone-burst wave
packet -- the classic guided-wave NDE defect signature: time-of-flight
to the reflected echo locates the defect, and the reflection/attenuation
amplitude relates to its severity. Solved via the same explicit FDTD
scheme used elsewhere in this package.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from .base import InspectionPINN, PhysicsLossFn, train_generic_inspection_pinn


def _tone_burst(t: np.ndarray, t0: float, f0: float, n_cycles: float) -> np.ndarray:
    dur = n_cycles / f0
    tau = t - t0
    window = np.where(np.abs(tau) <= dur / 2, 0.5 * (1 + np.cos(2 * np.pi * tau / dur)), 0.0)
    return window * np.sin(2 * np.pi * f0 * tau)


def _fdtd_klein_gordon_1d(x: np.ndarray, c0: float, beta: np.ndarray, dt: float, nt: int, source: np.ndarray):
    """Explicit FDTD for u_tt = c0^2 u_xx - beta(x) u, with the transmitter
    driving a Dirichlet displacement at x[0] and a first-order Mur
    absorbing boundary at the far (right) end."""
    nx = len(x)
    dx = x[1] - x[0]
    u = np.zeros(nx)
    u_prev = np.zeros(nx)
    history = np.zeros((nt, nx))
    r2 = (c0 * dt / dx) ** 2
    for n in range(1, nt):
        u_new = np.empty(nx)
        u_new[1:-1] = (
            2 * u[1:-1] - u_prev[1:-1]
            + r2 * (u[2:] - 2 * u[1:-1] + u[:-2])
            - (dt ** 2) * beta[1:-1] * u[1:-1]
        )
        u_new[0] = source[n]  # driven transmitter
        u_new[-1] = u[-2] + ((c0 * dt - dx) / (c0 * dt + dx)) * (u_new[-2] - u[-1])
        u_prev, u = u, u_new
        history[n] = u
    return history


def generate_guided_wave_synthetic(
    *,
    L: float = 1.0,
    c0: float = 3000.0,
    beta0: float = 0.0,        # background cutoff term (0 = non-dispersive background)
    n_points: int = 80,
    duration: float = 0.0006,
    burst_freq: float = 50e3,
    n_cycles: float = 4.0,
    defect_position: float = 0.5,   # fraction of L
    defect_width: float = 0.03,     # fraction of L
    defect_beta: float = 2.0e9,     # local impedance/cutoff bump at the defect
    n_supervised: int = 400,
    n_collocation: int = 200,
    seed: Optional[int] = 0,
) -> Dict[str, Any]:
    """Generate a synthetic guided-wave pulse-echo dataset: a tone burst
    is injected at x=0 and propagates toward a defect (a local `beta(x)`
    bump), which partially reflects it back to a receiver near the
    transmitter -- classic pulse-echo guided-wave NDE.

    Returns a dict with:
      x_grid, t_grid   -- 1D grids
      beta_field       -- (nx,) local dispersion/impedance map
      u_field          -- (nt, nx) displacement history
      receiver_trace   -- u(t) at the receiver (near the transmitter)
      x, y, x_col      -- InspectionPINN training data
    """
    rng = np.random.default_rng(seed)

    x = np.linspace(0.0, L, n_points)
    dx = x[1] - x[0]
    dt = 0.4 * dx / c0
    nt = max(16, int(round(duration / dt)))
    t = np.arange(nt) * dt

    defect_x0 = defect_position * L
    defect_halfwidth = 0.5 * defect_width * L
    beta_field = beta0 * np.ones(n_points)
    beta_field[np.abs(x - defect_x0) <= defect_halfwidth] = defect_beta

    t0_burst = 1.5 * n_cycles / burst_freq
    source = _tone_burst(t, t0_burst, burst_freq, n_cycles)

    u_field = _fdtd_klein_gordon_1d(x, c0, beta_field, dt, nt, source)

    receiver_idx = max(1, int(round(0.02 * (n_points - 1))))
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
        "beta_field": beta_field,
        "u_field": u_field,
        "receiver_trace": receiver_trace,
        "x": x_sup,
        "y": y_sup,
        "x_col": x_col,
        "u_scale": u_scale,
        "c0": float(c0),
    }


def make_guided_wave_physics_loss(x_grid: np.ndarray, beta_field: np.ndarray, c0: float) -> PhysicsLossFn:
    """Build a `(model, batch) -> (loss, comps)` physics loss enforcing
    the dispersive wave equation ``u_tt - c0^2 u_xx + beta(x) u = 0``."""
    def beta_of_x(x_np: np.ndarray) -> np.ndarray:
        return np.interp(x_np, x_grid, beta_field)

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

        beta_vals = beta_of_x(x_col[:, 0:1].detach().cpu().numpy())
        beta_t = torch.as_tensor(beta_vals, dtype=u.dtype, device=u.device).reshape(-1, 1)

        residual = d2u_dt2 - (c0 ** 2) * d2u_dx2 + beta_t * u
        loss = torch.mean(residual ** 2)
        comps = {"residual_rmse": float(torch.sqrt(torch.mean(residual.detach() ** 2)).item())}
        return loss, comps

    return physics_loss_fn


def train_guided_wave_ultrasonic(
    synthetic_data: Optional[Dict[str, Any]] = None,
    *,
    hidden=(64, 64, 64),
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cpu",
    seed: Optional[int] = 0,
    log_dir: str = "runs/inspection_guided_wave",
    run_name: str = "guided_wave_pinn",
    w_supervised: float = 1.0,
    w_physics: float = 1.0,
    **synthetic_kwargs: Any,
) -> Dict[str, Any]:
    if synthetic_data is None:
        synthetic_data = generate_guided_wave_synthetic(seed=seed, **synthetic_kwargs)

    model = InspectionPINN(in_dim=2, out_dim=1, hidden=list(hidden), modality="guided_wave_ultrasonic")
    physics_fn = make_guided_wave_physics_loss(
        synthetic_data["x_grid"], synthetic_data["beta_field"], synthetic_data["c0"],
    )
    train_out = train_generic_inspection_pinn(
        model, synthetic_data, physics_fn,
        epochs=epochs, lr=lr, batch_size=batch_size, device=device, seed=seed,
        log_dir=log_dir, run_name=run_name, w_supervised=w_supervised, w_physics=w_physics,
    )
    return {"model": model, "train_out": train_out, "synthetic_data": synthetic_data}
