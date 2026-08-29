"""pinneapple_systems.process_components.torsional_stickslip -- 1D
torsional-wave finite-difference simulation of a compliant rotating
shaft, driven at one end at a prescribed angular velocity and loaded at
the other end by Stribeck (velocity-weakening) friction acting on a
lumped end inertia -- the standard mechanism behind torsional stick-slip
in any long, compliant driveline with a friction load (a drill string is
one instance; so is any sufficiently long rotating shaft coupling a
speed-controlled driver to a friction-loaded end mass).

SELECTED FORMULATION
---------------------
Governing PDE: the 1D torsional wave equation for angular displacement
`phi(z,t)` (formulated here as a *deviation* from steady rotation,
`phi = theta - omega_set*t`, to avoid large-number cancellation when
`theta` itself grows without bound):

    rho*J * d2phi/dt2 = G*J * d2phi/dz2 - beta*rho*J*(omega_set + dphi/dt)

i.e. wave propagation (`G*J*d2phi/dz2`) with a distributed linear
viscous damping term, discretized by central differences in z and
explicit symplectic Euler in time (velocities updated from the current
state, then positions updated from the *new* velocities -- unlike plain
forward Euler, symplectic Euler doesn't accumulate energy on a
non-dissipative wave equation over long runs).

Boundary conditions:
- Driven end: angular velocity held at `omega_set` (+ optional zero-mean
  noise, representing real speed-control jitter) -- a velocity (Neumann-
  on-the-integrated-state) condition, not a position clamp, so that any
  injected noise actually propagates into the wave field rather than
  being silently discarded by a rigid `phi=0` condition.
- Loaded end: Newton's second law for the lumped end inertia,
  `I_end*d(omega_end)/dt = T_transmitted - T_friction(omega_end)`, with
  Stribeck friction `T_friction = sign(omega_end) * [T_kinetic +
  (T_static - T_kinetic)*exp(-|omega_end|/omega_breakout)]` -- the
  standard Stribeck curve (friction torque decays from a static peak to
  a lower kinetic plateau as sliding velocity increases), the mechanism
  that makes stick-slip self-sustaining: near zero velocity the load
  resists more than it does once moving, so the end alternately sticks
  (locks to the driven-end torque) and slips (accelerates past it).

Stick-slip severity index `SSI = (omega_max - omega_min) / |omega_mean|`
at the loaded end -- a standard normalized measure of end-speed
oscillation severity (0 = perfectly smooth rotation) -- evaluated only
over the *tail* of the run (`ssi_window_fraction`, default the last
half). The startup transient (the driven end's velocity is a step input
at t=0, which rings the torsional wave much like any suddenly-applied
load) can dominate a whole-window swing measurement regardless of the
friction model, since a lightly-damped wave's first-cycle overshoot is
generally larger than its eventual self-sustained friction-driven
oscillation; excluding it makes SSI track the mechanism it's meant to
diagnose (Stribeck-driven limit-cycle stick-slip) rather than initial-
condition ringing. The full time series (`omega_field_rad_s`,
`end_omega_ts_rad_s`) is still returned in full, so a caller wanting the
whole-window value (or a different tail fraction) can recompute it.

VALIDITY ENVELOPE: 1D torsional dynamics only (no lateral/axial
coupling), linear-elastic shaft, Stribeck friction model for the end
load (not a full elastic-plastic contact model). The explicit time
integration requires `dt` below the CFL limit for the wave speed
`sqrt(G/rho)` -- enforced automatically in `simulate_torsional_stickslip`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StickSlipResult:
    t_s: np.ndarray
    z_m: np.ndarray
    omega_field_rad_s: np.ndarray
    end_omega_ts_rad_s: np.ndarray
    surface_torque_ts_N_m: np.ndarray
    stick_slip_index: float


def stribeck_friction_torque(omega: float, T_static: float, T_kinetic: float, omega_breakout: float) -> float:
    """T_friction = sign(omega)*[T_kinetic + (T_static-T_kinetic)*exp(-|omega|/omega_breakout)]."""
    magnitude = T_kinetic + (T_static - T_kinetic) * np.exp(-abs(omega) / omega_breakout)
    return -magnitude * np.sign(omega + 1e-12)


def simulate_torsional_stickslip(
    length_m: float,
    G_Pa: float,
    J_m4: float,
    rho_kg_m3: float,
    omega_set_rad_s: float,
    T_static_N_m: float,
    T_kinetic_N_m: float,
    omega_breakout_rad_s: float,
    end_inertia_kg_m2: float,
    n_nodes: int = 60,
    n_steps: int = 4000,
    save_every: int = 40,
    damping_coeff: float = 0.05,
    speed_control_noise_std_rad_s: float = 0.0,
    seed: int = 42,
    ssi_window_fraction: float = 0.5,
) -> StickSlipResult:
    """Explicit finite-difference solve of the 1D torsional wave equation
    described in the module docstring, from a driven end (node 0, held
    at `omega_set_rad_s`) to a friction-loaded end (node -1, lumped
    inertia `end_inertia_kg_m2` under Stribeck friction). Uses a fixed
    uniform shaft cross-section (`J_m4`, `rho_kg_m3` constant along
    `length_m`) with the friction-end lumped inertia as the only
    additional discrete mass.
    """
    if n_nodes < 3:
        raise ValueError(f"n_nodes must be >= 3, got {n_nodes}")

    GJ = G_Pa * J_m4
    dz = length_m / (n_nodes - 1)
    wave_speed = np.sqrt(G_Pa / rho_kg_m3)
    dt = min(0.004, 0.9 * dz / wave_speed)
    c2_dz2 = (G_Pa / rho_kg_m3) / dz ** 2

    I_half_node = rho_kg_m3 * J_m4 * dz / 2.0
    I_end = I_half_node + end_inertia_kg_m2

    n_saved = max(1, n_steps // save_every)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, speed_control_noise_std_rad_s, n_steps)

    phi = np.zeros(n_nodes)
    omega_dev = np.zeros(n_nodes)
    phi_new = np.zeros_like(phi)
    omega_dev_new = np.zeros_like(omega_dev)

    t_saved = np.zeros(n_saved)
    omega_field = np.zeros((n_saved, n_nodes))
    surface_torque_ts = np.zeros(n_saved)
    save_idx = 0

    for step in range(n_steps):
        omega_act = omega_set_rad_s + omega_dev
        d2phi = phi[2:] - 2.0 * phi[1:-1] + phi[:-2]
        rhs_int = c2_dz2 * d2phi - damping_coeff * omega_act[1:-1]
        omega_dev_new[1:-1] = omega_dev[1:-1] + dt * rhs_int

        omega_dev_new[0] = noise[step]

        T_transmitted = GJ * (phi[-2] - phi[-1]) / dz
        omega_end = omega_set_rad_s + omega_dev[-1]
        T_friction = stribeck_friction_torque(omega_end, T_static_N_m, T_kinetic_N_m, omega_breakout_rad_s)
        alpha_end = (T_transmitted + T_friction) / I_end - damping_coeff * omega_end
        omega_dev_new[-1] = omega_dev[-1] + dt * alpha_end

        phi_new[1:-1] = phi[1:-1] + dt * omega_dev_new[1:-1]
        phi_new[0] = phi[0] + dt * omega_dev_new[0]
        phi_new[-1] = phi[-1] + dt * omega_dev_new[-1]

        phi, phi_new = phi_new, phi
        omega_dev, omega_dev_new = omega_dev_new, omega_dev

        if step % save_every == 0 and save_idx < n_saved:
            omega_field[save_idx] = omega_set_rad_s + omega_dev
            surface_torque_ts[save_idx] = -GJ * (phi[1] - phi[0]) / dz
            t_saved[save_idx] = step * dt
            save_idx += 1

    end_omega_ts = omega_field[:, -1]
    tail_start = int(len(end_omega_ts) * (1.0 - ssi_window_fraction))
    tail = end_omega_ts[tail_start:]
    omega_mean = tail.mean()
    ssi = float((tail.max() - tail.min()) / max(abs(omega_mean), 1e-3))

    return StickSlipResult(
        t_s=t_saved,
        z_m=np.linspace(0.0, length_m, n_nodes),
        omega_field_rad_s=omega_field,
        end_omega_ts_rad_s=end_omega_ts,
        surface_torque_ts_N_m=surface_torque_ts,
        stick_slip_index=ssi,
    )
