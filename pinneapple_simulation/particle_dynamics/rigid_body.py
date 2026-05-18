"""Differentiable rigid body dynamics for robot learning and physics coupling.

All state transitions are implemented with standard PyTorch operations, making
every quantity differentiable through the simulation trajectory.

Supported integrators
---------------------
- Symplectic (semi-implicit) Euler for position/orientation  (default)

Coordinates
-----------
- 2-D: position (x, y), scalar angle θ, velocity (vx, vy), angular vel ω
- 3-D: position (x, y, z), unit quaternion (w, x, y, z), linear vel, angular vel
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------


class RigidBodyState:
    """Mutable container for the state of *n_bodies* rigid bodies.

    For 2-D problems:
        - ``pos``   : ``(n_bodies, 2)``  –  (x, y)
        - ``angle`` : ``(n_bodies,)``    –  rotation angle θ in radians
        - ``vel``   : ``(n_bodies, 2)``  –  (vx, vy)
        - ``omega`` : ``(n_bodies,)``    –  angular velocity ω

    For 3-D problems:
        - ``pos``   : ``(n_bodies, 3)``
        - ``quat``  : ``(n_bodies, 4)``  –  unit quaternion (w, x, y, z)
        - ``vel``   : ``(n_bodies, 3)``
        - ``omega`` : ``(n_bodies, 3)``  –  body-frame angular velocity

    Parameters
    ----------
    n_bodies:
        Number of rigid bodies.
    dim:
        Spatial dimension (2 or 3).
    device:
        PyTorch device.
    dtype:
        PyTorch float dtype.
    """

    def __init__(
        self,
        n_bodies: int = 1,
        dim: int = 2,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.n_bodies = n_bodies
        self.dim = dim
        self.device = device
        self.dtype = dtype

        self.pos = torch.zeros(n_bodies, dim, device=device, dtype=dtype)

        if dim == 2:
            self.angle = torch.zeros(n_bodies, device=device, dtype=dtype)
            self.quat: Optional[torch.Tensor] = None
        else:
            self.angle = torch.zeros(n_bodies, device=device, dtype=dtype)
            # Identity quaternion (w=1, x=y=z=0)
            self.quat = torch.zeros(n_bodies, 4, device=device, dtype=dtype)
            self.quat[:, 0] = 1.0

        self.vel = torch.zeros(n_bodies, dim, device=device, dtype=dtype)
        self.omega = torch.zeros(n_bodies, device=device, dtype=dtype) if dim == 2 \
            else torch.zeros(n_bodies, 3, device=device, dtype=dtype)

    def clone(self) -> "RigidBodyState":
        """Return a deep copy of this state."""
        s = RigidBodyState(self.n_bodies, self.dim, self.device, self.dtype)
        s.pos = self.pos.clone()
        s.angle = self.angle.clone()
        if self.quat is not None:
            s.quat = self.quat.clone()
        s.vel = self.vel.clone()
        s.omega = self.omega.clone()
        return s

    def __repr__(self) -> str:
        return (
            f"RigidBodyState(n={self.n_bodies}, dim={self.dim}, "
            f"pos={self.pos.tolist()}, vel={self.vel.tolist()})"
        )


# ---------------------------------------------------------------------------
# Single rigid body
# ---------------------------------------------------------------------------


class RigidBody(nn.Module):
    """Single rigid body with mass, inertia tensor, and a symplectic integrator.

    Parameters
    ----------
    mass:
        Total mass (scalar).
    inertia:
        Moment of inertia. For 2-D pass a scalar tensor; for 3-D pass a
        ``(3, 3)`` inertia tensor.
    dim:
        Spatial dimension (2 or 3).
    """

    def __init__(
        self,
        mass: float,
        inertia: torch.Tensor,
        dim: int = 2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("mass", torch.tensor(mass, dtype=torch.float32))
        # Store inertia; make it a buffer so it moves with .to(device)
        self.register_buffer("inertia", inertia.float())
        # Inverse inertia (precomputed for efficiency)
        if inertia.ndim == 0 or inertia.numel() == 1:
            self.register_buffer("inertia_inv", 1.0 / inertia.float())
        else:
            self.register_buffer("inertia_inv", torch.linalg.inv(inertia.float()))

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def step(
        self,
        state: RigidBodyState,
        force: torch.Tensor,
        torque: torch.Tensor,
        dt: float,
    ) -> RigidBodyState:
        """Advance the state by *dt* using symplectic Euler integration.

        Symplectic Euler is first-order accurate and energy-preserving (for
        conservative systems), making it suitable for long-horizon simulations.

        Parameters
        ----------
        state:
            Current rigid body state.
        force:
            Applied forces, shape ``(n_bodies, dim)`` or ``(dim,)`` for a
            single body.
        torque:
            Applied torques; scalar ``(n_bodies,)`` for 2-D or
            ``(n_bodies, 3)`` for 3-D.
        dt:
            Time step in seconds.

        Returns
        -------
        RigidBodyState
            Updated state at time ``t + dt``.
        """
        new_state = state.clone()
        m = self.mass
        I_inv = self.inertia_inv

        # --- Velocity update (velocity-Verlet / symplectic: v first) ---
        accel = force / m                    # (n_bodies, dim)
        new_state.vel = state.vel + dt * accel

        # --- Position update ---
        new_state.pos = state.pos + dt * new_state.vel

        # --- Angular update (2-D scalar) ---
        if self.dim == 2:
            alpha = I_inv * torque           # angular acceleration (n_bodies,)
            new_state.omega = state.omega + dt * alpha
            new_state.angle = state.angle + dt * new_state.omega
        else:
            # 3-D: omega in body frame, integrate with explicit Euler
            alpha = (I_inv @ torque.unsqueeze(-1)).squeeze(-1)
            new_state.omega = state.omega + dt * alpha

            # Quaternion integration: q_new = q + 0.5 * dt * [0, omega] ⊗ q
            q = state.quat                   # (n_bodies, 4)
            w_body = new_state.omega         # (n_bodies, 3)
            # Build pure-quaternion form of angular velocity
            q_dot = 0.5 * _quat_mult_vec(q, w_body)
            new_state.quat = _quat_normalize(q + dt * q_dot)

        return new_state


# ---------------------------------------------------------------------------
# System of rigid bodies
# ---------------------------------------------------------------------------


class RigidBodySystem(nn.Module):
    """System of multiple rigid bodies with optional gravity.

    Parameters
    ----------
    bodies:
        List of :class:`RigidBody` instances.
    gravity:
        Gravitational acceleration tuple.  For 2-D, only the first two
        components are used.  Default: (0, -9.81, 0) – standard Earth gravity.
    """

    def __init__(
        self,
        bodies: List[RigidBody],
        gravity: Tuple[float, ...] = (0.0, -9.81, 0.0),
    ) -> None:
        super().__init__()
        self.bodies = nn.ModuleList(bodies)
        self.gravity = gravity

    def forward(
        self,
        states: List[RigidBodyState],
        forces: List[torch.Tensor],
        dt: float,
    ) -> List[RigidBodyState]:
        """Step all bodies forward by *dt*.

        Gravity is added automatically to each body's force.

        Parameters
        ----------
        states:
            List of :class:`RigidBodyState`, one per body.
        forces:
            List of external force tensors ``(dim,)`` per body.
        dt:
            Time step.

        Returns
        -------
        list of RigidBodyState
            Updated states.
        """
        new_states = []
        for i, (body, state, ext_force) in enumerate(
            zip(self.bodies, states, forces)
        ):
            dim = body.dim
            g = torch.tensor(self.gravity[:dim], dtype=torch.float32,
                             device=ext_force.device)
            total_force = ext_force + body.mass * g
            torque = torch.zeros(1, dtype=torch.float32, device=ext_force.device) \
                if dim == 2 else torch.zeros(3, dtype=torch.float32,
                                             device=ext_force.device)
            new_states.append(body.step(state, total_force.unsqueeze(0),
                                        torque.unsqueeze(0), dt))
        return new_states


# ---------------------------------------------------------------------------
# Quaternion helpers (private)
# ---------------------------------------------------------------------------


def _quat_normalize(q: torch.Tensor) -> torch.Tensor:
    """Normalise quaternion(s) to unit length.  ``q`` shape: ``(*, 4)``."""
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def _quat_mult_vec(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute the quaternion product ``q ⊗ [0, v]``.

    Parameters
    ----------
    q : ``(*, 4)``  –  (w, x, y, z) unit quaternion
    v : ``(*, 3)``  –  pure vector part

    Returns
    -------
    torch.Tensor
        ``(*, 4)`` result quaternion.
    """
    w, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    return torch.stack([
        - qx * vx - qy * vy - qz * vz,      # w
          w  * vx + qy * vz - qz * vy,      # x
          w  * vy + qz * vx - qx * vz,      # y
          w  * vz + qx * vy - qy * vx,      # z
    ], dim=-1)
