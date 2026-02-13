from __future__ import annotations
import torch
import torch.nn as nn

from .base import ContinuousModelBase, ContOutput


class SymplecticODENet(ContinuousModelBase):
    """
    Symplectic ODE-Net:

      - Learns separable Hamiltonian: H(q,p)=T(p)+V(q)
      - Field is Hamiltonian by construction
      - Provides symplectic integrator (Stormer–Verlet) for rollout
    """

    def __init__(self, dim_q: int, hidden: int = 128):
        super().__init__()
        self.dim_q = int(dim_q)

        self.T = nn.Sequential(
            nn.Linear(dim_q, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

        self.V = nn.Sequential(
            nn.Linear(dim_q, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    # --------------------------------------------------
    # Hamiltonian and vector field
    # --------------------------------------------------

    def hamiltonian(self, q: torch.Tensor, p: torch.Tensor):
        return self.T(p) + self.V(q)  # (N,1)

    def vector_field(self, z: torch.Tensor):
        assert z.shape[1] == 2 * self.dim_q, "Input must have dimension 2*dim_q"

        z = z.detach().requires_grad_(True)

        q = z[:, :self.dim_q]
        p = z[:, self.dim_q:]

        H = self.hamiltonian(q, p)  # (N,1)

        grad = torch.autograd.grad(
            H,
            z,
            grad_outputs=torch.ones_like(H),
            create_graph=True
        )[0]

        qdot = grad[:, self.dim_q:]
        pdot = -grad[:, :self.dim_q]

        dz = torch.cat([qdot, pdot], dim=-1)
        return dz, H

    # --------------------------------------------------
    # Forward (field evaluation)
    # --------------------------------------------------

    def forward(self, z: torch.Tensor, *, y_true=None, return_loss=False) -> ContOutput:
        dz, H = self.vector_field(z)

        losses = {"total": torch.tensor(0.0, device=z.device)}

        if return_loss and y_true is not None:
            mse = self.mse(dz, y_true)
            losses["mse"] = mse
            losses["total"] = mse

        return ContOutput(y=dz, losses=losses, extras={"H": H})

    # --------------------------------------------------
    # Symplectic Integrator (Stormer–Verlet)
    # --------------------------------------------------

    @torch.no_grad()
    def symplectic_step(self, z: torch.Tensor, dt: float):
        """
        One Stormer–Verlet step.
        z: (N, 2*dim_q)
        dt: timestep
        """
        q = z[:, :self.dim_q]
        p = z[:, self.dim_q:]

        # IMPORTANT:
        # We are inside no_grad(), but we still need autograd to compute dV/dq and dT/dp.
        # So we temporarily re-enable grad for those local derivatives.
        with torch.enable_grad():
            # p half-step
            q_req = q.detach().requires_grad_(True)
            V = self.V(q_req)  # (N,1)
            dVdq = torch.autograd.grad(
                V,
                q_req,
                grad_outputs=torch.ones_like(V),
                create_graph=False
            )[0]
            p_half = p - 0.5 * dt * dVdq

            # q full-step
            p_req = p_half.detach().requires_grad_(True)
            T = self.T(p_req)  # (N,1)
            dTdp = torch.autograd.grad(
                T,
                p_req,
                grad_outputs=torch.ones_like(T),
                create_graph=False
            )[0]
            q_new = q + dt * dTdp

            # p second half-step
            q_req2 = q_new.detach().requires_grad_(True)
            V_new = self.V(q_req2)  # (N,1)
            dVdq_new = torch.autograd.grad(
                V_new,
                q_req2,
                grad_outputs=torch.ones_like(V_new),
                create_graph=False
            )[0]
            p_new = p_half - 0.5 * dt * dVdq_new

        return torch.cat([q_new, p_new], dim=-1)

    @torch.no_grad()
    def rollout(self, z0: torch.Tensor, dt: float, steps: int):
        """
        Perform multiple symplectic steps.
        Returns trajectory tensor of shape (steps+1, N, 2*dim_q)
        """
        traj = [z0]
        z = z0

        for _ in range(steps):
            z = self.symplectic_step(z, dt)
            traj.append(z)

        return torch.stack(traj)
