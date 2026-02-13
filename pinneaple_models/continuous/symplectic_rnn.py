from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import ContinuousModelBase, ContOutput


class SymplecticRNN(ContinuousModelBase):
    """
    Symplectic RNN (MVP, leapfrog-style update):

    State z = [q, p], dim_q = dim_p.
    We learn a separable Hamiltonian H(q,p) = T(p) + V(q).

    Discrete symplectic update (leapfrog / Störmer–Verlet):
      p_{n+1/2} = p_n - (dt/2) * dV/dq(q_n)
      q_{n+1}   = q_n + dt * dT/dp(p_{n+1/2})
      p_{n+1}   = p_{n+1/2} - (dt/2) * dV/dq(q_{n+1})

    Inputs:
      z0: (B, 2*dim_q)
      t:  (T,) increasing  (dt can vary)
    Output:
      z_path: (B, T, 2*dim_q)
    """
    def __init__(
        self,
        dim_q: int,
        *,
        hidden: int = 128,
        num_layers: int = 2,
        activation: str = "tanh",
    ):
        super().__init__()
        self.dim_q = int(dim_q)

        act = (activation or "tanh").lower()
        act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}.get(act, nn.Tanh)

        def mlp(in_d: int) -> nn.Sequential:
            layers = [nn.Linear(in_d, hidden), act_fn()]
            for _ in range(max(int(num_layers) - 1, 0)):
                layers += [nn.Linear(hidden, hidden), act_fn()]
            layers += [nn.Linear(hidden, 1)]
            return nn.Sequential(*layers)

        # scalar energies
        self.T = mlp(self.dim_q)  # kinetic energy from p
        self.V = mlp(self.dim_q)  # potential energy from q

    # -------------------------
    # gradients of energies
    # -------------------------
    def dTdp(self, p: torch.Tensor) -> torch.Tensor:
        # IMPORTANT: make a leaf tensor for autograd.grad
        # keeps gradients w.r.t. network parameters, avoids non-leaf requires_grad_ errors
        p_ = p.detach().requires_grad_(True)
        T = self.T(p_).sum()
        (grad,) = torch.autograd.grad(T, p_, create_graph=True)
        return grad

    def dVdq(self, q: torch.Tensor) -> torch.Tensor:
        # IMPORTANT: make a leaf tensor for autograd.grad
        q_ = q.detach().requires_grad_(True)
        V = self.V(q_).sum()
        (grad,) = torch.autograd.grad(V, q_, create_graph=True)
        return grad

    def forward(
        self,
        z0: torch.Tensor,                    # (B,2*dim_q)
        t: torch.Tensor,                     # (T,)
        *,
        y_true: Optional[torch.Tensor] = None,  # (B,T,2*dim_q)
        return_loss: bool = False,
    ) -> ContOutput:
        B, D = z0.shape
        if D != 2 * self.dim_q:
            raise ValueError(f"Expected z0 dim {2*self.dim_q}, got {D}")

        if t.ndim != 1:
            raise ValueError(f"Expected t to be 1D (T,), got shape {tuple(t.shape)}")

        Tn = t.numel()
        if Tn < 1:
            raise ValueError("t must have at least one element.")
        if Tn > 1:
            # avoid dt.item() CPU sync; keep it tensor-based
            if not torch.all(t[1:] > t[:-1]):
                raise ValueError("t must be strictly increasing for SymplecticRNN update.")

        q = z0[:, : self.dim_q]
        p = z0[:, self.dim_q :]

        zs = [torch.cat([q, p], dim=-1)]

        for i in range(Tn - 1):
            # dt as scalar tensor on correct device/dtype
            dt = (t[i + 1] - t[i]).to(dtype=z0.dtype, device=z0.device)

            # p half step
            dV1 = self.dVdq(q)
            p_half = p - 0.5 * dt * dV1

            # q full step
            dT = self.dTdp(p_half)
            q_new = q + dt * dT

            # p second half step
            dV2 = self.dVdq(q_new)
            p_new = p_half - 0.5 * dt * dV2

            q, p = q_new, p_new
            zs.append(torch.cat([q, p], dim=-1))

        z_path = torch.stack(zs, dim=1)  # (B,T,2*dim_q)

        losses: Dict[str, torch.Tensor] = {"total": torch.zeros((), device=z_path.device, dtype=z_path.dtype)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(z_path, y_true)
            losses["total"] = losses["mse"]

        return ContOutput(y=z_path, losses=losses, extras={"t": t})
