from __future__ import annotations
"""Extended PINN for domain decomposition and multi-subnet architecture."""

from typing import Any, Dict, List, Optional, Callable, Union, Tuple

import torch
import torch.nn as nn

from .base import PINNBase, PINNOutput  # single source of truth
from .vanilla import VanillaPINN        # reuse canonical VanillaPINN


def _act(name: str) -> nn.Module:
    name = (name or "tanh").lower()
    return {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}.get(name, nn.Tanh())


class SubnetWrapper(PINNBase):
    def __init__(self, in_dim: int, out_dim: int, hidden=(128,128,128,128), activation="tanh"):
        super().__init__()
        self.net = VanillaPINN(in_dim=in_dim, out_dim=out_dim, hidden=list(hidden), activation=activation)

    def forward(self, *inputs: torch.Tensor) -> PINNOutput:
        if len(inputs) == 1:
            x = inputs[0]
        else:
            x = torch.cat(inputs, dim=1)

        y = self.net.predict(x)   # returns Tensor (PINNBase.predict unwraps PINNOutput.y)
        z = self._zeros(y)
        return PINNOutput(y=y, losses={"total": z}, extras={})


class XPINN(PINNBase):
    def __init__(
        self,
        n_subdomains: int,
        in_dim: int,
        out_dim: int,
        hidden=(128, 128, 128, 128),
        activation: str = "tanh",
        interface_weight: float = 1.0,
        interface_flux_weight: float = 1.0,
        physics_weight: float = 1.0,
    ):
        super().__init__()
        self.interface_weight = float(interface_weight)
        self.interface_flux_weight = float(interface_flux_weight)
        self.physics_weight = float(physics_weight)

        self.subnets = nn.ModuleList([
            SubnetWrapper(in_dim=in_dim, out_dim=out_dim, hidden=hidden, activation=activation)
            for _ in range(int(n_subdomains))
        ])

    def forward(
        self,
        x_list: List[torch.Tensor],
        *,
        interface_pairs: Optional[List[Tuple]] = None,
        physics_fn: Optional[Callable[..., Any]] = None,
        physics_data_list: Optional[List[Dict[str, Any]]] = None,
    ) -> PINNOutput:
        ys: List[torch.Tensor] = [self.subnets[i].predict(x_list[i]) for i in range(len(self.subnets))]

        ref = self._ref_tensor(ys[0] if len(ys) else None)
        zero = ref.new_zeros(())
        losses: Dict[str, torch.Tensor] = {"total": zero}

        if interface_pairs:
            iface_u = zero
            iface_flux = zero

            for item in interface_pairs:
                # (i, j, xi, xj, ni, nj)
                if len(item) == 4:
                    i, j, xi, xj = item
                    ni = nj = None
                elif len(item) == 6:
                    i, j, xi, xj, ni, nj = item
                else:
                    raise ValueError("interface_pairs must be (i,j,xi,xj) or (i,j,xi,xj,ni,nj)")

                yi = self.subnets[i].predict(xi)
                yj = self.subnets[j].predict(xj)
                iface_u = iface_u + torch.mean((yi - yj) ** 2)

                if self.interface_flux_weight > 0.0 and (ni is not None) and (nj is not None):
                    xi_g = self.ensure_requires_grad(xi)
                    xj_g = self.ensure_requires_grad(xj)
                    yi_g = self.subnets[i].predict(xi_g)
                    yj_g = self.subnets[j].predict(xj_g)

                    dni = self.normal_derivative(yi_g, xi_g, ni)
                    dnj = self.normal_derivative(yj_g, xj_g, nj)
                    iface_flux = iface_flux + torch.mean((dni - dnj) ** 2)

            losses["interface"] = iface_u
            losses["total"] = losses["total"] + self.interface_weight * iface_u

            if self.interface_flux_weight > 0.0:
                losses["interface_flux"] = iface_flux
                losses["total"] = losses["total"] + self.interface_flux_weight * iface_flux

        if physics_fn is not None and physics_data_list is not None:
            phys_total = zero
            phys_breakdown: Dict[str, torch.Tensor] = {}

            for i, pdata in enumerate(physics_data_list):
                pl = self.subnets[i].physics_loss(physics_fn=physics_fn, physics_data=pdata)

                p_main = pl.get("physics", zero)
                phys_total = phys_total + p_main

                for k, v in pl.items():
                    if isinstance(k, str) and k.startswith("physics/") and torch.is_tensor(v):
                        phys_breakdown[k] = phys_breakdown.get(k, zero) + v

            losses["physics"] = phys_total
            for k, v in phys_breakdown.items():
                losses[k] = v

            losses["total"] = losses["total"] + self.physics_weight * phys_total

        return PINNOutput(y=ys, losses=losses, extras={"n_subdomains": len(self.subnets)})
