from __future__ import annotations
"""Extended PINN for domain decomposition and multi-subnet architecture."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Union, Tuple

import torch
import torch.nn as nn

def _act(name: str) -> nn.Module:
    name = (name or "tanh").lower()
    return {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}.get(name, nn.Tanh())


class VanillaPINN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: List[int] = (128, 128, 128, 128),
        activation: str = "tanh",
    ):
        super().__init__()
        act = _act(activation)
        dims = [int(in_dim), *list(hidden), int(out_dim)]
        layers: List[nn.Module] = []
        for a, b in zip(dims[:-2], dims[1:-1]):
            layers += [nn.Linear(a, b), act]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class PINNOutput:
    y: Union[torch.Tensor, List[torch.Tensor], Any]
    losses: Dict[str, torch.Tensor]
    extras: Dict[str, Any]


class PINNBase(nn.Module):
    def __init__(self):
        super().__init__()

    def _ref_tensor(self, *maybe: Any) -> torch.Tensor:
        try:
            return next(self.parameters())
        except StopIteration:
            pass

        def _find(obj: Any) -> Optional[torch.Tensor]:
            if torch.is_tensor(obj):
                return obj
            if isinstance(obj, dict):
                for v in obj.values():
                    t = _find(v)
                    if t is not None:
                        return t
            if isinstance(obj, (list, tuple)):
                for v in obj:
                    t = _find(v)
                    if t is not None:
                        return t
            return None

        for obj in maybe:
            t = _find(obj)
            if t is not None:
                return t

        return torch.empty((), device="cpu", dtype=torch.float32)

    def _zeros(self, like: Any = None) -> torch.Tensor:
        ref = self._ref_tensor(like)
        return ref.new_zeros(())

    def predict(self, *inputs: torch.Tensor, **kwargs):
        out = self.forward(*inputs, **kwargs)
        if isinstance(out, PINNOutput):
            return out.y
        return out

    @staticmethod
    def ensure_requires_grad(x: torch.Tensor) -> torch.Tensor:
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        return x

    @staticmethod
    def grad(
        y: torch.Tensor,
        x: torch.Tensor,
        *,
        retain_graph: bool = True,
        create_graph: bool = True,
    ) -> torch.Tensor:
        y_ = y if y.ndim == 0 else y.sum()
        (g,) = torch.autograd.grad(
            y_,
            x,
            retain_graph=retain_graph,
            create_graph=create_graph,
            allow_unused=False,
        )
        return g

    @classmethod
    def normal_derivative(cls, y: torch.Tensor, x: torch.Tensor, n_hat: torch.Tensor) -> torch.Tensor:
        if n_hat.ndim == 1:
            n_hat = n_hat.unsqueeze(0).expand(x.shape[0], -1)
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        outs = []
        for k in range(y.shape[1]):
            gk = cls.grad(y[:, k], x)  # (B,in_dim)
            dn = (gk * n_hat).sum(dim=1, keepdim=True)  # (B,1)
            outs.append(dn)
        return torch.cat(outs, dim=1)

    def physics_loss(
        self,
        *,
        physics_fn: Optional[Callable[..., Any]] = None,
        physics_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if physics_fn is None or physics_data is None:
            z = self._zeros()
            return {"physics": z}

        ref = self._ref_tensor(physics_data)
        z0 = ref.new_zeros(())

        def _looks_like_factory(fn: Callable[..., Any], batch: Any) -> bool:
            if not isinstance(batch, dict):
                return False
            return True

        class _FactoryModelAdapter(nn.Module):
            def __init__(self, base: PINNBase):
                super().__init__()
                self.base = base
                inv = getattr(base, "inverse_params", None)
                self.inverse_params = inv if isinstance(inv, nn.ParameterDict) else nn.ParameterDict()

            def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
                y = self.base.predict(*inputs)
                if isinstance(y, (list, tuple)):
                    raise TypeError("Factory adapter expects Tensor; received list/tuple.")
                if not torch.is_tensor(y):
                    raise TypeError(f"Factory adapter expects Tensor; received {type(y)}.")
                return y

        if _looks_like_factory(physics_fn, physics_data) and len(kwargs) == 0:
            adapter = _FactoryModelAdapter(self)
            res = physics_fn(adapter, physics_data)
        else:
            res = physics_fn(self, physics_data, **kwargs)

        if isinstance(res, tuple) and len(res) == 2:
            total, comps = res
            if not torch.is_tensor(total):
                total = torch.tensor(float(total), device=ref.device, dtype=ref.dtype)
            out: Dict[str, torch.Tensor] = {"physics": total}
            if isinstance(comps, dict):
                for k, v in comps.items():
                    out[f"physics/{k}"] = v if torch.is_tensor(v) else torch.tensor(
                        float(v), device=total.device, dtype=total.dtype
                    )
            return out

        if isinstance(res, dict):
            out = dict(res)
            for k, v in list(out.items()):
                if torch.is_tensor(v):
                    continue
                try:
                    out[k] = torch.tensor(float(v), device=ref.device, dtype=ref.dtype)
                except Exception:
                    pass
            if "physics" not in out:
                total = None
                for k in ("total", "loss", "pde"):
                    if k in out and torch.is_tensor(out[k]):
                        total = out[k]
                        break
                out["physics"] = total if total is not None else z0
            return out

        return {"physics": z0}


class SubnetWrapper(PINNBase):
    def __init__(self, in_dim: int, out_dim: int, hidden=(128,128,128,128), activation="tanh"):
        super().__init__()
        self.net = VanillaPINN(in_dim=in_dim, out_dim=out_dim, hidden=list(hidden), activation=activation)

    def forward(self, *inputs: torch.Tensor) -> PINNOutput:
        if len(inputs) == 1:
            x = inputs[0]
        else:
            x = torch.cat(inputs, dim=1)

        y = self.net(x)
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
