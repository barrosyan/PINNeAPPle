from __future__ import annotations
"""Generic PINO wrapper: operator + physics residual losses on predicted field u."""

from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Mapping, Sequence, Tuple, Literal

import torch

from .base import NeuralOperatorBase, OperatorOutput


DerivMethod = Literal["spectral", "fd"]
PINOPhysicsFn = Callable[..., Mapping[str, torch.Tensor]]
ResidualFn = Callable[..., torch.Tensor]


class PhysicsInformedNeuralOperator(NeuralOperatorBase):
    """
    Generic PINO wrapper.

    physics_fn contract:
        physics_fn(u, **physics_data) -> dict of scalar tensors/numbers.
    Suggested keys (all optional):
        - "pde", "bc", "ic", "physics_total"
    If "physics_total" missing, wrapper builds it from weighted sum.
    """

    def __init__(
        self,
        operator: NeuralOperatorBase,
        *,
        lambda_pde: float = 1.0,
        lambda_bc: float = 1.0,
        lambda_ic: float = 1.0,
        lambda_physics: float = 1.0,
        prefix: str = "physics/",
    ):
        super().__init__()
        self.operator = operator
        self.lambda_pde = float(lambda_pde)
        self.lambda_bc = float(lambda_bc)
        self.lambda_ic = float(lambda_ic)
        self.lambda_physics = float(lambda_physics)
        self.prefix = str(prefix)

    @staticmethod
    def _zero_scalar_like(ref: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=ref.device, dtype=ref.dtype)

    @staticmethod
    def _to_scalar_tensor(v: Any, ref: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(v):
            t = v.to(device=ref.device, dtype=ref.dtype)
        else:
            t = torch.tensor(float(v), device=ref.device, dtype=ref.dtype)
        if t.ndim != 0:
            t = t.mean()
        return t

    def forward(
        self,
        *args,
        physics_fn: Optional[PINOPhysicsFn] = None,
        physics_data: Optional[Dict[str, Any]] = None,
        **kw,
    ) -> OperatorOutput:
        out = self.operator(*args, **kw)
        losses = dict(out.losses) if getattr(out, "losses", None) is not None else {}

        u = out.y
        z = self._zero_scalar_like(u)

        total = losses.get("total", z)
        total = self._to_scalar_tensor(total, z)
        losses["total"] = total

        if physics_fn is None or physics_data is None:
            return OperatorOutput(y=u, losses=losses, extras=out.extras)

        pl_raw = physics_fn(u, **physics_data)
        if not isinstance(pl_raw, dict):
            raise TypeError("PINO physics_fn must return a dict[str, Tensor/number].")

        pl: Dict[str, torch.Tensor] = {}
        for k, v in pl_raw.items():
            kk = k if str(k).startswith(self.prefix) else f"{self.prefix}{k}"
            pl[kk] = self._to_scalar_tensor(v, z)

        pde = pl.get(f"{self.prefix}pde", z)
        bc = pl.get(f"{self.prefix}bc", z)
        ic = pl.get(f"{self.prefix}ic", z)

        physics_total = pl.get(
            f"{self.prefix}physics_total",
            self.lambda_pde * pde + self.lambda_bc * bc + self.lambda_ic * ic,
        )

        pl[f"{self.prefix}physics_total"] = physics_total
        pl.setdefault(f"{self.prefix}pde", pde)
        pl.setdefault(f"{self.prefix}bc", bc)
        pl.setdefault(f"{self.prefix}ic", ic)

        losses.update(pl)
        losses["total"] = losses["total"] + self.lambda_physics * physics_total

        return OperatorOutput(y=u, losses=losses, extras=out.extras)


@dataclass
class GridSpec:
    """
    Descreve grid uniforme (PINO clássico).
    L: tamanho físico por eixo (mesmo length de dims).
    dims: quais dims de u correspondem aos eixos (ex.: (-2, -1) para 2D).
    """
    L: Sequence[float]
    dims: Sequence[int]


@dataclass
class ResidualSpec:
    """
    Spec genérico para PDE residual.
    residual(u=..., deriv=..., params=..., **extra) -> r
    onde deriv é o dict retornado por compute_derivatives.
    """
    residual: ResidualFn
    orders: Sequence[int] = (1, 2)
    method: DerivMethod = "spectral"


def pino_physics_fn(
    u: torch.Tensor,
    *,
    grid: GridSpec,
    spec: ResidualSpec,
    params: Optional[Dict[str, Any]] = None,
    bc_mask: Optional[torch.Tensor] = None,
    bc_value: Optional[torch.Tensor] = None,
    ic_mask: Optional[torch.Tensor] = None,
    ic_value: Optional[torch.Tensor] = None,
    pde_weight: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Generic PINO physics loss:
      - computes derivatives on grid (FFT/FD)
      - evaluates residual r
      - returns pde/bc/ic + physics_total
    """
    device, dtype = u.device, u.dtype
    z = torch.zeros((), device=device, dtype=dtype)

    deriv = compute_derivatives(u, grid=grid, orders=spec.orders, method=spec.method)

    r = spec.residual(u=u, deriv=deriv, params=params or {})
    if not torch.is_tensor(r):
        raise TypeError("ResidualSpec.residual must return a torch.Tensor residual field.")

    if pde_weight is not None:
        pw = pde_weight.to(device=device, dtype=dtype)
        pde = ((r ** 2) * pw).mean()
    else:
        pde = (r ** 2).mean()

    bc = z
    if bc_mask is not None and bc_value is not None:
        bc_value = bc_value.to(device=device, dtype=dtype)
        bc = ((u - bc_value)[bc_mask] ** 2).mean()

    ic = z
    if ic_mask is not None and ic_value is not None:
        ic_value = ic_value.to(device=device, dtype=dtype)
        ic = ((u - ic_value)[ic_mask] ** 2).mean()

    return {"pde": pde, "bc": bc, "ic": ic, "physics_total": pde + bc + ic}


def _wavenumbers(n: int, L: float, device, dtype) -> torch.Tensor:
    k = 2.0 * torch.pi * torch.fft.fftfreq(n, d=L / n, device=device)
    return k.to(dtype=dtype)


def spectral_derivative(u: torch.Tensor, *, dim: int, L: float, order: int = 1) -> torch.Tensor:
    device, dtype = u.device, u.dtype
    n = u.shape[dim]
    k = _wavenumbers(n, L, device, dtype)

    U = torch.fft.fft(u, dim=dim)
    ik = (1j * k) ** order

    shape = [1] * u.ndim
    shape[dim] = -1
    ik = ik.reshape(shape)

    du = torch.fft.ifft(U * ik, dim=dim).real
    return du


def fd_derivative(u: torch.Tensor, *, dim: int, dx: float, order: int = 1) -> torch.Tensor:
    if order not in (1, 2):
        raise ValueError("fd_derivative supports only order 1 or 2.")
    shift_p = torch.roll(u, shifts=-1, dims=dim)
    shift_m = torch.roll(u, shifts=+1, dims=dim)
    if order == 1:
        return (shift_p - shift_m) / (2.0 * dx)
    return (shift_p - 2.0 * u + shift_m) / (dx * dx)


def compute_derivatives(
    u: torch.Tensor,
    *,
    grid: GridSpec,
    orders: Sequence[int] = (1, 2),
    method: DerivMethod = "spectral",
) -> Dict[Tuple[int, int], torch.Tensor]:
    """
    Retorna derivadas por eixo e ordem:
      key = (axis_index, order)
    onde axis_index é o índice dentro de grid.dims/grid.L.
    """
    out: Dict[Tuple[int, int], torch.Tensor] = {}
    for ax_i, (dim, L) in enumerate(zip(grid.dims, grid.L)):
        n = u.shape[dim]
        dx = L / n
        for o in orders:
            if method == "spectral":
                out[(ax_i, o)] = spectral_derivative(u, dim=dim, L=L, order=o)
            elif method == "fd":
                out[(ax_i, o)] = fd_derivative(u, dim=dim, dx=dx, order=o)
            else:
                raise ValueError(f"Unknown method: {method}")
    return out