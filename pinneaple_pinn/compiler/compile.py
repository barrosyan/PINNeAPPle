from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from pinneaple_environment.spec import ProblemSpec
from pinneaple_environment.conditions import ConditionSpec

from .autograd_ops import (
    ensure_tensor,
    grad,
    jacobian,
    divergence,
    laplacian,
    time_derivative,
    norm_dot_grad,
    mse,
)
from .loss import LossWeights


def _coord_index(coords: Sequence[str], name: str) -> int:
    if name not in coords:
        raise KeyError(f"Coord '{name}' not in coords={coords}")
    return list(coords).index(name)


def _split_fields(y: torch.Tensor, field_names: Sequence[str]) -> Dict[str, torch.Tensor]:
    if y.ndim == 1:
        y = y[:, None]
    if y.shape[1] != len(field_names):
        raise ValueError(f"Model out_dim={y.shape[1]} != number of fields={len(field_names)} ({field_names})")
    out = {}
    for i, f in enumerate(field_names):
        out[f] = y[:, i:i + 1]
    return out


def _gather_condition_points(batch: Dict[str, Any], cond: ConditionSpec):
    if cond.kind in ("dirichlet", "neumann", "robin"):
        return batch.get("x_bc"), batch.get("y_bc")
    if cond.kind == "initial":
        return batch.get("x_ic"), batch.get("y_ic")
    return batch.get("x_data"), batch.get("y_data")


def compile_problem(
    spec: ProblemSpec,
    *,
    weights: Optional[LossWeights] = None,
) -> Callable[[torch.nn.Module, Any, Dict[str, Any]], Dict[str, torch.Tensor]]:
    w = weights or LossWeights()
    coords = spec.coords
    field_names = list(spec.fields)

    has_t = ("t" in coords)
    t_index = _coord_index(coords, "t") if has_t else None

    spatial_coord_names = [c for c in coords if c != "t"]
    spatial_dim = len(spatial_coord_names)

    def loss_fn(model: torch.nn.Module, y_hat: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        device = next(model.parameters()).device
        ctx = batch.get("ctx", {})

        xcol = batch.get("x_col")
        if xcol is None:
            raise KeyError("Batch missing 'x_col'")
        xcol = xcol.to(device).clone().detach().requires_grad_(True)

        ycol = ensure_tensor(model(xcol))
        fields = _split_fields(ycol, field_names)

        pde_kind = spec.pde.kind
        p = spec.pde.params
        res_list: List[torch.Tensor] = []

        if pde_kind == "laplace":
            if len(field_names) != 1:
                raise ValueError("Laplace expects 1 scalar field.")
            phi = fields[field_names[0]]
            res_list.append(laplacian(phi, xcol))

        elif pde_kind == "poisson":
            if len(field_names) != 1:
                raise ValueError("Poisson expects 1 scalar field.")
            phi = fields[field_names[0]]
            f_fn = ctx.get("source_fn") or ctx.get("f_fn")
            if f_fn is None:
                f = torch.zeros_like(phi)
            else:
                f_np = f_fn(xcol.detach().cpu().numpy(), ctx)
                f = torch.as_tensor(f_np, device=device, dtype=phi.dtype)
                if f.ndim == 1:
                    f = f[:, None]
            res_list.append(laplacian(phi, xcol) - f)

        elif pde_kind == "helmholtz":
            if len(field_names) != 1:
                raise ValueError("Helmholtz expects 1 scalar field.")
            u = fields[field_names[0]]
            k = float(p.get("k", 1.0))
            f_fn = ctx.get("source_fn") or ctx.get("f_fn")
            f = torch.zeros_like(u)
            if f_fn is not None:
                f_np = f_fn(xcol.detach().cpu().numpy(), ctx)
                f = torch.as_tensor(f_np, device=device, dtype=u.dtype)
                if f.ndim == 1:
                    f = f[:, None]
            res_list.append(laplacian(u, xcol) + (k * k) * u - f)

        elif pde_kind == "heat_equation":
            if not has_t:
                raise ValueError("Heat equation expects time coord 't'.")
            if len(field_names) != 1:
                raise ValueError("Heat equation expects scalar field (T).")
            T = fields[field_names[0]]
            alpha = float(p.get("alpha", p.get("kappa", 1e-3)))
            q_fn = ctx.get("source_fn") or ctx.get("q_fn")
            q = torch.zeros_like(T)
            if q_fn is not None:
                q_np = q_fn(xcol.detach().cpu().numpy(), ctx)
                q = torch.as_tensor(q_np, device=device, dtype=T.dtype)
                if q.ndim == 1:
                    q = q[:, None]
            Tt = time_derivative(T, xcol, t_index)  # type: ignore[arg-type]
            res_list.append(Tt - alpha * laplacian(T, xcol) - q)

        elif pde_kind == "wave_equation":
            if not has_t:
                raise ValueError("Wave equation expects time coord 't'.")
            if len(field_names) != 1:
                raise ValueError("Wave equation expects scalar field.")
            u = fields[field_names[0]]
            c = float(p.get("c", 1.0))
            f_fn = ctx.get("source_fn") or ctx.get("f_fn")
            f = torch.zeros_like(u)
            if f_fn is not None:
                f_np = f_fn(xcol.detach().cpu().numpy(), ctx)
                f = torch.as_tensor(f_np, device=device, dtype=u.dtype)
                if f.ndim == 1:
                    f = f[:, None]
            ut = time_derivative(u, xcol, t_index)  # type: ignore[arg-type]
            utt = time_derivative(ut, xcol, t_index)  # type: ignore[arg-type]
            res_list.append(utt - (c * c) * laplacian(u, xcol) - f)

        elif pde_kind == "advection_diffusion":
            if not has_t:
                raise ValueError("Advection-diffusion expects time coord 't'.")
            if len(field_names) != 1:
                raise ValueError("Advection-diffusion expects scalar field c.")
            c = fields[field_names[0]]
            Pe = p.get("Pe", None)
            kappa = float(p.get("kappa", 1.0 / float(Pe) if Pe is not None else 1e-3))
            c_t = time_derivative(c, xcol, t_index)  # type: ignore[arg-type]
            gc = grad(c, xcol)
            vel = []
            for name in spatial_coord_names:
                key = {"x": "u0", "y": "v0", "z": "w0"}.get(name)
                vel.append(float(p.get(key, 0.0)) if key else 0.0)
            vel_t = torch.tensor(vel, device=device, dtype=c.dtype)[None, :]
            sp_idx = [coords.index(n) for n in spatial_coord_names]
            adv = torch.sum(gc[:, sp_idx] * vel_t, dim=1, keepdim=True)
            res_list.append(c_t + adv - kappa * laplacian(c, xcol))

        elif pde_kind == "burgers":
            nu = float(p.get("nu", 0.01))
            if not has_t:
                raise ValueError("Burgers expects time coord 't'.")
            if spatial_dim == 1 and field_names == ["u"]:
                u = fields["u"]
                ut = time_derivative(u, xcol, t_index)  # type: ignore[arg-type]
                gu = grad(u, xcol)
                ix = _coord_index(coords, "x")
                ux = gu[:, ix:ix + 1]
                uxx = torch.autograd.grad(
                    outputs=ux,
                    inputs=xcol,
                    grad_outputs=torch.ones_like(ux),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=False,
                )[0][:, ix:ix + 1]
                res_list.append(ut + u * ux - nu * uxx)
            else:
                raise ValueError(f"Unsupported Burgers configuration: spatial_dim={spatial_dim}, fields={field_names}")

        elif pde_kind == "navier_stokes_incompressible":
            if not has_t:
                raise ValueError("Navier–Stokes expects time coord 't'.")
            Re = float(p.get("Re", 100.0))
            inv_Re = float(p.get("inv_Re", 1.0 / Re))

            if spatial_dim == 2:
                needed = ["u", "v", "p"]
            elif spatial_dim == 3:
                needed = ["u", "v", "w", "p"]
            else:
                raise ValueError("NS requires 2D or 3D spatial dims.")
            for n in needed:
                if n not in fields:
                    raise ValueError(f"NS expects field '{n}' in outputs.")

            if spatial_dim == 2:
                u, v, p_ = fields["u"], fields["v"], fields["p"]
                U = torch.cat([u, v], dim=1)
            else:
                u, v, wv, p_ = fields["u"], fields["v"], fields["w"], fields["p"]
                U = torch.cat([u, v, wv], dim=1)

            ut = time_derivative(u, xcol, t_index)  # type: ignore[arg-type]
            vt = time_derivative(v, xcol, t_index)  # type: ignore[arg-type]
            if spatial_dim == 3:
                wt = time_derivative(wv, xcol, t_index)  # type: ignore[arg-type]

            JU = jacobian(U, xcol)
            sp_idx = [coords.index(n) for n in spatial_coord_names]
            JUs = JU[:, :, sp_idx]

            conv = torch.zeros((xcol.shape[0], spatial_dim), device=device, dtype=xcol.dtype)
            for i in range(spatial_dim):
                for j in range(spatial_dim):
                    conv[:, i] = conv[:, i] + U[:, j] * JUs[:, i, j]

            conv_u = conv[:, 0:1]
            conv_v = conv[:, 1:2]
            if spatial_dim == 3:
                conv_w = conv[:, 2:3]

            gp = grad(p_, xcol)
            px = gp[:, sp_idx[0]:sp_idx[0] + 1]
            py = gp[:, sp_idx[1]:sp_idx[1] + 1]
            if spatial_dim == 3:
                pz = gp[:, sp_idx[2]:sp_idx[2] + 1]

            lap_u = laplacian(u, xcol)
            lap_v = laplacian(v, xcol)
            if spatial_dim == 3:
                lap_w = laplacian(wv, xcol)

            res_list.append(ut + conv_u + px - inv_Re * lap_u)
            res_list.append(vt + conv_v + py - inv_Re * lap_v)
            if spatial_dim == 3:
                res_list.append(wt + conv_w + pz - inv_Re * lap_w)

            res_list.append(divergence(U, xcol))

        elif pde_kind == "linear_elasticity":
            if spatial_dim not in (2, 3):
                raise ValueError("Elasticity expects 2D or 3D spatial dims.")
            needed = ["ux", "uy"] + (["uz"] if spatial_dim == 3 else [])
            for n in needed:
                if n not in fields:
                    raise ValueError(f"Elasticity expects '{n}'.")

            lam = float(p.get("lambda", 1.0))
            mu = float(p.get("mu", 1.0))

            U = torch.cat([fields["ux"], fields["uy"]] + ([fields["uz"]] if spatial_dim == 3 else []), dim=1)
            JU = jacobian(U, xcol)
            sp_idx = [coords.index(n) for n in spatial_coord_names]
            Gu = JU[:, :, sp_idx]

            eps = 0.5 * (Gu + torch.transpose(Gu, 1, 2))

            tr = torch.zeros((xcol.shape[0], 1), device=device, dtype=xcol.dtype)
            for i in range(spatial_dim):
                tr = tr + eps[:, i:i + 1, i:i + 1].reshape(-1, 1)

            sigma = torch.zeros_like(eps)
            for i in range(spatial_dim):
                sigma[:, i, i] = sigma[:, i, i] + lam * tr[:, 0]
            sigma = sigma + 2.0 * mu * eps

            b_fn = ctx.get("body_force_fn")
            b = torch.zeros((xcol.shape[0], spatial_dim), device=device, dtype=xcol.dtype)
            if b_fn is not None:
                b_np = b_fn(xcol.detach().cpu().numpy(), ctx)
                b = torch.as_tensor(b_np, device=device, dtype=xcol.dtype)
                if b.ndim == 1:
                    b = b[:, None].repeat(1, spatial_dim)

            for i in range(spatial_dim):
                div_si = torch.zeros((xcol.shape[0], 1), device=device, dtype=xcol.dtype)
                for j in range(spatial_dim):
                    sij = sigma[:, i:i + 1, j:j + 1].reshape(-1, 1)
                    g = grad(sij, xcol)
                    div_si = div_si + g[:, sp_idx[j]:sp_idx[j] + 1]
                res_list.append(div_si + b[:, i:i + 1])

        elif pde_kind == "darcy":
            mode = spec.pde.meta.get("mode", "pressure_only")
            k = float(p.get("k", 1.0))
            mu = float(p.get("mu", 1.0))
            s_fn = ctx.get("source_fn") or ctx.get("s_fn")

            if mode == "pressure_only":
                if len(field_names) != 1:
                    raise ValueError("Darcy pressure_only expects scalar p.")
                pfield = fields[field_names[0]]
                s = torch.zeros_like(pfield)
                if s_fn is not None:
                    s_np = s_fn(xcol.detach().cpu().numpy(), ctx)
                    s = torch.as_tensor(s_np, device=device, dtype=pfield.dtype)
                    if s.ndim == 1:
                        s = s[:, None]
                res_list.append(k * laplacian(pfield, xcol) - s)
            else:
                raise ValueError("Darcy mixed mode not implemented in this patch.")

        else:
            raise ValueError(f"Unsupported PDE kind: {pde_kind}")

        pde_res = torch.cat([r if r.ndim == 2 else r[:, None] for r in res_list], dim=1)
        l_pde = torch.mean(pde_res ** 2)

        total = w.w_pde * l_pde
        out: Dict[str, torch.Tensor] = {"pde": l_pde}

        def eval_fields(X: torch.Tensor) -> Dict[str, torch.Tensor]:
            y = ensure_tensor(model(X))
            return _split_fields(y, field_names)

        for cond in spec.conditions:
            Xc, Yc = _gather_condition_points(batch, cond)
            if Xc is None:
                continue
            Xc = Xc.to(device)

            mask_key = f"mask_{cond.name}"
            if mask_key in batch:
                m = batch[mask_key].to(device).bool()
                Xc = Xc[m]
                if Yc is not None:
                    Yc = Yc.to(device)[m]
            else:
                if Yc is not None:
                    Yc = Yc.to(device)

            if Xc.numel() == 0:
                continue

            fvals = eval_fields(Xc)
            pred = torch.cat([fvals[f] for f in cond.fields], dim=1)

            if Yc is None:
                Y_np = cond.values(Xc.detach().cpu().numpy(), ctx)
                Yc = torch.as_tensor(Y_np, device=device, dtype=Xc.dtype)
            else:
                if Yc.ndim == 1:
                    Yc = Yc[:, None]

                # if y_bc is full out_dim, slice matching cond.fields
                if Yc.shape[1] == len(field_names) and len(cond.fields) != len(field_names):
                    idxs = [field_names.index(f) for f in cond.fields]
                    Yc = Yc[:, idxs]

            if cond.kind == "dirichlet":
                l = mse(pred, Yc)
                out[f"bc_{cond.name}"] = l.detach()
                total = total + (w.w_bc * float(cond.weight)) * l

            elif cond.kind == "neumann":
                n = batch.get("n_bc")
                if n is None:
                    raise KeyError("NeumannBC requires batch['n_bc']")
                n = n.to(device)
                if mask_key in batch:
                    n = n[batch[mask_key].to(device).bool()]
                parts = []
                for fname in cond.fields:
                    Xr = Xc.clone().detach().requires_grad_(True)
                    u = eval_fields(Xr)[fname]
                    parts.append(norm_dot_grad(u, Xr, n))
                flux_pred = torch.cat(parts, dim=1)
                l = mse(flux_pred, Yc)
                out[f"bc_{cond.name}"] = l.detach()
                total = total + (w.w_bc * float(cond.weight)) * l

            elif cond.kind == "robin":
                coeffs = (ctx.get("robin_coeffs") or {}).get(cond.name, {"a": 1.0, "b": 1.0})
                a = float(coeffs.get("a", 1.0))
                b_ = float(coeffs.get("b", 1.0))
                n = batch.get("n_bc")
                if n is None:
                    raise KeyError("RobinBC requires batch['n_bc']")
                n = n.to(device)
                if mask_key in batch:
                    n = n[batch[mask_key].to(device).bool()]
                parts = []
                for fname in cond.fields:
                    Xr = Xc.clone().detach().requires_grad_(True)
                    u = eval_fields(Xr)[fname]
                    flux = norm_dot_grad(u, Xr, n)
                    parts.append(a * u + b_ * flux)
                lhs = torch.cat(parts, dim=1)
                l = mse(lhs, Yc)
                out[f"bc_{cond.name}"] = l.detach()
                total = total + (w.w_bc * float(cond.weight)) * l

            elif cond.kind == "initial":
                l = mse(pred, Yc)
                out[f"ic_{cond.name}"] = l.detach()
                total = total + (w.w_ic * float(cond.weight)) * l

            else:
                l = mse(pred, Yc)
                out[f"data_{cond.name}"] = l.detach()
                total = total + (w.w_data * float(cond.weight)) * l

        out["total"] = total
        return out

    return loss_fn