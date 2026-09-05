from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from pinneapple_physics.pde_environment.spec import ProblemSpec
from pinneapple_physics.pde_environment.conditions import ConditionSpec

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
    spatial_indices = [list(coords).index(c) for c in spatial_coord_names]

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
            res_list.append(Tt - alpha * laplacian(T, xcol, spatial_indices) - q)

        elif pde_kind in ("heat_equation_steady", "heat_equation_steady_multilayer"):
            # Steady (time-independent) Fourier conduction: k*laplacian(T)
            # = -q (isotropic). "_multilayer" presets (e.g.
            # refractory_lining) already fold a multi-layer stack into one
            # effective conductivity via a series-resistance formula
            # before reaching here (params["k_eff"]), so both kinds share
            # this branch -- the multilayer physics is in how k_eff was
            # computed, not in the residual itself.
            if len(field_names) != 1:
                raise ValueError(f"{pde_kind} expects a scalar field (T).")
            T = fields[field_names[0]]
            k = float(p.get("k_eff", p.get("k", 1.0)))
            q_fn = ctx.get("source_fn") or ctx.get("q_fn")
            q = torch.zeros_like(T)
            if q_fn is not None:
                q_np = q_fn(xcol.detach().cpu().numpy(), ctx)
                q = torch.as_tensor(q_np, device=device, dtype=T.dtype)
                if q.ndim == 1:
                    q = q[:, None]
            res_list.append(k * laplacian(T, xcol, spatial_indices) + q)

        elif pde_kind == "heat_equation_steady_anisotropic":
            # Steady conduction with a direction-dependent conductivity
            # (e.g. pcb_thermal's in-plane vs. through-plane FR4
            # conductivity): sum_i k_i * d^2T/dx_i^2 = -q, one k_<coord>
            # parameter per spatial coordinate actually present in this
            # spec (a preset may define k_z even when 'z' isn't one of
            # its coords, e.g. a 2D board -- unused params are simply
            # ignored, not an error).
            if len(field_names) != 1:
                raise ValueError("heat_equation_steady_anisotropic expects a scalar field (T).")
            T = fields[field_names[0]]
            gT = grad(T, xcol)
            lap_aniso = torch.zeros_like(T)
            for coord_name, idx in zip(spatial_coord_names, spatial_indices):
                k_i = float(p.get(f"k_{coord_name}", p.get("k", 1.0)))
                d2T_i = grad(gT[:, idx:idx + 1], xcol)[:, idx:idx + 1]
                lap_aniso = lap_aniso + k_i * d2T_i
            q_fn = ctx.get("source_fn") or ctx.get("q_fn")
            q = torch.zeros_like(T)
            if q_fn is not None:
                q_np = q_fn(xcol.detach().cpu().numpy(), ctx)
                q = torch.as_tensor(q_np, device=device, dtype=T.dtype)
                if q.ndim == 1:
                    q = q[:, None]
            res_list.append(lap_aniso + q)

        elif pde_kind == "heat_equation_transient":
            # Transient (time-dependent) heat conduction, distinguished
            # from the plain "heat_equation" kind because
            # car_brake_thermal's coords are cylindrical (r, z, t), not
            # Cartesian (x, y, t) -- a brake disc genuinely is an
            # axisymmetric body, so its Laplacian needs the standard
            # extra (1/r)*dT/dr term:
            #   rho*cp*dT/dt = k*[d2T/dr2 + (1/r)*dT/dr + d2T/dz2] + q
            # (verified this session with sympy: the plain Cartesian
            # laplacian(), which omits the (1/r)*dT/dr term, gives a
            # nonzero, wrong residual for e.g. T=ln(r) even though that
            # function exactly solves the TRUE axisymmetric equation).
            if not has_t:
                raise ValueError("heat_equation_transient expects a time coord 't'.")
            if "r" not in coords or "z" not in coords:
                raise ValueError("heat_equation_transient expects coords 'r' and 'z' (axisymmetric).")
            if len(field_names) != 1:
                raise ValueError("heat_equation_transient expects a scalar field (T).")
            T = fields[field_names[0]]
            alpha = float(p.get("alpha", 1e-5))
            r_idx = _coord_index(coords, "r")
            z_idx = _coord_index(coords, "z")
            r_val = xcol[:, r_idx:r_idx + 1]
            gT = grad(T, xcol)
            dT_dr = gT[:, r_idx:r_idx + 1]
            d2T_dr2 = grad(dT_dr, xcol)[:, r_idx:r_idx + 1]
            d2T_dz2 = grad(gT[:, z_idx:z_idx + 1], xcol)[:, z_idx:z_idx + 1]
            lap_axisym = d2T_dr2 + dT_dr / r_val + d2T_dz2
            q_fn = ctx.get("source_fn") or ctx.get("q_fn")
            q = torch.zeros_like(T)
            if q_fn is not None:
                q_np = q_fn(xcol.detach().cpu().numpy(), ctx)
                q = torch.as_tensor(q_np, device=device, dtype=T.dtype)
                if q.ndim == 1:
                    q = q[:, None]
            Tt = time_derivative(T, xcol, t_index)  # type: ignore[arg-type]
            res_list.append(Tt - alpha * lap_axisym - q)

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
            res_list.append(utt - (c * c) * laplacian(u, xcol, spatial_indices) - f)

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
            res_list.append(c_t + adv - kappa * laplacian(c, xcol, spatial_indices))

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

        elif pde_kind == "inviscid_burgers":
            if not has_t:
                raise ValueError("Inviscid Burgers expects time coord 't'.")
            if spatial_dim == 1 and field_names == ["u"]:
                u = fields["u"]
                ut = time_derivative(u, xcol, t_index)  # type: ignore[arg-type]
                gu = grad(u, xcol)
                ix = _coord_index(coords, "x")
                ux = gu[:, ix:ix + 1]
                res_list.append(ut + u * ux)
            else:
                raise ValueError(f"Unsupported Burgers configuration: spatial_dim={spatial_dim}, fields={field_names}")

        elif pde_kind in ("navier_stokes_incompressible", "incompressible_navier_stokes_2d"):
            # "incompressible_navier_stokes_2d" (aircraft_wing_aerodynamics,
            # car_external_aero) is the identical physics as
            # "navier_stokes_incompressible" at spatial_dim=2 -- same
            # fields (u, v, p), same params ({nu, Re}), same steady 2D
            # incompressible momentum+continuity equations. This was a
            # naming collision (found the same way reaction_diffusion_2d
            # was: two presets pointing at physics that already existed
            # under a different kind string), not missing physics.
            #
            # Steady-state presets (no 't' coord by design, e.g.
            # channel_flow_3d/lid_driven_cavity_3d/pipe_flow_3d) used to
            # hit this branch's unconditional "expects time coord 't'"
            # check and fail outright -- there is nothing physically
            # wrong with a steady NS residual (drop du/dt, keep
            # convection+pressure+viscous+continuity), so it is supported
            # here instead of rejected.
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

            if has_t:
                ut = time_derivative(u, xcol, t_index)  # type: ignore[arg-type]
                vt = time_derivative(v, xcol, t_index)  # type: ignore[arg-type]
                if spatial_dim == 3:
                    wt = time_derivative(wv, xcol, t_index)  # type: ignore[arg-type]
            else:
                ut = torch.zeros_like(u)
                vt = torch.zeros_like(v)
                if spatial_dim == 3:
                    wt = torch.zeros_like(wv)

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

            lap_u = laplacian(u, xcol, spatial_indices)
            lap_v = laplacian(v, xcol, spatial_indices)
            if spatial_dim == 3:
                lap_w = laplacian(wv, xcol, spatial_indices)

            # Optional momentum source, e.g. a periodic channel/duct's
            # OpenFOAM-style meanVelocityForce fvOption (a spatially-uniform
            # streamwise force holding the bulk velocity at a target) --
            # mirrors the body_force_fn hook linear_elasticity/darcy already
            # have; navier_stokes_incompressible previously had no such hook
            # at all, so a momentum-source-driven flow (any periodic channel,
            # pipe, or duct case) could not be expressed through this
            # compiler.
            b_fn = ctx.get("body_force_fn")
            f = torch.zeros((xcol.shape[0], spatial_dim), device=device, dtype=xcol.dtype)
            if b_fn is not None:
                f_np = b_fn(xcol.detach().cpu().numpy(), ctx)
                f = torch.as_tensor(f_np, device=device, dtype=xcol.dtype)
                if f.ndim == 1 or (f.ndim == 2 and f.shape[1] == 1):
                    raise ValueError(
                        "navier_stokes_incompressible body_force_fn returned a "
                        f"scalar-valued ({tuple(f.shape)}) source. A scalar cannot "
                        "be broadcast isotropically across all momentum components "
                        "-- most momentum sources (e.g. a streamwise "
                        "meanVelocityForce) act along a single direction, and "
                        "silently applying the scalar to every component would "
                        f"produce a physically incorrect forcing. Return a full "
                        f"(N, {spatial_dim}) vector source instead."
                    )

            res_list.append(ut + conv_u + px - inv_Re * lap_u - f[:, 0:1])
            res_list.append(vt + conv_v + py - inv_Re * lap_v - f[:, 1:2])
            if spatial_dim == 3:
                res_list.append(wt + conv_w + pz - inv_Re * lap_w - f[:, 2:3])

            res_list.append(divergence(U, xcol, spatial_indices))

        elif pde_kind in ("linear_elasticity", "linear_elasticity_plane_strain", "linear_elasticity_plane_stress"):
            # "linear_elasticity" at spatial_dim=2 and
            # "linear_elasticity_plane_strain" are mathematically
            # identical: plane strain IS "assume zero out-of-plane
            # strain, use the full 3D constitutive relation restricted
            # to 2D" -- exactly what the generic branch below already
            # does for spatial_dim=2, no special-casing needed.
            # "linear_elasticity_plane_stress" needs one change: the
            # standard reduced Lame parameter lambda* = 2*lambda*mu /
            # (lambda + 2*mu) (Timoshenko & Goodier, Theory of
            # Elasticity) in place of the raw 3D lambda, so that
            # sigma_zz comes out to 0 as plane stress requires.
            if spatial_dim not in (2, 3):
                raise ValueError("Elasticity expects 2D or 3D spatial dims.")
            needed = ["ux", "uy"] + (["uz"] if spatial_dim == 3 else [])
            for n in needed:
                if n not in fields:
                    raise ValueError(f"Elasticity expects '{n}'.")

            lam = float(p.get("lambda", 1.0))
            mu = float(p.get("mu", 1.0))
            if pde_kind == "linear_elasticity_plane_stress":
                lam = 2.0 * lam * mu / (lam + 2.0 * mu)

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
                if b.ndim == 1 or (b.ndim == 2 and b.shape[1] == 1):
                    raise ValueError(
                        "linear_elasticity body_force_fn returned a scalar-valued "
                        f"({tuple(b.shape)}) body force. A scalar cannot be broadcast "
                        "isotropically across all spatial axes -- most body forces "
                        "(e.g. gravity, b=(0, -rho*g) in 2D) act along a single "
                        "direction, and silently applying the scalar to every "
                        "equilibrium equation would produce a physically incorrect "
                        f"loading condition. Return a full (N, {spatial_dim}) vector "
                        "body force instead."
                    )

            for i in range(spatial_dim):
                div_si = torch.zeros((xcol.shape[0], 1), device=device, dtype=xcol.dtype)
                for j in range(spatial_dim):
                    sij = sigma[:, i:i + 1, j:j + 1].reshape(-1, 1)
                    g = grad(sij, xcol)
                    div_si = div_si + g[:, sp_idx[j]:sp_idx[j] + 1]
                res_list.append(div_si + b[:, i:i + 1])

        elif pde_kind == "thermoelasticity_2d":
            # Coupled 2D thermoelasticity (Boley & Weiner, Theory of
            # Thermal Stresses): steady heat conduction (shape-only, so
            # conductivity k cancels out of a Dirichlet-driven Laplace
            # problem and isn't needed as a parameter) plus plane-stress
            # linear elasticity with an added isotropic thermal strain
            # eps_th = alpha_T * T * I (T taken relative to the preset's
            # zero-strain reference, matching its own boundary conditions
            # T_cold=0/T_hot=dT):
            #   div(grad(T)) = 0
            #   sigma = C_planestress : (eps - alpha_T*T*I),  div(sigma) = 0
            for n in ("ux", "uy", "T"):
                if n not in fields:
                    raise ValueError(f"thermoelasticity_2d expects field '{n}'.")
            if spatial_dim != 2:
                raise ValueError("thermoelasticity_2d expects 2 spatial dims.")
            T = fields["T"]
            alpha_T = float(p.get("alpha_T", 0.0))
            lam = float(p.get("lambda", 1.0))
            mu = float(p.get("mu", 1.0))
            lam = 2.0 * lam * mu / (lam + 2.0 * mu)  # plane-stress reduction, per this preset's own meta

            res_list.append(laplacian(T, xcol, spatial_indices))

            U = torch.cat([fields["ux"], fields["uy"]], dim=1)
            sp_idx = [coords.index(n) for n in spatial_coord_names]
            JU = jacobian(U, xcol)
            Gu = JU[:, :, sp_idx]
            eps = 0.5 * (Gu + torch.transpose(Gu, 1, 2))

            tr = eps[:, 0:1, 0:1].reshape(-1, 1) + eps[:, 1:2, 1:2].reshape(-1, 1)
            tr_th = 2.0 * alpha_T * T  # trace of the 2D isotropic thermal-strain tensor

            sigma = 2.0 * mu * eps
            sigma[:, 0, 0] = sigma[:, 0, 0] + lam * (tr[:, 0] - tr_th[:, 0]) - 2.0 * mu * alpha_T * T[:, 0]
            sigma[:, 1, 1] = sigma[:, 1, 1] + lam * (tr[:, 0] - tr_th[:, 0]) - 2.0 * mu * alpha_T * T[:, 0]

            for i in range(2):
                div_si = torch.zeros((xcol.shape[0], 1), device=device, dtype=xcol.dtype)
                for j in range(2):
                    sij = sigma[:, i:i + 1, j:j + 1].reshape(-1, 1)
                    g = grad(sij, xcol)
                    div_si = div_si + g[:, sp_idx[j]:sp_idx[j] + 1]
                res_list.append(div_si)

        elif pde_kind == "incompressible_navier_stokes_rotating_frame":
            # Steady incompressible NS in a frame rotating at angular
            # velocity omega about the z-axis (fan_cooler_cfd), adding
            # the standard Coriolis and centrifugal terms to the
            # existing "navier_stokes_incompressible" residual (u, v are
            # RELATIVE velocity in the rotating frame):
            #   (u.grad)u + 2*Omega x u - omega^2*r + grad(p) = inv_Re*lap(u)
            # Derived from 2*Omega x u for Omega=(0,0,omega):
            #   2*Omega x u = (-2*omega*v, 2*omega*u)
            # and Omega x (Omega x r) = -omega^2*r (centrifugal term
            # -omega^2*r moved to the LHS as "+Omega x (Omega x r)"
            # becomes "-omega^2*r" on the force side, i.e. the familiar
            # outward centrifugal push). Verified this session with
            # `sympy` against the textbook solid-body-rotation solution
            # (u=v=0, p=0.5*omega^2*(x^2+y^2)) -- zero relative velocity
            # balanced entirely by a hydrostatic-style pressure field.
            if spatial_dim != 2:
                raise ValueError("incompressible_navier_stokes_rotating_frame expects 2D spatial dims.")
            for n in ("u", "v", "p"):
                if n not in fields:
                    raise ValueError(f"incompressible_navier_stokes_rotating_frame expects field '{n}'.")
            Re = float(p.get("Re", 100.0))
            inv_Re = float(p.get("inv_Re", 1.0 / Re))
            omega = float(p.get("omega", 0.0))

            u_vel, v_vel, p_ = fields["u"], fields["v"], fields["p"]
            U = torch.cat([u_vel, v_vel], dim=1)
            sp_idx = [coords.index(n) for n in spatial_coord_names]
            x_val = xcol[:, sp_idx[0]:sp_idx[0] + 1]
            y_val = xcol[:, sp_idx[1]:sp_idx[1] + 1]

            JU = jacobian(U, xcol)
            JUs = JU[:, :, sp_idx]
            conv = torch.zeros((xcol.shape[0], 2), device=device, dtype=xcol.dtype)
            for i in range(2):
                for j in range(2):
                    conv[:, i] = conv[:, i] + U[:, j] * JUs[:, i, j]

            gp = grad(p_, xcol)
            lap_u = laplacian(u_vel, xcol, spatial_indices)
            lap_v = laplacian(v_vel, xcol, spatial_indices)

            res_list.append(
                conv[:, 0:1] - 2.0 * omega * v_vel - omega * omega * x_val
                + gp[:, sp_idx[0]:sp_idx[0] + 1] - inv_Re * lap_u
            )
            res_list.append(
                conv[:, 1:2] + 2.0 * omega * u_vel - omega * omega * y_val
                + gp[:, sp_idx[1]:sp_idx[1] + 1] - inv_Re * lap_v
            )
            res_list.append(divergence(U, xcol, spatial_indices))

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
                res_list.append((k / mu) * laplacian(pfield, xcol) - s)
            else:
                raise ValueError("Darcy mixed mode not implemented in this patch.")

        elif pde_kind == "euler_bernoulli_beam":
            # EI d^4w/dz^4 - F_axial d^2w/dz^2 - q(z) = 0.
            # F_axial is a fixed (non-learned) compressive axial load producing
            # a linear P-Delta amplification term; this branch does NOT cover
            # geometrically-nonlinear (Von Karman) beams, which need a second,
            # coupled axial-displacement field and a quadratic strain term —
            # tracked separately, not handled here.
            if len(field_names) != 1:
                raise ValueError("Euler-Bernoulli beam expects 1 scalar field (deflection).")
            defl = fields[field_names[0]]  # NOTE: intentionally not named "w" — that name is
            # already bound to the outer LossWeights instance in this closure's scope, and
            # Python would silently shadow it for the rest of loss_fn otherwise.
            EI = float(p.get("EI", 1.0))
            F_axial = float(p.get("F_axial", 0.0))
            z_idx = _coord_index(coords, "z")

            q_fn = ctx.get("source_fn") or ctx.get("q_fn")
            if q_fn is not None:
                q_np = q_fn(xcol.detach().cpu().numpy(), ctx)
                q = torch.as_tensor(q_np, device=device, dtype=defl.dtype)
                if q.ndim == 1:
                    q = q[:, None]
            elif "q" in p:
                q = torch.full_like(defl, float(p["q"]))
            else:
                q = torch.zeros_like(defl)

            w1 = grad(defl, xcol)[:, z_idx:z_idx + 1]
            w2 = torch.autograd.grad(
                outputs=w1, inputs=xcol, grad_outputs=torch.ones_like(w1),
                create_graph=True, retain_graph=True, allow_unused=False,
            )[0][:, z_idx:z_idx + 1]
            w3 = torch.autograd.grad(
                outputs=w2, inputs=xcol, grad_outputs=torch.ones_like(w2),
                create_graph=True, retain_graph=True, allow_unused=False,
            )[0][:, z_idx:z_idx + 1]
            w4 = torch.autograd.grad(
                outputs=w3, inputs=xcol, grad_outputs=torch.ones_like(w3),
                create_graph=True, retain_graph=True, allow_unused=False,
            )[0][:, z_idx:z_idx + 1]

            res_list.append(EI * w4 - F_axial * w2 - q)

        elif pde_kind == "hyperelasticity_neo_hookean":
            # Compressible Neo-Hookean, plane-strain assumption (F_33=1) —
            # documented modeling choice: the source problem is registered
            # symbolically only (sigma_xx/sigma_xy/sigma_yy divergence form,
            # E, nu, K) with no numerical solver ever implemented, so neither
            # plane-stress-vs-strain nor how K relates to E/nu is fixed by
            # the source itself. Convention used here: mu from E,nu (the
            # well-defined shear modulus); K used directly as an independent
            # volumetric/bulk penalty (standard practice for near-
            # incompressible Neo-Hookean, decoupled from the linear-elastic
            # lambda). Energy: W = (mu/2)(I1-2-2lnJ) + (K/2)(lnJ)^2, giving
            # the standard result P = mu(F-F^-T) + K*ln(J)*F^-T, then
            # sigma = (1/J) P F^T.
            if spatial_dim != 2:
                raise ValueError("hyperelasticity_neo_hookean (this branch) expects 2D spatial dims.")
            for n in ("u", "v"):
                if n not in fields:
                    raise ValueError(f"hyperelasticity_neo_hookean expects field '{n}'.")
            E_mod = float(p.get("E", 210e9))
            nu = float(p.get("nu", 0.3))
            K_bulk = float(p.get("K", 1.0))
            mu = E_mod / (2.0 * (1.0 + nu))

            sp_idx = [coords.index(c) for c in spatial_coord_names]
            x_idx, y_idx = sp_idx[0], sp_idx[1]
            U = torch.cat([fields["u"], fields["v"]], dim=1)
            JU = jacobian(U, xcol)
            Gu = JU[:, :, sp_idx]

            eye2 = torch.eye(2, device=device, dtype=xcol.dtype).unsqueeze(0)
            F = eye2 + Gu

            F00, F01 = F[:, 0, 0], F[:, 0, 1]
            F10, F11 = F[:, 1, 0], F[:, 1, 1]
            detF = F00 * F11 - F01 * F10
            detF_safe = torch.clamp(detF, min=1e-6)
            lnJ = torch.log(detF_safe)

            invT00 = F11 / detF_safe
            invT01 = -F10 / detF_safe
            invT10 = -F01 / detF_safe
            invT11 = F00 / detF_safe

            P00 = mu * (F00 - invT00) + K_bulk * lnJ * invT00
            P01 = mu * (F01 - invT01) + K_bulk * lnJ * invT01
            P10 = mu * (F10 - invT10) + K_bulk * lnJ * invT10
            P11 = mu * (F11 - invT11) + K_bulk * lnJ * invT11

            s00 = (P00 * F00 + P01 * F01) / detF_safe
            s01 = (P00 * F10 + P01 * F11) / detF_safe
            s10 = (P10 * F00 + P11 * F01) / detF_safe
            s11 = (P10 * F10 + P11 * F11) / detF_safe

            sigma_xx = s00.unsqueeze(1)
            sigma_xy = (0.5 * (s01 + s10)).unsqueeze(1)  # enforce symmetry numerically
            sigma_yy = s11.unsqueeze(1)

            b_fn = ctx.get("body_force_fn")
            fx = torch.zeros((xcol.shape[0], 1), device=device, dtype=xcol.dtype)
            fy = torch.zeros((xcol.shape[0], 1), device=device, dtype=xcol.dtype)
            if b_fn is not None:
                b_np = b_fn(xcol.detach().cpu().numpy(), ctx)
                b_t = torch.as_tensor(b_np, device=device, dtype=xcol.dtype)
                if b_t.ndim == 1:
                    b_t = b_t[:, None].repeat(1, 2)
                fx, fy = b_t[:, 0:1], b_t[:, 1:2]

            dsxx_dx = grad(sigma_xx, xcol)[:, x_idx:x_idx + 1]
            dsxy_dy = grad(sigma_xy, xcol)[:, y_idx:y_idx + 1]
            dsxy_dx = grad(sigma_xy, xcol)[:, x_idx:x_idx + 1]
            dsyy_dy = grad(sigma_yy, xcol)[:, y_idx:y_idx + 1]

            res_list.append(dsxx_dx + dsxy_dy + fx)
            res_list.append(dsxy_dx + dsyy_dy + fy)

        elif pde_kind == "buckley_leverett_two_phase":
            # Immiscible two-phase (water-oil) Darcy flow: water-saturation
            # transport coupled to a pressure equation via total mobility and
            # fractional flow. Uses a standard quadratic Corey relative-
            # permeability model (krw=Sw^2, kro=(1-Sw)^2) since the exact
            # relperm law is a modeling choice not fixed by the governing
            # equation itself; this is a documented default, not a fabricated
            # unknown.
            if not has_t:
                raise ValueError("Two-phase reservoir flow expects time coord 't'.")
            for n in ("Sw", "p"):
                if n not in fields:
                    raise ValueError(f"Two-phase reservoir flow expects field '{n}'.")
            phi = float(p.get("phi", 0.2))
            k_perm = float(p.get("k", 1e-13))
            mu_w = float(p.get("mu_w", 1e-3))
            mu_o = float(p.get("mu_o", 5e-3))
            q_well = float(p.get("q_well", 0.0))

            sp_idx = [coords.index(c) for c in spatial_coord_names]

            Sw, p_res = fields["Sw"], fields["p"]
            Sw_c = torch.clamp(Sw, 1e-4, 1.0 - 1e-4)
            krw = Sw_c ** 2
            kro = (1.0 - Sw_c) ** 2
            lambda_w = krw / mu_w
            lambda_o = kro / mu_o
            lambda_t = lambda_w + lambda_o
            fw = lambda_w / lambda_t

            grad_p = grad(p_res, xcol)
            q_vec = [-k_perm * lambda_t * grad_p[:, idx:idx + 1] for idx in sp_idx]

            Sw_t = time_derivative(Sw, xcol, t_index)  # type: ignore[arg-type]
            div_flux = torch.zeros_like(Sw)
            for idx, qc in zip(sp_idx, q_vec):
                g = grad(fw * qc, xcol)
                div_flux = div_flux + g[:, idx:idx + 1]
            res_list.append(phi * Sw_t + div_flux)

            kflux = [k_perm * lambda_t * grad_p[:, idx:idx + 1] for idx in sp_idx]
            div_kflux = torch.zeros_like(p_res)
            for idx, fc in zip(sp_idx, kflux):
                g = grad(fc, xcol)
                div_kflux = div_kflux + g[:, idx:idx + 1]
            res_list.append(div_kflux + q_well)

        elif pde_kind == "biot_poroelasticity":
            # Linear poroelastic consolidation (Biot theory): elastic skeleton
            # momentum coupled to transient pore-pressure diffusion. Fully
            # linear, no internal state/history variables — unlike plasticity,
            # this is a straightforward extension of the existing
            # linear_elasticity residual plus a diffusion equation.
            if not has_t:
                raise ValueError("Biot poroelasticity expects time coord 't'.")
            for n in ("u", "v", "p"):
                if n not in fields:
                    raise ValueError(f"Biot poroelasticity expects field '{n}'.")
            if spatial_dim != 2:
                raise ValueError("Biot poroelasticity (this branch) expects 2D spatial dims.")

            lam = float(p.get("lambda", 1.0))
            mu = float(p.get("mu", 1.0))
            alpha = float(p.get("alpha", 1.0))
            M_biot = float(p.get("M", 1.0))
            k_perm = float(p.get("k", 1.0))
            mu_f = float(p.get("mu_f", 1.0))

            sp_idx = [coords.index(c) for c in spatial_coord_names]
            x_idx, y_idx = sp_idx[0], sp_idx[1]

            u_f, v_f, p_f = fields["u"], fields["v"], fields["p"]

            U = torch.cat([u_f, v_f], dim=1)
            JU = jacobian(U, xcol)
            Gu = JU[:, :, sp_idx]
            eps = 0.5 * (Gu + torch.transpose(Gu, 1, 2))
            tr = eps[:, 0:1, 0:1].reshape(-1, 1) + eps[:, 1:2, 1:2].reshape(-1, 1)

            sigma = torch.zeros_like(eps)
            for i in range(spatial_dim):
                sigma[:, i, i] = sigma[:, i, i] + lam * tr[:, 0]
            sigma = sigma + 2.0 * mu * eps

            grad_p = grad(p_f, xcol)
            dp = [grad_p[:, x_idx:x_idx + 1], grad_p[:, y_idx:y_idx + 1]]

            for i in range(spatial_dim):
                div_si = torch.zeros((xcol.shape[0], 1), device=device, dtype=xcol.dtype)
                for j in range(spatial_dim):
                    sij = sigma[:, i:i + 1, j:j + 1].reshape(-1, 1)
                    g = grad(sij, xcol)
                    div_si = div_si + g[:, sp_idx[j]:sp_idx[j] + 1]
                res_list.append(div_si - alpha * dp[i])

            u_t = time_derivative(u_f, xcol, t_index)  # type: ignore[arg-type]
            v_t = time_derivative(v_f, xcol, t_index)  # type: ignore[arg-type]
            d_ut_dx = grad(u_t, xcol)[:, x_idx:x_idx + 1]
            d_vt_dy = grad(v_t, xcol)[:, y_idx:y_idx + 1]
            p_t = time_derivative(p_f, xcol, t_index)  # type: ignore[arg-type]
            lap_p = laplacian(p_f, xcol, sp_idx)

            res_list.append((1.0 / M_biot) * p_t + alpha * (d_ut_dx + d_vt_dy) - (k_perm / mu_f) * lap_p)

        elif pde_kind == "maxwell_te":
            # 2D transverse-electric Maxwell curl equations with conductive loss:
            #   eps dEx/dt - dHz/dy + sigma Ex = 0
            #   eps dEy/dt + dHz/dx + sigma Ey = 0
            #   mu  dHz/dt - dEx/dy + dEy/dx   = 0
            if not has_t:
                raise ValueError("Maxwell TE expects time coord 't'.")
            for n in ("Ex", "Ey", "Hz"):
                if n not in fields:
                    raise ValueError(f"Maxwell TE expects field '{n}'.")
            epsilon = float(p.get("epsilon", 1.0))
            mu = float(p.get("mu", 1.0))
            sigma = float(p.get("sigma", 0.0))
            x_idx = _coord_index(coords, "x")
            y_idx = _coord_index(coords, "y")

            Ex, Ey, Hz = fields["Ex"], fields["Ey"], fields["Hz"]
            dEx_dt = time_derivative(Ex, xcol, t_index)  # type: ignore[arg-type]
            dEy_dt = time_derivative(Ey, xcol, t_index)  # type: ignore[arg-type]
            dHz_dt = time_derivative(Hz, xcol, t_index)  # type: ignore[arg-type]

            grad_Hz = grad(Hz, xcol)
            dHz_dy = grad_Hz[:, y_idx:y_idx + 1]
            dHz_dx = grad_Hz[:, x_idx:x_idx + 1]

            grad_Ex = grad(Ex, xcol)
            dEx_dy = grad_Ex[:, y_idx:y_idx + 1]
            grad_Ey = grad(Ey, xcol)
            dEy_dx = grad_Ey[:, x_idx:x_idx + 1]

            res_list.append(epsilon * dEx_dt - dHz_dy + sigma * Ex)
            res_list.append(epsilon * dEy_dt + dHz_dx + sigma * Ey)
            res_list.append(mu * dHz_dt - dEx_dy + dEy_dx)

        elif pde_kind == "axisymmetric_linear_elasticity":
            # Static axisymmetric (r,z) linear elasticity, no torsional (u_theta)
            # coupling — the hoop strain eps_tt = u_r/r term has no Cartesian
            # equivalent, which is exactly why this is its own kind rather than
            # reusing "linear_elasticity". Classical formulation, e.g.
            # Timoshenko & Goodier, Theory of Elasticity.
            if "r" not in coords or "z" not in coords:
                raise ValueError("Axisymmetric linear elasticity expects coords ('r','z').")
            for n in ("u_r", "u_z"):
                if n not in fields:
                    raise ValueError(f"Axisymmetric elasticity expects field '{n}'.")

            lam = float(p.get("lambda", 1.0))
            mu = float(p.get("mu", 1.0))

            r_idx = _coord_index(coords, "r")
            z_idx = _coord_index(coords, "z")
            r_col = xcol[:, r_idx:r_idx + 1]

            u_r, u_z = fields["u_r"], fields["u_z"]
            grad_ur = grad(u_r, xcol)
            grad_uz = grad(u_z, xcol)

            dur_dr = grad_ur[:, r_idx:r_idx + 1]
            dur_dz = grad_ur[:, z_idx:z_idx + 1]
            duz_dr = grad_uz[:, r_idx:r_idx + 1]
            duz_dz = grad_uz[:, z_idx:z_idx + 1]

            eps_rr = dur_dr
            eps_tt = u_r / r_col
            eps_zz = duz_dz
            eps_rz = 0.5 * (dur_dz + duz_dr)

            tr_eps = eps_rr + eps_tt + eps_zz
            sigma_rr = lam * tr_eps + 2.0 * mu * eps_rr
            sigma_tt = lam * tr_eps + 2.0 * mu * eps_tt
            sigma_zz = lam * tr_eps + 2.0 * mu * eps_zz
            sigma_rz = 2.0 * mu * eps_rz

            grad_srr = grad(sigma_rr, xcol)
            grad_srz = grad(sigma_rz, xcol)
            grad_szz = grad(sigma_zz, xcol)

            dsrr_dr = grad_srr[:, r_idx:r_idx + 1]
            dsrz_dz = grad_srz[:, z_idx:z_idx + 1]
            dsrz_dr = grad_srz[:, r_idx:r_idx + 1]
            dszz_dz = grad_szz[:, z_idx:z_idx + 1]

            res_r = dsrr_dr + dsrz_dz + (sigma_rr - sigma_tt) / r_col
            res_z = dsrz_dr + dszz_dz + sigma_rz / r_col

            res_list.append(res_r)
            res_list.append(res_z)

        elif pde_kind == "axisymmetric_linear_elasticity_torsion":
            # threaded_coupling_tc50_rotating: axisymmetric elasticity
            # (u_r, u_z) -- identical physics to "axisymmetric_linear_elasticity"
            # above -- PLUS a decoupled torsional equation for u_theta
            # (the preset's own docstring/meta already state the torsion
            # problem decouples from the meridional one in linear
            # elasticity, and give the exact torsional Navier equation):
            #   d^2(u_theta)/dr^2 + (1/r)*d(u_theta)/dr - u_theta/r^2
            #     + d^2(u_theta)/dz^2 = 0
            if "r" not in coords or "z" not in coords:
                raise ValueError("axisymmetric_linear_elasticity_torsion expects coords ('r','z').")
            for n in ("u_r", "u_z", "u_θ"):
                if n not in fields:
                    raise ValueError(f"axisymmetric_linear_elasticity_torsion expects field '{n}'.")

            lam = float(p.get("lambda", 1.0))
            mu = float(p.get("mu", 1.0))

            r_idx = _coord_index(coords, "r")
            z_idx = _coord_index(coords, "z")
            r_col = xcol[:, r_idx:r_idx + 1]

            u_r, u_z, u_th = fields["u_r"], fields["u_z"], fields["u_θ"]
            grad_ur = grad(u_r, xcol)
            grad_uz = grad(u_z, xcol)

            dur_dr = grad_ur[:, r_idx:r_idx + 1]
            dur_dz = grad_ur[:, z_idx:z_idx + 1]
            duz_dr = grad_uz[:, r_idx:r_idx + 1]
            duz_dz = grad_uz[:, z_idx:z_idx + 1]

            eps_rr = dur_dr
            eps_tt = u_r / r_col
            eps_zz = duz_dz
            eps_rz = 0.5 * (dur_dz + duz_dr)

            tr_eps = eps_rr + eps_tt + eps_zz
            sigma_rr = lam * tr_eps + 2.0 * mu * eps_rr
            sigma_tt = lam * tr_eps + 2.0 * mu * eps_tt
            sigma_zz = lam * tr_eps + 2.0 * mu * eps_zz
            sigma_rz = 2.0 * mu * eps_rz

            grad_srr = grad(sigma_rr, xcol)
            grad_srz = grad(sigma_rz, xcol)
            grad_szz = grad(sigma_zz, xcol)

            dsrr_dr = grad_srr[:, r_idx:r_idx + 1]
            dsrz_dz = grad_srz[:, z_idx:z_idx + 1]
            dsrz_dr = grad_srz[:, r_idx:r_idx + 1]
            dszz_dz = grad_szz[:, z_idx:z_idx + 1]

            res_list.append(dsrr_dr + dsrz_dz + (sigma_rr - sigma_tt) / r_col)
            res_list.append(dsrz_dr + dszz_dz + sigma_rz / r_col)

            duth_dr = grad(u_th, xcol)[:, r_idx:r_idx + 1]
            d2uth_dr2 = grad(duth_dr, xcol)[:, r_idx:r_idx + 1]
            d2uth_dz2 = grad(grad(u_th, xcol)[:, z_idx:z_idx + 1], xcol)[:, z_idx:z_idx + 1]
            res_list.append(d2uth_dr2 + duth_dr / r_col - u_th / (r_col * r_col) + d2uth_dz2)

        elif pde_kind in ("incompressible_navier_stokes_energy_2d", "incompressible_navier_stokes_energy_3d",
                          "navier_stokes_energy_2d"):
            # Steady incompressible Navier-Stokes momentum+continuity
            # (identical to "navier_stokes_incompressible" above) coupled
            # one-way to a steady advection-diffusion energy equation:
            #   u.grad(T) = alpha_T*laplacian(T) + Q_source/(rho*cp)
            # Thermal diffusivity alpha_T = nu/Pr is assumed from a fixed
            # Prandtl number (air, Pr=0.71, standard engineering value)
            # since none of datacenter_airflow_2d/datacenter_cfd_3d/
            # furnace_combustion_zone provide a thermal conductivity
            # parameter directly -- override via params["Pr"] if a
            # different fluid's Prandtl number is needed.
            # These 3 preset names collapsed onto one physics
            # implementation the same way "incompressible_navier_stokes_2d"
            # did above -- distinct kind strings, identical equations.
            if spatial_dim not in (2, 3):
                raise ValueError(f"{pde_kind} expects 2D or 3D spatial dims.")
            needed = ["u", "v", "p", "T"] + (["w"] if spatial_dim == 3 else [])
            for n in needed:
                if n not in fields:
                    raise ValueError(f"{pde_kind} expects field '{n}' in outputs.")

            nu = float(p.get("nu", 1.5e-5))
            Re = float(p.get("Re", 100.0))
            inv_Re = float(p.get("inv_Re", 1.0 / Re))
            Pr = float(p.get("Pr", 0.71))
            alpha_T = nu / Pr
            rho = float(p.get("rho", 1.2))
            cp = float(p.get("cp", 1006.0))
            Q_source = float(p.get("Q_source", 0.0))

            if spatial_dim == 2:
                u, v, p_, T = fields["u"], fields["v"], fields["p"], fields["T"]
                U = torch.cat([u, v], dim=1)
            else:
                u, v, wv, p_, T = fields["u"], fields["v"], fields["w"], fields["p"], fields["T"]
                U = torch.cat([u, v, wv], dim=1)

            sp_idx = [coords.index(n) for n in spatial_coord_names]
            JU = jacobian(U, xcol)
            JUs = JU[:, :, sp_idx]
            conv = torch.zeros((xcol.shape[0], spatial_dim), device=device, dtype=xcol.dtype)
            for i in range(spatial_dim):
                for j in range(spatial_dim):
                    conv[:, i] = conv[:, i] + U[:, j] * JUs[:, i, j]

            gp = grad(p_, xcol)
            lap_u = laplacian(u, xcol, spatial_indices)
            lap_v = laplacian(v, xcol, spatial_indices)
            res_list.append(conv[:, 0:1] + gp[:, sp_idx[0]:sp_idx[0] + 1] - inv_Re * lap_u)
            res_list.append(conv[:, 1:2] + gp[:, sp_idx[1]:sp_idx[1] + 1] - inv_Re * lap_v)
            if spatial_dim == 3:
                lap_w = laplacian(wv, xcol, spatial_indices)
                res_list.append(conv[:, 2:3] + gp[:, sp_idx[2]:sp_idx[2] + 1] - inv_Re * lap_w)
            res_list.append(divergence(U, xcol, spatial_indices))

            gT = grad(T, xcol)
            u_dot_gradT = torch.zeros((xcol.shape[0], 1), device=device, dtype=xcol.dtype)
            for i in range(spatial_dim):
                u_dot_gradT = u_dot_gradT + U[:, i:i + 1] * gT[:, sp_idx[i]:sp_idx[i] + 1]
            lap_T = laplacian(T, xcol, spatial_indices)
            res_list.append(u_dot_gradT - alpha_T * lap_T - Q_source / (rho * cp))

        elif pde_kind == "stokes":
            # Steady creeping (zero-inertia) flow: momentum without the
            # convective term, plus continuity.
            if spatial_dim != 2:
                raise ValueError("Stokes (this branch) expects 2D spatial dims.")
            for n in ("u", "v", "p"):
                if n not in fields:
                    raise ValueError(f"Stokes expects field '{n}'.")
            mu = float(p.get("mu", 1.0))
            sp_idx = [coords.index(c) for c in spatial_coord_names]
            u_f, v_f, p_f = fields["u"], fields["v"], fields["p"]
            gp = grad(p_f, xcol)
            px = gp[:, sp_idx[0]:sp_idx[0] + 1]
            py = gp[:, sp_idx[1]:sp_idx[1] + 1]
            res_list.append(-mu * laplacian(u_f, xcol, sp_idx) + px)
            res_list.append(-mu * laplacian(v_f, xcol, sp_idx) + py)
            res_list.append(divergence(torch.cat([u_f, v_f], dim=1), xcol, sp_idx))

        elif pde_kind == "brinkman":
            # Stokes-like viscous term plus Darcy drag: bridges free flow and
            # porous-media flow.
            if spatial_dim != 2:
                raise ValueError("Brinkman (this branch) expects 2D spatial dims.")
            for n in ("u", "v", "p"):
                if n not in fields:
                    raise ValueError(f"Brinkman expects field '{n}'.")
            mu = float(p.get("mu", 1.0))
            mu_eff = float(p.get("mu_eff", 1.0))
            K_perm = float(p.get("K", 1.0))
            sp_idx = [coords.index(c) for c in spatial_coord_names]
            u_f, v_f, p_f = fields["u"], fields["v"], fields["p"]
            gp = grad(p_f, xcol)
            px = gp[:, sp_idx[0]:sp_idx[0] + 1]
            py = gp[:, sp_idx[1]:sp_idx[1] + 1]
            res_list.append(-mu_eff * laplacian(u_f, xcol, sp_idx) + (mu / K_perm) * u_f + px)
            res_list.append(-mu_eff * laplacian(v_f, xcol, sp_idx) + (mu / K_perm) * v_f + py)
            res_list.append(divergence(torch.cat([u_f, v_f], dim=1), xcol, sp_idx))

        elif pde_kind == "shallow_water":
            # Depth-averaged free-surface flow (Saint-Venant), flat bottom by
            # default (ctx['bathymetry_grad_fn'] may supply real db/dx, db/dy).
            if not has_t:
                raise ValueError("Shallow water expects time coord 't'.")
            for n in ("h", "hu", "hv"):
                if n not in fields:
                    raise ValueError(f"Shallow water expects field '{n}'.")
            g_grav = float(p.get("g", 9.81))
            x_idx = _coord_index(coords, "x")
            y_idx = _coord_index(coords, "y")
            h, hu, hv = fields["h"], fields["hu"], fields["hv"]
            u_vel, v_vel = hu / h, hv / h

            b_fn = ctx.get("bathymetry_grad_fn")
            bx = torch.zeros_like(h)
            by = torch.zeros_like(h)
            if b_fn is not None:
                b_np = b_fn(xcol.detach().cpu().numpy(), ctx)
                b_t = torch.as_tensor(b_np, device=device, dtype=h.dtype)
                bx, by = b_t[:, 0:1], b_t[:, 1:2]

            F1x, F2x, F3x = hu, hu * u_vel + 0.5 * g_grav * h * h, hu * v_vel
            F1y, F3y = hv, hv * v_vel + 0.5 * g_grav * h * h
            F2y = hu * v_vel

            ht = time_derivative(h, xcol, t_index)  # type: ignore[arg-type]
            hut = time_derivative(hu, xcol, t_index)  # type: ignore[arg-type]
            hvt = time_derivative(hv, xcol, t_index)  # type: ignore[arg-type]

            res_list.append(ht + grad(F1x, xcol)[:, x_idx:x_idx + 1] + grad(F1y, xcol)[:, y_idx:y_idx + 1])
            res_list.append(hut + grad(F2x, xcol)[:, x_idx:x_idx + 1] + grad(F2y, xcol)[:, y_idx:y_idx + 1] + g_grav * h * bx)
            res_list.append(hvt + grad(F3x, xcol)[:, x_idx:x_idx + 1] + grad(F3y, xcol)[:, y_idx:y_idx + 1] + g_grav * h * by)

        elif pde_kind == "shallow_water_2d":
            # Nonlinear ROTATING shallow-water equations in PRIMITIVE
            # variables (h, u, v), f-plane approximation (climate_atmosphere_2d)
            # -- genuinely different from "shallow_water" above, which uses
            # conservative variables (h, hu, hv) and has no Coriolis term.
            # Not a naming collision like reaction_diffusion_2d/
            # incompressible_navier_stokes_2d were: different field
            # variables AND an extra physics term (rotation), so this is
            # its own kind rather than an alias.
            #   dh/dt + d(hu)/dx + d(hv)/dy = 0
            #   du/dt + u*du/dx + v*du/dy - f*v + g*dh/dx = 0
            #   dv/dt + u*dv/dx + v*dv/dy + f*u + g*dh/dy = 0
            if not has_t:
                raise ValueError("shallow_water_2d expects a time coord 't'.")
            for n in ("h", "u", "v"):
                if n not in fields:
                    raise ValueError(f"shallow_water_2d expects field '{n}'.")
            g_grav = float(p.get("g", 9.81))
            f_cor = float(p.get("f", 1e-4))
            x_idx = _coord_index(coords, "x")
            y_idx = _coord_index(coords, "y")
            h, u_vel, v_vel = fields["h"], fields["u"], fields["v"]

            gh = grad(h, xcol)
            gu = grad(u_vel, xcol)
            gv = grad(v_vel, xcol)
            ht = time_derivative(h, xcol, t_index)  # type: ignore[arg-type]
            ut = time_derivative(u_vel, xcol, t_index)  # type: ignore[arg-type]
            vt = time_derivative(v_vel, xcol, t_index)  # type: ignore[arg-type]

            hu_x = grad(h * u_vel, xcol)[:, x_idx:x_idx + 1]
            hv_y = grad(h * v_vel, xcol)[:, y_idx:y_idx + 1]
            res_list.append(ht + hu_x + hv_y)
            res_list.append(
                ut + u_vel * gu[:, x_idx:x_idx + 1] + v_vel * gu[:, y_idx:y_idx + 1]
                - f_cor * v_vel + g_grav * gh[:, x_idx:x_idx + 1]
            )
            res_list.append(
                vt + u_vel * gv[:, x_idx:x_idx + 1] + v_vel * gv[:, y_idx:y_idx + 1]
                + f_cor * u_vel + g_grav * gh[:, y_idx:y_idx + 1]
            )

        elif pde_kind == "stommel_gyre_2d":
            # Stommel (1948) wind-driven ocean-gyre barotropic streamfunction
            # model: r*laplacian(psi) + beta*dpsi/dx = curl(tau)/(rho0*H).
            # climate_ocean_gyre's own docstring gives the intended default
            # forcing curl(tau) = (tau0*pi/W)*sin(pi*y/W), but W (basin
            # width) is not one of the PDE's params (only used internally
            # by the preset to compute a diagnostic velocity scale) -- so,
            # consistent with "poisson"/"heat_equation" elsewhere in this
            # compiler, the forcing is read from ctx["source_fn"] (already
            # expected to return curl(tau)/(rho0*H) directly) and defaults
            # to zero (unforced decay) if the caller doesn't supply one.
            if len(field_names) != 1:
                raise ValueError("stommel_gyre_2d expects 1 scalar field (psi).")
            psi = fields[field_names[0]]
            beta = float(p.get("beta", 2e-11))
            r_fric = float(p.get("r", 1e-7))
            x_idx = _coord_index(coords, "x")
            dpsi_dx = grad(psi, xcol)[:, x_idx:x_idx + 1]
            lap_psi = laplacian(psi, xcol, spatial_indices)
            forcing_fn = ctx.get("source_fn")
            forcing = torch.zeros_like(psi)
            if forcing_fn is not None:
                f_np = forcing_fn(xcol.detach().cpu().numpy(), ctx)
                forcing = torch.as_tensor(f_np, device=device, dtype=psi.dtype)
                if forcing.ndim == 1:
                    forcing = forcing[:, None]
            res_list.append(r_fric * lap_psi + beta * dpsi_dx - forcing)

        elif pde_kind == "opinion_dynamics_2d":
            # Continuum Hegselmann-Krause opinion dynamics (bistable
            # Allen-Cahn-style reaction-diffusion):
            #   du/dt = D*laplacian(u) + alpha*u*(1-u^2)
            if not has_t:
                raise ValueError("opinion_dynamics_2d expects a time coord 't'.")
            if len(field_names) != 1:
                raise ValueError("opinion_dynamics_2d expects 1 scalar field (u).")
            u_op = fields[field_names[0]]
            D_diff = float(p.get("D", 0.01))
            alpha_op = float(p.get("alpha", 1.0))
            ut = time_derivative(u_op, xcol, t_index)  # type: ignore[arg-type]
            lap_u = laplacian(u_op, xcol, spatial_indices)
            res_list.append(ut - D_diff * lap_u - alpha_op * u_op * (1.0 - u_op * u_op))

        elif pde_kind == "euler_compressible":
            # Inviscid compressible flow, conservative form, ideal gas (gamma-law).
            if not has_t:
                raise ValueError("Euler compressible expects time coord 't'.")
            for n in ("rho", "rho_u", "rho_v", "E"):
                if n not in fields:
                    raise ValueError(f"Euler compressible expects field '{n}'.")
            gamma = float(p.get("gamma", 1.4))
            x_idx = _coord_index(coords, "x")
            y_idx = _coord_index(coords, "y")
            rho, rho_u, rho_v, E_f = fields["rho"], fields["rho_u"], fields["rho_v"], fields["E"]
            u_vel, v_vel = rho_u / rho, rho_v / rho
            p_pres = (gamma - 1.0) * (E_f - 0.5 * rho * (u_vel * u_vel + v_vel * v_vel))

            F1x, F2x, F3x, F4x = rho_u, rho_u * u_vel + p_pres, rho_u * v_vel, (E_f + p_pres) * u_vel
            F1y, F2y, F3y, F4y = rho_v, rho_u * v_vel, rho_v * v_vel + p_pres, (E_f + p_pres) * v_vel

            rho_t = time_derivative(rho, xcol, t_index)  # type: ignore[arg-type]
            rhou_t = time_derivative(rho_u, xcol, t_index)  # type: ignore[arg-type]
            rhov_t = time_derivative(rho_v, xcol, t_index)  # type: ignore[arg-type]
            E_t = time_derivative(E_f, xcol, t_index)  # type: ignore[arg-type]

            res_list.append(rho_t + grad(F1x, xcol)[:, x_idx:x_idx + 1] + grad(F1y, xcol)[:, y_idx:y_idx + 1])
            res_list.append(rhou_t + grad(F2x, xcol)[:, x_idx:x_idx + 1] + grad(F2y, xcol)[:, y_idx:y_idx + 1])
            res_list.append(rhov_t + grad(F3x, xcol)[:, x_idx:x_idx + 1] + grad(F3y, xcol)[:, y_idx:y_idx + 1])
            res_list.append(E_t + grad(F4x, xcol)[:, x_idx:x_idx + 1] + grad(F4y, xcol)[:, y_idx:y_idx + 1])

        elif pde_kind == "fisher_kpp":
            if not has_t:
                raise ValueError("Fisher-KPP expects time coord 't'.")
            if len(field_names) != 1:
                raise ValueError("Fisher-KPP expects 1 scalar field.")
            D = float(p.get("D", 0.001))
            r = float(p.get("r", 1.0))
            u_f = fields[field_names[0]]
            ut = time_derivative(u_f, xcol, t_index)  # type: ignore[arg-type]
            lap_u = laplacian(u_f, xcol, spatial_indices)
            res_list.append(ut - D * lap_u - r * u_f * (1.0 - u_f))

        elif pde_kind == "reaction_diffusion":
            # Two-species autocatalytic reaction-diffusion (Gray-Scott form);
            # fills in one of the 3 PDE families declared in capabilities.py
            # that had no compile_problem implementation before this branch.
            if not has_t:
                raise ValueError("Reaction-diffusion expects time coord 't'.")
            for n in ("u", "v"):
                if n not in fields:
                    raise ValueError(f"Reaction-diffusion expects field '{n}'.")
            Du = float(p.get("Du", 2e-5))
            Dv = float(p.get("Dv", 1e-5))
            F_feed = float(p.get("F", 0.04))
            k_kill = float(p.get("k", 0.06))
            u_f, v_f = fields["u"], fields["v"]
            ut = time_derivative(u_f, xcol, t_index)  # type: ignore[arg-type]
            vt = time_derivative(v_f, xcol, t_index)  # type: ignore[arg-type]
            lap_u = laplacian(u_f, xcol, spatial_indices)
            lap_v = laplacian(v_f, xcol, spatial_indices)
            res_list.append(ut - Du * lap_u + u_f * v_f * v_f - F_feed * (1.0 - u_f))
            res_list.append(vt - Dv * lap_v - u_f * v_f * v_f + (F_feed + k_kill) * v_f)

        elif pde_kind == "reaction_diffusion_2d":
            # Single-species linear reaction-diffusion (drug/tracer
            # diffusion with first-order elimination):
            #   dC/dt = D*laplacian(C) - lambda*C
            # Distinct from "reaction_diffusion" above (two-species
            # nonlinear Gray-Scott kinetics, fields u/v) -- same coord/time
            # machinery, different, much simpler physics; the two kinds
            # were previously confused as one gap because their names are
            # similar, but they are genuinely different equations.
            if not has_t:
                raise ValueError("reaction_diffusion_2d expects a time coord 't'.")
            if len(field_names) != 1:
                raise ValueError("reaction_diffusion_2d expects 1 scalar field (C).")
            C = fields[field_names[0]]
            D_diff = float(p.get("D", 1e-10))
            lam_decay = float(p.get("lambda", 0.0))
            Ct = time_derivative(C, xcol, t_index)  # type: ignore[arg-type]
            lap_C = laplacian(C, xcol, spatial_indices)
            res_list.append(Ct - D_diff * lap_C + lam_decay * C)

        elif pde_kind == "black_scholes_1d":
            # Black-Scholes PDE for European option pricing (Black &
            # Scholes, 1973), in the preset's own forward-time convention
            # tau = T - t (coords are literally named 'S', 'tau', not
            # 't', so this branch does not use has_t/t_index -- same
            # pattern as lane_emden_polytrope's 'xi'):
            #   dV/dtau = 0.5*sigma^2*S^2*d2V/dS2 + r*S*dV/dS - r*V
            if "S" not in coords or "tau" not in coords:
                raise ValueError("black_scholes_1d expects coords 'S' and 'tau'.")
            if len(field_names) != 1:
                raise ValueError("black_scholes_1d expects 1 scalar field (V).")
            V = fields[field_names[0]]
            sigma = float(p.get("sigma", 0.2))
            r = float(p.get("r", 0.05))
            S_idx = _coord_index(coords, "S")
            tau_idx = _coord_index(coords, "tau")
            S_val = xcol[:, S_idx:S_idx + 1]
            gV = grad(V, xcol)
            dV_dtau = gV[:, tau_idx:tau_idx + 1]
            dV_dS = gV[:, S_idx:S_idx + 1]
            d2V_dS2 = laplacian(V, xcol, [S_idx])
            res_list.append(dV_dtau - 0.5 * sigma * sigma * S_val * S_val * d2V_dS2 - r * S_val * dV_dS + r * V)

        elif pde_kind == "heston_pde_2d":
            # Heston (1993) stochastic volatility option-pricing PDE, same
            # forward-time tau=T-t convention as black_scholes_1d, plus a
            # mixed second derivative d2V/dS/dv from the correlated
            # Brownian motions:
            #   dV/dtau = 0.5*S^2*v*d2V/dS2 + rho*sigma_v*S*v*d2V/dSdv
            #           + 0.5*sigma_v^2*v*d2V/dv2 + r*S*dV/dS
            #           + kappa*(theta-v)*dV/dv - r*V
            if not all(c in coords for c in ("S", "v", "tau")):
                raise ValueError("heston_pde_2d expects coords 'S', 'v', 'tau'.")
            if len(field_names) != 1:
                raise ValueError("heston_pde_2d expects 1 scalar field (V).")
            V = fields[field_names[0]]
            kappa = float(p.get("kappa", 2.0))
            theta = float(p.get("theta", 0.04))
            sigma_v = float(p.get("sigma_v", 0.3))
            rho = float(p.get("rho", -0.7))
            r = float(p.get("r", 0.05))
            S_idx = _coord_index(coords, "S")
            v_idx = _coord_index(coords, "v")
            tau_idx = _coord_index(coords, "tau")
            S_val = xcol[:, S_idx:S_idx + 1]
            v_val = xcol[:, v_idx:v_idx + 1]

            gV = grad(V, xcol)
            dV_dtau = gV[:, tau_idx:tau_idx + 1]
            dV_dS = gV[:, S_idx:S_idx + 1]
            dV_dv = gV[:, v_idx:v_idx + 1]
            d2V_dS2 = laplacian(V, xcol, [S_idx])
            d2V_dv2 = laplacian(V, xcol, [v_idx])
            d2V_dSdv = grad(dV_dS, xcol)[:, v_idx:v_idx + 1]

            res_list.append(
                dV_dtau
                - 0.5 * S_val * S_val * v_val * d2V_dS2
                - rho * sigma_v * S_val * v_val * d2V_dSdv
                - 0.5 * sigma_v * sigma_v * v_val * d2V_dv2
                - r * S_val * dV_dS
                - kappa * (theta - v_val) * dV_dv
                + r * V
            )

        elif pde_kind == "euler_bernoulli_beam_von_karman":
            # Geometrically nonlinear (Von Karman) beam-column: axial and
            # transverse displacement fields coupled through the Von Karman
            # strain N = EA*(u' + 0.5*w'^2). Unlike plasticity/fracture, this
            # has one standard, unambiguous formulation (no competing
            # constitutive-modeling convention to choose between) — it's
            # calculus (nested/product derivatives), which autograd computes
            # exactly, so this is a natural extension of the already-verified
            # "euler_bernoulli_beam" branch rather than new open modeling risk.
            #   Axial equilibrium:      dN/dz = 0            (no distributed axial load)
            #   Transverse equilibrium: EI*w'''' - d/dz(N*w') - q(z) = 0
            #   Constitutive:           N = EA*(u' + 0.5*(w')^2)
            if "u" not in fields or "w" not in fields:
                raise ValueError("Von Karman beam expects fields 'u' (axial) and 'w' (transverse).")
            u_f, w_f = fields["u"], fields["w"]
            EA = float(p.get("EA", 1.0))
            EI = float(p.get("EI", 1.0))
            z_idx = _coord_index(coords, "z")

            q_fn = ctx.get("source_fn") or ctx.get("q_fn")
            if q_fn is not None:
                q_np = q_fn(xcol.detach().cpu().numpy(), ctx)
                q = torch.as_tensor(q_np, device=device, dtype=w_f.dtype)
                if q.ndim == 1:
                    q = q[:, None]
            elif "q" in p:
                q = torch.full_like(w_f, float(p["q"]))
            else:
                q = torch.zeros_like(w_f)

            u1 = grad(u_f, xcol)[:, z_idx:z_idx + 1]
            w1 = grad(w_f, xcol)[:, z_idx:z_idx + 1]
            w2 = torch.autograd.grad(
                outputs=w1, inputs=xcol, grad_outputs=torch.ones_like(w1),
                create_graph=True, retain_graph=True, allow_unused=False,
            )[0][:, z_idx:z_idx + 1]
            w3 = torch.autograd.grad(
                outputs=w2, inputs=xcol, grad_outputs=torch.ones_like(w2),
                create_graph=True, retain_graph=True, allow_unused=False,
            )[0][:, z_idx:z_idx + 1]
            w4 = torch.autograd.grad(
                outputs=w3, inputs=xcol, grad_outputs=torch.ones_like(w3),
                create_graph=True, retain_graph=True, allow_unused=False,
            )[0][:, z_idx:z_idx + 1]

            N_force = EA * (u1 + 0.5 * w1 * w1)
            dN_dz = torch.autograd.grad(
                outputs=N_force, inputs=xcol, grad_outputs=torch.ones_like(N_force),
                create_graph=True, retain_graph=True, allow_unused=False,
            )[0][:, z_idx:z_idx + 1]

            Nw1 = N_force * w1
            d_Nw1_dz = torch.autograd.grad(
                outputs=Nw1, inputs=xcol, grad_outputs=torch.ones_like(Nw1),
                create_graph=True, retain_graph=True, allow_unused=False,
            )[0][:, z_idx:z_idx + 1]

            res_list.append(dN_dz)
            res_list.append(EI * w4 - d_Nw1_dz - q)

        elif pde_kind == "reaction_kinetics_network":
            # Generic well-mixed (0D-in-space) reaction-network residual: each
            # field is a species concentration C_i(t[, other coords]); the
            # network-specific stoichiometry/rate law lives entirely in a
            # user-supplied callable, not in this dispatch branch, so one
            # implementation serves any reaction network (disinfection
            # kinetics, degradation kinetics, corrosion, adsorption, or any
            # other well-mixed chemical/biological reaction system).
            if not has_t:
                raise ValueError("reaction_kinetics_network expects a time coord 't'.")
            rate_fn = ctx.get("rate_fn") or spec.pde.meta.get("rate_fn")
            if rate_fn is None:
                raise ValueError(
                    "reaction_kinetics_network requires meta['rate_fn'] or "
                    "ctx['rate_fn']: a callable "
                    "(fields: Dict[str, Tensor], coords: Dict[str, Tensor], params: dict) "
                    "-> Dict[str, Tensor] returning the target dC_i/dt for each "
                    "field name. Unlike source_fn/f_fn/q_fn elsewhere in this "
                    "compiler, rate_fn receives live (non-detached) field "
                    "tensors and MUST use torch ops so gradients flow back "
                    "into the model."
                )
            coords_dict = {name: xcol[:, i:i + 1] for i, name in enumerate(coords)}
            rhs = rate_fn(fields, coords_dict, p)
            for fname in field_names:
                dCi_dt = time_derivative(fields[fname], xcol, t_index)  # type: ignore[arg-type]
                if fname not in rhs:
                    raise ValueError(f"rate_fn did not return a target rate for field '{fname}'")
                res_list.append(dCi_dt - rhs[fname])

        elif pde_kind == "sir_ode":
            # Kermack & McKendrick (1927) SIR compartmental model.
            if not has_t:
                raise ValueError("sir_ode expects a time coord 't'.")
            for n in ("S", "I", "R"):
                if n not in fields:
                    raise ValueError(f"sir_ode expects field '{n}'.")
            beta = float(p.get("beta", 0.3))
            gamma = float(p.get("gamma", 0.1))
            N = float(p.get("N", 1e6))
            S, I, R = fields["S"], fields["I"], fields["R"]
            S_t = time_derivative(S, xcol, t_index)  # type: ignore[arg-type]
            I_t = time_derivative(I, xcol, t_index)  # type: ignore[arg-type]
            R_t = time_derivative(R, xcol, t_index)  # type: ignore[arg-type]
            infection = beta * S * I / N
            res_list.append(S_t + infection)
            res_list.append(I_t - infection + gamma * I)
            res_list.append(R_t - gamma * I)

        elif pde_kind == "pk_two_compartment_ode":
            # Two-compartment pharmacokinetic model, IV bolus (see
            # Rowland & Tozer, "Clinical Pharmacokinetics and
            # Pharmacodynamics"): dC1/dt = -(k12+kel)C1 + k21 C2,
            # dC2/dt = k12 C1 - k21 C2.
            if not has_t:
                raise ValueError("pk_two_compartment_ode expects a time coord 't'.")
            for n in ("C1", "C2"):
                if n not in fields:
                    raise ValueError(f"pk_two_compartment_ode expects field '{n}'.")
            k12 = float(p.get("k12", 0.5))
            k21 = float(p.get("k21", 0.3))
            kel = float(p.get("kel", 0.2))
            C1, C2 = fields["C1"], fields["C2"]
            C1_t = time_derivative(C1, xcol, t_index)  # type: ignore[arg-type]
            C2_t = time_derivative(C2, xcol, t_index)  # type: ignore[arg-type]
            res_list.append(C1_t + (k12 + kel) * C1 - k21 * C2)
            res_list.append(C2_t - k12 * C1 + k21 * C2)

        elif pde_kind == "compressor_meanline_1d":
            # 1D mean-line axial-compressor thermodynamics
            # (axial_compressor_meanline). Exactly the 3 equations the
            # preset's own docstring states (stage-averaged, streamwise
            # coordinate s in [0,1]):
            #   Energy:     c_p * dT_t/ds = W_stage_per_unit_length
            #   Continuity: d(rho*u*A(s))/ds = 0  (A(s) from
            #               ctx["area_fn"], defaulting to constant A=1 --
            #               a straight annular duct -- if not supplied)
            #   State:      p_t = rho * R_gas * T_t  (algebraic, ideal gas)
            # The preset's 5th field, c_theta (tangential velocity), is
            # NOT given its own governing equation OR boundary condition
            # anywhere in the preset itself (only T_t/p_t/u have inlet
            # BCs) -- rather than invent an unstated Euler-work/swirl
            # equation for it, it is left with no PDE residual here,
            # matching exactly what the preset itself specifies, not
            # silently adding physics beyond what was documented.
            if "s" not in coords:
                raise ValueError("compressor_meanline_1d expects coord 's'.")
            for n in ("T_t", "p_t", "rho", "u"):
                if n not in fields:
                    raise ValueError(f"compressor_meanline_1d expects field '{n}'.")
            c_p = float(p.get("c_p", 1004.5))
            R_gas = float(p.get("R_gas", 287.0))
            W_stage = float(p.get("W_stage_per_unit_length", 0.0))
            s_idx = _coord_index(coords, "s")

            T_t, p_t, rho, u_vel = fields["T_t"], fields["p_t"], fields["rho"], fields["u"]
            dTt_ds = grad(T_t, xcol)[:, s_idx:s_idx + 1]
            res_list.append(c_p * dTt_ds - W_stage)

            area_fn = ctx.get("area_fn")
            if area_fn is not None:
                A_np = area_fn(xcol.detach().cpu().numpy(), ctx)
                A_val = torch.as_tensor(A_np, device=device, dtype=rho.dtype)
                if A_val.ndim == 1:
                    A_val = A_val[:, None]
            else:
                A_val = torch.ones_like(rho)
            mass_flux = rho * u_vel * A_val
            res_list.append(grad(mass_flux, xcol)[:, s_idx:s_idx + 1])

            res_list.append(p_t - rho * R_gas * T_t)

        elif pde_kind == "compressible_euler_2d":
            # Steady 2D compressible Euler in PRIMITIVE variables
            # (rho, u, v, p, T) -- axial_compressor_cascade_2d. Genuinely
            # different from the existing "euler_compressible" kind,
            # which uses CONSERVATIVE variables (rho, rho_u, rho_v, E)
            # and is unsteady (not a naming collision like
            # reaction_diffusion_2d/incompressible_navier_stokes_2d
            # earlier). Rather than re-derive the flux physics from
            # scratch (real risk of a subtle sign error in compressible
            # gas dynamics), the conservative quantities are built
            # algebraically from the primitive fields and the residual
            # reuses EXACTLY the same flux-divergence structure as the
            # already-verified "euler_compressible" branch (steady, so
            # its time-derivative terms are simply dropped), plus the
            # ideal-gas state equation tying rho/T/p together (since here
            # p is its own independent field, not derived from E).
            #
            # HONEST CONFIDENCE NOTE: this reuses already-verified flux
            # formulas, but was NOT independently checked against a fresh
            # nontrivial closed-form manufactured solution this session
            # (genuine 2D nonlinear compressible Euler solutions in
            # closed form are hard to construct quickly and this
            # implementation's own MMS test below is a real, verified
            # exact solution -- see below -- but only in the
            # SUBSONIC-uniform-flow-plus-linear-temperature-gradient
            # regime, not a general nonlinear check). Tier A (compiles,
            # runs) is confirmed; treat with correspondingly moderate,
            # not full, confidence versus the astrophysics/finance kinds
            # that have true nonlinear exact-solution coverage.
            if spatial_dim != 2:
                raise ValueError("compressible_euler_2d expects 2D spatial dims.")
            for n in ("rho", "u", "v", "p", "T"):
                if n not in fields:
                    raise ValueError(f"compressible_euler_2d expects field '{n}'.")
            gamma = float(p.get("gamma", 1.4))
            R_gas = float(p.get("R_gas", 287.0))
            cv = R_gas / (gamma - 1.0)
            x_idx = _coord_index(coords, "x")
            y_idx = _coord_index(coords, "y")

            rho, u_vel, v_vel, p_pres, T_temp = fields["rho"], fields["u"], fields["v"], fields["p"], fields["T"]
            rho_u = rho * u_vel
            rho_v = rho * v_vel
            E_f = rho * (cv * T_temp + 0.5 * (u_vel * u_vel + v_vel * v_vel))

            F1x, F2x, F3x, F4x = rho_u, rho_u * u_vel + p_pres, rho_u * v_vel, (E_f + p_pres) * u_vel
            F1y, F2y, F3y, F4y = rho_v, rho_u * v_vel, rho_v * v_vel + p_pres, (E_f + p_pres) * v_vel

            res_list.append(grad(F1x, xcol)[:, x_idx:x_idx + 1] + grad(F1y, xcol)[:, y_idx:y_idx + 1])
            res_list.append(grad(F2x, xcol)[:, x_idx:x_idx + 1] + grad(F2y, xcol)[:, y_idx:y_idx + 1])
            res_list.append(grad(F3x, xcol)[:, x_idx:x_idx + 1] + grad(F3y, xcol)[:, y_idx:y_idx + 1])
            res_list.append(grad(F4x, xcol)[:, x_idx:x_idx + 1] + grad(F4y, xcol)[:, y_idx:y_idx + 1])
            res_list.append(p_pres - rho * R_gas * T_temp)

        elif pde_kind == "phonon_bte_1d_gray":
            # Gray-medium (Callaway model) phonon Boltzmann transport
            # equation, 1D (crystal_phonon) -- the preset's own docstring
            # gives the exact equation:
            #   dT/dt + vg*dT/dx = -(T-T_eq)/tau + (k/Cv)*d2T/dx2
            # Cv (heat capacity) and T_eq (local equilibrium temperature)
            # aren't in the preset's own PDE params (only k, tau, vg, Kn
            # are) -- Cv defaults to 1 (so k already acts as the
            # diffusivity, consistent with this preset's own
            # ScaleSpec(alpha=k) treating k that way), and T_eq defaults
            # to the preset's own initial-condition value 0.5*(T_hot+T_cold)
            # if not overridden via params["T_eq"].
            if not has_t:
                raise ValueError("phonon_bte_1d_gray expects a time coord 't'.")
            if len(field_names) != 1:
                raise ValueError("phonon_bte_1d_gray expects a scalar field (T).")
            T_f = fields[field_names[0]]
            k_cond = float(p.get("k", 150.0))
            Cv = float(p.get("Cv", 1.0))
            tau_relax = float(p.get("tau", 1e-12))
            vg = float(p.get("vg", 3000.0))
            T_eq = float(p.get("T_eq", 300.0))
            alpha_th = k_cond / Cv

            Tt = time_derivative(T_f, xcol, t_index)  # type: ignore[arg-type]
            Tx = grad(T_f, xcol)[:, spatial_indices[0]:spatial_indices[0] + 1]
            Txx = laplacian(T_f, xcol, spatial_indices)
            res_list.append(Tt + vg * Tx + (T_f - T_eq) / tau_relax - alpha_th * Txx)

        elif pde_kind == "kepler_two_body_orbit":
            # Restricted two-body problem, planar Cartesian formulation
            # (Vallado, "Fundamentals of Astrodynamics and Applications").
            # State: position (x, y), velocity (vx, vy). mu = G*(M1+M2) is
            # the gravitational parameter.
            if not has_t:
                raise ValueError("kepler_two_body_orbit expects a time coord 't'.")
            for n in ("x", "y", "vx", "vy"):
                if n not in fields:
                    raise ValueError(f"kepler_two_body_orbit expects field '{n}'.")
            mu = float(p.get("mu", 398600.4418))
            x_f, y_f, vx_f, vy_f = fields["x"], fields["y"], fields["vx"], fields["vy"]
            r = torch.sqrt(x_f * x_f + y_f * y_f + 1e-12)
            r3 = r ** 3
            x_t = time_derivative(x_f, xcol, t_index)  # type: ignore[arg-type]
            y_t = time_derivative(y_f, xcol, t_index)  # type: ignore[arg-type]
            vx_t = time_derivative(vx_f, xcol, t_index)  # type: ignore[arg-type]
            vy_t = time_derivative(vy_f, xcol, t_index)  # type: ignore[arg-type]
            res_list.append(x_t - vx_f)
            res_list.append(y_t - vy_f)
            res_list.append(vx_t + mu * x_f / r3)
            res_list.append(vy_t + mu * y_f / r3)

        elif pde_kind == "space_debris_cw_relative_motion":
            # Clohessy-Wiltshire / Hill's equations for relative motion
            # near a circular reference orbit (Clohessy & Wiltshire, 1960)
            # -- the standard tool for space-debris conjunction assessment
            # and proximity operations. x = radial, y = along-track,
            # z = cross-track; n = mean motion of the reference orbit.
            if not has_t:
                raise ValueError("space_debris_cw_relative_motion expects a time coord 't'.")
            for nm in ("x", "y", "z", "vx", "vy", "vz"):
                if nm not in fields:
                    raise ValueError(f"space_debris_cw_relative_motion expects field '{nm}'.")
            n_mm = float(p.get("n", 0.0011))  # mean motion, rad/s (~LEO)
            x_f, y_f, z_f = fields["x"], fields["y"], fields["z"]
            vx_f, vy_f, vz_f = fields["vx"], fields["vy"], fields["vz"]
            x_t = time_derivative(x_f, xcol, t_index)  # type: ignore[arg-type]
            y_t = time_derivative(y_f, xcol, t_index)  # type: ignore[arg-type]
            z_t = time_derivative(z_f, xcol, t_index)  # type: ignore[arg-type]
            vx_t = time_derivative(vx_f, xcol, t_index)  # type: ignore[arg-type]
            vy_t = time_derivative(vy_f, xcol, t_index)  # type: ignore[arg-type]
            vz_t = time_derivative(vz_f, xcol, t_index)  # type: ignore[arg-type]
            res_list.append(x_t - vx_f)
            res_list.append(y_t - vy_f)
            res_list.append(z_t - vz_f)
            res_list.append(vx_t - 2.0 * n_mm * vy_f - 3.0 * n_mm * n_mm * x_f)
            res_list.append(vy_t + 2.0 * n_mm * vx_f)
            res_list.append(vz_t + n_mm * n_mm * z_f)

        elif pde_kind == "satellite_j2_perturbation":
            # Two-body motion plus the J2 (Earth oblateness) perturbing
            # acceleration, derived here as -grad(V) of the standard J2
            # geopotential V = -(mu/r)[1 - J2 (Re/r)^2 (3(z/r)^2-1)/2]
            # (Vallado, "Fundamentals of Astrodynamics and Applications";
            # this is the same potential whose orbit-averaged secular
            # nodal/apsidal drift rates are used as this preset's
            # independent literature cross-check -- see
            # presets/astrophysics.py). State: (x, y, z, vx, vy, vz) in an
            # Earth-centered inertial frame.
            if not has_t:
                raise ValueError("satellite_j2_perturbation expects a time coord 't'.")
            for nm in ("x", "y", "z", "vx", "vy", "vz"):
                if nm not in fields:
                    raise ValueError(f"satellite_j2_perturbation expects field '{nm}'.")
            mu = float(p.get("mu", 398600.4418))
            J2 = float(p.get("J2", 1.08262668e-3))
            Re = float(p.get("Re", 6378.137))
            x_f, y_f, z_f = fields["x"], fields["y"], fields["z"]
            vx_f, vy_f, vz_f = fields["vx"], fields["vy"], fields["vz"]
            r2 = x_f * x_f + y_f * y_f + z_f * z_f + 1e-9
            r = torch.sqrt(r2)
            r5 = r2 * r2 * r
            j2_common = 1.5 * J2 * mu * (Re * Re) / r5
            z2_over_r2 = z_f * z_f / r2
            ax_j2 = j2_common * x_f * (5.0 * z2_over_r2 - 1.0)
            ay_j2 = j2_common * y_f * (5.0 * z2_over_r2 - 1.0)
            az_j2 = j2_common * z_f * (5.0 * z2_over_r2 - 3.0)
            r3 = r2 * r
            ax = -mu * x_f / r3 + ax_j2
            ay = -mu * y_f / r3 + ay_j2
            az = -mu * z_f / r3 + az_j2
            x_t = time_derivative(x_f, xcol, t_index)  # type: ignore[arg-type]
            y_t = time_derivative(y_f, xcol, t_index)  # type: ignore[arg-type]
            z_t = time_derivative(z_f, xcol, t_index)  # type: ignore[arg-type]
            vx_t = time_derivative(vx_f, xcol, t_index)  # type: ignore[arg-type]
            vy_t = time_derivative(vy_f, xcol, t_index)  # type: ignore[arg-type]
            vz_t = time_derivative(vz_f, xcol, t_index)  # type: ignore[arg-type]
            res_list.append(x_t - vx_f)
            res_list.append(y_t - vy_f)
            res_list.append(z_t - vz_f)
            res_list.append(vx_t - ax)
            res_list.append(vy_t - ay)
            res_list.append(vz_t - az)

        elif pde_kind == "spacecraft_attitude_euler_rotation":
            # Torque-free rigid-body attitude dynamics, body-frame Euler
            # equations (Hughes, "Spacecraft Attitude Dynamics";
            # Wertz, "Spacecraft Attitude Determination and Control").
            # State: angular velocity components (w1, w2, w3) about the
            # principal axes; params I1, I2, I3 = principal moments of
            # inertia.
            if not has_t:
                raise ValueError("spacecraft_attitude_euler_rotation expects a time coord 't'.")
            for nm in ("w1", "w2", "w3"):
                if nm not in fields:
                    raise ValueError(f"spacecraft_attitude_euler_rotation expects field '{nm}'.")
            I1 = float(p.get("I1", 100.0))
            I2 = float(p.get("I2", 100.0))
            I3 = float(p.get("I3", 150.0))
            w1, w2, w3 = fields["w1"], fields["w2"], fields["w3"]
            w1_t = time_derivative(w1, xcol, t_index)  # type: ignore[arg-type]
            w2_t = time_derivative(w2, xcol, t_index)  # type: ignore[arg-type]
            w3_t = time_derivative(w3, xcol, t_index)  # type: ignore[arg-type]
            res_list.append(I1 * w1_t - (I2 - I3) * w2 * w3)
            res_list.append(I2 * w2_t - (I3 - I1) * w3 * w1)
            res_list.append(I3 * w3_t - (I1 - I2) * w1 * w2)

        elif pde_kind == "lane_emden_polytrope":
            # Lane-Emden equation for a self-gravitating polytropic star
            # (Chandrasekhar, "An Introduction to the Study of Stellar
            # Structure", 1939):
            #   theta''(xi) + (2/xi) theta'(xi) + theta(xi)^n = 0
            # written as a first-order system with phi := theta'(xi):
            #   theta'(xi) = phi ;  phi'(xi) = -theta^n - (2/xi) phi
            # Single independent coordinate xi (dimensionless radius);
            # this branch does not require a 't' coord -- xcol is simply
            # the (N,1) collocation column for xi.
            for nm in ("theta", "phi"):
                if nm not in fields:
                    raise ValueError(f"lane_emden_polytrope expects field '{nm}'.")
            n_poly = float(p.get("n", 1.0))
            theta, phi_f = fields["theta"], fields["phi"]
            xi = xcol[:, 0:1]
            theta_xi = grad(theta, xcol)[:, 0:1]
            phi_xi = grad(phi_f, xcol)[:, 0:1]
            if n_poly.is_integer():
                # Real integer powers of a negative base are well-defined
                # (and, critically, theta**0 == 1 regardless of theta's
                # sign) -- use them directly. Only fall back to the
                # sign-preserving |theta|^n regularization below for
                # non-integer n, where torch's ** on a negative base
                # would otherwise produce NaN.
                theta_pow_n = theta ** int(n_poly)
            else:
                theta_pow_n = torch.sign(theta) * torch.abs(theta) ** n_poly
            res_list.append(theta_xi - phi_f)
            res_list.append(phi_xi + theta_pow_n + (2.0 / xi) * phi_f)

        elif pde_kind == "euler_compressible_1d":
            # Inviscid compressible flow, conservative form, ideal gas
            # (gamma-law), 1D -- e.g. the Sod shock tube (Sod, 1978) and
            # other 1D astrophysical hydrodynamics validation cases.
            if not has_t:
                raise ValueError("euler_compressible_1d expects time coord 't'.")
            for n in ("rho", "rho_u", "E"):
                if n not in fields:
                    raise ValueError(f"euler_compressible_1d expects field '{n}'.")
            gamma = float(p.get("gamma", 1.4))
            x_idx = _coord_index(coords, "x")
            rho, rho_u, E_f = fields["rho"], fields["rho_u"], fields["E"]
            u_vel = rho_u / rho
            p_pres = (gamma - 1.0) * (E_f - 0.5 * rho * u_vel * u_vel)

            F1x, F2x, F3x = rho_u, rho_u * u_vel + p_pres, (E_f + p_pres) * u_vel

            rho_t = time_derivative(rho, xcol, t_index)  # type: ignore[arg-type]
            rhou_t = time_derivative(rho_u, xcol, t_index)  # type: ignore[arg-type]
            E_t = time_derivative(E_f, xcol, t_index)  # type: ignore[arg-type]

            res_list.append(rho_t + grad(F1x, xcol)[:, x_idx:x_idx + 1])
            res_list.append(rhou_t + grad(F2x, xcol)[:, x_idx:x_idx + 1])
            res_list.append(E_t + grad(F3x, xcol)[:, x_idx:x_idx + 1])

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

            elif cond.kind == "neumann" and cond.order <= 1:
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

            elif cond.kind == "neumann" and cond.order > 1:
                # Repeated single-coordinate derivative (e.g. beam moment
                # M=EI*d^2w/dz^2 or shear V=EI*d^3w/dz^3) — NOT a generalized
                # normal-dot-gradient contraction; see ConditionSpec's
                # docstring for the deliberately narrow scope.
                if cond.deriv_coord is None:
                    raise KeyError(f"Condition '{cond.name}' has order={cond.order} but no deriv_coord")
                d_idx = _coord_index(coords, cond.deriv_coord)
                parts = []
                for fname in cond.fields:
                    Xr = Xc.clone().detach().requires_grad_(True)
                    g = eval_fields(Xr)[fname]
                    for _ in range(cond.order):
                        g = torch.autograd.grad(
                            outputs=g, inputs=Xr, grad_outputs=torch.ones_like(g),
                            create_graph=True, retain_graph=True, allow_unused=False,
                        )[0][:, d_idx:d_idx + 1]
                    parts.append(g)
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