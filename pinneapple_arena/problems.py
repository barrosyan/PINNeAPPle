"""Arena physics problem registry.

22 problems across 8 physics domains:

  Fluid Mechanics         kovasznay_ns, stokes_2d, lid_driven_cavity_ns
  Diffusion / Heat        heat_1d, heat_2d, convection_diffusion_1d, darcy_flow_2d
  Elliptic                poisson_2d, laplace_2d, biharmonic_2d, helmholtz_2d
  Transport               advection_diffusion_1d
  Waves                   wave_1d, klein_gordon_1d, nls_1d
  Nonlinear / Solitons    burgers_1d, kdv_1d
  Reaction-Diffusion      allen_cahn, fisher_kpp_1d, fitzhugh_nagumo_1d
  Structural / Finance    linear_elasticity_2d, black_scholes_1d

Each problem exposes:
  - analytical(x, y, **params) -> dict[str, ndarray] | None
  - pinn_residuals(net, xy_int, xy_bc, uv_bc, **params) -> (res_loss, bc_loss)
  - supervised_data(n_train, n_bc, grid_n, **params)
        -> (X_int, Y_int, X_bc, Y_bc, X_eval, Y_eval, field_names)
  - compiled_losses(**params) -> dict | None   (via pinneapple_physics)
  - physics_preset: Optional[str]
  - description, input_dim, output_dim
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


# ── helpers ───────────────────────────────────────────────────────────────────

def _unwrap(out):
    """Extract a plain tensor from PINNOutput, OperatorOutput, or plain tensor."""
    if torch.is_tensor(out):
        return out
    if hasattr(out, "y") and torch.is_tensor(out.y):
        return out.y
    if hasattr(out, "x") and torch.is_tensor(out.x):
        return out.x
    if isinstance(out, dict):
        return torch.stack(list(out.values()), dim=-1)
    return out


def _grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                create_graph=True)[0]


def _boundary_points_rect(xmin, xmax, ymin, ymax, n):
    rng = np.random.default_rng(0)
    n_each = max(n // 4, 1)
    xl = np.full(n_each, xmin); yl = rng.uniform(ymin, ymax, n_each)
    xr = np.full(n_each, xmax); yr = rng.uniform(ymin, ymax, n_each)
    xb = rng.uniform(xmin, xmax, n_each); yb = np.full(n_each, ymin)
    xt = rng.uniform(xmin, xmax, n_each); yt = np.full(n_each, ymax)
    return np.stack([np.concatenate([xl, xr, xb, xt]),
                     np.concatenate([yl, yr, yb, yt])], axis=1)


def _eval_grid_2d(xmin, xmax, ymin, ymax, grid_n):
    gx = np.linspace(xmin, xmax, grid_n)
    gy = np.linspace(ymin, ymax, grid_n)
    GX, GY = np.meshgrid(gx, gy)
    return np.stack([GX.ravel(), GY.ravel()], axis=1)


# ── Problem base ───────────────────────────────────────────────────────────────

class ArenaProblem:
    name: str = "base"
    description: str = ""
    domain: str = ""        # human-readable physics domain
    input_dim: int = 2
    output_dim: int = 1
    physics_preset: Optional[str] = None

    def analytical(self, x: np.ndarray, y: np.ndarray, **params
                   ) -> Optional[Dict[str, np.ndarray]]:
        return None

    def pinn_residuals(self, net, xy_int: torch.Tensor, xy_bc: torch.Tensor,
                       uv_bc: torch.Tensor, **params
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def supervised_data(self, n_train: int, n_bc: int, grid_n: int, **params
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                   np.ndarray, np.ndarray, List[str]]:
        raise NotImplementedError

    def compiled_losses(self, physics_preset_override: Optional[str] = None,
                        **params) -> Optional[Dict[str, Callable]]:
        preset_name = physics_preset_override or self.physics_preset
        if preset_name is None:
            return None
        try:
            from pinneapple_physics import compile_physics, get_preset
            return compile_physics(get_preset(preset_name))
        except Exception:
            return None

    def to_problem_spec(self, physics_preset_override: Optional[str] = None, **params):
        preset_name = physics_preset_override or self.physics_preset
        if preset_name is None:
            return None
        try:
            from pinneapple_physics import get_preset
            return get_preset(preset_name)
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════════════════
# FLUID MECHANICS
# ══════════════════════════════════════════════════════════════════════════════

class KovasznayNS(ArenaProblem):
    name = "kovasznay_ns"
    description = "2D Kovasznay Navier-Stokes with analytical solution (Re=40)"
    domain = "Fluid Mechanics"
    input_dim = 2
    output_dim = 3  # u, v, p
    physics_preset = "ns_incompressible_2d_default"

    def _lambda(self, re): return re / 2 - math.sqrt(re**2 / 4 + 4*math.pi**2)

    def analytical(self, x, y, re=40.0, **kw):
        lam = self._lambda(re)
        u = 1 - np.exp(lam*x) * np.cos(2*np.pi*y)
        v = lam/(2*np.pi) * np.exp(lam*x) * np.sin(2*np.pi*y)
        p = 0.5*(1 - np.exp(2*lam*x))
        return {"u": u, "v": v, "p": p}

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, re=40.0, **kw):
        nu = 1.0/re
        xy = xy_int.clone().requires_grad_(True)
        out = _unwrap(net(xy))
        u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
        u_x = _grad(u, xy)[:, 0:1]; u_y = _grad(u, xy)[:, 1:2]
        v_x = _grad(v, xy)[:, 0:1]; v_y = _grad(v, xy)[:, 1:2]
        p_x = _grad(p, xy)[:, 0:1]; p_y = _grad(p, xy)[:, 1:2]
        u_xx = _grad(u_x, xy)[:, 0:1]; u_yy = _grad(u_y, xy)[:, 1:2]
        v_xx = _grad(v_x, xy)[:, 0:1]; v_yy = _grad(v_y, xy)[:, 1:2]
        r1 = u*u_x + v*u_y + p_x - nu*(u_xx + u_yy)
        r2 = u*v_x + v*v_y + p_y - nu*(v_xx + v_yy)
        r3 = u_x + v_y
        res = (r1**2 + r2**2 + r3**2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc)**2).mean()
        return res, bc

    def supervised_data(self, n_train=200, n_bc=500, grid_n=40, re=40.0, **kw):
        lam = self._lambda(re); rng = np.random.default_rng(42)
        x = rng.uniform(-0.5, 1.0, n_train); y = rng.uniform(-0.5, 1.5, n_train)
        xy_int = np.stack([x, y], axis=1)
        f = self.analytical(x, y, re=re)
        Y_int = np.stack([f["u"], f["v"], f["p"]], axis=1)
        pts = _boundary_points_rect(-0.5, 1.0, -0.5, 1.5, n_bc)
        bf = self.analytical(pts[:,0], pts[:,1], re=re)
        Y_bc = np.stack([bf["u"], bf["v"], bf["p"]], axis=1)
        xy_e = _eval_grid_2d(-0.5, 1.0, -0.5, 1.5, grid_n)
        ef = self.analytical(xy_e[:,0], xy_e[:,1], re=re)
        Y_e = np.stack([ef["u"], ef["v"], ef["p"]], axis=1)
        return xy_int, Y_int, pts, Y_bc, xy_e, Y_e, ["u", "v", "p"]


class Stokes2D(ArenaProblem):
    """2D Stokes (Poiseuille) flow: u = 1-y², v = 0, p = -2x."""
    name = "stokes_2d"
    description = "2D Poiseuille/Stokes flow: u=1-y^2, v=0, p=-2x (Re->0)"
    domain = "Fluid Mechanics"
    input_dim = 2
    output_dim = 3  # u, v, p

    def analytical(self, x, y, mu=1.0, **kw):
        # Poiseuille: no-slip at y=±1, pressure-driven in x
        u = (1 - y**2)
        v = np.zeros_like(x)
        p = -2*mu*x
        return {"u": u, "v": v, "p": p}

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, mu=1.0, **kw):
        xy = xy_int.clone().requires_grad_(True)
        out = _unwrap(net(xy))
        u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
        u_x = _grad(u, xy)[:, 0:1]; u_y = _grad(u, xy)[:, 1:2]
        v_x = _grad(v, xy)[:, 0:1]; v_y = _grad(v, xy)[:, 1:2]
        p_x = _grad(p, xy)[:, 0:1]; p_y = _grad(p, xy)[:, 1:2]
        u_xx = _grad(u_x, xy)[:, 0:1]; u_yy = _grad(u_y, xy)[:, 1:2]
        v_xx = _grad(v_x, xy)[:, 0:1]; v_yy = _grad(v_y, xy)[:, 1:2]
        # Stokes: -mu*Δu + ∇p = 0,  div u = 0
        r1 = -mu*(u_xx + u_yy) + p_x
        r2 = -mu*(v_xx + v_yy) + p_y
        r3 = u_x + v_y
        res = (r1**2 + r2**2 + r3**2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc)**2).mean()
        return res, bc

    def supervised_data(self, n_train=200, n_bc=400, grid_n=40, mu=1.0, **kw):
        rng = np.random.default_rng(42)
        x = rng.uniform(0.0, 2.0, n_train); y = rng.uniform(-1.0, 1.0, n_train)
        xy_int = np.stack([x, y], axis=1)
        f = self.analytical(x, y, mu=mu)
        Y_int = np.stack([f["u"], f["v"], f["p"]], axis=1)
        pts = _boundary_points_rect(0.0, 2.0, -1.0, 1.0, n_bc)
        bf = self.analytical(pts[:,0], pts[:,1], mu=mu)
        Y_bc = np.stack([bf["u"], bf["v"], bf["p"]], axis=1)
        xy_e = _eval_grid_2d(0.0, 2.0, -1.0, 1.0, grid_n)
        ef = self.analytical(xy_e[:,0], xy_e[:,1], mu=mu)
        Y_e = np.stack([ef["u"], ef["v"], ef["p"]], axis=1)
        return xy_int, Y_int, pts, Y_bc, xy_e, Y_e, ["u", "v", "p"]


# ══════════════════════════════════════════════════════════════════════════════
# DIFFUSION / HEAT
# ══════════════════════════════════════════════════════════════════════════════

class Heat1D(ArenaProblem):
    """1D heat: u_t = alpha*u_xx,  u = exp(-alpha*pi^2*t)*sin(pi*x)."""
    name = "heat_1d"
    description = "1D heat equation u_t = alpha*u_xx with Fourier solution"
    domain = "Diffusion / Heat"
    input_dim = 2   # (x, t)
    output_dim = 1

    def analytical(self, x, t, alpha=0.1, **kw):
        return {"u": np.exp(-alpha*math.pi**2*t) * np.sin(math.pi*x)}

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, alpha=0.1, **kw):
        xt = xy_int.clone().requires_grad_(True)
        u = _unwrap(net(xt))
        u_t = _grad(u, xt)[:, 1:2]; u_x = _grad(u, xt)[:, 0:1]
        u_xx = _grad(u_x, xt)[:, 0:1]
        res = ((u_t - alpha*u_xx)**2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc)**2).mean()
        return res, bc

    def supervised_data(self, n_train=200, n_bc=400, grid_n=40, alpha=0.1, **kw):
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, n_train); t = rng.uniform(0, 1, n_train)
        xt = np.stack([x, t], axis=1)
        Y_int = self.analytical(x, t, alpha=alpha)["u"].reshape(-1, 1)
        n_e = n_bc // 3
        t_s = rng.uniform(0, 1, n_e); x_ic = rng.uniform(0, 1, n_e)
        xb = np.concatenate([np.zeros(n_e), np.ones(n_e), x_ic])
        tb = np.concatenate([t_s, t_s, np.zeros(n_e)])
        pts = np.stack([xb, tb], axis=1)
        ub = np.concatenate([np.zeros(2*n_e), np.sin(math.pi*x_ic)])
        Y_bc = ub.reshape(-1, 1)
        gx = np.linspace(0, 1, grid_n); gt = np.linspace(0, 1, grid_n)
        GX, GT = np.meshgrid(gx, gt)
        xy_e = np.stack([GX.ravel(), GT.ravel()], axis=1)
        Y_e = self.analytical(xy_e[:,0], xy_e[:,1], alpha=alpha)["u"].reshape(-1, 1)
        return xt, Y_int, pts, Y_bc, xy_e, Y_e, ["u"]


class Heat2D(ArenaProblem):
    """2D heat: u_t = alpha*(u_xx+u_yy)."""
    name = "heat_2d"
    description = "2D heat equation u_t = alpha*(u_xx+u_yy) with Fourier solution"
    domain = "Diffusion / Heat"
    input_dim = 3   # (x, y, t)
    output_dim = 1

    def analytical(self, x, y, t=None, alpha=0.1, **kw):
        if t is None: t = np.zeros_like(x)
        return {"u": np.sin(math.pi*x)*np.sin(math.pi*y)*np.exp(-2*alpha*math.pi**2*t)}

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, alpha=0.1, **kw):
        xyt = xy_int.clone().requires_grad_(True)
        u = _unwrap(net(xyt))
        u_t = _grad(u, xyt)[:, 2:3]
        u_x = _grad(u, xyt)[:, 0:1]; u_y = _grad(u, xyt)[:, 1:2]
        u_xx = _grad(u_x, xyt)[:, 0:1]; u_yy = _grad(u_y, xyt)[:, 1:2]
        res = ((u_t - alpha*(u_xx + u_yy))**2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc)**2).mean()
        return res, bc

    def supervised_data(self, n_train=200, n_bc=500, grid_n=20, alpha=0.1, **kw):
        rng = np.random.default_rng(42)
        x = rng.uniform(0,1,n_train); y = rng.uniform(0,1,n_train); t = rng.uniform(0,1,n_train)
        xyt = np.stack([x,y,t], axis=1)
        Y_int = self.analytical(x, y, t, alpha=alpha)["u"].reshape(-1, 1)
        n_e = n_bc // 5; t_s = rng.uniform(0,1,n_e)
        y_s = rng.uniform(0,1,n_e); x_s = rng.uniform(0,1,n_e)
        x_ic = rng.uniform(0,1,n_e); y_ic = rng.uniform(0,1,n_e)
        xb = np.concatenate([np.zeros(n_e), np.ones(n_e), x_s, x_s, x_ic])
        yb = np.concatenate([y_s, y_s, np.zeros(n_e), np.ones(n_e), y_ic])
        tb = np.concatenate([t_s, t_s, t_s, t_s, np.zeros(n_e)])
        pts = np.stack([xb, yb, tb], axis=1)
        ub = np.concatenate([np.zeros(4*n_e), self.analytical(x_ic, y_ic, np.zeros(n_e), alpha=alpha)["u"]])
        Y_bc = ub.reshape(-1, 1)
        g = np.linspace(0,1,grid_n)
        GX, GY, GT = np.meshgrid(g, g, [0.5])
        xy_e = np.stack([GX.ravel(), GY.ravel(), GT.ravel()], axis=1)
        Y_e = self.analytical(xy_e[:,0], xy_e[:,1], xy_e[:,2], alpha=alpha)["u"].reshape(-1, 1)
        return xyt, Y_int, pts, Y_bc, xy_e, Y_e, ["u"]


class ConvectionDiffusion1D(ArenaProblem):
    """Steady 1D convection-diffusion: c*u_x = eps*u_xx.
    Analytical: u = (exp(c*x/eps) - 1)/(exp(c/eps) - 1) on [0,1], u(0)=0, u(1)=1."""
    name = "convection_diffusion_1d"
    description = "Steady 1D convection-diffusion boundary layer (Peclet layer)"
    domain = "Diffusion / Heat"
    input_dim = 1   # (x,)
    output_dim = 1

    def analytical(self, x, t=None, c=1.0, eps=0.05, **kw):
        denom = math.exp(c/eps) - 1
        u = (np.exp(c*x/eps) - 1) / (denom + 1e-40)
        return {"u": u}

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, c=1.0, eps=0.05, **kw):
        xi = xy_int.clone().requires_grad_(True)
        u = _unwrap(net(xi))
        u_x = _grad(u, xi)[:, 0:1]
        u_xx = _grad(u_x, xi)[:, 0:1]
        res = ((c*u_x - eps*u_xx)**2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc)**2).mean()
        return res, bc

    def supervised_data(self, n_train=200, n_bc=50, grid_n=100, c=1.0, eps=0.05, **kw):
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, n_train).reshape(-1, 1)
        Y_int = self.analytical(x.ravel(), c=c, eps=eps)["u"].reshape(-1, 1)
        x_bc = np.array([[0.0], [1.0]])
        Y_bc = np.array([[0.0], [1.0]])
        x_e = np.linspace(0, 1, grid_n).reshape(-1, 1)
        Y_e = self.analytical(x_e.ravel(), c=c, eps=eps)["u"].reshape(-1, 1)
        return x, Y_int, x_bc, Y_bc, x_e, Y_e, ["u"]


class DarcyFlow2D(ArenaProblem):
    """2D Darcy: -Δp = f,  p = sin(πx)sin(πy),  f = 2π²sin(πx)sin(πy)."""
    name = "darcy_flow_2d"
    description = "2D Darcy flow (pressure Poisson) with manufactured solution"
    domain = "Diffusion / Heat"
    input_dim = 2
    output_dim = 1  # pressure p
    physics_preset = "darcy_pressure_only_3d_default"

    def analytical(self, x, y, **kw):
        return {"p": np.sin(math.pi*x)*np.sin(math.pi*y)}

    def _source(self, x, y):
        return 2*math.pi**2 * np.sin(math.pi*x)*np.sin(math.pi*y)

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, **kw):
        xy = xy_int.clone().requires_grad_(True)
        p = _unwrap(net(xy))
        p_x = _grad(p, xy)[:, 0:1]; p_y = _grad(p, xy)[:, 1:2]
        p_xx = _grad(p_x, xy)[:, 0:1]; p_yy = _grad(p_y, xy)[:, 1:2]
        f = torch.tensor(self._source(xy_int[:,0].detach().cpu().numpy(),
                                      xy_int[:,1].detach().cpu().numpy()),
                         dtype=torch.float32, device=xy.device).unsqueeze(1)
        res = ((-(p_xx + p_yy) - f)**2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc)**2).mean()
        return res, bc

    def supervised_data(self, n_train=200, n_bc=400, grid_n=40, **kw):
        rng = np.random.default_rng(42)
        x = rng.uniform(0,1,n_train); y = rng.uniform(0,1,n_train)
        xy_int = np.stack([x,y], axis=1)
        Y_int = self.analytical(x, y)["p"].reshape(-1, 1)
        pts = _boundary_points_rect(0,1,0,1,n_bc)
        Y_bc = self.analytical(pts[:,0], pts[:,1])["p"].reshape(-1, 1)
        xy_e = _eval_grid_2d(0,1,0,1,grid_n)
        Y_e = self.analytical(xy_e[:,0], xy_e[:,1])["p"].reshape(-1, 1)
        return xy_int, Y_int, pts, Y_bc, xy_e, Y_e, ["p"]


# ══════════════════════════════════════════════════════════════════════════════
# ELLIPTIC
# ══════════════════════════════════════════════════════════════════════════════

class Laplace2D(ArenaProblem):
    """2D Laplace: Δu = 0,  u = sin(πx)sinh(πy)/sinh(π)."""
    name = "laplace_2d"
    description = "2D Laplace equation with Dirichlet BCs (potential flow)"
    domain = "Elliptic"
    input_dim = 2
    output_dim = 1
    physics_preset = "laplace_2d_default"

    def analytical(self, x, y, **kw):
        u = np.sin(math.pi*x) * np.sinh(math.pi*y) / math.sinh(math.pi)
        return {"u": u}

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, **kw):
        xy = xy_int.clone().requires_grad_(True)
        u = _unwrap(net(xy))
        u_x = _grad(u, xy)[:, 0:1]; u_y = _grad(u, xy)[:, 1:2]
        u_xx = _grad(u_x, xy)[:, 0:1]; u_yy = _grad(u_y, xy)[:, 1:2]
        res = ((u_xx + u_yy)**2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc)**2).mean()
        return res, bc

    def supervised_data(self, n_train=200, n_bc=400, grid_n=40, **kw):
        rng = np.random.default_rng(42)
        x = rng.uniform(0,1,n_train); y = rng.uniform(0,1,n_train)
        xy_int = np.stack([x,y], axis=1)
        Y_int = self.analytical(x, y)["u"].reshape(-1, 1)
        pts = _boundary_points_rect(0,1,0,1,n_bc)
        Y_bc = self.analytical(pts[:,0], pts[:,1])["u"].reshape(-1, 1)
        xy_e = _eval_grid_2d(0,1,0,1,grid_n)
        Y_e = self.analytical(xy_e[:,0], xy_e[:,1])["u"].reshape(-1, 1)
        return xy_int, Y_int, pts, Y_bc, xy_e, Y_e, ["u"]


class Poisson2D(ArenaProblem):
    """2D Poisson: Δu = -2π²sin(πx)sin(πy),  u = sin(πx)sin(πy)."""
    name = "poisson_2d"
    description = "2D Poisson equation with analytical manufactured solution"
    domain = "Elliptic"
    input_dim = 2
    output_dim = 1
    physics_preset = "poisson_2d_default"

    def analytical(self, x, y, **kw):
        return {"u": np.sin(math.pi*x)*np.sin(math.pi*y)}

    def _source(self, x, y):
        return -2*math.pi**2 * np.sin(math.pi*x)*np.sin(math.pi*y)

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, **kw):
        xy = xy_int.clone().requires_grad_(True)
        u = _unwrap(net(xy))
        u_x = _grad(u, xy)[:, 0:1]; u_y = _grad(u, xy)[:, 1:2]
        u_xx = _grad(u_x, xy)[:, 0:1]; u_yy = _grad(u_y, xy)[:, 1:2]
        f = torch.tensor(self._source(xy_int[:,0].detach().cpu().numpy(),
                                      xy_int[:,1].detach().cpu().numpy()),
                         dtype=torch.float32, device=xy.device).unsqueeze(1)
        res = ((u_xx + u_yy - f)**2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc)**2).mean()
        return res, bc

    def supervised_data(self, n_train=200, n_bc=400, grid_n=40, **kw):
        rng = np.random.default_rng(42)
        x = rng.uniform(0,1,n_train); y = rng.uniform(0,1,n_train)
        xy_int = np.stack([x,y], axis=1)
        Y_int = self.analytical(x,y)["u"].reshape(-1,1)
        pts = _boundary_points_rect(0,1,0,1,n_bc)
        Y_bc = self.analytical(pts[:,0], pts[:,1])["u"].reshape(-1,1)
        xy_e = _eval_grid_2d(0,1,0,1,grid_n)
        Y_e = self.analytical(xy_e[:,0], xy_e[:,1])["u"].reshape(-1,1)
        return xy_int, Y_int, pts, Y_bc, xy_e, Y_e, ["u"]


class Helmholtz2D(ArenaProblem):
    """2D Helmholtz: u_xx + u_yy + k²u = f,  u = sin(πx)sin(πy)."""
    name = "helmholtz_2d"
    description = "2D Helmholtz equation with manufactured solution"
    domain = "Elliptic"
    input_dim = 2
    output_dim = 1
    physics_preset = "helmholtz_acoustics_3d_default"

    def analytical(self, x, y, k=1.0, **kw):
        return {"u": np.sin(math.pi*x)*np.sin(math.pi*y)}

    def _source(self, x, y, k):
        return (k**2 - 2*math.pi**2)*np.sin(math.pi*x)*np.sin(math.pi*y)

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, k=1.0, **kw):
        xy = xy_int.clone().requires_grad_(True)
        u = _unwrap(net(xy))
        u_x = _grad(u, xy)[:, 0:1]; u_y = _grad(u, xy)[:, 1:2]
        u_xx = _grad(u_x, xy)[:, 0:1]; u_yy = _grad(u_y, xy)[:, 1:2]
        f = torch.tensor(self._source(xy_int[:,0].detach().cpu().numpy(),
                                      xy_int[:,1].detach().cpu().numpy(), k),
                         dtype=torch.float32, device=xy.device).unsqueeze(1)
        res = ((u_xx + u_yy + k**2*u - f)**2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc)**2).mean()
        return res, bc

    def supervised_data(self, n_train=200, n_bc=400, grid_n=40, k=1.0, **kw):
        rng = np.random.default_rng(42)
        x = rng.uniform(0,1,n_train); y = rng.uniform(0,1,n_train)
        xy_int = np.stack([x,y], axis=1)
        Y_int = self.analytical(x,y,k=k)["u"].reshape(-1,1)
        pts = _boundary_points_rect(0,1,0,1,n_bc)
        Y_bc = self.analytical(pts[:,0],pts[:,1],k=k)["u"].reshape(-1,1)
        xy_e = _eval_grid_2d(0,1,0,1,grid_n)
        Y_e = self.analytical(xy_e[:,0],xy_e[:,1],k=k)["u"].reshape(-1,1)
        return xy_int, Y_int, pts, Y_bc, xy_e, Y_e, ["u"]


class Biharmonic2D(ArenaProblem):
    """2D Biharmonic (thin plate): Δ²u = f, u = sin(πx)sin(πy), f = 4π⁴*u."""
    name = "biharmonic_2d"
    description = "2D Biharmonic equation (thin plate bending) Δ²u = f"
    domain = "Elliptic"
    input_dim = 2
    output_dim = 1

    def analytical(self, x, y, **kw):
        return {"u": np.sin(math.pi*x)*np.sin(math.pi*y)}

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, **kw):
        xy = xy_int.clone().requires_grad_(True)
        u = _unwrap(net(xy))
        u_x = _grad(u, xy)[:, 0:1]; u_y = _grad(u, xy)[:, 1:2]
        u_xx = _grad(u_x, xy)[:, 0:1]; u_yy = _grad(u_y, xy)[:, 1:2]
        lap = u_xx + u_yy
        lap_x = _grad(lap, xy)[:, 0:1]; lap_y = _grad(lap, xy)[:, 1:2]
        lap_xx = _grad(lap_x, xy)[:, 0:1]; lap_yy = _grad(lap_y, xy)[:, 1:2]
        bihar = lap_xx + lap_yy
        f = 4*math.pi**4 * u   # source for u = sin(πx)sin(πy)
        res = ((bihar - f)**2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc)**2).mean()
        return res, bc

    def supervised_data(self, n_train=200, n_bc=400, grid_n=40, **kw):
        rng = np.random.default_rng(42)
        x = rng.uniform(0,1,n_train); y = rng.uniform(0,1,n_train)
        xy_int = np.stack([x,y], axis=1)
        Y_int = self.analytical(x,y)["u"].reshape(-1,1)
        pts = _boundary_points_rect(0,1,0,1,n_bc)
        Y_bc = self.analytical(pts[:,0],pts[:,1])["u"].reshape(-1,1)
        xy_e = _eval_grid_2d(0,1,0,1,grid_n)
        Y_e = self.analytical(xy_e[:,0],xy_e[:,1])["u"].reshape(-1,1)
        return xy_int, Y_int, pts, Y_bc, xy_e, Y_e, ["u"]


# ══════════════════════════════════════════════════════════════════════════════
# TRANSPORT
# ══════════════════════════════════════════════════════════════════════════════

class AdvectionDiffusion1D(ArenaProblem):
    """1D advection-diffusion: u_t + c*u_x = nu*u_xx.
    Analytical: u = exp(-nu*pi²*t)*sin(pi*(x - c*t))."""
    name = "advection_diffusion_1d"
    description = "1D advection-diffusion: u_t + c*u_x = nu*u_xx"
    domain = "Transport"
    input_dim = 2   # (x, t)
    output_dim = 1

    def analytical(self, x, t, c=1.0, nu=0.01, **kw):
        u = np.exp(-nu*math.pi**2*t) * np.sin(math.pi*(x - c*t))
        return {"u": u}

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, c=1.0, nu=0.01, **kw):
        xt = xy_int.clone().requires_grad_(True)
        u = _unwrap(net(xt))
        u_t = _grad(u, xt)[:, 1:2]; u_x = _grad(u, xt)[:, 0:1]
        u_xx = _grad(u_x, xt)[:, 0:1]
        res = ((u_t + c*u_x - nu*u_xx)**2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc)**2).mean()
        return res, bc

    def supervised_data(self, n_train=200, n_bc=400, grid_n=40, c=1.0, nu=0.01, **kw):
        rng = np.random.default_rng(42)
        x = rng.uniform(-1,1,n_train); t = rng.uniform(0,1,n_train)
        xt = np.stack([x,t], axis=1)
        Y_int = self.analytical(x, t, c=c, nu=nu)["u"].reshape(-1,1)
        # BCs: periodic approximation — use x=-1 and x=1 (zeros at t=0 ends)
        n_e = n_bc // 3; t_s = rng.uniform(0,1,n_e); x_ic = rng.uniform(-1,1,n_e)
        xb = np.concatenate([-np.ones(n_e), np.ones(n_e), x_ic])
        tb = np.concatenate([t_s, t_s, np.zeros(n_e)])
        pts = np.stack([xb, tb], axis=1)
        ub = np.concatenate([
            self.analytical(-np.ones(n_e), t_s, c=c, nu=nu)["u"],
            self.analytical( np.ones(n_e), t_s, c=c, nu=nu)["u"],
            np.sin(math.pi*x_ic),   # IC: u(x,0)=sin(πx)
        ])
        Y_bc = ub.reshape(-1,1)
        gx = np.linspace(-1,1,grid_n); gt = np.linspace(0,1,grid_n)
        GX, GT = np.meshgrid(gx, gt)
        xy_e = np.stack([GX.ravel(), GT.ravel()], axis=1)
        Y_e = self.analytical(xy_e[:,0], xy_e[:,1], c=c, nu=nu)["u"].reshape(-1,1)
        return xt, Y_int, pts, Y_bc, xy_e, Y_e, ["u"]


# ══════════════════════════════════════════════════════════════════════════════
# WAVES
# ══════════════════════════════════════════════════════════════════════════════

class Wave1D(ArenaProblem):
    """1D wave: u_tt = c²u_xx,  u = sin(πx)cos(πct)."""
    name = "wave_1d"
    description = "1D wave equation u_tt = c^2*u_xx with sinusoidal solution"
    domain = "Waves"
    input_dim = 2
    output_dim = 1

    def analytical(self, x, t, c=1.0, **kw):
        return {"u": np.sin(math.pi*x)*np.cos(math.pi*c*t)}

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, c=1.0, **kw):
        xt = xy_int.clone().requires_grad_(True)
        u = _unwrap(net(xt))
        u_t = _grad(u, xt)[:, 1:2]; u_x = _grad(u, xt)[:, 0:1]
        u_tt = _grad(u_t, xt)[:, 1:2]; u_xx = _grad(u_x, xt)[:, 0:1]
        res = ((u_tt - c**2*u_xx)**2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc)**2).mean()
        return res, bc

    def supervised_data(self, n_train=200, n_bc=400, grid_n=40, c=1.0, **kw):
        rng = np.random.default_rng(42)
        x = rng.uniform(0,1,n_train); t = rng.uniform(0,1,n_train)
        xt = np.stack([x,t], axis=1)
        Y_int = self.analytical(x, t, c=c)["u"].reshape(-1,1)
        n_e = n_bc//4; t_s = rng.uniform(0,1,n_e); x_ic = rng.uniform(0,1,n_e)
        xb = np.concatenate([np.zeros(n_e), np.ones(n_e), x_ic, x_ic])
        tb = np.concatenate([t_s, t_s, np.zeros(n_e), np.zeros(n_e)])
        pts = np.stack([xb, tb], axis=1)
        ub = np.concatenate([np.zeros(2*n_e), np.sin(math.pi*x_ic), np.zeros(n_e)])
        Y_bc = ub.reshape(-1,1)
        gx = np.linspace(0,1,grid_n); gt = np.linspace(0,1,grid_n)
        GX, GT = np.meshgrid(gx, gt)
        xy_e = np.stack([GX.ravel(), GT.ravel()], axis=1)
        Y_e = self.analytical(xy_e[:,0], xy_e[:,1], c=c)["u"].reshape(-1,1)
        return xt, Y_int, pts, Y_bc, xy_e, Y_e, ["u"]


class KleinGordon1D(ArenaProblem):
    """1D Klein-Gordon: u_tt - u_xx + m²u = 0.
    Analytical: u = cos(kx)cos(ωt),  ω² = k² + m²."""
    name = "klein_gordon_1d"
    description = "1D Klein-Gordon equation u_tt - c^2*u_xx + m^2*u = 0"
    domain = "Waves"
    input_dim = 2
    output_dim = 1

    def analytical(self, x, t, c=1.0, m=1.0, k=math.pi, **kw):
        omega = math.sqrt(c**2*k**2 + m**2)
        return {"u": np.cos(k*x)*np.cos(omega*t)}

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, c=1.0, m=1.0, k=math.pi, **kw):
        omega = math.sqrt(c**2*k**2 + m**2)
        xt = xy_int.clone().requires_grad_(True)
        u = _unwrap(net(xt))
        u_t = _grad(u, xt)[:, 1:2]; u_x = _grad(u, xt)[:, 0:1]
        u_tt = _grad(u_t, xt)[:, 1:2]; u_xx = _grad(u_x, xt)[:, 0:1]
        res = ((u_tt - c**2*u_xx + m**2*u)**2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc)**2).mean()
        return res, bc

    def supervised_data(self, n_train=200, n_bc=400, grid_n=40, c=1.0, m=1.0, k=math.pi, **kw):
        omega = math.sqrt(c**2*k**2 + m**2)
        rng = np.random.default_rng(42)
        x = rng.uniform(-1,1,n_train); t = rng.uniform(0,2,n_train)
        xt = np.stack([x,t], axis=1)
        Y_int = self.analytical(x, t, c=c, m=m, k=k)["u"].reshape(-1,1)
        n_e = n_bc//4; t_s = rng.uniform(0,2,n_e); x_ic = rng.uniform(-1,1,n_e)
        xb = np.concatenate([-np.ones(n_e), np.ones(n_e), x_ic, x_ic])
        tb = np.concatenate([t_s, t_s, np.zeros(n_e), np.zeros(n_e)])
        pts = np.stack([xb, tb], axis=1)
        ub = np.concatenate([
            self.analytical(-np.ones(n_e), t_s, c=c, m=m, k=k)["u"],
            self.analytical( np.ones(n_e), t_s, c=c, m=m, k=k)["u"],
            np.cos(k*x_ic),     # IC: u(x,0) = cos(kx)
            np.zeros(n_e),      # IC: u_t(x,0) = 0
        ])
        Y_bc = ub.reshape(-1,1)
        gx = np.linspace(-1,1,grid_n); gt = np.linspace(0,2,grid_n)
        GX, GT = np.meshgrid(gx, gt)
        xy_e = np.stack([GX.ravel(), GT.ravel()], axis=1)
        Y_e = self.analytical(xy_e[:,0], xy_e[:,1], c=c, m=m, k=k)["u"].reshape(-1,1)
        return xt, Y_int, pts, Y_bc, xy_e, Y_e, ["u"]


class NLS1D(ArenaProblem):
    """1D NLS bright soliton: |ψ(x,t)| = A*sech(A*(x-vt)).
    We solve for (u, v) = (Re ψ, Im ψ) using the real/imaginary split."""
    name = "nls_1d"
    description = "1D Nonlinear Schrodinger soliton: i*psi_t + psi_xx + |psi|^2*psi = 0"
    domain = "Waves"
    input_dim = 2
    output_dim = 2  # (Re ψ, Im ψ)

    def analytical(self, x, t, A=1.0, v=0.0, **kw):
        # Bright soliton: ψ = A*sech(A*(x-vt))*exp(i*(v/2*x - (v²/4 - A²/2)*t))
        phase_x = v/2 * x
        phase_t = -(v**2/4 - A**2/2) * t
        amp = A / np.cosh(A*(x - v*t))
        u = amp * np.cos(phase_x + phase_t)
        w = amp * np.sin(phase_x + phase_t)
        return {"u": u, "v": w}

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, A=1.0, v=0.0, **kw):
        # NLS split: u_t = -v_xx - (u²+v²)*v,  v_t = u_xx + (u²+v²)*u
        xt = xy_int.clone().requires_grad_(True)
        out = _unwrap(net(xt))
        u, wi = out[:, 0:1], out[:, 1:2]
        u_t = _grad(u, xt)[:, 1:2]; wi_t = _grad(wi, xt)[:, 1:2]
        u_x = _grad(u, xt)[:, 0:1]; wi_x = _grad(wi, xt)[:, 0:1]
        u_xx = _grad(u_x, xt)[:, 0:1]; wi_xx = _grad(wi_x, xt)[:, 0:1]
        rho2 = u**2 + wi**2
        r1 = u_t + wi_xx + rho2*wi
        r2 = wi_t - u_xx - rho2*u
        res = (r1**2 + r2**2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc)**2).mean()
        return res, bc

    def supervised_data(self, n_train=300, n_bc=400, grid_n=40, A=1.0, v=0.0, **kw):
        rng = np.random.default_rng(42)
        x = rng.uniform(-5,5,n_train); t = rng.uniform(0,2,n_train)
        xt = np.stack([x,t], axis=1)
        f = self.analytical(x, t, A=A, v=v)
        Y_int = np.stack([f["u"], f["v"]], axis=1)
        n_e = n_bc//3; t_s = rng.uniform(0,2,n_e); x_ic = rng.uniform(-5,5,n_e)
        xb = np.concatenate([-5*np.ones(n_e), 5*np.ones(n_e), x_ic])
        tb = np.concatenate([t_s, t_s, np.zeros(n_e)])
        pts = np.stack([xb, tb], axis=1)
        bf = self.analytical(pts[:,0], pts[:,1], A=A, v=v)
        Y_bc = np.stack([bf["u"], bf["v"]], axis=1)
        gx = np.linspace(-5,5,grid_n); gt = np.linspace(0,2,grid_n)
        GX, GT = np.meshgrid(gx, gt)
        xy_e = np.stack([GX.ravel(), GT.ravel()], axis=1)
        ef = self.analytical(xy_e[:,0], xy_e[:,1], A=A, v=v)
        Y_e = np.stack([ef["u"], ef["v"]], axis=1)
        return xt, Y_int, pts, Y_bc, xy_e, Y_e, ["u_Re", "u_Im"]


# ══════════════════════════════════════════════════════════════════════════════
# NONLINEAR / SOLITONS
# ══════════════════════════════════════════════════════════════════════════════

class Burgers1D(ArenaProblem):
    """1D Burgers: u_t + u*u_x = nu*u_xx."""
    name = "burgers_1d"
    description = "1D viscous Burgers equation u_t + u*u_x = nu*u_xx"
    domain = "Nonlinear / Solitons"
    input_dim = 2
    output_dim = 1
    physics_preset = "burgers_1d_default"

    def analytical(self, x, t, nu=0.01, **kw):
        u = -np.sin(math.pi*x) / (1 + t*math.pi*np.cos(math.pi*x) + 1e-8)
        return {"u": u}

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, nu=0.01, **kw):
        xt = xy_int.clone().requires_grad_(True)
        u = _unwrap(net(xt))
        u_t = _grad(u, xt)[:, 1:2]; u_x = _grad(u, xt)[:, 0:1]
        u_xx = _grad(u_x, xt)[:, 0:1]
        res = ((u_t + u*u_x - nu*u_xx)**2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc)**2).mean()
        return res, bc

    def supervised_data(self, n_train=200, n_bc=400, grid_n=40, nu=0.01, **kw):
        rng = np.random.default_rng(42)
        x = rng.uniform(-1,1,n_train); t = rng.uniform(0,1,n_train)
        xt = np.stack([x,t], axis=1)
        Y_int = self.analytical(x, t, nu=nu)["u"].reshape(-1,1)
        n_e = n_bc//3; t_s = rng.uniform(0,1,n_e); x_ic = rng.uniform(-1,1,n_e)
        xb = np.concatenate([-np.ones(n_e), np.ones(n_e), x_ic])
        tb = np.concatenate([t_s, t_s, np.zeros(n_e)])
        pts = np.stack([xb, tb], axis=1)
        ub = np.concatenate([np.zeros(2*n_e), -np.sin(math.pi*x_ic)])
        Y_bc = ub.reshape(-1,1)
        gx = np.linspace(-1,1,grid_n); gt = np.linspace(0,1,grid_n)
        GX, GT = np.meshgrid(gx, gt)
        xy_e = np.stack([GX.ravel(), GT.ravel()], axis=1)
        Y_e = self.analytical(xy_e[:,0], xy_e[:,1], nu=nu)["u"].reshape(-1,1)
        return xt, Y_int, pts, Y_bc, xy_e, Y_e, ["u"]


class KdV1D(ArenaProblem):
    """1D KdV: u_t + 6u*u_x + u_xxx = 0.
    Soliton: u = -2c*sech²(sqrt(c)*(x - 4c*t))."""
    name = "kdv_1d"
    description = "1D KdV soliton: u_t + 6*u*u_x + u_xxx = 0"
    domain = "Nonlinear / Solitons"
    input_dim = 2
    output_dim = 1

    def analytical(self, x, t, c=1.0, x0=0.0, **kw):
        xi = np.sqrt(c)*(x - 4*c*t - x0)
        return {"u": -2*c / np.cosh(xi)**2}

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, c=1.0, x0=0.0, **kw):
        xt = xy_int.clone().requires_grad_(True)
        u = _unwrap(net(xt))
        u_t = _grad(u, xt)[:, 1:2]; u_x = _grad(u, xt)[:, 0:1]
        u_xx = _grad(u_x, xt)[:, 0:1]; u_xxx = _grad(u_xx, xt)[:, 0:1]
        res = ((u_t + 6*u*u_x + u_xxx)**2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc)**2).mean()
        return res, bc

    def supervised_data(self, n_train=300, n_bc=400, grid_n=40, c=1.0, x0=0.0, **kw):
        rng = np.random.default_rng(42)
        x = rng.uniform(-8,8,n_train); t = rng.uniform(0,1,n_train)
        xt = np.stack([x,t], axis=1)
        Y_int = self.analytical(x, t, c=c, x0=x0)["u"].reshape(-1,1)
        n_e = n_bc//3; t_s = rng.uniform(0,1,n_e); x_ic = rng.uniform(-8,8,n_e)
        xb = np.concatenate([-8*np.ones(n_e), 8*np.ones(n_e), x_ic])
        tb = np.concatenate([t_s, t_s, np.zeros(n_e)])
        pts = np.stack([xb, tb], axis=1)
        bf = self.analytical(pts[:,0], pts[:,1], c=c, x0=x0)
        Y_bc = bf["u"].reshape(-1,1)
        gx = np.linspace(-8,8,grid_n); gt = np.linspace(0,1,grid_n)
        GX, GT = np.meshgrid(gx, gt)
        xy_e = np.stack([GX.ravel(), GT.ravel()], axis=1)
        Y_e = self.analytical(xy_e[:,0], xy_e[:,1], c=c, x0=x0)["u"].reshape(-1,1)
        return xt, Y_int, pts, Y_bc, xy_e, Y_e, ["u"]


# ══════════════════════════════════════════════════════════════════════════════
# REACTION-DIFFUSION
# ══════════════════════════════════════════════════════════════════════════════

class AllenCahn(AreraProblem := ArenaProblem):
    """1D Allen-Cahn: u_t = eps²u_xx + u - u³.  Tanh interface."""
    name = "allen_cahn"
    description = "1D Allen-Cahn phase-field equation: u_t = eps^2*u_xx + u - u^3"
    domain = "Reaction-Diffusion"
    input_dim = 2
    output_dim = 1

    def analytical(self, x, t, eps=0.01, **kw):
        return {"u": np.tanh(x / (math.sqrt(2)*eps))}

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, eps=0.01, **kw):
        xt = xy_int.clone().requires_grad_(True)
        u = _unwrap(net(xt))
        u_t = _grad(u, xt)[:, 1:2]; u_x = _grad(u, xt)[:, 0:1]
        u_xx = _grad(u_x, xt)[:, 0:1]
        res = ((u_t - eps**2*u_xx - u + u**3)**2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc)**2).mean()
        return res, bc

    def supervised_data(self, n_train=200, n_bc=400, grid_n=40, eps=0.01, **kw):
        rng = np.random.default_rng(42)
        x = rng.uniform(-1,1,n_train); t = rng.uniform(0,1,n_train)
        xt = np.stack([x,t], axis=1)
        Y_int = self.analytical(x, t, eps=eps)["u"].reshape(-1,1)
        n_e = n_bc//3; t_s = rng.uniform(0,1,n_e); x_ic = rng.uniform(-1,1,n_e)
        xb = np.concatenate([-np.ones(n_e), np.ones(n_e), x_ic])
        tb = np.concatenate([t_s, t_s, np.zeros(n_e)])
        pts = np.stack([xb, tb], axis=1)
        ub = np.concatenate([-np.ones(n_e), np.ones(n_e), np.tanh(x_ic/(math.sqrt(2)*eps))])
        Y_bc = ub.reshape(-1,1)
        gx = np.linspace(-1,1,grid_n); gt = np.linspace(0,1,grid_n)
        GX, GT = np.meshgrid(gx, gt)
        xy_e = np.stack([GX.ravel(), GT.ravel()], axis=1)
        Y_e = self.analytical(xy_e[:,0], xy_e[:,1], eps=eps)["u"].reshape(-1,1)
        return xt, Y_int, pts, Y_bc, xy_e, Y_e, ["u"]


class FisherKPP1D(ArenaProblem):
    """1D Fisher-KPP: u_t = D*u_xx + r*u*(1-u).
    Traveling wave: u ≈ 1/(1+exp(x - v*t)) with v = 5/sqrt(6)."""
    name = "fisher_kpp_1d"
    description = "1D Fisher-KPP traveling wave: u_t = D*u_xx + r*u*(1-u)"
    domain = "Reaction-Diffusion"
    input_dim = 2
    output_dim = 1

    def _speed(self, D, r): return 5*math.sqrt(D*r/6)

    def analytical(self, x, t, D=1.0, r=1.0, **kw):
        v = self._speed(D, r)
        w = 1/math.sqrt(6*D/r)   # sharpness parameter
        u = 1.0 / (1.0 + np.exp(w*(x - v*t)))
        return {"u": u}

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, D=1.0, r=1.0, **kw):
        xt = xy_int.clone().requires_grad_(True)
        u = _unwrap(net(xt))
        u_t = _grad(u, xt)[:, 1:2]; u_x = _grad(u, xt)[:, 0:1]
        u_xx = _grad(u_x, xt)[:, 0:1]
        res = ((u_t - D*u_xx - r*u*(1-u))**2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc)**2).mean()
        return res, bc

    def supervised_data(self, n_train=200, n_bc=400, grid_n=40, D=1.0, r=1.0, **kw):
        rng = np.random.default_rng(42)
        x = rng.uniform(-5,5,n_train); t = rng.uniform(0,2,n_train)
        xt = np.stack([x,t], axis=1)
        Y_int = self.analytical(x, t, D=D, r=r)["u"].reshape(-1,1)
        n_e = n_bc//3; t_s = rng.uniform(0,2,n_e); x_ic = rng.uniform(-5,5,n_e)
        xb = np.concatenate([-5*np.ones(n_e), 5*np.ones(n_e), x_ic])
        tb = np.concatenate([t_s, t_s, np.zeros(n_e)])
        pts = np.stack([xb, tb], axis=1)
        bf = self.analytical(pts[:,0], pts[:,1], D=D, r=r)
        Y_bc = bf["u"].reshape(-1,1)
        gx = np.linspace(-5,5,grid_n); gt = np.linspace(0,2,grid_n)
        GX, GT = np.meshgrid(gx, gt)
        xy_e = np.stack([GX.ravel(), GT.ravel()], axis=1)
        Y_e = self.analytical(xy_e[:,0], xy_e[:,1], D=D, r=r)["u"].reshape(-1,1)
        return xt, Y_int, pts, Y_bc, xy_e, Y_e, ["u"]


class FitzHughNagumo1D(ArenaProblem):
    """1D FitzHugh-Nagumo excitable media (two fields: v, w).
    v_t = v_xx + v(1-v)(v-a) - w + I,  w_t = eps*(v - gamma*w)."""
    name = "fitzhugh_nagumo_1d"
    description = "1D FitzHugh-Nagumo excitable media: coupled v-w PDE system"
    domain = "Reaction-Diffusion"
    input_dim = 2
    output_dim = 2  # (v, w)
    physics_preset = None

    def analytical(self, x, t, a=0.1, eps=0.01, gamma=0.5, I=0.0, **kw):
        # Approximate: plane wave solution with tanh front for v, w follows slowly
        v0 = 0.5*(1 + np.tanh(5*(x - 0.5)))
        w0 = eps*v0 / (eps + gamma*eps)   # slow manifold approximation
        v = v0 * np.exp(-eps*t)
        w = w0 * (1 - np.exp(-eps*t)) + w0*np.exp(-eps*t)
        return {"v": v, "w": w}

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, a=0.1, eps=0.01, gamma=0.5, I=0.0, **kw):
        xt = xy_int.clone().requires_grad_(True)
        out = _unwrap(net(xt))
        v, w = out[:, 0:1], out[:, 1:2]
        v_t = _grad(v, xt)[:, 1:2]; v_x = _grad(v, xt)[:, 0:1]
        v_xx = _grad(v_x, xt)[:, 0:1]
        w_t = _grad(w, xt)[:, 1:2]
        r1 = v_t - v_xx - v*(1-v)*(v-a) + w - I
        r2 = w_t - eps*(v - gamma*w)
        res = (r1**2 + r2**2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc)**2).mean()
        return res, bc

    def supervised_data(self, n_train=300, n_bc=400, grid_n=40, a=0.1, eps=0.01,
                        gamma=0.5, I=0.0, **kw):
        rng = np.random.default_rng(42)
        x = rng.uniform(0,1,n_train); t = rng.uniform(0,2,n_train)
        xt = np.stack([x,t], axis=1)
        f = self.analytical(x, t, a=a, eps=eps, gamma=gamma, I=I)
        Y_int = np.stack([f["v"], f["w"]], axis=1)
        n_e = n_bc//3; t_s = rng.uniform(0,2,n_e); x_ic = rng.uniform(0,1,n_e)
        xb = np.concatenate([np.zeros(n_e), np.ones(n_e), x_ic])
        tb = np.concatenate([t_s, t_s, np.zeros(n_e)])
        pts = np.stack([xb, tb], axis=1)
        bf = self.analytical(pts[:,0], pts[:,1], a=a, eps=eps, gamma=gamma, I=I)
        Y_bc = np.stack([bf["v"], bf["w"]], axis=1)
        gx = np.linspace(0,1,grid_n); gt = np.linspace(0,2,grid_n)
        GX, GT = np.meshgrid(gx, gt)
        xy_e = np.stack([GX.ravel(), GT.ravel()], axis=1)
        ef = self.analytical(xy_e[:,0], xy_e[:,1], a=a, eps=eps, gamma=gamma, I=I)
        Y_e = np.stack([ef["v"], ef["w"]], axis=1)
        return xt, Y_int, pts, Y_bc, xy_e, Y_e, ["v", "w"]


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURAL / FINANCE
# ══════════════════════════════════════════════════════════════════════════════

class LinearElasticity2D(ArenaProblem):
    """2D plane-stress linear elasticity (manufactured solution).
    Displacement: u = sin(πx)sin(πy), v = 0.
    Body force computed from Lamé equations."""
    name = "linear_elasticity_2d"
    description = "2D plane-stress linear elasticity with manufactured displacement"
    domain = "Structural"
    input_dim = 2
    output_dim = 2  # (u_x, u_y) displacements
    physics_preset = None

    def _lame(self, E, nu):
        lam = E*nu/((1+nu)*(1-2*nu))
        mu  = E/(2*(1+nu))
        return lam, mu

    def analytical(self, x, y, E=1.0, nu_p=0.3, **kw):
        # Manufactured: u_x = sin(πx)sin(πy), u_y = cos(πx)cos(πy)/π
        ux = np.sin(math.pi*x)*np.sin(math.pi*y)
        uy = np.cos(math.pi*x)*np.cos(math.pi*y) / math.pi
        return {"ux": ux, "uy": uy}

    def _body_force(self, x, y, E=1.0, nu_p=0.3):
        lam, mu = self._lame(E, nu_p)
        p = math.pi
        # For u_x = sin(πx)sin(πy), u_y = cos(πx)cos(πy)/π:
        # div(eps) computed symbolically → body force
        fx = (lam+2*mu)*p**2*np.sin(p*x)*np.sin(p*y) + mu*p**2*np.sin(p*x)*np.sin(p*y) \
             - lam*p**2*np.sin(p*x)*np.sin(p*y)
        fy = (lam+2*mu)*p*np.cos(p*x)*np.cos(p*y) + mu*p*np.cos(p*x)*np.cos(p*y) \
             + lam*p*np.cos(p*x)*np.cos(p*y)
        return fx, fy

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, E=1.0, nu_p=0.3, **kw):
        lam, mu = self._lame(E, nu_p)
        xy = xy_int.clone().requires_grad_(True)
        out = _unwrap(net(xy))
        ux, uy = out[:, 0:1], out[:, 1:2]
        # strains
        ux_x = _grad(ux, xy)[:, 0:1]; ux_y = _grad(ux, xy)[:, 1:2]
        uy_x = _grad(uy, xy)[:, 0:1]; uy_y = _grad(uy, xy)[:, 1:2]
        div_u = ux_x + uy_y
        # stresses σ = λ*div(u)*I + 2μ*ε
        sx_x = _grad((lam*div_u + 2*mu*ux_x), xy)[:, 0:1]
        sxy_y = _grad((mu*(ux_y + uy_x)), xy)[:, 1:2]
        sy_y = _grad((lam*div_u + 2*mu*uy_y), xy)[:, 1:2]
        sxy_x = _grad((mu*(ux_y + uy_x)), xy)[:, 0:1]
        x_np = xy_int[:,0].detach().cpu().numpy()
        y_np = xy_int[:,1].detach().cpu().numpy()
        fx, fy = self._body_force(x_np, y_np, E=E, nu_p=nu_p)
        fx_t = torch.tensor(fx, dtype=torch.float32, device=xy.device).unsqueeze(1)
        fy_t = torch.tensor(fy, dtype=torch.float32, device=xy.device).unsqueeze(1)
        r1 = sx_x + sxy_y + fx_t
        r2 = sxy_x + sy_y + fy_t
        res = (r1**2 + r2**2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc)**2).mean()
        return res, bc

    def supervised_data(self, n_train=200, n_bc=400, grid_n=40, E=1.0, nu_p=0.3, **kw):
        rng = np.random.default_rng(42)
        x = rng.uniform(0,1,n_train); y = rng.uniform(0,1,n_train)
        xy_int = np.stack([x,y], axis=1)
        f = self.analytical(x, y, E=E, nu_p=nu_p)
        Y_int = np.stack([f["ux"], f["uy"]], axis=1)
        pts = _boundary_points_rect(0,1,0,1,n_bc)
        bf = self.analytical(pts[:,0], pts[:,1], E=E, nu_p=nu_p)
        Y_bc = np.stack([bf["ux"], bf["uy"]], axis=1)
        xy_e = _eval_grid_2d(0,1,0,1,grid_n)
        ef = self.analytical(xy_e[:,0], xy_e[:,1], E=E, nu_p=nu_p)
        Y_e = np.stack([ef["ux"], ef["uy"]], axis=1)
        return xy_int, Y_int, pts, Y_bc, xy_e, Y_e, ["ux", "uy"]


class BlackScholes1D(ArenaProblem):
    """1D Black-Scholes PDE for a European call option.
    V_t + r*S*V_S + sigma^2*S^2/2*V_SS - r*V = 0.
    Analytical via BS formula (evaluated on (log-price, time) grid)."""
    name = "black_scholes_1d"
    description = "1D Black-Scholes PDE: European call option pricing"
    domain = "Finance"
    input_dim = 2   # (S, t) — stock price and time-to-expiry
    output_dim = 1  # V (option value)

    def _norm_cdf(self, x):
        from scipy.special import ndtr
        return ndtr(x)

    def analytical(self, S, t, K=1.0, r=0.05, sigma=0.2, **kw):
        eps = 1e-8
        S = np.clip(S, eps, None); t = np.clip(t, eps, None)
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*t) / (sigma*np.sqrt(t) + eps)
        d2 = d1 - sigma*np.sqrt(t)
        try:
            N = self._norm_cdf
            V = S*N(d1) - K*np.exp(-r*t)*N(d2)
        except Exception:
            from math import erf, sqrt
            N_scalar = lambda z: 0.5*(1 + erf(z/sqrt(2)))
            N_vec    = np.vectorize(N_scalar)
            V = S*N_vec(d1) - K*np.exp(-r*t)*N_vec(d2)
        return {"V": np.maximum(V, 0)}

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, K=1.0, r=0.05, sigma=0.2, **kw):
        St = xy_int.clone().requires_grad_(True)
        V = _unwrap(net(St))
        V_t = _grad(V, St)[:, 1:2]
        V_S = _grad(V, St)[:, 0:1]
        V_SS = _grad(V_S, St)[:, 0:1]
        S = St[:, 0:1]
        res = ((V_t + r*S*V_S + 0.5*sigma**2*S**2*V_SS - r*V)**2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc)**2).mean()
        return res, bc

    def supervised_data(self, n_train=200, n_bc=400, grid_n=40,
                        K=1.0, r=0.05, sigma=0.2, **kw):
        rng = np.random.default_rng(42)
        S = rng.uniform(0.2, 2.0, n_train); t = rng.uniform(0.01, 1.0, n_train)
        St = np.stack([S, t], axis=1)
        Y_int = self.analytical(S, t, K=K, r=r, sigma=sigma)["V"].reshape(-1,1)
        # boundary: S=0 (V=0), S→large (V=S-K*exp(-rt)), t→0 (V=max(S-K,0))
        n_e = n_bc//3; S_bc_hi = 2.0; t_s = rng.uniform(0.01,1.0,n_e)
        S_ic = rng.uniform(0.2, 2.0, n_e)
        S_b = np.concatenate([0.2*np.ones(n_e), S_bc_hi*np.ones(n_e), S_ic])
        t_b = np.concatenate([t_s, t_s, 0.01*np.ones(n_e)])
        pts = np.stack([S_b, t_b], axis=1)
        bf = self.analytical(pts[:,0], pts[:,1], K=K, r=r, sigma=sigma)
        Y_bc = bf["V"].reshape(-1,1)
        gS = np.linspace(0.2, 2.0, grid_n); gt = np.linspace(0.01, 1.0, grid_n)
        GS, GT = np.meshgrid(gS, gt)
        xy_e = np.stack([GS.ravel(), GT.ravel()], axis=1)
        Y_e = self.analytical(xy_e[:,0], xy_e[:,1], K=K, r=r, sigma=sigma)["V"].reshape(-1,1)
        return St, Y_int, pts, Y_bc, xy_e, Y_e, ["V"]


# ══════════════════════════════════════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════════════════════════════════════

# Fix AllenCahn class definition (used alias trick above — clean up)
class AllenCahn(ArenaProblem):
    name = "allen_cahn"
    description = "1D Allen-Cahn phase-field equation: u_t = eps^2*u_xx + u - u^3"
    domain = "Reaction-Diffusion"
    input_dim = 2
    output_dim = 1

    def analytical(self, x, t, eps=0.01, **kw):
        return {"u": np.tanh(x / (math.sqrt(2)*eps))}

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, eps=0.01, **kw):
        xt = xy_int.clone().requires_grad_(True)
        u = _unwrap(net(xt))
        u_t = _grad(u, xt)[:, 1:2]; u_x = _grad(u, xt)[:, 0:1]
        u_xx = _grad(u_x, xt)[:, 0:1]
        res = ((u_t - eps**2*u_xx - u + u**3)**2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc)**2).mean()
        return res, bc

    def supervised_data(self, n_train=200, n_bc=400, grid_n=40, eps=0.01, **kw):
        rng = np.random.default_rng(42)
        x = rng.uniform(-1,1,n_train); t = rng.uniform(0,1,n_train)
        xt = np.stack([x,t], axis=1)
        Y_int = self.analytical(x, t, eps=eps)["u"].reshape(-1,1)
        n_e = n_bc//3; t_s = rng.uniform(0,1,n_e); x_ic = rng.uniform(-1,1,n_e)
        xb = np.concatenate([-np.ones(n_e), np.ones(n_e), x_ic])
        tb = np.concatenate([t_s, t_s, np.zeros(n_e)])
        pts = np.stack([xb, tb], axis=1)
        ub = np.concatenate([-np.ones(n_e), np.ones(n_e), np.tanh(x_ic/(math.sqrt(2)*eps))])
        Y_bc = ub.reshape(-1,1)
        gx = np.linspace(-1,1,grid_n); gt = np.linspace(0,1,grid_n)
        GX, GT = np.meshgrid(gx, gt)
        xy_e = np.stack([GX.ravel(), GT.ravel()], axis=1)
        Y_e = self.analytical(xy_e[:,0], xy_e[:,1], eps=eps)["u"].reshape(-1,1)
        return xt, Y_int, pts, Y_bc, xy_e, Y_e, ["u"]


_PROBLEM_REGISTRY: Dict[str, ArenaProblem] = {}

def _reg(cls):
    inst = cls()
    _PROBLEM_REGISTRY[inst.name] = inst
    return cls

# Fluid Mechanics
_reg(KovasznayNS)
_reg(Stokes2D)
# Diffusion / Heat
_reg(Heat1D)
_reg(Heat2D)
_reg(ConvectionDiffusion1D)
_reg(DarcyFlow2D)
# Elliptic
_reg(Laplace2D)
_reg(Poisson2D)
_reg(Helmholtz2D)
_reg(Biharmonic2D)
# Transport
_reg(AdvectionDiffusion1D)
# Waves
_reg(Wave1D)
_reg(KleinGordon1D)
_reg(NLS1D)
# Nonlinear / Solitons
_reg(Burgers1D)
_reg(KdV1D)
# Reaction-Diffusion
_reg(AllenCahn)
_reg(FisherKPP1D)
_reg(FitzHughNagumo1D)
# Structural / Finance
_reg(LinearElasticity2D)
_reg(BlackScholes1D)


def get_problem(name: str) -> ArenaProblem:
    if name not in _PROBLEM_REGISTRY:
        raise KeyError(f"Unknown problem '{name}'. Available: {sorted(_PROBLEM_REGISTRY)}")
    return _PROBLEM_REGISTRY[name]


def register_problem(problem: "ArenaProblem") -> "ArenaProblem":
    """Register a custom ArenaProblem instance so Arena can find it by name.

    Usage::

        from pinneapple_arena.problems import register_problem

        my_problem = MyCustomProblem()
        register_problem(my_problem)
        # now use {"name": my_problem.name} in your ArenaConfig
    """
    _PROBLEM_REGISTRY[problem.name] = problem
    return problem


def list_problems() -> List[str]:
    return sorted(_PROBLEM_REGISTRY.keys())


def list_problems_by_domain() -> Dict[str, List[str]]:
    domains: Dict[str, List[str]] = {}
    for name, p in _PROBLEM_REGISTRY.items():
        domains.setdefault(getattr(p, "domain", "Other"), []).append(name)
    return domains
