"""RANS turbulence closure presets for PINN training.

Implements k-omega SST (Menter 1994) and Spalart-Allmaras (SA, 1992) models
as PDE residual functions compatible with PINNeAPPle's PINNFactory.

Each residual class is callable:
    residuals = KOmegaSSTResiduals(nu=1e-5, rho=1.0)
    r_dict = residuals(model, x_col)   # returns dict of residual tensors

The dict keys match the equation names and can be directly summed into a
PINN loss::

    loss = sum(r.pow(2).mean() for r in r_dict.values())
"""
from __future__ import annotations

import torch
from typing import Callable, Dict, Optional


# ---------------------------------------------------------------------------
# k-omega SST constants  (Menter 1994)
# ---------------------------------------------------------------------------

SST_CONSTS: Dict[str, float] = {
    "sigma_k1":  0.85,
    "sigma_k2":  1.0,
    "sigma_w1":  0.5,
    "sigma_w2":  0.856,
    "beta1":     0.075,
    "beta2":     0.0828,
    "beta_star": 0.09,
    "kappa":     0.41,
    "a1":        0.31,
    "gamma1":    5.0 / 9.0,
    "gamma2":    0.44,
}


# ---------------------------------------------------------------------------
# Helper: first / second-order autograd
# ---------------------------------------------------------------------------

def _grad1(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Gradient of scalar/vector field f w.r.t. x (same-shape as x)."""
    return torch.autograd.grad(
        f, x,
        grad_outputs=torch.ones_like(f),
        create_graph=True,
        retain_graph=True,
    )[0]


def _laplacian(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Scalar Laplacian of f w.r.t. x: sum_i d^2f/dx_i^2. f must be (N,1)."""
    gf = _grad1(f, x)  # (N, D)
    lap = torch.zeros(f.shape[0], 1, device=x.device, dtype=x.dtype)
    for i in range(x.shape[1]):
        g_i = gf[:, i : i + 1]
        lap = lap + _grad1(g_i, x)[:, i : i + 1]
    return lap


# ---------------------------------------------------------------------------
# Feature 10a: k-omega SST
# ---------------------------------------------------------------------------

class KOmegaSSTResiduals:
    """k-omega SST two-equation turbulence model residuals for PINN.

    Equations solved (2D, incompressible, steady RANS):

        Momentum-x:  rho*(u u_x + v u_y) + p_x - d/dx[(mu+mu_t) 2 u_x]
                     - d/dy[(mu+mu_t)(u_y + v_x)] = 0
        Momentum-y:  rho*(u v_x + v v_y) + p_y - d/dx[(mu+mu_t)(u_y + v_x)]
                     - d/dy[(mu+mu_t) 2 v_y] = 0
        Continuity:  u_x + v_y = 0
        k-equation:  u k_x + v k_y - d/dx[(nu+sigma_k nu_t) k_x]
                     - d/dy[(nu+sigma_k nu_t) k_y] - P_k + beta* k omega = 0
        omega-eq:    u w_x + v w_y - d/dx[(nu+sigma_w nu_t) w_x]
                     - d/dy[(nu+sigma_w nu_t) w_y]
                     - gamma P_k/nu_t + beta omega^2 - CD_kw = 0

    Usage::

        residuals = KOmegaSSTResiduals(nu=1e-5, rho=1.0)
        r = residuals(model_uvpkw, x_col)
        # r is a dict with keys: momentum_x, momentum_y, continuity, k_eq, omega_eq
        loss = sum(v.pow(2).mean() for v in r.values())
    """

    def __init__(
        self,
        nu: float = 1e-5,
        rho: float = 1.0,
        consts: Optional[Dict[str, float]] = None,
    ) -> None:
        self.nu = nu          # kinematic viscosity
        self.rho = rho        # density
        self.mu = nu * rho    # dynamic viscosity
        self.c = {**SST_CONSTS, **(consts or {})}

    # ------------------------------------------------------------------
    # Blending function helpers (simplified: use inner-layer constants)
    # A full implementation requires wall-distance; here we default F1=0
    # (outer layer), giving the k-omega SST outer constants.
    # Pass consts={'F1': 1.0} to force inner-layer constants.
    # ------------------------------------------------------------------

    def _blend_const(self, name1: str, name2: str, F1: float = 0.0) -> float:
        """Blend constant: phi = F1*phi1 + (1-F1)*phi2."""
        return F1 * self.c[name1] + (1.0 - F1) * self.c[name2]

    def eddy_viscosity(
        self,
        k: torch.Tensor,
        omega: torch.Tensor,
        S_mag: torch.Tensor,
        F2: float = 1.0,
    ) -> torch.Tensor:
        """Compute turbulent viscosity mu_t (Bradshaw limiter included).

        Parameters
        ----------
        k, omega : (N,1) turbulence fields
        S_mag    : (N,1) magnitude of strain-rate tensor S = |S_ij|
        F2       : blending function (0=outer, 1=inner wall region)
        """
        a1 = self.c["a1"]
        denom = torch.clamp(
            torch.maximum(a1 * omega, S_mag * F2),
            min=1e-10,
        )
        return self.rho * a1 * k / denom

    def __call__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        x_col: torch.Tensor,
        F1: float = 0.0,
        F2: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute all RANS + k-omega SST residuals via autograd.

        Parameters
        ----------
        model  : callable (N,2) -> (N,5): outputs (u, v, p, k, omega)
        x_col  : (N,2) collocation points, will get requires_grad=True
        F1, F2 : SST blending functions (scalars; provide per-point tensors
                 if desired by monkey-patching or subclassing)

        Returns
        -------
        dict with keys: 'momentum_x', 'momentum_y', 'continuity',
                        'k_eq', 'omega_eq'
        """
        x = x_col.clone().requires_grad_(True)
        out = model(x)                          # (N, 5)
        u     = out[:, 0:1]
        v     = out[:, 1:2]
        p     = out[:, 2:3]
        k     = torch.clamp(out[:, 3:4], min=0.0)    # k >= 0
        omega = torch.clamp(out[:, 4:5], min=1e-6)   # omega > 0

        # ---- first derivatives ----
        u_grad = _grad1(u, x)              # (N, 2)
        v_grad = _grad1(v, x)
        p_grad = _grad1(p, x)
        k_grad = _grad1(k, x)
        w_grad = _grad1(omega, x)

        u_x = u_grad[:, 0:1];  u_y = u_grad[:, 1:2]
        v_x = v_grad[:, 0:1];  v_y = v_grad[:, 1:2]
        p_x = p_grad[:, 0:1];  p_y = p_grad[:, 1:2]
        k_x = k_grad[:, 0:1];  k_y = k_grad[:, 1:2]
        w_x = w_grad[:, 0:1];  w_y = w_grad[:, 1:2]

        # ---- blended constants ----
        sigma_k = self._blend_const("sigma_k1", "sigma_k2", F1)
        sigma_w = self._blend_const("sigma_w1", "sigma_w2", F1)
        beta    = self._blend_const("beta1",    "beta2",    F1)
        gamma   = self._blend_const("gamma1",   "gamma2",   F1)
        beta_s  = self.c["beta_star"]

        # ---- strain-rate magnitude |S| = sqrt(2 S_ij S_ij) ----
        # 2D: S_11=u_x, S_22=v_y, S_12=S_21=0.5(u_y+v_x)
        S_mag = torch.sqrt(
            2.0 * (u_x**2 + v_y**2 + 0.5 * (u_y + v_x)**2) + 1e-12
        )

        # ---- eddy viscosity ----
        nu_t = self.eddy_viscosity(k, omega, S_mag, F2) / self.rho  # kinematic
        mu_t = nu_t * self.rho                                        # dynamic
        mu_eff = self.mu + mu_t

        # ---- production of k: P_k = mu_t * S^2 (with realizability clip) ----
        P_k = torch.clamp(mu_t * S_mag**2, max=10.0 * beta_s * self.rho * k * omega + 1e-12)

        # ---- continuity ----
        cont = u_x + v_y

        # ---- momentum-x: rho(u u_x + v u_y) + p_x - div[(mu_eff)(grad u + ...)]=0 ----
        # viscous term (Boussinesq): div[(mu+mu_t) grad u] = mu_eff * lap(u) (simplified)
        lap_u = _laplacian(u, x)
        # grad(nu_eff) . grad(u) cross-term (d mu_eff / dx * u_x + ...)
        mu_eff_grad = _grad1(mu_eff, x)         # (N, 2)
        visc_x = mu_eff * lap_u + mu_eff_grad[:, 0:1] * u_x + mu_eff_grad[:, 1:2] * u_y

        mom_x = (
            self.rho * (u * u_x + v * u_y)
            + p_x
            - visc_x
        )

        # ---- momentum-y ----
        lap_v = _laplacian(v, x)
        visc_y = mu_eff * lap_v + mu_eff_grad[:, 0:1] * v_x + mu_eff_grad[:, 1:2] * v_y
        mom_y = (
            self.rho * (u * v_x + v * v_y)
            + p_y
            - visc_y
        )

        # ---- k-equation ----
        nu_k = self.nu + sigma_k * nu_t
        lap_k = _laplacian(k, x)
        nu_k_grad = _grad1(
            (self.nu + sigma_k * nu_t) * torch.ones_like(k), x
        )
        diff_k = (self.nu + sigma_k * nu_t) * lap_k + nu_k_grad[:, 0:1] * k_x + nu_k_grad[:, 1:2] * k_y
        k_eq = (
            u * k_x + v * k_y
            - diff_k
            - P_k / self.rho
            + beta_s * k * omega
        )

        # ---- omega-equation ----
        # cross-diffusion term CD_kw = 2 sigma_w2 / omega * (k_x w_x + k_y w_y)
        CD_kw = torch.clamp(
            2.0 * self.c["sigma_w2"] / omega * (k_x * w_x + k_y * w_y),
            min=0.0,
        )
        lap_w = _laplacian(omega, x)
        nu_w_grad = _grad1(
            (self.nu + sigma_w * nu_t) * torch.ones_like(omega), x
        )
        diff_w = (self.nu + sigma_w * nu_t) * lap_w + nu_w_grad[:, 0:1] * w_x + nu_w_grad[:, 1:2] * w_y
        omega_eq = (
            u * w_x + v * w_y
            - diff_w
            - gamma * self.rho * S_mag**2
            + beta * omega**2
            - (1.0 - F1) * CD_kw
        )

        return {
            "momentum_x": mom_x,
            "momentum_y": mom_y,
            "continuity":  cont,
            "k_eq":        k_eq,
            "omega_eq":    omega_eq,
        }


# ---------------------------------------------------------------------------
# Feature 10b: Spalart-Allmaras (SA-1992)
# ---------------------------------------------------------------------------

class SpalartAllmarasResiduals:
    """Spalart-Allmaras one-equation turbulence model residuals.

    Transport equation for modified eddy viscosity nu_tilde::

        D nu_tilde / Dt = cb1 * S_tilde * nu_tilde
                        - cw1 * fw * (nu_tilde / d)^2
                        + (1/sigma) * div[(nu + nu_tilde) grad nu_tilde]
                        + (cb2/sigma) |grad nu_tilde|^2

    Usage::

        sa = SpalartAllmarasResiduals(nu=1e-5)
        r = sa(model_uvpn, x_col, wall_distance_fn=lambda x: x[:,1:2])
        # r: dict with 'momentum_x', 'momentum_y', 'continuity', 'sa_eq'
    """

    SA_CONSTS: Dict[str, float] = {
        "cb1":   0.1355,
        "cb2":   0.622,
        "sigma": 2.0 / 3.0,
        "kappa": 0.41,
        "cw2":   0.3,
        "cw3":   2.0,
        "cv1":   7.1,
        "ct3":   1.2,
        "ct4":   0.5,
        # cw1 computed in __init__
    }

    def __init__(self, nu: float = 1e-5) -> None:
        self.nu = nu
        self.c = dict(self.SA_CONSTS)
        cb1   = self.c["cb1"]
        cb2   = self.c["cb2"]
        sigma = self.c["sigma"]
        kappa = self.c["kappa"]
        self.c["cw1"] = cb1 / kappa**2 + (1.0 + cb2) / sigma

    # ------------------------------------------------------------------
    # SA auxiliary functions
    # ------------------------------------------------------------------

    def _fv1(self, chi: torch.Tensor) -> torch.Tensor:
        cv1 = self.c["cv1"]
        return chi**3 / (chi**3 + cv1**3)

    def _fv2(self, chi: torch.Tensor, fv1: torch.Tensor) -> torch.Tensor:
        return 1.0 - chi / (1.0 + chi * fv1)

    def _fw(self, r: torch.Tensor) -> torch.Tensor:
        cw2 = self.c["cw2"]
        cw3 = self.c["cw3"]
        g = r + cw2 * (r**6 - r)
        return g * ((1.0 + cw3**6) / (g**6 + cw3**6)) ** (1.0 / 6.0)

    def __call__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        x_col: torch.Tensor,
        wall_distance_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute SA residuals.

        Parameters
        ----------
        model            : callable (N,2) -> (N,4): (u, v, p, nu_tilde)
        x_col            : (N,2) collocation points
        wall_distance_fn : callable (N,2) -> (N,1) returning wall distance d.
                           If None, uses x[:,1:2] (distance from lower wall).

        Returns
        -------
        dict with keys: 'momentum_x', 'momentum_y', 'continuity', 'sa_eq'
        """
        x = x_col.clone().requires_grad_(True)
        out = model(x)                          # (N, 4)
        u      = out[:, 0:1]
        v      = out[:, 1:2]
        p      = out[:, 2:3]
        nu_t   = torch.clamp(out[:, 3:4], min=0.0)   # nu_tilde >= 0

        # wall distance
        if wall_distance_fn is not None:
            d = torch.clamp(wall_distance_fn(x), min=1e-6)
        else:
            d = torch.clamp(x[:, 1:2], min=1e-6)

        # ---- first derivatives ----
        u_grad  = _grad1(u,    x)
        v_grad  = _grad1(v,    x)
        p_grad  = _grad1(p,    x)
        nt_grad = _grad1(nu_t, x)

        u_x = u_grad[:, 0:1];  u_y = u_grad[:, 1:2]
        v_x = v_grad[:, 0:1];  v_y = v_grad[:, 1:2]
        p_x = p_grad[:, 0:1];  p_y = p_grad[:, 1:2]
        nt_x = nt_grad[:, 0:1]; nt_y = nt_grad[:, 1:2]

        # ---- SA auxiliary ----
        chi  = nu_t / self.nu
        fv1  = self._fv1(chi)
        fv2  = self._fv2(chi, fv1)
        kappa = self.c["kappa"]

        # vorticity magnitude omega = |dv/dx - du/dy|
        vort = torch.abs(v_x - u_y) + 1e-12

        # modified strain rate S_tilde = omega + nu_tilde / (kappa^2 d^2) * fv2
        S_tilde = torch.clamp(
            vort + nu_t / (kappa**2 * d**2) * fv2,
            min=1e-10,
        )

        r = torch.clamp(nu_t / (S_tilde * kappa**2 * d**2 + 1e-12), max=10.0)
        fw = self._fw(r)

        # production / destruction
        cb1   = self.c["cb1"]
        cw1   = self.c["cw1"]
        cb2   = self.c["cb2"]
        sigma = self.c["sigma"]

        production   = cb1 * S_tilde * nu_t
        destruction  = cw1 * fw * (nu_t / d)**2

        # diffusion term: (1/sigma) div[(nu + nu_tilde) grad nu_tilde]
        nu_eff = self.nu + nu_t                   # (N,1)
        lap_nt = _laplacian(nu_t, x)
        nu_eff_grad = _grad1(nu_eff, x)           # (N, 2)
        diffusion = (1.0 / sigma) * (
            nu_eff * lap_nt
            + nu_eff_grad[:, 0:1] * nt_x
            + nu_eff_grad[:, 1:2] * nt_y
        )

        # cb2/sigma |grad nu_tilde|^2
        cross = (cb2 / sigma) * (nt_x**2 + nt_y**2)

        sa_eq = (
            u * nt_x + v * nt_y
            - production
            + destruction
            - diffusion
            - cross
        )

        # ---- RANS momentum with SA eddy viscosity ----
        nu_turb = nu_t * fv1                      # actual eddy viscosity
        nu_total = self.nu + nu_turb
        lap_u = _laplacian(u, x)
        lap_v = _laplacian(v, x)
        nu_tot_grad = _grad1(nu_total, x)

        mom_x = (
            u * u_x + v * u_y
            + p_x
            - (nu_total * lap_u + nu_tot_grad[:, 0:1] * u_x + nu_tot_grad[:, 1:2] * u_y)
        )
        mom_y = (
            u * v_x + v * v_y
            + p_y
            - (nu_total * lap_v + nu_tot_grad[:, 0:1] * v_x + nu_tot_grad[:, 1:2] * v_y)
        )
        cont = u_x + v_y

        return {
            "momentum_x": mom_x,
            "momentum_y": mom_y,
            "continuity":  cont,
            "sa_eq":       sa_eq,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_rans_preset(model_name: str = "k-omega-sst", **kwargs):
    """Get a RANS turbulence closure residual object by name.

    Parameters
    ----------
    model_name : one of 'k-omega-sst', 'spalart-allmaras', 'sa'
    **kwargs   : forwarded to the residual class constructor

    Returns
    -------
    Callable residual object (KOmegaSSTResiduals or SpalartAllmarasResiduals)

    Examples
    --------
    >>> res = get_rans_preset("k-omega-sst", nu=1e-5, rho=1.0)
    >>> res = get_rans_preset("spalart-allmaras", nu=1.5e-5)
    """
    _models = {
        "k-omega-sst":       KOmegaSSTResiduals,
        "spalart-allmaras":  SpalartAllmarasResiduals,
        "sa":                SpalartAllmarasResiduals,
    }
    if model_name not in _models:
        raise ValueError(
            f"Unknown RANS model: '{model_name}'. "
            f"Available: {sorted(_models.keys())}"
        )
    return _models[model_name](**kwargs)


__all__ = [
    "SST_CONSTS",
    "KOmegaSSTResiduals",
    "SpalartAllmarasResiduals",
    "get_rans_preset",
]
