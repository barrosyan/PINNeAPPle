"""Physics losses for the Schrödinger equation and quantum Hamiltonians."""
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn


# ── Built-in potential functions ──────────────────────────────────────────────

def harmonic_potential(x: torch.Tensor, omega: float = 1.0, mass: float = 1.0) -> torch.Tensor:
    """V(x) = ½mω²‖x‖² — quantum harmonic oscillator."""
    return 0.5 * mass * omega ** 2 * (x ** 2).sum(dim=-1)


def infinite_well_potential(x: torch.Tensor, width: float = 1.0) -> torch.Tensor:
    """V(x) = 0 inside (0, L), handled via Dirichlet BCs. Returns 0 in domain."""
    return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)


def coulomb_potential(x: torch.Tensor, charge: float = 1.0, eps: float = 1e-3) -> torch.Tensor:
    """V(r) = −e²/r — hydrogen-like atom potential (regularized at origin)."""
    r = (x ** 2).sum(dim=-1).sqrt().clamp_min(eps)
    return -charge / r


def double_well_potential(x: torch.Tensor, a: float = 1.0, b: float = 0.0) -> torch.Tensor:
    """V(x) = (x² − a)² + bx — symmetric / asymmetric double well."""
    x1 = x[..., 0]
    return (x1 ** 2 - a) ** 2 + b * x1


POTENTIALS = {
    "harmonic_oscillator": harmonic_potential,
    "harmonic":            harmonic_potential,
    "infinite_well":       infinite_well_potential,
    "coulomb":             coulomb_potential,
    "double_well":         double_well_potential,
}


def get_potential(name_or_fn) -> Callable:
    """Resolve a potential by name string or return it directly if callable."""
    if callable(name_or_fn):
        return name_or_fn
    key = name_or_fn.lower().replace(" ", "_")
    if key not in POTENTIALS:
        raise KeyError(
            f"Unknown potential {name_or_fn!r}. "
            f"Available: {list(POTENTIALS.keys())}"
        )
    return POTENTIALS[key]


# ── Autograd helpers ──────────────────────────────────────────────────────────

def _laplacian(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute ∇²u via second-order autograd, summed over spatial dimensions.

    Parameters
    ----------
    u : (N,) or (N, 1)
    x : (N, dim), requires_grad=True

    Returns
    -------
    (N,) — ∇²u at each collocation point
    """
    if u.ndim > 1:
        u = u.squeeze(-1)
    grads = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True,
    )[0]  # (N, dim)

    lap = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    for i in range(x.shape[1]):
        lap_i = torch.autograd.grad(
            grads[:, i], x,
            grad_outputs=torch.ones_like(grads[:, i]),
            create_graph=True, retain_graph=True,
        )[0][:, i]
        lap = lap + lap_i
    return lap


# ── Physics losses ────────────────────────────────────────────────────────────

def energy_loss(
    model: nn.Module,
    x: torch.Tensor,
    potential: str | Callable = "harmonic_oscillator",
    hbar: float = 1.0,
    mass: float = 1.0,
    **potential_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Variational energy (Rayleigh quotient) for the time-independent Schrödinger equation.

    Minimising this loss drives the model toward the ground-state wavefunction
    ψ₀(x) — the standard Variational Quantum Eigensolver (VQE) objective.

    .. math::

        E[ψ] = \\frac{\\langle ψ | H | ψ \\rangle}{\\langle ψ | ψ \\rangle}
              = \\frac{\\int ψ^* \\left(-\\frac{ℏ^2}{2m}∇^2 + V\\right) ψ \\,dx}
                      {\\int |ψ|^2 \\,dx}

    Parameters
    ----------
    model : nn.Module
        Wavefunction approximator ψ(x). Should output a real or complex scalar.
    x : Tensor (N, dim)
        Collocation points (domain interior). Must *not* have requires_grad set;
        it is cloned and grad-enabled internally.
    potential : str or Callable
        Potential function V(x) → (N,). Built-in names: ``"harmonic_oscillator"``,
        ``"coulomb"``, ``"infinite_well"``, ``"double_well"``.
    hbar : float
        Reduced Planck constant (natural units: 1.0).
    mass : float
        Particle mass (natural units: 1.0).
    **potential_kwargs
        Extra keyword arguments forwarded to the potential function.

    Returns
    -------
    energy : scalar Tensor
        Variational energy estimate ⟨H⟩ (minimise this).
    norm_sq : scalar Tensor
        ∫|ψ|² dx (for diagnostics; should stay close to 1).
    """
    V_fn = get_potential(potential)
    x_col = x.clone().requires_grad_(True)

    psi = model(x_col)
    if psi.ndim > 1:
        psi = psi.squeeze(-1)

    psi_sq = psi ** 2
    norm_sq = psi_sq.mean()

    # Kinetic term: T = −ℏ²/(2m) ∇²ψ
    lap = _laplacian(psi, x_col)                          # (N,)
    T_integrand = -(hbar ** 2) / (2.0 * mass) * lap * psi  # ψ·(−ℏ²/2m ∇²ψ)

    # Potential term: V(x)|ψ|²
    V_vals = V_fn(x_col.detach(), **potential_kwargs)
    V_integrand = V_vals * psi_sq

    # Rayleigh quotient (Monte Carlo integration = mean)
    energy = (T_integrand + V_integrand).mean() / (norm_sq + 1e-10)
    return energy, norm_sq


def pde_residual_loss(
    model: nn.Module,
    x: torch.Tensor,
    eigenvalue: float,
    potential: str | Callable = "harmonic_oscillator",
    hbar: float = 1.0,
    mass: float = 1.0,
    **potential_kwargs,
) -> torch.Tensor:
    """
    Strong-form PDE residual for the time-independent Schrödinger equation.

    .. math::

        \\mathcal{L}_{\\text{PDE}} = \\left\\|
            -\\frac{ℏ^2}{2m}∇^2 ψ + V ψ - E ψ
        \\right\\|^2

    Parameters
    ----------
    model : nn.Module
        Wavefunction approximator ψ(x).
    x : Tensor (N, dim)
        Collocation points.
    eigenvalue : float
        Target energy eigenvalue E. Use ``energy_loss`` to estimate this.
    potential, hbar, mass, **potential_kwargs
        As in :func:`energy_loss`.

    Returns
    -------
    scalar Tensor — mean squared PDE residual.
    """
    V_fn = get_potential(potential)
    x_col = x.clone().requires_grad_(True)

    psi = model(x_col)
    if psi.ndim > 1:
        psi = psi.squeeze(-1)

    lap = _laplacian(psi, x_col)
    V_vals = V_fn(x_col.detach(), **potential_kwargs)

    # Hψ − Eψ = (−ℏ²/2m ∇²ψ + Vψ) − Eψ
    residual = -(hbar ** 2) / (2.0 * mass) * lap + V_vals * psi - eigenvalue * psi
    return residual.pow(2).mean()


def time_dependent_loss(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    potential: str | Callable = "harmonic_oscillator",
    hbar: float = 1.0,
    mass: float = 1.0,
    **potential_kwargs,
) -> torch.Tensor:
    """
    PDE residual for the time-dependent Schrödinger equation.

    .. math::

        \\mathcal{L} = \\left\\| i ℏ \\frac{∂ψ}{∂t} - H ψ \\right\\|^2

    For real-valued networks, decomposes ψ = ψ_R + i ψ_I and solves the
    coupled real system:
        ℏ ∂ψ_R/∂t = H ψ_I
        ℏ ∂ψ_I/∂t = −H ψ_R

    Parameters
    ----------
    model : nn.Module
        Wavefunction approximator ``model(xt) → (N, 2)`` where
        output[:, 0] = ψ_R, output[:, 1] = ψ_I.
    x : Tensor (N, dim)
    t : Tensor (N, 1)
    potential, hbar, mass, **potential_kwargs
        As in :func:`energy_loss`.

    Returns
    -------
    scalar Tensor — mean squared residual (real + imaginary parts).
    """
    V_fn = get_potential(potential)
    x_col = x.clone().requires_grad_(True)
    t_col = t.clone().requires_grad_(True)

    xt = torch.cat([x_col, t_col], dim=-1)
    psi = model(xt)

    if psi.shape[-1] != 2:
        raise ValueError(
            "time_dependent_loss expects model output of shape (N, 2): "
            "[ψ_real, ψ_imag]"
        )

    psi_R, psi_I = psi[:, 0], psi[:, 1]
    V_vals = V_fn(x_col.detach(), **potential_kwargs)

    def _H(u):
        """Apply −ℏ²/2m ∇²u + Vu."""
        lap = _laplacian(u, x_col)
        return -(hbar ** 2) / (2.0 * mass) * lap + V_vals * u

    H_psiR = _H(psi_R)
    H_psiI = _H(psi_I)

    # Time derivatives via autograd
    dpsiR_dt = torch.autograd.grad(
        psi_R, t_col,
        grad_outputs=torch.ones_like(psi_R),
        create_graph=True, retain_graph=True,
    )[0].squeeze(-1)

    dpsiI_dt = torch.autograd.grad(
        psi_I, t_col,
        grad_outputs=torch.ones_like(psi_I),
        create_graph=True, retain_graph=True,
    )[0].squeeze(-1)

    # iℏ∂ψ/∂t = Hψ  ⟺  ℏ∂ψ_R/∂t = H ψ_I,  ℏ∂ψ_I/∂t = −H ψ_R
    res_R = hbar * dpsiR_dt - H_psiI
    res_I = hbar * dpsiI_dt + H_psiR
    return (res_R.pow(2) + res_I.pow(2)).mean()


def normalization_loss(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Soft constraint enforcing wavefunction normalization: ∫|ψ|² dx ≈ 1.

    Approximated via Monte Carlo: E[|ψ(x)|²] ≈ 1 / |Ω| where |Ω| is the
    domain measure. For a unit domain this reduces to mean(ψ²) ≈ 1.

    Returns
    -------
    scalar Tensor — (mean(ψ²) − 1)²
    """
    psi = model(x)
    if psi.ndim > 1:
        psi = psi.squeeze(-1)
    return (psi.pow(2).mean() - 1.0).pow(2)


def hamiltonian_expectation(
    model: nn.Module,
    x: torch.Tensor,
    H_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Discrete Hamiltonian expectation ⟨ψ|H|ψ⟩ for matrix Hamiltonians.

    Useful for spin systems, tight-binding models, or any problem where the
    Hamiltonian is given as a finite-dimensional operator.

    Parameters
    ----------
    model : nn.Module
        Outputs a state vector ψ ∈ ℝᵈ (or ℂᵈ).
    x : Tensor (N, input_dim)
        Batch of inputs (each produces one state vector).
    H_matrix : Tensor (d, d)
        Hamiltonian matrix (should be Hermitian).

    Returns
    -------
    scalar Tensor — mean energy ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩
    """
    psi = model(x)  # (N, d)
    norm_sq = (psi ** 2).sum(dim=-1, keepdim=True).clamp_min(1e-10)  # (N, 1)
    psi_norm = psi / norm_sq.sqrt()
    H_psi = psi_norm @ H_matrix.T  # (N, d)
    energy_per_sample = (psi_norm * H_psi).sum(dim=-1)  # (N,)
    return energy_per_sample.mean()
