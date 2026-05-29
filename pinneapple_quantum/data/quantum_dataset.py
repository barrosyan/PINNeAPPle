"""Quantum dataset layer — states, operators, measurements, and domain samplers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


# ── Quantum state representation ──────────────────────────────────────────────

@dataclass
class QuantumState:
    """
    A pure quantum state ψ(x) over a spatial domain.

    Attributes
    ----------
    coords : Tensor (N, dim)
        Spatial coordinates.
    psi_real : Tensor (N,)
        Real part of the wavefunction.
    psi_imag : Tensor (N,), optional
        Imaginary part. None for real-valued wavefunctions.
    meta : dict
        Metadata (equation kind, parameters, etc.).
    """
    coords:   torch.Tensor
    psi_real: torch.Tensor
    psi_imag: Optional[torch.Tensor] = None
    meta:     Dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.meta is None:
            self.meta = {}

    @property
    def is_complex(self) -> bool:
        return self.psi_imag is not None

    @property
    def norm_sq(self) -> torch.Tensor:
        """∫|ψ|² dx approximated by mean(|ψ|²)."""
        psi2 = self.psi_real ** 2
        if self.psi_imag is not None:
            psi2 = psi2 + self.psi_imag ** 2
        return psi2.mean()

    def to_tensor(self) -> torch.Tensor:
        """Return ψ as a real-valued Tensor of shape (N, 2) [real, imag] or (N, 1)."""
        if self.psi_imag is not None:
            return torch.stack([self.psi_real, self.psi_imag], dim=-1)
        return self.psi_real.unsqueeze(-1)


@dataclass
class QuantumObservable:
    """
    A measured observable produced by a quantum circuit.

    Attributes
    ----------
    expectation : Tensor (n_observables,) or (batch, n_observables)
        ⟨ψ|Oᵢ|ψ⟩ for each observable Oᵢ.
    variance : Tensor, optional
        ⟨O²⟩ − ⟨O⟩² for each observable.
    operator_names : list of str
        Human-readable names for each observable (e.g. ``["Z₀", "Z₁"]``).
    """
    expectation:    torch.Tensor
    variance:       Optional[torch.Tensor] = None
    operator_names: List[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.operator_names is None:
            n = self.expectation.shape[-1] if self.expectation.ndim > 0 else 1
            self.operator_names = [f"Z{i}" for i in range(n)]


# ── Domain samplers ───────────────────────────────────────────────────────────

def sample_domain_1d(
    x_min: float = 0.0,
    x_max: float = 1.0,
    n_col: int = 1000,
    n_bc: int = 50,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample collocation and boundary points for a 1D quantum domain.

    Returns
    -------
    x_col : Tensor (n_col, 1) — interior collocation points
    x_bc  : Tensor (n_bc, 1) — boundary / initial condition points
    """
    x_col = torch.rand(n_col, 1, device=device, dtype=dtype) * (x_max - x_min) + x_min
    # Boundary: endpoints and their neighborhoods
    x_bc  = torch.cat([
        torch.full((n_bc // 2, 1), x_min, device=device, dtype=dtype),
        torch.full((n_bc - n_bc // 2, 1), x_max, device=device, dtype=dtype),
    ])
    return x_col, x_bc


def sample_domain_2d(
    x_min: float = -2.0, x_max: float = 2.0,
    y_min: float = -2.0, y_max: float = 2.0,
    n_col: int = 4096,
    n_bc: int = 200,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample collocation and boundary points for a 2D quantum domain.

    Returns
    -------
    x_col : Tensor (n_col, 2)
    x_bc  : Tensor (n_bc, 2)
    """
    xy = torch.stack([
        torch.rand(n_col, device=device, dtype=dtype) * (x_max - x_min) + x_min,
        torch.rand(n_col, device=device, dtype=dtype) * (y_max - y_min) + y_min,
    ], dim=-1)

    # Boundary: four edges
    n_edge = n_bc // 4
    t = torch.rand(n_edge, device=device, dtype=dtype)
    edges = [
        torch.stack([x_min + t * (x_max - x_min), torch.full_like(t, y_min)], dim=-1),
        torch.stack([x_min + t * (x_max - x_min), torch.full_like(t, y_max)], dim=-1),
        torch.stack([torch.full_like(t, x_min), y_min + t * (y_max - y_min)], dim=-1),
        torch.stack([torch.full_like(t, x_max), y_min + t * (y_max - y_min)], dim=-1),
    ]
    x_bc = torch.cat(edges[:n_bc // n_edge], dim=0)[:n_bc]
    return xy, x_bc


def exact_eigenstates_1d_harmonic(
    x: torch.Tensor,
    n_state: int = 0,
    omega: float = 1.0,
    hbar: float = 1.0,
    mass: float = 1.0,
) -> Tuple[torch.Tensor, float]:
    """
    Analytically compute the n-th eigenstate of the 1D harmonic oscillator.

    ψ_n(x) = Nₙ Hₙ(α x) exp(−α²x²/2)
    Eₙ = ℏω(n + ½)

    where α = sqrt(mω/ℏ), Hₙ = Hermite polynomial.

    Parameters
    ----------
    x : Tensor (N, 1) or (N,)
    n_state : int
        Quantum number (0 = ground state).

    Returns
    -------
    psi : Tensor (N,) — normalized wavefunction values
    E_n : float       — exact energy eigenvalue
    """
    import math

    x_flat = x.squeeze(-1) if x.ndim > 1 else x
    alpha  = math.sqrt(mass * omega / hbar)
    xi     = alpha * x_flat.cpu().float()

    # Hermite polynomial via recursion
    if n_state == 0:
        H = torch.ones_like(xi)
    elif n_state == 1:
        H = 2.0 * xi
    else:
        H_prev2 = torch.ones_like(xi)
        H_prev1 = 2.0 * xi
        for k in range(2, n_state + 1):
            H = 2.0 * xi * H_prev1 - 2.0 * (k - 1) * H_prev2
            H_prev2, H_prev1 = H_prev1, H

    norm = (alpha / math.sqrt(math.pi)) ** 0.5 / math.sqrt(
        2 ** n_state * math.factorial(n_state)
    )
    psi = norm * H * torch.exp(-0.5 * xi ** 2)
    E_n = hbar * omega * (n_state + 0.5)
    return psi.to(x.device, dtype=x.dtype), float(E_n)


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class QuantumCollocationDataset(Dataset):
    """
    Dataset of domain collocation points for quantum PINN training.

    Stores (x_col, x_bc, psi_exact) where ``psi_exact`` is optional
    analytic reference data for supervised pre-training or validation.

    Parameters
    ----------
    x_col : Tensor (N_col, dim)
        Interior collocation points.
    x_bc : Tensor (N_bc, dim)
        Boundary / initial condition points.
    psi_exact : Tensor (N_col,), optional
        Analytic wavefunction values at x_col (for supervised guidance).
    batch_size : int
        Number of collocation points per mini-batch.
    """

    def __init__(
        self,
        x_col: torch.Tensor,
        x_bc: torch.Tensor,
        psi_exact: Optional[torch.Tensor] = None,
        batch_size: int = 256,
    ):
        self.x_col      = x_col
        self.x_bc       = x_bc
        self.psi_exact  = psi_exact
        self.batch_size = batch_size
        self._n = len(x_col)

    def __len__(self) -> int:
        return max(1, self._n // self.batch_size)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Random mini-batch of collocation points
        perm  = torch.randperm(self._n)[: self.batch_size]
        batch = {"x_col": self.x_col[perm], "x_bc": self.x_bc}
        if self.psi_exact is not None:
            batch["psi_exact"] = self.psi_exact[perm]
        return batch


class SpinChainDataset(Dataset):
    """
    Dataset of spin-chain configurations and their energy labels.

    Useful for Hamiltonian learning tasks where many-body ground states
    must be approximated from measurement data.

    Parameters
    ----------
    n_spins : int
        Number of qubits / spin sites.
    n_samples : int
        Number of random spin configurations.
    J : float
        Ising coupling strength.
    h : float
        Transverse field strength.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        n_spins: int = 4,
        n_samples: int = 1024,
        J: float = 1.0,
        h: float = 0.5,
        seed: int = 42,
    ):
        self.n_spins   = n_spins
        self.J         = J
        self.h         = h

        rng = torch.Generator().manual_seed(seed)
        # Random ±1 spin configurations
        self.configs = (torch.randint(0, 2, (n_samples, n_spins),
                                      generator=rng).float() * 2 - 1)
        # Ising energy: E = -J Σ sᵢsᵢ₊₁ − h Σ sᵢ
        nn_coupling = (self.configs[:, :-1] * self.configs[:, 1:]).sum(dim=-1)
        field_term  = self.configs.sum(dim=-1)
        self.energies = -J * nn_coupling - h * field_term  # (n_samples,)

    def __len__(self) -> int:
        return len(self.configs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "config": self.configs[idx],
            "energy": self.energies[idx].unsqueeze(0),
        }
