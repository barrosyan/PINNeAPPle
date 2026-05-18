"""Sensitivity analysis for inverse problems.

Quantifies how sensitive the observables are to each physical parameter,
providing essential information for:
- Identifiability analysis (can parameters be recovered from available data?)
- Experimental design (which sensors / measurements are most informative?)
- Uncertainty propagation (which parameters drive output uncertainty?)
- Prioritizing regularization (ill-conditioned directions need stronger reg)

Three levels of analysis
------------------------
LocalSensitivity       Jacobian J = ∂G/∂θ at a nominal point; FIM = J^T J / σ²
IdentifiabilityAnalyzer FIM eigenanalysis; D/A/E-optimality criteria
GlobalSensitivity      Sobol variance-based indices via Saltelli's method

References
----------
- Saltelli et al. (2010) "Variance based sensitivity analysis of model output"
- Kaipio & Somersalo (2005) "Statistical and Computational Inverse Problems"
- Pukelsheim (1993) "Optimal Design of Experiments" (D/A/E-optimality)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Local sensitivity analysis (Jacobian + Fisher information)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LocalSensitivityResult:
    """Results of local sensitivity analysis.

    Attributes
    ----------
    jacobian : np.ndarray, shape (k, p)
        J = ∂G/∂θ at the nominal parameter values.
    fisher_information : np.ndarray, shape (p, p)
        FIM = J^T Γ⁻¹ J.
    singular_values : np.ndarray, shape (min(k,p),)
        Singular values of J.
    sensitivity_indices : np.ndarray, shape (p,)
        Per-parameter sensitivity: ‖Jᵢ‖ / ‖J‖_F  (column norms).
    param_names : list of str
        Names of the parameters in order.
    """
    jacobian: np.ndarray
    fisher_information: np.ndarray
    singular_values: np.ndarray
    sensitivity_indices: np.ndarray
    param_names: List[str]


class LocalSensitivity:
    """Local (linearized) sensitivity analysis around a nominal parameter point.

    Computes the Jacobian J = ∂G/∂θ using PyTorch's reverse-mode AD, then
    derives the Fisher Information Matrix (FIM) for parameter identifiability.

    Parameters
    ----------
    forward_fn : callable(theta: Tensor) -> Tensor
        Maps parameter vector θ ∈ R^p to observable vector G(θ) ∈ R^k.
    noise_std : float or np.ndarray
        Observation noise standard deviation (scalar or (k,) array).
        Used to form Γ⁻¹ = diag(1/σ²) in the FIM.
    param_names : list of str, optional
        Names of the p parameters (for reporting).
    """

    def __init__(
        self,
        forward_fn: Callable[[torch.Tensor], torch.Tensor],
        noise_std: float = 1.0,
        param_names: Optional[List[str]] = None,
    ) -> None:
        self.forward_fn = forward_fn
        self.noise_std = noise_std
        self.param_names = param_names

    def compute(self, theta_nominal: torch.Tensor) -> LocalSensitivityResult:
        """Compute local sensitivity at ``theta_nominal``.

        Parameters
        ----------
        theta_nominal : torch.Tensor, shape (p,)
            Nominal parameter values θ₀.

        Returns
        -------
        LocalSensitivityResult
        """
        theta = theta_nominal.clone().detach().requires_grad_(True)

        # Jacobian via torch.autograd.functional.jacobian.
        # Shape is (output_shape, param_shape); flatten to (k, p).
        J_t = torch.autograd.functional.jacobian(
            self.forward_fn, theta, create_graph=False, strict=False
        )
        p = theta.numel()
        J = J_t.detach().reshape(-1, p).numpy()   # (k, p)

        # Noise precision (diagonal Γ⁻¹)
        k = J.shape[0]
        p = J.shape[1]
        if np.isscalar(self.noise_std):
            prec = np.ones(k) / self.noise_std ** 2
        else:
            prec = 1.0 / np.asarray(self.noise_std) ** 2

        # Fisher Information Matrix F = J^T diag(prec) J
        JT_prec = J.T * prec[np.newaxis, :]   # (p, k)
        FIM = JT_prec @ J                       # (p, p)

        # SVD of J for sensitivity indices
        _, sv, _ = np.linalg.svd(J, full_matrices=False)

        # Per-parameter sensitivity: column norms of J
        col_norms = np.linalg.norm(J, axis=0)   # (p,)
        total = np.linalg.norm(col_norms) + 1e-30
        sensitivity_idx = col_norms / total

        names = self.param_names or [f"theta_{i}" for i in range(p)]

        return LocalSensitivityResult(
            jacobian=J,
            fisher_information=FIM,
            singular_values=sv,
            sensitivity_indices=sensitivity_idx,
            param_names=names,
        )

    def finite_difference_jacobian(
        self,
        theta_nominal: np.ndarray,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Finite-difference Jacobian (fallback; no autograd required).

        Returns shape (k, p).
        """
        p = len(theta_nominal)
        theta0 = torch.tensor(theta_nominal, dtype=torch.float32)
        g0 = self.forward_fn(theta0).detach().numpy()
        k = g0.size
        J = np.zeros((k, p))
        for i in range(p):
            theta_p = theta0.clone()
            theta_p[i] += eps
            gp = self.forward_fn(theta_p).detach().numpy()
            J[:, i] = (gp - g0) / eps
        return J


# ──────────────────────────────────────────────────────────────────────────────
# Identifiability analyzer
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IdentifiabilityResult:
    """Output of identifiability analysis.

    Attributes
    ----------
    eigenvalues : np.ndarray, shape (p,)
        Eigenvalues of FIM, ascending order.
    eigenvectors : np.ndarray, shape (p, p)
        Columns = eigenvectors of FIM.
    condition_number : float
        κ(FIM) = λ_max / λ_min (large → ill-conditioned).
    d_optimality : float
        det(FIM) — larger is better (D-optimal design criterion).
    a_optimality : float
        trace(FIM⁻¹) — smaller is better (A-optimal).
    e_optimality : float
        λ_min(FIM) — larger is better (E-optimal).
    identifiable_indices : list of int
        Indices of *identifiable* parameters (eigenvalue > tol).
    unidentifiable_indices : list of int
        Indices of parameters that cannot be recovered from the data.
    param_names : list of str
    """
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    condition_number: float
    d_optimality: float
    a_optimality: float
    e_optimality: float
    identifiable_indices: List[int]
    unidentifiable_indices: List[int]
    param_names: List[str]

    def report(self) -> str:
        lines = ["Identifiability Analysis", "=" * 40]
        lines.append(f"  Condition number : {self.condition_number:.3e}")
        lines.append(f"  D-optimality     : {self.d_optimality:.3e}")
        lines.append(f"  A-optimality     : {self.a_optimality:.3e}")
        lines.append(f"  E-optimality     : {self.e_optimality:.3e}")
        lines.append(f"  Identifiable     : {[self.param_names[i] for i in self.identifiable_indices]}")
        lines.append(f"  Not identifiable : {[self.param_names[i] for i in self.unidentifiable_indices]}")
        return "\n".join(lines)


class IdentifiabilityAnalyzer:
    """Analyse parameter identifiability from the Fisher Information Matrix.

    Parameters
    ----------
    tol : float
        Eigenvalue threshold: parameters with eigenvalue < tol · λ_max are
        considered unidentifiable (zero in effective rank sense).
    """

    def __init__(self, tol: float = 1e-6) -> None:
        self.tol = float(tol)

    def analyze(
        self,
        fim: np.ndarray,
        param_names: Optional[List[str]] = None,
    ) -> IdentifiabilityResult:
        """Perform identifiability analysis on a Fisher Information Matrix.

        Parameters
        ----------
        fim : np.ndarray, shape (p, p)
        param_names : list of str, optional
        """
        p = fim.shape[0]
        names = param_names or [f"theta_{i}" for i in range(p)]

        # Eigen-decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(fim)  # ascending order
        eigenvalues = np.maximum(eigenvalues, 0.0)        # clamp numerical negatives

        lam_max = eigenvalues[-1] if eigenvalues[-1] > 0 else 1e-30
        lam_min = eigenvalues[0]

        threshold = self.tol * lam_max
        identifiable = [i for i, ev in enumerate(eigenvalues) if ev > threshold]
        not_identifiable = [i for i, ev in enumerate(eigenvalues) if ev <= threshold]

        # D-optimality: det(FIM); use sum of log eigenvalues for numerical stability
        pos_eigs = eigenvalues[eigenvalues > 0]
        d_opt = float(np.exp(np.sum(np.log(pos_eigs)))) if len(pos_eigs) > 0 else 0.0

        # A-optimality: trace(FIM^{-1})
        try:
            fim_inv = np.linalg.pinv(fim, rcond=self.tol)
            a_opt = float(np.trace(fim_inv))
        except np.linalg.LinAlgError:
            a_opt = float("inf")

        # E-optimality: smallest eigenvalue
        e_opt = float(lam_min)

        # Condition number
        cond = float(lam_max / max(lam_min, 1e-30))

        return IdentifiabilityResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            condition_number=cond,
            d_optimality=d_opt,
            a_optimality=a_opt,
            e_optimality=e_opt,
            identifiable_indices=identifiable,
            unidentifiable_indices=not_identifiable,
            param_names=names,
        )

    def effective_rank(self, fim: np.ndarray) -> int:
        """Number of identifiable parameters (eigenvalues above threshold)."""
        result = self.analyze(fim)
        return len(result.identifiable_indices)


# ──────────────────────────────────────────────────────────────────────────────
# Global sensitivity: Sobol variance-based indices
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SobolResult:
    """Sobol sensitivity indices.

    Attributes
    ----------
    S1 : np.ndarray, shape (p,)
        First-order (main effect) indices: fraction of variance explained by θᵢ alone.
    ST : np.ndarray, shape (p,)
        Total effect indices: fraction including all interactions with θᵢ.
    param_names : list of str
    n_samples : int
        Number of Monte Carlo samples used.
    """
    S1: np.ndarray
    ST: np.ndarray
    param_names: List[str]
    n_samples: int

    def report(self) -> str:
        lines = ["Global Sensitivity (Sobol indices)", "=" * 40]
        lines.append(f"  {'Parameter':<20}  S1 (main)   ST (total)")
        lines.append("  " + "-" * 38)
        for name, s1, st in zip(self.param_names, self.S1, self.ST):
            lines.append(f"  {name:<20}  {s1:.4f}      {st:.4f}")
        return "\n".join(lines)


class GlobalSensitivity:
    """Variance-based global sensitivity analysis (Sobol indices).

    Uses Saltelli's (2010) estimator with two random matrices A, B.

    Parameters
    ----------
    forward_fn : callable(theta_batch: np.ndarray) -> np.ndarray
        Vectorized forward model; input (N, p), output (N,) or (N, k).
        If output is (N, k), Sobol indices are averaged over outputs.
    param_bounds : list of (lo, hi)
        Uniform prior bounds for each parameter.
    n_samples : int
        Base sample count N.  Total evaluations = N * (p + 2).
    seed : int, optional
    """

    def __init__(
        self,
        forward_fn: Callable[[np.ndarray], np.ndarray],
        param_bounds: List[Tuple[float, float]],
        n_samples: int = 1024,
        seed: Optional[int] = None,
    ) -> None:
        self.forward_fn = forward_fn
        self.param_bounds = param_bounds
        self.n_samples = int(n_samples)
        self.seed = seed

    def compute(
        self,
        param_names: Optional[List[str]] = None,
    ) -> SobolResult:
        """Run Sobol analysis and return indices.

        Returns
        -------
        SobolResult
        """
        rng = np.random.default_rng(self.seed)
        p = len(self.param_bounds)
        N = self.n_samples
        names = param_names or [f"theta_{i}" for i in range(p)]

        lo = np.array([b[0] for b in self.param_bounds])
        hi = np.array([b[1] for b in self.param_bounds])

        # Sample two independent N×p matrices in [0,1], then scale
        A = rng.random((N, p)) * (hi - lo) + lo
        B = rng.random((N, p)) * (hi - lo) + lo

        # Evaluate A and B
        yA = self._eval(A)   # (N,)
        yB = self._eval(B)   # (N,)

        total_var = np.var(np.concatenate([yA, yB]))
        if total_var < 1e-30:
            warnings.warn("GlobalSensitivity: near-zero variance — model may be constant.")
            total_var = 1.0

        S1 = np.zeros(p)
        ST = np.zeros(p)

        for i in range(p):
            # AB_i: matrix A with column i replaced by B's column i
            AB_i = A.copy()
            AB_i[:, i] = B[:, i]
            yAB_i = self._eval(AB_i)

            # BA_i: matrix B with column i replaced by A's column i
            BA_i = B.copy()
            BA_i[:, i] = A[:, i]
            yBA_i = self._eval(BA_i)

            # Saltelli (2010) estimators
            S1[i] = float(np.mean(yB * (yAB_i - yA)) / total_var)
            ST[i] = float(np.mean((yA - yAB_i) ** 2) / (2 * total_var))

        # Clamp to [0, 1]
        S1 = np.clip(S1, 0, 1)
        ST = np.clip(ST, 0, 1)

        return SobolResult(S1=S1, ST=ST, param_names=names, n_samples=N)

    def _eval(self, theta_batch: np.ndarray) -> np.ndarray:
        """Evaluate forward model and reduce to scalar per sample."""
        out = self.forward_fn(theta_batch)
        out = np.asarray(out)
        if out.ndim > 1:
            out = out.mean(axis=1)
        return out.ravel()


__all__ = [
    "LocalSensitivity",
    "LocalSensitivityResult",
    "IdentifiabilityAnalyzer",
    "IdentifiabilityResult",
    "GlobalSensitivity",
    "SobolResult",
]
