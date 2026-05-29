"""Conservation law checks for trained PINN models.

All checks use Monte Carlo integration over the domain and PyTorch autograd
for computing spatial derivatives.  ``scipy`` is used for quasi-random
(Sobol) sampling when available, falling back to uniform random otherwise.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .core import CheckResult

# ---------------------------------------------------------------------------
# Optional scipy for Sobol low-discrepancy sampling
# ---------------------------------------------------------------------------
try:
    from scipy.stats.qmc import Sobol as _Sobol  # type: ignore

    def _sample_domain(
        n_points: int, n_dims: int, bounds: List[Tuple[float, float]], device: str
    ) -> Tensor:
        """Return quasi-random Sobol samples mapped to *bounds*."""
        sampler = _Sobol(d=n_dims, scramble=True)
        raw = sampler.random(n_points)  # (n, d) in [0, 1)
        import numpy as np  # already available if scipy is
        lo = torch.tensor([b[0] for b in bounds], dtype=torch.float32)
        hi = torch.tensor([b[1] for b in bounds], dtype=torch.float32)
        pts = torch.from_numpy(raw.astype("float32")) * (hi - lo) + lo
        return pts.to(device)

except ImportError:  # pragma: no cover

    def _sample_domain(  # type: ignore[misc]
        n_points: int, n_dims: int, bounds: List[Tuple[float, float]], device: str
    ) -> Tensor:
        """Return uniform random samples mapped to *bounds*."""
        lo = torch.tensor([b[0] for b in bounds], dtype=torch.float32, device=device)
        hi = torch.tensor([b[1] for b in bounds], dtype=torch.float32, device=device)
        return torch.rand(n_points, n_dims, device=device) * (hi - lo) + lo


def _domain_volume(bounds: List[Tuple[float, float]]) -> float:
    """Return the hyper-volume of the axis-aligned bounding box."""
    vol = 1.0
    for lo, hi in bounds:
        vol *= hi - lo
    return vol


def _forward_tensor(model: object, x: Tensor) -> Tensor:
    """Call the model and extract a plain Tensor regardless of output type."""
    out = model(x)  # type: ignore[operator]
    if isinstance(out, Tensor):
        return out
    # Support PINNOutput / OperatorOutput with a `.y` attribute
    if hasattr(out, "y"):
        return out.y
    raise TypeError(
        f"Model returned {type(out)!r}; expected Tensor or an object with a `.y` attribute."
    )


class ConservationCheck:
    """Checks conservation laws via numerical integration over the domain.

    Parameters
    ----------
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, …).
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_mass_conservation(
        self,
        model: object,
        coord_names: List[str],
        domain_bounds: Dict[str, Tuple[float, float]],
        field_name: str = "u",
        n_points: int = 10_000,
        tolerance: float = 1e-3,
    ) -> CheckResult:
        """Check mass conservation (∇·u ≈ 0) for incompressible flow.

        Computes the mean absolute divergence over Monte Carlo sample points.
        For scalar ``u`` this reduces to |∂u/∂x₁ + … + ∂u/∂xₙ|.

        Parameters
        ----------
        model:
            Trained model with ``forward`` / callable interface.
        coord_names:
            Ordered list of coordinate names matching the model input columns.
        domain_bounds:
            Dict mapping each coordinate name to ``(lo, hi)`` bounds.
        field_name:
            Name of the conserved field (informational only).
        n_points:
            Number of Monte Carlo sample points.
        tolerance:
            Acceptance threshold on the mean |∇·u|.

        Returns
        -------
        CheckResult
        """
        bounds = [domain_bounds[c] for c in coord_names]
        n_dims = len(coord_names)
        x = _sample_domain(n_points, n_dims, bounds, self.device)
        x.requires_grad_(True)

        u = _forward_tensor(model, x)  # (n, D) or (n,)
        if u.dim() == 1:
            u = u.unsqueeze(-1)  # treat scalar as (n, 1)

        # Compute divergence: sum_i ∂u_i/∂x_i  (component i of output wrt coord i)
        n_out = u.shape[-1]
        n_div_components = min(n_dims, n_out)
        div = torch.zeros(n_points, device=self.device)
        for i in range(n_div_components):
            grad_i = torch.autograd.grad(
                u[:, i].sum(), x, create_graph=False, retain_graph=True
            )[0]  # (n, n_dims)
            div = div + grad_i[:, i]

        mean_abs_div = div.abs().mean().item()
        passed = mean_abs_div <= tolerance
        return CheckResult(
            name="mass_conservation",
            passed=passed,
            value=mean_abs_div,
            threshold=tolerance,
            description=f"Mean |∇·{field_name}| over {n_points} Monte Carlo points",
        )

    def check_energy_conservation(
        self,
        model: object,
        coord_names: List[str],
        domain_bounds: Dict[str, Tuple[float, float]],
        field_name: str = "u",
        n_points: int = 10_000,
        tolerance: float = 1e-3,
    ) -> CheckResult:
        """Check energy conservation in integral form: ∂E/∂t + ∇·F ≈ 0.

        The energy density is approximated as ``E = ½ ‖u‖²`` and the flux
        divergence as ``∇·(u E)``.  The time derivative is estimated via
        autograd assuming the **last** coordinate in ``coord_names`` is time.

        Parameters
        ----------
        model:
            Trained model.
        coord_names:
            Coordinate names; the last entry is treated as time.
        domain_bounds:
            Dict mapping each coordinate name to ``(lo, hi)`` bounds.
        field_name:
            Name of the field (informational only).
        n_points:
            Number of Monte Carlo sample points.
        tolerance:
            Acceptance threshold on mean |∂E/∂t + ∇·(uE)|.

        Returns
        -------
        CheckResult
        """
        bounds = [domain_bounds[c] for c in coord_names]
        n_dims = len(coord_names)
        x = _sample_domain(n_points, n_dims, bounds, self.device)
        x.requires_grad_(True)

        u = _forward_tensor(model, x)  # (n, D)
        if u.dim() == 1:
            u = u.unsqueeze(-1)

        E = 0.5 * (u ** 2).sum(dim=-1)  # (n,)

        # ∂E/∂t — time is the last coordinate
        t_idx = n_dims - 1
        dE_dt = torch.autograd.grad(
            E.sum(), x, create_graph=False, retain_graph=True
        )[0][:, t_idx]  # (n,)

        # ∇·(u E): sum_i ∂(u_i * E)/∂x_i over spatial dims
        n_spatial = min(n_dims - 1, u.shape[-1])
        flux_div = torch.zeros(n_points, device=self.device)
        for i in range(n_spatial):
            flux_i = u[:, i] * E  # (n,)
            grad_i = torch.autograd.grad(
                flux_i.sum(), x, create_graph=False, retain_graph=True
            )[0][:, i]
            flux_div = flux_div + grad_i

        residual = (dE_dt + flux_div).abs().mean().item()
        passed = residual <= tolerance
        return CheckResult(
            name="energy_conservation",
            passed=passed,
            value=residual,
            threshold=tolerance,
            description=(
                f"Mean |∂E/∂t + ∇·({field_name}·E)| over {n_points} points; "
                f"E=½‖{field_name}‖²"
            ),
        )

    def check_integral_quantity(
        self,
        model: object,
        coord_names: List[str],
        domain_bounds: Dict[str, Tuple[float, float]],
        integrand_fn: Callable[[Tensor], Tensor],
        expected_value: float,
        tolerance: float,
        name: str,
        n_points: int = 10_000,
    ) -> CheckResult:
        """Generic Monte Carlo integral check.

        Computes ``V * mean(integrand_fn(u(x)))`` over the domain, where *V*
        is the domain hyper-volume, and compares to *expected_value*.

        Parameters
        ----------
        model:
            Trained model.
        coord_names:
            Coordinate names.
        domain_bounds:
            Dict mapping each coordinate name to ``(lo, hi)`` bounds.
        integrand_fn:
            A callable ``f(u) -> Tensor`` of shape ``(n,)`` or scalar.
        expected_value:
            The expected value of the integral.
        tolerance:
            Acceptance threshold on ``|integral - expected_value|``.
        name:
            Check identifier (used in the returned :class:`CheckResult`).
        n_points:
            Number of Monte Carlo sample points.

        Returns
        -------
        CheckResult
        """
        bounds = [domain_bounds[c] for c in coord_names]
        n_dims = len(coord_names)
        volume = _domain_volume(bounds)

        x = _sample_domain(n_points, n_dims, bounds, self.device)
        with torch.no_grad():
            u = _forward_tensor(model, x)
            integrand = integrand_fn(u)
            integral = volume * integrand.mean().item()

        error = abs(integral - expected_value)
        passed = error <= tolerance
        return CheckResult(
            name=name,
            passed=passed,
            value=error,
            threshold=tolerance,
            description=(
                f"∫ integrand dΩ = {integral:.4e}, expected {expected_value:.4e}; "
                f"|error| = {error:.4e} over {n_points} MC points"
            ),
        )
