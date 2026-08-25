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
        spatial_coord_names: Optional[List[str]] = None,
        velocity_indices: Optional[List[int]] = None,
    ) -> CheckResult:
        """Check mass conservation (∇·u ≈ 0) for incompressible flow.

        Computes the mean absolute divergence ``div = sum_k ∂u[velocity_indices[k]]
        / ∂x[spatial_coord_names[k]]`` over Monte Carlo sample points, where the
        pairing between velocity output columns and spatial coordinate columns is
        made explicitly by *name*/*index* rather than by raw positional order.

        This is only a meaningful "mass conservation" statement when ``u`` is a
        genuine velocity vector field with one component per spatial direction
        (``len(velocity_indices) == len(spatial_coord_names)``); it is not defined
        for an arbitrary scalar output.

        Parameters
        ----------
        model:
            Trained model with ``forward`` / callable interface.
        coord_names:
            Ordered list of coordinate names matching the model input columns.
            May include non-spatial coordinates such as time.
        domain_bounds:
            Dict mapping each coordinate name to ``(lo, hi)`` bounds.
        field_name:
            Name of the conserved field (informational only).
        n_points:
            Number of Monte Carlo sample points.
        tolerance:
            Acceptance threshold on the mean |∇·u|.
        spatial_coord_names:
            Names (subset of ``coord_names``) of the spatial coordinates that
            participate in the divergence. Defaults to every entry of
            ``coord_names`` except ones named ``"t"`` or ``"time"``
            (case-insensitive), so a time coordinate is excluded automatically.
        velocity_indices:
            Output-column indices of the velocity components, in the same
            order as ``spatial_coord_names`` (i.e. ``velocity_indices[k]`` is
            paired with ``spatial_coord_names[k]``). Defaults to
            ``[0, 1, ..., len(spatial_coord_names) - 1]``. Must have the same
            length as ``spatial_coord_names``.

        Returns
        -------
        CheckResult

        Raises
        ------
        ValueError
            If ``velocity_indices`` and ``spatial_coord_names`` have different
            lengths, or if a name in ``spatial_coord_names`` is not present in
            ``coord_names``.
        """
        if spatial_coord_names is None:
            spatial_coord_names = [
                c for c in coord_names if c.lower() not in ("t", "time")
            ]
        if velocity_indices is None:
            velocity_indices = list(range(len(spatial_coord_names)))

        if len(velocity_indices) != len(spatial_coord_names):
            raise ValueError(
                "velocity_indices and spatial_coord_names must have the same "
                f"length (a genuine velocity vector field needs one output "
                f"component per spatial direction); got "
                f"{len(velocity_indices)} velocity indices vs "
                f"{len(spatial_coord_names)} spatial coordinates "
                f"({spatial_coord_names!r})."
            )

        spatial_coord_idx = []
        for name in spatial_coord_names:
            try:
                spatial_coord_idx.append(coord_names.index(name))
            except ValueError as exc:
                raise ValueError(
                    f"spatial coordinate {name!r} not found in coord_names={coord_names!r}"
                ) from exc

        bounds = [domain_bounds[c] for c in coord_names]
        n_dims = len(coord_names)
        x = _sample_domain(n_points, n_dims, bounds, self.device)
        x.requires_grad_(True)

        u = _forward_tensor(model, x)  # (n, D) or (n,)
        if u.dim() == 1:
            u = u.unsqueeze(-1)  # treat scalar as (n, 1)

        n_out = u.shape[-1]
        for vi in velocity_indices:
            if vi >= n_out:
                raise ValueError(
                    f"velocity index {vi} out of range for model output with "
                    f"{n_out} column(s)."
                )

        # Compute divergence: sum_k ∂u[velocity_indices[k]] / ∂x[spatial_coord_idx[k]]
        div = torch.zeros(n_points, device=self.device)
        for out_col, coord_col in zip(velocity_indices, spatial_coord_idx):
            grad_i = torch.autograd.grad(
                u[:, out_col].sum(), x, create_graph=False, retain_graph=True
            )[0]  # (n, n_dims)
            div = div + grad_i[:, coord_col]

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
        velocity_indices: Optional[List[int]] = None,
    ) -> CheckResult:
        """Check the restricted, source-free energy balance: ∂E/∂t + ∇·(uE) ≈ 0.

        The energy density ``E = ½ ‖u_velocity‖²`` and the advective flux
        ``∇·(u_velocity E)`` are both built from **only** ``velocity_indices``
        output columns, so a non-velocity output (e.g. pressure) never leaks
        into either term. The time derivative is estimated via autograd
        assuming the **last** coordinate in ``coord_names`` is time.

        IMPORTANT — this checks the *source-free scalar advection* form of
        the energy balance only. It omits the pressure-work term
        ``-∇·(u·p)`` and any viscous dissipation/diffusion term that appear
        in the true incompressible Navier-Stokes kinetic-energy balance. A
        physically energy-conserving flow with a non-uniform pressure field
        (e.g. Poiseuille/channel flow, or any flow doing pressure work) can
        legitimately show a **nonzero** residual here even though no physics
        is being violated. Treat a FAIL from this check as "the source-free
        advective energy balance is not satisfied", not as definitive
        evidence of a genuine energy-conservation violation, unless you know
        the modeled flow truly has no pressure-work or viscous terms.

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
        velocity_indices:
            Output-column indices that make up the velocity vector used for
            both ``E`` and the flux. Defaults to all spatial (non-time)
            coordinate slots, i.e. ``[0, ..., len(coord_names) - 2]``
            (assuming the last coordinate is time), clipped to the number of
            output columns available.

        Returns
        -------
        CheckResult
        """
        bounds = [domain_bounds[c] for c in coord_names]
        n_dims = len(coord_names)
        x = _sample_domain(n_points, n_dims, bounds, self.device)
        x.requires_grad_(True)

        u_full = _forward_tensor(model, x)  # (n, D)
        if u_full.dim() == 1:
            u_full = u_full.unsqueeze(-1)

        t_idx = n_dims - 1
        if velocity_indices is None:
            n_spatial_default = min(n_dims - 1, u_full.shape[-1])
            velocity_indices = list(range(n_spatial_default))

        n_out = u_full.shape[-1]
        for vi in velocity_indices:
            if vi >= n_out:
                raise ValueError(
                    f"velocity index {vi} out of range for model output with "
                    f"{n_out} column(s)."
                )

        u = u_full[:, velocity_indices]  # (n, n_velocity) — velocity-only columns

        E = 0.5 * (u ** 2).sum(dim=-1)  # (n,) — kinetic energy density, velocity-only

        # ∂E/∂t — time is the last coordinate
        dE_dt = torch.autograd.grad(
            E.sum(), x, create_graph=False, retain_graph=True
        )[0][:, t_idx]  # (n,)

        # ∇·(u E): sum over the same velocity columns, paired with the
        # spatial coordinate at the same position (coordinate i for velocity
        # column i), consistent with the columns used to build E above.
        flux_div = torch.zeros(n_points, device=self.device)
        for k in range(u.shape[-1]):
            flux_i = u[:, k] * E  # (n,)
            grad_i = torch.autograd.grad(
                flux_i.sum(), x, create_graph=False, retain_graph=True
            )[0][:, k]
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
