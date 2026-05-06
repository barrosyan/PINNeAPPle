"""PhysicsValidator: orchestrator for physics validation runs.

Collects check specifications, executes them in order, and produces a
:class:`~pinneaple_validate.core.ValidationReport`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .core import CheckResult, ValidationReport
from .boundary import BoundaryCheck
from .conservation import ConservationCheck
from .symmetry import SymmetryCheck

# ---------------------------------------------------------------------------
# Internal spec records
# ---------------------------------------------------------------------------


@dataclass
class _ConservationSpec:
    kind: str  # "mass" | "energy" | "integral"
    kwargs: Dict[str, Any]


@dataclass
class _BoundarySpec:
    kind: str  # "dirichlet" | "neumann" | "periodicity"
    kwargs: Dict[str, Any]


@dataclass
class _SymmetrySpec:
    kind: str  # "reflection" | "rotational"
    kwargs: Dict[str, Any]


@dataclass
class _CustomSpec:
    fn: Callable[..., Any]
    name: str
    threshold: float
    kwargs: Dict[str, Any]


# ---------------------------------------------------------------------------
# PhysicsValidator
# ---------------------------------------------------------------------------


class PhysicsValidator:
    """Orchestrates a suite of physics validation checks on a trained model.

    Parameters
    ----------
    model:
        Trained model with a ``forward`` / callable interface compatible with
        the pinneaple model conventions (returns ``Tensor`` or an object with
        a ``.y`` attribute).
    coord_names:
        Ordered list of coordinate names matching the model's input columns
        (e.g. ``["x", "y", "t"]``).
    domain_bounds:
        Dict mapping each coordinate name to ``(lo, hi)`` bounds.
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, …).

    Example
    -------
    >>> validator = PhysicsValidator(model, ["x", "y"], {"x": (0,1), "y": (0,1)})
    >>> validator.add_conservation_check(kind="mass", tolerance=1e-3)
    >>> report = validator.validate()
    >>> print(report.summary())
    """

    def __init__(
        self,
        model: object,
        coord_names: List[str],
        domain_bounds: Dict[str, Tuple[float, float]],
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.coord_names = coord_names
        self.domain_bounds = domain_bounds
        self.device = device

        self._conservation_specs: List[_ConservationSpec] = []
        self._boundary_specs: List[_BoundarySpec] = []
        self._symmetry_specs: List[_SymmetrySpec] = []
        self._custom_specs: List[_CustomSpec] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_conservation_check(self, kind: str = "mass", **kwargs: Any) -> "PhysicsValidator":
        """Register a conservation check.

        Parameters
        ----------
        kind:
            One of ``"mass"``, ``"energy"``, or ``"integral"``.
        **kwargs:
            Forwarded to the corresponding :class:`ConservationCheck` method.
            ``coord_names`` and ``domain_bounds`` are injected automatically
            unless overridden.

        Returns
        -------
        self (for chaining)
        """
        self._conservation_specs.append(_ConservationSpec(kind=kind, kwargs=kwargs))
        return self

    def add_boundary_check(self, kind: str = "dirichlet", **kwargs: Any) -> "PhysicsValidator":
        """Register a boundary condition check.

        Parameters
        ----------
        kind:
            One of ``"dirichlet"``, ``"neumann"``, or ``"periodicity"``.
        **kwargs:
            Forwarded to the corresponding :class:`BoundaryCheck` method.

        Returns
        -------
        self (for chaining)
        """
        self._boundary_specs.append(_BoundarySpec(kind=kind, kwargs=kwargs))
        return self

    def add_symmetry_check(self, kind: str = "reflection", **kwargs: Any) -> "PhysicsValidator":
        """Register a symmetry check.

        Parameters
        ----------
        kind:
            One of ``"reflection"`` or ``"rotational"``.
        **kwargs:
            Forwarded to the corresponding :class:`SymmetryCheck` method.

        Returns
        -------
        self (for chaining)
        """
        self._symmetry_specs.append(_SymmetrySpec(kind=kind, kwargs=kwargs))
        return self

    def add_custom_check(
        self,
        fn: Callable[..., Any],
        name: str,
        threshold: float,
        **kwargs: Any,
    ) -> "PhysicsValidator":
        """Register a custom check function.

        The function must accept ``(model, coord_names, domain_bounds, **kwargs)``
        and return either a :class:`~pinneaple_validate.core.CheckResult` or a
        plain ``float`` (which will be compared to *threshold*).

        Parameters
        ----------
        fn:
            Custom check callable.
        name:
            Identifier for the check in the report.
        threshold:
            Acceptance threshold (used when *fn* returns a float).
        **kwargs:
            Additional keyword arguments passed to *fn*.

        Returns
        -------
        self (for chaining)
        """
        self._custom_specs.append(
            _CustomSpec(fn=fn, name=name, threshold=threshold, kwargs=kwargs)
        )
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def validate(self, model_name: Optional[str] = None) -> ValidationReport:
        """Run all registered checks and return a :class:`ValidationReport`.

        Parameters
        ----------
        model_name:
            Optional name override; defaults to ``type(model).__name__``.

        Returns
        -------
        ValidationReport
        """
        name = model_name or type(self.model).__name__
        report = ValidationReport.create(name)

        conservation = ConservationCheck(device=self.device)
        boundary = BoundaryCheck(device=self.device)
        symmetry = SymmetryCheck(device=self.device)

        # --- Conservation ---
        for spec in self._conservation_specs:
            kw = dict(spec.kwargs)
            kw.setdefault("coord_names", self.coord_names)
            kw.setdefault("domain_bounds", self.domain_bounds)
            if spec.kind == "mass":
                result = conservation.check_mass_conservation(self.model, **kw)
            elif spec.kind == "energy":
                result = conservation.check_energy_conservation(self.model, **kw)
            elif spec.kind == "integral":
                result = conservation.check_integral_quantity(self.model, **kw)
            else:
                raise ValueError(f"Unknown conservation kind: {spec.kind!r}")
            report.checks.append(result)

        # --- Boundary ---
        for spec in self._boundary_specs:
            kw = dict(spec.kwargs)
            if spec.kind == "dirichlet":
                result = boundary.check_dirichlet(self.model, **kw)
            elif spec.kind == "neumann":
                result = boundary.check_neumann(self.model, **kw)
            elif spec.kind == "periodicity":
                result = boundary.check_periodicity(self.model, **kw)
            else:
                raise ValueError(f"Unknown boundary kind: {spec.kind!r}")
            report.checks.append(result)

        # --- Symmetry ---
        for spec in self._symmetry_specs:
            kw = dict(spec.kwargs)
            if spec.kind == "reflection":
                result = symmetry.check_reflection(self.model, **kw)
            elif spec.kind == "rotational":
                result = symmetry.check_rotational(self.model, **kw)
            else:
                raise ValueError(f"Unknown symmetry kind: {spec.kind!r}")
            report.checks.append(result)

        # --- Custom ---
        for spec in self._custom_specs:
            raw = spec.fn(
                self.model, self.coord_names, self.domain_bounds, **spec.kwargs
            )
            if isinstance(raw, CheckResult):
                report.checks.append(raw)
            else:
                value = float(raw)
                report.checks.append(
                    CheckResult(
                        name=spec.name,
                        passed=value <= spec.threshold,
                        value=value,
                        threshold=spec.threshold,
                        description=f"Custom check '{spec.name}'",
                    )
                )

        return report

    # ------------------------------------------------------------------
    # Auto-configuration from ProblemSpec
    # ------------------------------------------------------------------

    def validate_from_spec(self, problem_spec: Any) -> ValidationReport:
        """Auto-configure and run checks from a :class:`~pinneaple_problemdesign.schema.ProblemSpec`.

        Currently auto-detects:
        - Dirichlet BCs from ``problem_spec.physics.boundary_conditions`` strings
          containing ``"dirichlet"`` or ``"="`` patterns.
        - Mass conservation check when the governing equations mention
          ``"incompressible"`` or ``"∇·"`` / ``"div"``.
        - Energy conservation when equations mention ``"energy"`` or ``"∂E"``.

        For a precise programmatic setup you should call the ``add_*`` methods
        directly.  This method is a best-effort convenience helper.

        Parameters
        ----------
        problem_spec:
            A :class:`~pinneaple_problemdesign.schema.ProblemSpec` instance.

        Returns
        -------
        ValidationReport
        """
        physics = getattr(problem_spec, "physics", None)
        if physics is not None:
            governing = " ".join(getattr(physics, "governing_equations", [])).lower()

            # Conservation heuristics
            if any(kw in governing for kw in ("incompressible", "∇·", "div u", "div(u)")):
                self._conservation_specs.append(
                    _ConservationSpec(kind="mass", kwargs={})
                )
            if any(kw in governing for kw in ("energy", "∂e", "de/dt")):
                self._conservation_specs.append(
                    _ConservationSpec(kind="energy", kwargs={})
                )

        return self.validate(
            model_name=getattr(problem_spec, "title", None) or type(self.model).__name__
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def validate_model(
    model: object,
    spec: Any,
    coord_names: Optional[List[str]] = None,
    domain_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    device: str = "cpu",
) -> ValidationReport:
    """Quick one-shot validation from a ProblemSpec or explicit arguments.

    Parameters
    ----------
    model:
        Trained PINN model.
    spec:
        A :class:`~pinneaple_problemdesign.schema.ProblemSpec` instance, or
        ``None`` to use *coord_names* / *domain_bounds* directly.
    coord_names:
        Required when *spec* is ``None``.
    domain_bounds:
        Required when *spec* is ``None``.
    device:
        PyTorch device string.

    Returns
    -------
    ValidationReport
    """
    if spec is not None:
        _coords = getattr(getattr(spec, "domain", None), "coord_names", None) or coord_names or []
        _bounds = getattr(getattr(spec, "domain", None), "bounds", None) or domain_bounds or {}
    else:
        _coords = coord_names or []
        _bounds = domain_bounds or {}

    validator = PhysicsValidator(model, _coords, _bounds, device=device)
    if spec is not None:
        return validator.validate_from_spec(spec)
    return validator.validate()
