"""ProblemBuilder — fluent API for constructing ProblemSpec objects.

This module provides the researcher-facing interface for defining physics
problems without touching the raw dataclass constructors.

Usage
-----
>>> from pinneaple_environment import ProblemBuilder
>>> import numpy as np

>>> spec = (
...     ProblemBuilder("my_heat")
...     .domain(x=(0.0, 1.0), t=(0.0, 1.0))
...     .fields("u")
...     .pde("heat_1d", alpha=0.01)
...     .ic(field="u", fn=lambda X: np.sin(np.pi * X[:, 0:1]))
...     .bc("dirichlet", field="u", value=0.0, on="x_boundary")
...     .sample(interior=4000, boundary=800, ic=800)
...     .build()
... )

The resulting ``ProblemSpec`` is directly accepted by:
- ``Arena.from_spec(spec).run(...)``
- ``run_arena_experiment(...)`` via YAML
- ``generate_pinn_dataset(spec, ...)``

The builder can also register the spec as a named preset::

    builder.register("my_heat_preset")
    # now available via: get_preset("my_heat_preset")
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .conditions import ConditionSpec, DirichletBC, InitialCondition, NeumannBC, DataConstraint
from .scales import ScaleSpec
from .spec import PDETermSpec, ProblemSpec
from .typing import CoordNames, FieldNames

# ── value helpers ──────────────────────────────────────────────────────────────

def _make_value_fn(
    value: Union[float, Callable, Dict[str, float]],
    n_fields: int,
) -> Callable[[np.ndarray, Dict[str, Any]], np.ndarray]:
    """Convert scalar / callable / dict into a (X, ctx) -> (N, n_fields) callable."""
    if callable(value):
        def _wrap_callable(X, ctx, _fn=value):
            v = _fn(X)
            v = np.asarray(v, dtype=np.float32)
            if v.ndim == 1:
                v = v[:, None]
            return v
        return _wrap_callable
    elif isinstance(value, dict):
        # dict maps field_name -> scalar/array; order by field index
        vals = list(value.values())
        def _dict_fn(X, ctx, _vals=vals, _nf=n_fields):
            out = np.zeros((X.shape[0], _nf), dtype=np.float32)
            for i, v in enumerate(_vals[:_nf]):
                out[:, i] = float(v)
            return out
        return _dict_fn
    else:
        # scalar constant
        const = float(value)
        def _const_fn(X, ctx, _c=const, _nf=n_fields):
            return np.full((X.shape[0], _nf), _c, dtype=np.float32)
        return _const_fn


def _make_selector(
    on: Union[str, Callable, Tuple],
    domain_bounds: Dict[str, Tuple[float, float]],
    coords: List[str],
) -> Tuple[str, Optional[Any]]:
    """
    Convert an `on` spec into (selector_type, selector) pair.

    Supported forms:
    - "boundary"                    → tag selector with tag="boundary"
    - "x_min" / "x_max"            → callable checking coord x at its min/max
    - "t_boundary" / "t=0"         → callable for time-like coordinate
    - ("x", "min") / ("x", 0.0)    → callable for coord at value
    - callable                      → passed through as callable selector
    """
    if on is None or on == "boundary":
        return "tag", {"tag": "boundary"}

    if callable(on):
        return "callable", on

    if isinstance(on, (list, tuple)) and len(on) == 2:
        coord, val = on
        if isinstance(val, str):
            if val == "min":
                val = domain_bounds.get(coord, (0.0, 1.0))[0]
            elif val == "max":
                val = domain_bounds.get(coord, (0.0, 1.0))[1]
            else:
                val = float(val)
        else:
            val = float(val)
        coord_idx = coords.index(coord) if coord in coords else 0
        _tol = 1e-6
        def _coord_sel(X, ctx, _ci=coord_idx, _v=val, _tol=_tol):
            return np.abs(X[:, _ci] - _v) < _tol
        return "callable", _coord_sel

    # String shorthand: "x_min", "x_max", "t=0", "t=1", etc.
    if isinstance(on, str):
        # "tag:name" → explicit tag
        if on.startswith("tag:"):
            return "tag", {"tag": on[4:]}

        # "coord=value" → e.g. "t=0"
        if "=" in on:
            coord, val_str = on.split("=", 1)
            coord = coord.strip()
            val = float(val_str.strip())
            coord_idx = coords.index(coord) if coord in coords else 0
            _tol = 1e-6
            def _eq_sel(X, ctx, _ci=coord_idx, _v=val, _tol=_tol):
                return np.abs(X[:, _ci] - _v) < _tol
            return "callable", _eq_sel

        # "coord_min" or "coord_max"
        for c in coords:
            if on == f"{c}_min":
                val = domain_bounds.get(c, (0.0, 1.0))[0]
                ci = coords.index(c)
                def _min_sel(X, ctx, _ci=ci, _v=val):
                    return np.abs(X[:, _ci] - _v) < 1e-6
                return "callable", _min_sel
            if on == f"{c}_max":
                val = domain_bounds.get(c, (0.0, 1.0))[1]
                ci = coords.index(c)
                def _max_sel(X, ctx, _ci=ci, _v=val):
                    return np.abs(X[:, _ci] - _v) < 1e-6
                return "callable", _max_sel

        # "x_boundary" → all x boundaries (min + max)
        if on.endswith("_boundary"):
            coord = on[: -len("_boundary")]
            if coord in coords:
                lo, hi = domain_bounds.get(coord, (0.0, 1.0))
                ci = coords.index(coord)
                def _both_sel(X, ctx, _ci=ci, _lo=lo, _hi=hi):
                    return (np.abs(X[:, _ci] - _lo) < 1e-6) | (np.abs(X[:, _ci] - _hi) < 1e-6)
                return "callable", _both_sel

        # fallback: treat as tag
        return "tag", {"tag": on}

    # fallback
    return "tag", {"tag": "boundary"}


# ══════════════════════════════════════════════════════════════════════════════
# ProblemBuilder
# ══════════════════════════════════════════════════════════════════════════════

class ProblemBuilder:
    """
    Fluent builder for ``ProblemSpec``.

    Every method returns ``self`` to allow method chaining.
    Call ``.build()`` at the end to get a frozen ``ProblemSpec``.

    Example
    -------
    >>> spec = (
    ...     ProblemBuilder("burgers_custom")
    ...     .domain(x=(-1.0, 1.0), t=(0.0, 1.0))
    ...     .fields("u")
    ...     .pde("burgers", nu=0.01)
    ...     .ic(field="u", fn=lambda X: -np.sin(np.pi * X[:, 0:1]))
    ...     .bc("dirichlet", field="u", value=0.0, on="x_boundary")
    ...     .sample(interior=4000, boundary=800, ic=800)
    ...     .solver(name="fdm", method="burgers_1d", nx=256, nt=256)
    ...     .build()
    ... )
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._coords: List[str] = []
        self._domain_bounds: Dict[str, Tuple[float, float]] = {}
        self._fields: List[str] = []
        self._pde_kind: str = ""
        self._pde_params: Dict[str, Any] = {}
        self._pde_meta: Dict[str, Any] = {}
        self._conditions: List[ConditionSpec] = []
        self._sample_defaults: Dict[str, int] = {}
        self._field_ranges: Dict[str, Tuple[float, float]] = {}
        self._solver_spec: Dict[str, Any] = {}
        self._references: List[str] = []
        self._scales: ScaleSpec = ScaleSpec()

    # ── Domain ────────────────────────────────────────────────────────────────

    def domain(self, **bounds: Tuple[float, float]) -> "ProblemBuilder":
        """
        Set coordinate domain bounds via keyword arguments.

        Example::

            .domain(x=(-1.0, 1.0), t=(0.0, 1.0))
        """
        self._domain_bounds = {k: (float(v[0]), float(v[1])) for k, v in bounds.items()}
        self._coords = list(bounds.keys())
        return self

    def domain_bounds(self, bounds: Dict[str, Tuple[float, float]]) -> "ProblemBuilder":
        """
        Set coordinate domain bounds via a dict (paper-style API).

        Example::

            .domain_bounds({"x": (0, 1), "y": (0, 1)})
        """
        self._domain_bounds = {k: (float(v[0]), float(v[1])) for k, v in bounds.items()}
        if not self._coords:
            self._coords = list(bounds.keys())
        return self

    def coords(self, *names: str) -> "ProblemBuilder":
        """Explicitly set coordinate ordering (optional if using .domain())."""
        self._coords = list(names)
        return self

    # ── Fields ────────────────────────────────────────────────────────────────

    def fields(self, *names: str) -> "ProblemBuilder":
        """
        Set output field names.

        Example::

            .fields("u", "v", "p")
        """
        self._fields = list(names)
        return self

    def field_range(self, **ranges: Tuple[float, float]) -> "ProblemBuilder":
        """Set expected value ranges per field (used for normalisation hints)."""
        self._field_ranges.update({k: (float(v[0]), float(v[1])) for k, v in ranges.items()})
        return self

    # ── PDE ──────────────────────────────────────────────────────────────────

    def pde(
        self,
        kind: str,
        *,
        fields: Optional[Sequence[str]] = None,
        coords: Optional[Sequence[str]] = None,
        **params: Any,
    ) -> "ProblemBuilder":
        """
        Set PDE type and parameters.

        ``kind`` must match a key known to the physics loss compiler
        (e.g. ``"burgers"``, ``"heat_1d"``, ``"laplace"``,
        ``"navier_stokes_incompressible"``).

        The optional ``fields`` and ``coords`` keyword arguments allow setting
        field and coordinate names inline (as in the paper)::

            .pde("poisson", fields=("u",), coords=("x", "y"))

        Example::

            .pde("burgers", nu=0.01)
            .pde("poisson", fields=("u",), coords=("x", "y"))
            .pde("navier_stokes_incompressible", Re=100.0)
        """
        self._pde_kind = kind
        self._pde_params = dict(params)
        if fields is not None:
            self._fields = list(fields)
        if coords is not None:
            self._coords = list(coords)
            # fill domain bounds with unit intervals if not already set
            for c in self._coords:
                if c not in self._domain_bounds:
                    self._domain_bounds[c] = (0.0, 1.0)
        return self

    def param(self, name: str, value: Any) -> "ProblemBuilder":
        """
        Add a single named PDE parameter.

        Equivalent to passing it as a keyword argument to ``.pde()``::

            .param("f", 1.0)  # same as .pde("poisson", f=1.0)
        """
        self._pde_params[name] = value
        return self

    def pde_meta(self, **meta: Any) -> "ProblemBuilder":
        """Set optional PDE metadata (passed through to compiler)."""
        self._pde_meta.update(meta)
        return self

    # ── Conditions ────────────────────────────────────────────────────────────

    def bc(
        self,
        kind: str = "dirichlet",
        *,
        field: Union[str, Sequence[str]],
        value: Union[float, Callable, Dict[str, float]] = 0.0,
        on: Union[str, Callable, Tuple, None] = "boundary",
        weight: float = 10.0,
        name: Optional[str] = None,
    ) -> "ProblemBuilder":
        """
        Add a boundary condition.

        Parameters
        ----------
        kind    : "dirichlet" | "neumann" | "robin"
        field   : field name(s) this condition applies to
        value   : constant scalar, per-field dict, or callable (X) -> array
        on      : selector spec — see ``_make_selector`` for all forms:
                  - "boundary"         → all boundary points (tag selector)
                  - "x_min" / "x_max" → left/right edge on coordinate x
                  - "x_boundary"       → both edges of coordinate x
                  - "t=0"              → where t == 0
                  - ("x", "min")       → tuple form
                  - callable(X) → bool mask
        weight  : loss weight for this condition

        Examples::

            .bc("dirichlet", field="u", value=0.0, on="boundary")
            .bc("dirichlet", field="u", value=0.0, on="x_boundary")
            .bc("neumann",   field="u", value=1.0, on="x_max")
        """
        fields_tuple = (field,) if isinstance(field, str) else tuple(field)
        n_fields = len(fields_tuple)
        sel_type, selector = _make_selector(on, self._domain_bounds, self._coords)
        value_fn = _make_value_fn(value, n_fields)
        cond_name = name or f"bc_{kind}_{len(self._conditions)}"

        if kind == "dirichlet":
            cond = DirichletBC(cond_name, fields_tuple, sel_type, selector, value_fn, weight)
        elif kind == "neumann":
            cond = NeumannBC(cond_name, fields_tuple, sel_type, selector, value_fn, weight)
        else:
            cond = ConditionSpec(
                name=cond_name,
                kind=kind,
                fields=fields_tuple,
                selector_type=sel_type,
                selector=selector,
                value_fn=value_fn,
                weight=weight,
            )
        self._conditions.append(cond)
        return self

    def ic(
        self,
        *,
        field: Union[str, Sequence[str]],
        fn: Optional[Callable] = None,
        value: Union[float, Callable, Dict[str, float]] = 0.0,
        weight: float = 10.0,
        name: Optional[str] = None,
        at_t: float = 0.0,
    ) -> "ProblemBuilder":
        """
        Add an initial condition (applied at t=``at_t``).

        Parameters
        ----------
        field  : field name(s)
        fn     : callable (X) -> array (takes precedence over ``value``)
        value  : scalar fallback if ``fn`` is None
        weight : loss weight
        at_t   : time coordinate value for initial condition

        Example::

            .ic(field="u", fn=lambda X: -np.sin(np.pi * X[:, 0:1]))
            .ic(field="u", value=0.0)
        """
        fields_tuple = (field,) if isinstance(field, str) else tuple(field)
        n_fields = len(fields_tuple)

        # Selector: t == at_t
        t_idx = self._coords.index("t") if "t" in self._coords else -1
        if t_idx >= 0:
            _ti, _at = t_idx, float(at_t)
            sel = lambda X, ctx, _ti=_ti, _at=_at: np.abs(X[:, _ti] - _at) < 1e-6
            sel_type = "callable"
        else:
            sel = None
            sel_type = "all"

        value_fn = _make_value_fn(fn if fn is not None else value, n_fields)
        cond_name = name or f"ic_{len(self._conditions)}"
        cond = InitialCondition(cond_name, fields_tuple, sel_type, sel, value_fn, weight)
        self._conditions.append(cond)
        return self

    def data(
        self,
        *,
        field: Union[str, Sequence[str]],
        fn: Optional[Callable] = None,
        value: Union[float, Callable] = 0.0,
        on: Union[str, Callable, None] = None,
        weight: float = 1.0,
        name: Optional[str] = None,
    ) -> "ProblemBuilder":
        """Add a data constraint (supervised loss at specific points)."""
        fields_tuple = (field,) if isinstance(field, str) else tuple(field)
        n_fields = len(fields_tuple)
        sel_type, selector = _make_selector(on, self._domain_bounds, self._coords)
        value_fn = _make_value_fn(fn if fn is not None else value, n_fields)
        cond_name = name or f"data_{len(self._conditions)}"
        cond = DataConstraint(cond_name, fields_tuple, sel_type, selector, value_fn, weight)
        self._conditions.append(cond)
        return self

    def add_condition(self, cond: ConditionSpec) -> "ProblemBuilder":
        """Add a raw ConditionSpec directly."""
        self._conditions.append(cond)
        return self

    def boundary_condition(
        self,
        name: str,
        condition: ConditionSpec,
    ) -> "ProblemBuilder":
        """
        Add a named boundary condition (paper-style API).

        Accepts a ConditionSpec produced by DirichletBC, NeumannBC, etc.
        The ``name`` overrides the condition's internal name.

        Example::

            from pinneaple_environment import DirichletBC
            .boundary_condition("wall", DirichletBC({"u": 0.0}))
        """
        import dataclasses
        named = dataclasses.replace(condition, name=name)
        self._conditions.append(named)
        return self

    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample(self, **counts: int) -> "ProblemBuilder":
        """
        Set default sampling counts.

        Recognised keys: ``interior``, ``boundary``, ``ic``, ``data``,
        ``n_col`` (alias for interior), ``n_bc`` (alias for boundary),
        ``n_ic`` (alias for ic).

        Example::

            .sample(interior=4000, boundary=800, ic=800)
        """
        aliases = {"interior": "n_col", "boundary": "n_bc", "ic": "n_ic", "data": "n_data"}
        for k, v in counts.items():
            self._sample_defaults[aliases.get(k, k)] = int(v)
        return self

    # ── Solver hint ───────────────────────────────────────────────────────────

    def solver(self, name: str = "fdm", method: str = "", **params: Any) -> "ProblemBuilder":
        """
        Hint which reference solver to use for generating training data.

        Example::

            .solver(name="fdm", method="burgers_1d", nx=256, nt=256)
            .solver(name="fenics", method="poisson_2d")
        """
        self._solver_spec = {"name": name, "method": method, "params": dict(params)}
        return self

    # ── References ────────────────────────────────────────────────────────────

    def reference(self, *refs: str) -> "ProblemBuilder":
        """Add reference strings (papers, notes) for documentation."""
        self._references.extend(refs)
        return self

    # ── Dim detection ─────────────────────────────────────────────────────────

    @property
    def _dim(self) -> int:
        """Spatial dimension = coords excluding 't'."""
        return sum(1 for c in self._coords if c != "t")

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self) -> ProblemSpec:
        """
        Validate and build the frozen ``ProblemSpec``.

        Raises
        ------
        ValueError
            If required fields are missing.
        """
        if not self._coords:
            raise ValueError("ProblemBuilder: call .domain() first to set coordinates.")
        if not self._fields:
            raise ValueError("ProblemBuilder: call .fields() to set output field names.")
        if not self._pde_kind:
            raise ValueError("ProblemBuilder: call .pde() to specify the PDE type.")

        # Fill sample_defaults with sensible fallbacks
        sd = {
            "n_col": 4000,
            "n_bc": 800,
            "n_ic": 800,
        }
        sd.update(self._sample_defaults)

        pde_term = PDETermSpec(
            kind=self._pde_kind,
            fields=tuple(self._fields),
            coords=tuple(self._coords),
            params=dict(self._pde_params),
            meta=dict(self._pde_meta),
        )

        return ProblemSpec(
            name=self._name,
            dim=self._dim,
            coords=tuple(self._coords),
            fields=tuple(self._fields),
            pde=pde_term,
            conditions=tuple(self._conditions),
            sample_defaults=sd,
            scales=self._scales,
            field_ranges=dict(self._field_ranges),
            references=tuple(self._references),
            domain_bounds=dict(self._domain_bounds),
            solver_spec=dict(self._solver_spec),
        )

    # ── Registry integration ───────────────────────────────────────────────────

    def register(self, preset_id: Optional[str] = None) -> "ProblemBuilder":
        """
        Register this problem as a named preset in the global registry.

        After registering, the problem is accessible via::

            from pinneaple_environment.presets.registry import get_preset
            spec = get_preset("my_preset_id")

            # or in YAML:
            # problem:
            #   id: my_preset_id

        Parameters
        ----------
        preset_id : str, optional
            Registry key. Defaults to the builder name.
        """
        from .presets.registry import register_preset as _register

        pid = preset_id or self._name
        spec = self.build()

        @_register(pid)
        def _factory() -> ProblemSpec:
            return spec

        return self

    # ── Quick-solve shortcut ──────────────────────────────────────────────────

    def solve(
        self,
        model: str = "VanillaPINN",
        *,
        epochs: int = 5000,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int = 42,
        n_col: Optional[int] = None,
        n_bc: Optional[int] = None,
        n_ic: Optional[int] = None,
        verbose: bool = True,
        **train_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Build the spec and run the resolution pipeline in one call.

        This is a convenience wrapper equivalent to::

            Arena.from_spec(builder.build()).run(model=model, epochs=epochs, ...)

        Returns
        -------
        dict with keys: spec, model, history, metrics
        """
        spec = self.build()
        from pinneaple_arena.api import Arena
        return Arena.from_spec(spec).run(
            model=model,
            epochs=epochs,
            lr=lr,
            device=device,
            seed=seed,
            n_col=n_col,
            n_bc=n_bc,
            n_ic=n_ic,
            verbose=verbose,
            **train_kwargs,
        )

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"ProblemBuilder(name={self._name!r}, "
            f"pde={self._pde_kind!r}, "
            f"coords={self._coords}, "
            f"fields={self._fields}, "
            f"conditions={len(self._conditions)})"
        )
