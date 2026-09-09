"""``ProblemSpec`` -> real DeepXDE ``dde.Model`` translation and training.

Translates a real PINNeAPPle :class:`~pinneapple_physics.pde_environment.spec.ProblemSpec`
(coords, fields, :class:`~pinneapple_physics.pde_environment.spec.PDETermSpec`,
and :class:`~pinneapple_physics.pde_environment.conditions.ConditionSpec`
boundary/initial conditions) into DeepXDE's own real objects -- a
``dde.geometry`` geometry, a ``dde.data.PDE``/``dde.data.TimePDE``, a
``dde.nn.FNN``, and a ``dde.Model`` -- and actually runs DeepXDE's own
``.train()`` on it. This is a genuine alternative solving backend, not a
re-implementation of PINN training: DeepXDE does the sampling, the network,
the optimizer loop, and the loss bookkeeping; this module only does the
one-time translation at the boundary.

Only two ``pde.kind`` values are translated for real here: ``"burgers"``
(1D viscous Burgers, e.g. the ``burgers_1d`` preset) and ``"poisson"`` (2D
Poisson, e.g. the ``poisson_2d`` preset) -- both residual forms are taken
directly from ``pinneapple_physics.pinn_solver.compiler.compile`` so the
sign conventions match PINNeAPPle's own native compiler
(``u_t + u*u_x - nu*u_xx = 0`` for Burgers; ``laplacian(u) - f = 0`` for
Poisson, ``f`` defaulting to zero). Any other ``pde.kind`` raises a clear
``NotImplementedError`` naming the unsupported kind rather than silently
mistranslating it -- generalizing this to every PDE kind PINNeAPPle's own
compiler supports is future work, not something this bridge fakes.

Condition translation (``ConditionSpec`` -> ``dde.icbc.*``) supports
``"dirichlet"``, ``"neumann"``, and ``"initial"`` conditions whose
``selector_type`` is ``"all"``, ``"callable"``, or ``"tag"`` with
``tag == "boundary"`` (the convention used by e.g. ``poisson_2d``'s BC,
meaning "the geometric domain boundary", which is exactly what DeepXDE's
own ``geom.on_boundary`` already computes). Other tags, condition kinds
(``"robin"``, ``"interface"``, ``"data"``), or multi-field conditions raise
``NotImplementedError`` rather than being silently dropped.

Optional dependency
--------------------
``deepxde`` is genuinely optional: importing this module never requires it.
Only :func:`solve_with_deepxde` (and :func:`require_deepxde`) perform a real
``import deepxde`` and raise a clear, actionable ``ImportError`` if it is
missing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch

from pinneapple_physics.pde_environment.conditions import ConditionSpec
from pinneapple_physics.pde_environment.spec import ProblemSpec

# PDE kinds this bridge can honestly translate into a real DeepXDE residual.
SUPPORTED_PDE_KINDS = ("burgers", "poisson")


# ---------------------------------------------------------------------------
# Optional deepxde import
# ---------------------------------------------------------------------------

def require_deepxde() -> Any:
    """Import ``deepxde``.

    Raises a clear, actionable ``ImportError`` if it is not installed. This
    is the ONLY place in :mod:`pinneapple_physics.deepxde_bridge` that
    performs a real ``import deepxde``.
    """
    try:
        import deepxde  # type: ignore
        return deepxde
    except ImportError as e:
        raise ImportError(
            "deepxde is required for pinneapple_physics.deepxde_bridge."
            "solve_with_deepxde but is not installed. Install with: "
            "pip install deepxde (see https://github.com/lululxvi/deepxde). "
            "Note: importing pinneapple_physics.deepxde_bridge itself never "
            "requires deepxde -- only actually solving a problem with it "
            "does."
        ) from e


def is_deepxde_available() -> bool:
    """Return ``True`` if ``deepxde`` can be imported."""
    try:
        require_deepxde()
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class DeepXDESolveResult:
    """Outcome of :func:`solve_with_deepxde`.

    Attributes
    ----------
    deepxde_model:
        The real, trained ``dde.Model`` (call ``.predict(x)`` on it directly).
    spec:
        The :class:`ProblemSpec` that was translated and solved.
    final_loss:
        Sum of the training-loss components (``train_state.loss_train``) at
        the last recorded training step.
    train_time_s:
        Wall-clock seconds spent in ``dde.Model.train()``.
    """

    deepxde_model: Any
    spec: ProblemSpec
    final_loss: float
    train_time_s: float


# ---------------------------------------------------------------------------
# Geometry translation
# ---------------------------------------------------------------------------

def _build_geometry(spec: ProblemSpec, dde: Any) -> tuple:
    """Return ``(geometry, is_time_dependent)`` from ``spec.coords``/``domain_bounds``.

    Time-dependent problems (``"t"`` in ``spec.coords``) build a
    ``dde.geometry.GeometryXTime`` over the remaining spatial coordinate(s)
    (1D -> ``Interval``, 2D -> ``Rectangle``) and a ``dde.geometry.TimeDomain``.
    Static problems build the spatial geometry directly.
    """
    coords = spec.coords
    bounds = spec.domain_bounds
    missing = [c for c in coords if c not in bounds]
    if missing:
        raise ValueError(
            f"solve_with_deepxde requires spec.domain_bounds for every coord; "
            f"missing bounds for {missing} (spec={spec.name!r})"
        )

    is_time = "t" in coords
    spatial_coords = [c for c in coords if c != "t"]

    if len(spatial_coords) == 1:
        (c,) = spatial_coords
        lo, hi = bounds[c]
        spatial_geom = dde.geometry.Interval(float(lo), float(hi))
    elif len(spatial_coords) == 2:
        c1, c2 = spatial_coords
        lo1, hi1 = bounds[c1]
        lo2, hi2 = bounds[c2]
        spatial_geom = dde.geometry.Rectangle(
            [float(lo1), float(lo2)], [float(hi1), float(hi2)]
        )
    else:
        raise NotImplementedError(
            f"solve_with_deepxde only translates 1D or 2D spatial geometry; "
            f"got spatial coords {spatial_coords} for spec {spec.name!r}"
        )

    if not is_time:
        return spatial_geom, False

    t_lo, t_hi = bounds["t"]
    timedomain = dde.geometry.TimeDomain(float(t_lo), float(t_hi))
    geomtime = dde.geometry.GeometryXTime(spatial_geom, timedomain)
    return geomtime, True


# ---------------------------------------------------------------------------
# Condition translation (ConditionSpec -> dde.icbc.*)
# ---------------------------------------------------------------------------

def _make_value_func(cond: ConditionSpec) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap ``cond.values`` (which takes ``(X, ctx)``) as DeepXDE's ``func(X)``.

    DeepXDE inspects the wrapped function's *signature* (via
    ``utils.get_num_args``, which counts all parameters, defaults included)
    to decide whether to call it as ``func(X)`` or ``func(X, aux_var)`` --
    so this must be a plain one-parameter closure, not a function with a
    second (even defaulted) parameter binding ``cond``.
    """

    def _func(X: np.ndarray) -> np.ndarray:
        return cond.values(np.asarray(X), {})

    return _func


def _make_predicate(cond: ConditionSpec, *, kind: str) -> Callable[[np.ndarray, bool], bool]:
    """Build DeepXDE's per-point ``on_boundary(x, on)`` / ``on_initial(x, on)`` predicate.

    - ``selector_type == "callable"``: evaluate ``cond.selector`` on the
      single point directly (DeepXDE calls this once per point).
    - ``selector_type == "all"``: defer entirely to DeepXDE's own geometry
      flag (``on``), matching "applies to all points of the corresponding
      set".
    - ``selector_type == "tag"`` with ``tag == "boundary"``: also defers to
      DeepXDE's own geometry flag -- this is the one PINNeAPPle tag
      convention (see e.g. the ``poisson_2d`` preset) that means exactly
      "the geometric domain boundary", i.e. what ``geom.on_boundary``
      already computes. Any other tag cannot be translated without the
      native compiler's ``ctx["tag_masks"]``, which this bridge does not
      have, so it raises ``NotImplementedError``.
    """
    if cond.selector_type == "callable":
        selector = cond.selector

        def _pred(x: np.ndarray, on: bool) -> bool:
            return bool(selector(np.asarray(x)[None, :], {})[0])

        return _pred

    if cond.selector_type == "all":
        return lambda x, on: bool(on)

    if cond.selector_type == "tag":
        tag = cond.selector.get("tag") if isinstance(cond.selector, dict) else None
        if tag == "boundary":
            return lambda x, on: bool(on)
        raise NotImplementedError(
            f"solve_with_deepxde cannot translate condition {cond.name!r} "
            f"(selector_type='tag', tag={tag!r}) -- only tag='boundary' "
            "(meaning DeepXDE's own geometry.on_boundary) is supported "
            "without the native compiler's ctx['tag_masks']."
        )

    raise NotImplementedError(
        f"solve_with_deepxde does not support selector_type={cond.selector_type!r} "
        f"(condition {cond.name!r})"
    )


def _translate_conditions(
    spec: ProblemSpec, geom: Any, is_time: bool, dde: Any
) -> list:
    icbcs = []
    for cond in spec.conditions:
        if len(cond.fields) != 1:
            raise NotImplementedError(
                f"solve_with_deepxde only supports single-field boundary/"
                f"initial conditions; condition {cond.name!r} has "
                f"fields={cond.fields}"
            )
        field = cond.fields[0]
        if field not in spec.fields:
            raise ValueError(
                f"Condition {cond.name!r} references unknown field {field!r}; "
                f"spec.fields={spec.fields}"
            )
        component = spec.fields.index(field)
        func = _make_value_func(cond)

        if cond.kind == "dirichlet":
            pred = _make_predicate(cond, kind="boundary")
            icbcs.append(dde.icbc.DirichletBC(geom, func, pred, component=component))
        elif cond.kind == "neumann":
            if cond.order != 1 or cond.deriv_coord is not None:
                raise NotImplementedError(
                    f"solve_with_deepxde only translates order=1 Neumann "
                    f"conditions (normal derivative); condition "
                    f"{cond.name!r} has order={cond.order}, "
                    f"deriv_coord={cond.deriv_coord!r}"
                )
            pred = _make_predicate(cond, kind="boundary")
            icbcs.append(dde.icbc.NeumannBC(geom, func, pred, component=component))
        elif cond.kind == "initial":
            if not is_time:
                raise ValueError(
                    f"Condition {cond.name!r} is an initial condition but "
                    f"spec {spec.name!r} has no 't' coordinate"
                )
            pred = _make_predicate(cond, kind="initial")
            icbcs.append(dde.icbc.IC(geom, func, pred, component=component))
        else:
            raise NotImplementedError(
                f"solve_with_deepxde does not support condition kind "
                f"{cond.kind!r} (condition {cond.name!r}); supported "
                "kinds: 'dirichlet', 'neumann', 'initial'"
            )
    return icbcs


# ---------------------------------------------------------------------------
# PDE residual translation (pde.kind -> a real dde-callable residual)
# ---------------------------------------------------------------------------

def _build_pde_residual(
    spec: ProblemSpec, dde: Any, *, source_fn: Optional[Callable] = None
) -> Callable:
    """Build the ``pde(x, y) -> residual`` callable DeepXDE's data classes need.

    Only ``"burgers"`` and ``"poisson"`` are translated; both residual forms
    match ``pinneapple_physics.pinn_solver.compiler.compile`` exactly (same
    sign conventions), so a spec solved via this bridge and via the native
    compiler are solving the same equation.
    """
    kind = spec.pde.kind
    coords = spec.coords
    fields = spec.fields

    if kind == "burgers":
        if fields != ("u",):
            raise NotImplementedError(
                f"solve_with_deepxde's 'burgers' translation only supports "
                f"a single field named 'u'; got fields={fields}"
            )
        if "x" not in coords or "t" not in coords:
            raise ValueError(
                f"'burgers' requires coords ('x', 't'); got {coords}"
            )
        nu = float(spec.pde.params.get("nu", 0.01))
        ix = coords.index("x")
        it = coords.index("t")

        def pde(x, y):
            dy_x = dde.grad.jacobian(y, x, i=0, j=ix)
            dy_t = dde.grad.jacobian(y, x, i=0, j=it)
            dy_xx = dde.grad.hessian(y, x, component=0, i=ix, j=ix)
            return dy_t + y * dy_x - nu * dy_xx

        return pde

    if kind == "poisson":
        if fields != ("u",):
            raise NotImplementedError(
                f"solve_with_deepxde's 'poisson' translation only supports "
                f"a single field named 'u'; got fields={fields}"
            )
        if "x" not in coords or "y" not in coords:
            raise ValueError(
                f"'poisson' requires coords ('x', 'y'); got {coords}"
            )
        ix = coords.index("x")
        iy = coords.index("y")
        src = source_fn or spec.pde.meta.get("source_fn") or spec.meta.get("source_fn")

        def pde(x, y):
            dy_xx = dde.grad.hessian(y, x, component=0, i=ix, j=ix)
            dy_yy = dde.grad.hessian(y, x, component=0, i=iy, j=iy)
            laplacian = dy_xx + dy_yy
            if src is None:
                return laplacian
            x_np = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
            f_np = np.asarray(src(x_np, {}), dtype=np.float32)
            if f_np.ndim == 1:
                f_np = f_np[:, None]
            f = torch.as_tensor(f_np, dtype=laplacian.dtype, device=laplacian.device)
            return laplacian - f

        return pde

    raise NotImplementedError(
        f"solve_with_deepxde does not support pde.kind={kind!r} "
        f"(spec {spec.name!r}); supported kinds: {SUPPORTED_PDE_KINDS}"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def solve_with_deepxde(
    spec: ProblemSpec,
    *,
    n_domain: int = 2000,
    n_boundary: int = 200,
    n_initial: int = 100,
    layers: Sequence[int] = (50, 50, 50, 50),
    epochs: int = 10000,
    activation: str = "tanh",
    kernel_initializer: str = "Glorot normal",
    optimizer: str = "adam",
    lr: float = 1e-3,
    source_fn: Optional[Callable[[np.ndarray, dict], np.ndarray]] = None,
    verbose: int = 0,
    **deepxde_kwargs: Any,
) -> DeepXDESolveResult:
    """Translate ``spec`` into a real DeepXDE problem and train it for real.

    Builds a ``dde.geometry`` geometry from ``spec.coords``/``domain_bounds``,
    a residual callable from ``spec.pde`` (see :func:`_build_pde_residual`
    for the supported ``pde.kind`` values), boundary/initial conditions from
    ``spec.conditions`` (see :func:`_translate_conditions`), a
    ``dde.data.PDE``/``dde.data.TimePDE``, a ``dde.nn.FNN`` sized
    ``[len(coords)] + list(layers) + [len(fields)]``, and a ``dde.Model`` --
    then calls DeepXDE's own ``model.compile()``/``model.train()``.

    Parameters
    ----------
    spec:
        The problem to solve. Only ``spec.pde.kind in {"burgers", "poisson"}``
        is translated; anything else raises ``NotImplementedError`` naming
        the unsupported kind.
    n_domain, n_boundary, n_initial:
        Passed through to ``dde.data.PDE``/``TimePDE`` as
        ``num_domain``/``num_boundary``/``num_initial``. ``n_initial`` is
        ignored for time-independent problems (no ``dde.data.PDE`` argument
        for it).
    layers:
        Hidden-layer widths of the ``dde.nn.FNN``.
    epochs:
        Training iterations, passed to ``model.train(iterations=epochs)``.
    activation, kernel_initializer:
        Passed to ``dde.nn.FNN``.
    optimizer, lr:
        Passed to ``model.compile()``.
    source_fn:
        For ``pde.kind == "poisson"`` only: ``f(X, ctx) -> array`` giving the
        RHS forcing (``laplacian(u) - f = 0``); defaults to
        ``spec.pde.meta.get("source_fn")`` and then zero, matching
        ``pinneapple_physics.pinn_solver.compiler.compile``'s convention.
    verbose:
        Passed to both ``model.compile()`` and ``model.train()``.
    **deepxde_kwargs:
        Forwarded to ``model.train()`` (e.g. ``display_every``, ``callbacks``).

    Raises
    ------
    ImportError
        If ``deepxde`` is not installed.
    NotImplementedError
        If ``spec.pde.kind``, a condition kind/selector, or the geometry
        shape cannot be honestly translated.
    """
    dde = require_deepxde()

    geom, is_time = _build_geometry(spec, dde)
    pde_fn = _build_pde_residual(spec, dde, source_fn=source_fn)
    icbcs = _translate_conditions(spec, geom, is_time, dde)

    layer_sizes = [len(spec.coords)] + list(layers) + [len(spec.fields)]
    net = dde.nn.FNN(layer_sizes, activation, kernel_initializer)

    if is_time:
        data = dde.data.TimePDE(
            geom,
            pde_fn,
            icbcs,
            num_domain=n_domain,
            num_boundary=n_boundary,
            num_initial=n_initial,
        )
    else:
        data = dde.data.PDE(
            geom,
            pde_fn,
            icbcs,
            num_domain=n_domain,
            num_boundary=n_boundary,
        )

    model = dde.Model(data, net)
    model.compile(optimizer, lr=lr, verbose=verbose)

    start = time.perf_counter()
    _losshistory, train_state = model.train(
        iterations=epochs, verbose=verbose, **deepxde_kwargs
    )
    train_time_s = time.perf_counter() - start

    final_loss = float(np.sum(np.asarray(train_state.loss_train, dtype=np.float64)))

    return DeepXDESolveResult(
        deepxde_model=model,
        spec=spec,
        final_loss=final_loss,
        train_time_s=train_time_s,
    )
