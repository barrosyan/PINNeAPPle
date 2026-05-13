"""High-level API for defining custom physics problems.

The user provides:
  - equation name and description
  - coordinate names and bounds
  - field names (unknowns)
  - PDE as a string (e.g. ``"eps * u_xx - k * u + f"``)
  - boundary/initial conditions as simple dicts
  - optional analytical solution for metrics

Everything else (loss generation, data sampling, Arena pipeline) is automatic.

Derivative notation in PDE strings
-----------------------------------
``u_x``  → ∂u/∂x       ``u_xx`` → ∂²u/∂x²
``u_t``  → ∂u/∂t       ``v_xy`` → ∂²v/∂x∂y
``p_yy`` → ∂²p/∂y²     etc.

All coordinate names and param names are available in the expression.
Math functions available: ``sin``, ``cos``, ``exp``, ``sqrt``, ``tanh``, ``log``,
``abs``, ``pi``, ``e``.

Examples
--------
1-D reaction-diffusion::

    from pinneaple_arena import define_problem
    import numpy as np

    prob = define_problem(
        name="reaction_diffusion",
        description="eps·u'' − k·u = f  on [0,1],  u(0)=u(1)=0",
        coords={"x": (0.0, 1.0)},
        fields=["u"],
        pde="eps * u_xx - k * u + (eps * pi**2 + k) * sin(pi * x)",
        bcs=[
            {"type": "dirichlet", "at": "x_min", "field": "u", "value": 0.0},
            {"type": "dirichlet", "at": "x_max", "field": "u", "value": 0.0},
        ],
        params={"eps": 0.1, "k": 1.0},
        analytical=lambda x, eps=0.1, k=1.0: {"u": np.sin(np.pi * x)},
    )
    prob.solve(models=["VanillaPINN", "SIREN"], epochs=3000)

2-D Poisson::

    prob = define_problem(
        name="poisson_2d",
        coords={"x": (0.0, 1.0), "y": (0.0, 1.0)},
        fields=["u"],
        pde="u_xx + u_yy + 2 * pi**2 * sin(pi * x) * sin(pi * y)",
        bcs=[{"type": "dirichlet", "at": "boundary", "field": "u", "value": 0.0}],
        params={},
        analytical=lambda x, y: {"u": np.sin(np.pi * x) * np.sin(np.pi * y)},
    )
    prob.solve(epochs=4000, output_dir="outputs/poisson/")

1-D+time heat equation::

    prob = define_problem(
        name="heat_1d_custom",
        coords={"x": (0.0, 1.0), "t": (0.0, 1.0)},
        fields=["u"],
        pde="u_t - alpha * u_xx",
        bcs=[
            {"type": "dirichlet", "at": "x_min", "field": "u", "value": 0.0},
            {"type": "dirichlet", "at": "x_max", "field": "u", "value": 0.0},
            {"type": "initial",   "at": "t_min", "field": "u",
             "value": lambda x, **kw: np.sin(np.pi * x)},
        ],
        params={"alpha": 0.1},
    )
    prob.solve()
"""
from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .problems import ArenaProblem, register_problem


# ─────────────────────────────────────────────────────────────────────────────
# PDE string → torch autograd evaluator
# ─────────────────────────────────────────────────────────────────────────────

def _build_pde_evaluator(
    pde_str: str,
    fields: List[str],
    coord_names: List[str],
) -> Callable:
    """Parse a PDE string and return a callable residual evaluator.

    Derivative tokens ``field_coordsuffix`` are replaced with autograd calls
    at parse time; everything else is evaluated dynamically with coordinate
    tensors and param values injected into the eval namespace.
    """
    coord_idx: Dict[str, int] = {c: i for i, c in enumerate(coord_names)}
    coord_chars_re = "".join(re.escape(c) for c in coord_names)

    # Replace derivative tokens: field_suffix → _D(_f["field"], _xy, [indices])
    # Sort fields by length descending to avoid partial matches (e.g. "u" vs "uv")
    processed = pde_str
    for fname in sorted(fields, key=len, reverse=True):
        pattern = re.compile(rf'\b{re.escape(fname)}_([{coord_chars_re}]+)\b')

        def _replace(m, _fname=fname):
            suffix = m.group(1)
            try:
                indices = [coord_idx[c] for c in suffix]
            except KeyError as exc:
                raise ValueError(
                    f"Derivative '{m.group(0)}': character '{exc.args[0]}' "
                    f"is not a coordinate. Coords are: {coord_names}"
                ) from exc
            return f'_D(_f["{_fname}"], _xy, {indices})'

        processed = pattern.sub(_replace, processed)

    pde_code = processed

    def _D(fv, xy, indices):
        """Chain partial derivatives via autograd."""
        import torch
        result = fv
        for idx in indices:
            gs = torch.autograd.grad(
                result, xy,
                grad_outputs=torch.ones_like(result),
                create_graph=True, retain_graph=True,
            )[0]
            result = gs[:, idx:idx + 1]
        return result

    def evaluator(net, xy_int, **params):
        import torch

        _xy = xy_int.clone().requires_grad_(True)
        raw = net(_xy)
        if hasattr(raw, "y"):
            raw = raw.y
        if isinstance(raw, dict):
            raw = torch.stack(list(raw.values()), dim=-1)

        # Field tensors
        _f: Dict[str, Any] = {}
        for fi, fname in enumerate(fields):
            if raw.ndim == 1:
                _f[fname] = raw.unsqueeze(1)
            elif raw.shape[1] == 1:
                _f[fname] = raw
            else:
                _f[fname] = raw[:, fi:fi + 1]

        import torch as _torch
        _ns: Dict[str, Any] = {
            # autograd helper
            "_D": _D, "_f": _f, "_xy": _xy,
            # math
            "sin": _torch.sin, "cos": _torch.cos, "exp": _torch.exp,
            "sqrt": _torch.sqrt, "abs": _torch.abs, "tanh": _torch.tanh,
            "log": _torch.log, "pi": math.pi, "e": math.e,
            # coords as column tensors
            **{cn: _xy[:, ci:ci + 1] for ci, cn in enumerate(coord_names)},
            # bare field names
            **{fn: _f[fn] for fn in fields},
            # physical params
            **{k: float(v) for k, v in params.items()},
            "__builtins__": {},
        }

        try:
            residual = eval(pde_code, _ns)  # noqa: S307 controlled namespace
        except Exception as exc:
            raise RuntimeError(
                f"[define_problem] Failed to evaluate PDE expression:\n"
                f"  original : {pde_str}\n"
                f"  processed: {pde_code}\n"
                f"  error    : {exc}"
            ) from exc

        return (residual ** 2).mean()

    evaluator._original = pde_str
    evaluator._processed = pde_code
    return evaluator


# ─────────────────────────────────────────────────────────────────────────────
# Boundary / initial condition helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sample_bc_points(
    at: str,
    n: int,
    coords: Dict[str, Tuple[float, float]],
    coord_names: List[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample *n* boundary points for a given location specifier.

    ``at`` accepts:
      ``"x_min"``, ``"x_max"``, ``"y_min"``, ``"y_max"``,
      ``"t_min"``, ``"boundary"``, ``"coord=value"`` (e.g. ``"x=0.5"``).
    """
    bounds = [coords[c] for c in coord_names]
    dim = len(coord_names)

    if at == "boundary":
        n_faces = 2 * dim
        n_per = max(n // n_faces, 1)
        parts = []
        for d in range(dim):
            for fixed_val in [bounds[d][0], bounds[d][1]]:
                pts = np.zeros((n_per, dim), dtype=np.float32)
                for d2 in range(dim):
                    if d2 == d:
                        pts[:, d2] = fixed_val
                    else:
                        lo, hi = bounds[d2]
                        pts[:, d2] = rng.uniform(lo, hi, n_per).astype(np.float32)
                parts.append(pts)
        return np.concatenate(parts, axis=0)

    m_mm = re.match(r'^(\w+)_(min|max)$', at)
    if m_mm:
        cname, side = m_mm.group(1), m_mm.group(2)
        if cname not in coord_names:
            raise ValueError(f"BC 'at={at}': '{cname}' not in coords {coord_names}")
        d = coord_names.index(cname)
        fixed_val = bounds[d][0] if side == "min" else bounds[d][1]
        pts = np.zeros((n, dim), dtype=np.float32)
        for d2 in range(dim):
            if d2 == d:
                pts[:, d2] = fixed_val
            else:
                lo, hi = bounds[d2]
                pts[:, d2] = rng.uniform(lo, hi, n).astype(np.float32)
        return pts

    m_eq = re.match(r'^(\w+)=([\d.eE+\-]+)$', at)
    if m_eq:
        cname, val_str = m_eq.group(1), m_eq.group(2)
        if cname not in coord_names:
            raise ValueError(f"BC 'at={at}': '{cname}' not in coords {coord_names}")
        d = coord_names.index(cname)
        fixed_val = float(val_str)
        pts = np.zeros((n, dim), dtype=np.float32)
        for d2 in range(dim):
            if d2 == d:
                pts[:, d2] = fixed_val
            else:
                lo, hi = bounds[d2]
                pts[:, d2] = rng.uniform(lo, hi, n).astype(np.float32)
        return pts

    raise ValueError(
        f"Cannot parse BC location '{at}'. "
        "Use: 'x_min', 'x_max', 'y_min', 'y_max', 't_min', 'boundary', or 'x=0.5'."
    )


def _eval_bc_value(
    val: Any,
    pts: np.ndarray,
    coord_names: List[str],
    params: Dict[str, float],
) -> np.ndarray:
    """Evaluate a BC value spec at boundary points → (N,) float32 array."""
    n = len(pts)

    if isinstance(val, (int, float)):
        return np.full(n, float(val), dtype=np.float32)

    if isinstance(val, str):
        result = np.zeros(n, dtype=np.float32)
        for i, pt in enumerate(pts):
            ns = {cn: float(pt[ci]) for ci, cn in enumerate(coord_names)}
            ns.update(params)
            ns.update({"sin": math.sin, "cos": math.cos, "pi": math.pi,
                        "exp": math.exp, "sqrt": math.sqrt, "__builtins__": {}})
            result[i] = float(eval(val, ns))  # noqa: S307
        return result

    if callable(val):
        coord_arrays = {cn: pts[:, ci] for ci, cn in enumerate(coord_names)}
        v = val(**coord_arrays, **{k: float(p) for k, p in params.items()})
        return np.asarray(v, dtype=np.float32).ravel()

    return np.zeros(n, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# EasyArenaProblem
# ─────────────────────────────────────────────────────────────────────────────

class EasyArenaProblem(ArenaProblem):
    """Auto-generated ArenaProblem from :func:`define_problem`.

    Do not instantiate directly — use :func:`define_problem` instead.
    """

    def __init__(
        self,
        name: str,
        description: str,
        coords: Dict[str, Tuple[float, float]],
        fields: List[str],
        pde_evaluator: Callable,
        bcs: List[Dict[str, Any]],
        params: Dict[str, float],
        analytical_fn: Optional[Callable],
        bc_weight: float,
    ):
        self.name = name
        self.description = description
        self.domain = "Custom"
        self._coords = coords
        self._coord_names = list(coords.keys())
        self._fields = fields
        self.input_dim = len(self._coord_names)
        self.output_dim = len(fields)
        self._pde_eval = pde_evaluator
        self._bcs = bcs
        self._params = dict(params)
        self._analytical_fn = analytical_fn
        self._bc_weight = float(bc_weight)

    # ── ArenaProblem interface ────────────────────────────────────────────────

    def analytical(self, *args, **kwargs) -> Optional[Dict[str, np.ndarray]]:
        if self._analytical_fn is None:
            return None
        coord_kw = {cn: args[i] for i, cn in enumerate(self._coord_names) if i < len(args)}
        coord_kw.update({k: v for k, v in kwargs.items() if k in self._coord_names})
        merged = {**self._params, **kwargs}
        return self._analytical_fn(**coord_kw, **{k: v for k, v in merged.items()
                                                   if k not in coord_kw})

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, **params):
        import torch
        merged = {**self._params, **params}

        pde_loss = self._pde_eval(net, xy_int, **merged)

        bc_out = net(xy_bc)
        if hasattr(bc_out, "y"):
            bc_out = bc_out.y
        if isinstance(bc_out, dict):
            bc_out = torch.stack(list(bc_out.values()), dim=-1)
        if bc_out.ndim == 1:
            bc_out = bc_out.unsqueeze(1)
        if uv_bc.shape[1] < bc_out.shape[1]:
            bc_out = bc_out[:, :uv_bc.shape[1]]

        bc_loss = ((bc_out - uv_bc) ** 2).mean()
        return pde_loss, self._bc_weight * bc_loss

    def supervised_data(
        self, n_train: int = 400, n_bc: int = 200, grid_n: int = 40, **params
    ):
        merged = {**self._params, **params}
        rng = np.random.default_rng(42)
        coord_names = self._coord_names
        bounds = [self._coords[c] for c in coord_names]
        dim = len(coord_names)

        # ── Interior points ───────────────────────────────────────────────────
        xy_int = np.column_stack([
            rng.uniform(lo, hi, n_train).astype(np.float32)
            for lo, hi in bounds
        ])

        if self._analytical_fn is not None:
            coord_kw = {cn: xy_int[:, ci] for ci, cn in enumerate(coord_names)}
            sol = self._analytical_fn(**coord_kw,
                                      **{k: float(v) for k, v in merged.items()
                                         if k not in coord_kw})
            Y_int = np.column_stack([
                np.asarray(sol[f], dtype=np.float32).ravel() for f in self._fields
            ])
        else:
            Y_int = np.zeros((n_train, self.output_dim), dtype=np.float32)

        # ── Boundary / initial conditions ─────────────────────────────────────
        n_per_bc = max(n_bc // max(len(self._bcs), 1), 4)
        bc_x_parts: List[np.ndarray] = []
        bc_y_parts: List[np.ndarray] = []

        for bc in self._bcs:
            at = bc.get("at", "boundary")
            pts = _sample_bc_points(at, n_per_bc, self._coords, coord_names, rng)

            Y_pts = np.zeros((len(pts), self.output_dim), dtype=np.float32)
            fname = bc.get("field", self._fields[0])
            fi = self._fields.index(fname) if fname in self._fields else 0
            Y_pts[:, fi] = _eval_bc_value(
                bc.get("value", 0.0), pts, coord_names, merged
            )
            bc_x_parts.append(pts)
            bc_y_parts.append(Y_pts)

        if bc_x_parts:
            xy_bc = np.concatenate(bc_x_parts, axis=0)
            Y_bc = np.concatenate(bc_y_parts, axis=0)
        else:
            xy_bc = np.zeros((0, dim), dtype=np.float32)
            Y_bc = np.zeros((0, self.output_dim), dtype=np.float32)

        # ── Evaluation grid ───────────────────────────────────────────────────
        if dim == 1:
            xy_eval = np.linspace(bounds[0][0], bounds[0][1], grid_n,
                                  dtype=np.float32).reshape(-1, 1)
        elif dim == 2:
            g0 = np.linspace(bounds[0][0], bounds[0][1], grid_n, dtype=np.float32)
            g1 = np.linspace(bounds[1][0], bounds[1][1], grid_n, dtype=np.float32)
            G0, G1 = np.meshgrid(g0, g1)
            xy_eval = np.stack([G0.ravel(), G1.ravel()], axis=1)
        else:
            # For 3D+ just random uniform eval points
            n_eval = grid_n ** 2
            xy_eval = np.column_stack([
                rng.uniform(lo, hi, n_eval).astype(np.float32)
                for lo, hi in bounds
            ])

        if self._analytical_fn is not None:
            coord_kw = {cn: xy_eval[:, ci] for ci, cn in enumerate(coord_names)}
            sol = self._analytical_fn(**coord_kw,
                                      **{k: float(v) for k, v in merged.items()
                                         if k not in coord_kw})
            Y_eval = np.column_stack([
                np.asarray(sol[f], dtype=np.float32).ravel() for f in self._fields
            ])
        else:
            Y_eval = np.zeros((len(xy_eval), self.output_dim), dtype=np.float32)

        return xy_int, Y_int, xy_bc, Y_bc, xy_eval, Y_eval, self._fields

    # ── High-level solve() ────────────────────────────────────────────────────

    def solve(
        self,
        models: Optional[List[Union[str, Dict[str, Any]]]] = None,
        epochs: int = 3000,
        lr: float = 1e-3,
        hidden: Optional[List[int]] = None,
        output_dir: str = "outputs/custom/",
        prefix: Optional[str] = None,
        grid_n: int = 40,
        n_col: int = 1000,
        n_bc: int = 200,
        n_train: int = 400,
        dark_theme: bool = True,
        save_figures: bool = True,
        show: bool = False,
        uq: bool = False,
    ):
        """Train models and produce figures in one call.

        Parameters
        ----------
        models     : list of model name strings or full model config dicts.
                     Defaults to ``["VanillaPINN", "SIREN"]``.
        epochs     : training epochs per model.
        lr         : learning rate.
        hidden     : MLP hidden layer sizes. Default ``[64, 64, 64]``.
        output_dir : directory for saved figures and logs.
        prefix     : filename prefix (defaults to ``name``).
        grid_n     : evaluation grid resolution.
        n_col      : collocation points.
        n_bc       : boundary points.
        n_train    : supervised interior points.
        dark_theme : dark matplotlib theme.
        save_figures : save PNG figures to ``output_dir``.
        show       : display figures interactively.
        uq         : enable Monte-Carlo dropout uncertainty quantification.

        Returns
        -------
        Arena  instance after ``run()``.
        """
        from .arena import Arena
        from .config import ArenaConfig

        if models is None:
            models = ["VanillaPINN", "SIREN"]
        if hidden is None:
            hidden = [64, 64, 64]
        if prefix is None:
            prefix = re.sub(r'\W+', '_', self.name)

        model_cfgs: List[Dict[str, Any]] = []
        for m in models:
            if isinstance(m, dict):
                model_cfgs.append(m)
            else:
                mtype = m.lower().replace(" ", "_").replace("-", "_")
                model_cfgs.append({
                    "name": m,
                    "type": mtype,
                    "network": {"hidden": hidden, "activation": "tanh"},
                    "training": {"epochs": epochs, "lr": lr},
                })

        cfg_dict: Dict[str, Any] = {
            "problem": {
                "name": self.name,
                "params": self._params,
                "grid_n": grid_n,
                "n_col": n_col,
                "n_bc": n_bc,
                "n_train_supervised": n_train,
            },
            "models": model_cfgs,
            "output": {
                "dir": output_dir,
                "prefix": prefix,
                "save_figures": save_figures,
                "dark_theme": dark_theme,
                "show": show,
            },
        }
        if uq:
            cfg_dict["uq"] = {"enabled": True, "method": "mc_dropout", "n_samples": 50}

        arena = Arena(ArenaConfig.from_dict(cfg_dict))
        arena.run()
        return arena

    def __repr__(self) -> str:
        return (
            f"EasyArenaProblem(name={self.name!r}, "
            f"coords={list(self._coords.keys())}, "
            f"fields={self._fields})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public factory
# ─────────────────────────────────────────────────────────────────────────────

def define_problem(
    name: str,
    coords: Dict[str, Tuple[float, float]],
    fields: List[str],
    pde: str,
    bcs: List[Dict[str, Any]],
    params: Optional[Dict[str, float]] = None,
    description: str = "",
    analytical: Optional[Callable] = None,
    bc_weight: float = 10.0,
) -> EasyArenaProblem:
    """Define a custom physics problem with a declarative, high-level API.

    All heavy lifting — loss compilation, training data generation, Arena
    pipeline construction — is handled automatically.

    Parameters
    ----------
    name        : unique problem identifier (used in output filenames).
    coords      : ordered mapping of coordinate name → (min, max) bounds.

                  Examples::

                      {"x": (0.0, 1.0)}                     # 1D spatial
                      {"x": (0.0, 1.0), "y": (0.0, 1.0)}   # 2D spatial
                      {"x": (-1.0, 1.0), "t": (0.0, 1.0)}  # 1D + time
                      {"x": (0,1), "y": (0,1), "z": (0,1)} # 3D

    fields      : list of unknown field names (e.g. ``["u"]``, ``["u", "v", "p"]``).

    pde         : PDE as a string expression **equal to zero** (residual form).

                  Derivative notation::

                      u_x   → ∂u/∂x        u_xx  → ∂²u/∂x²
                      u_t   → ∂u/∂t        v_xy  → ∂²v/∂x∂y

                  Math available: ``sin``, ``cos``, ``exp``, ``sqrt``,
                  ``tanh``, ``log``, ``abs``, ``pi``, ``e``.

                  All coord names and param names are in scope.

    bcs         : list of boundary/initial condition dicts::

                      {
                          "type" : "dirichlet",   # ignored by solver, kept for docs
                          "at"   : "x_min",       # location specifier (see below)
                          "field": "u",           # which field
                          "value": 0.0,           # float | str-expr | callable
                      }

                  ``at`` options:

                  * ``"x_min"`` / ``"x_max"`` / ``"y_min"`` / … — axis-aligned boundary
                  * ``"t_min"`` — initial condition (alias for ``"t_min"``)
                  * ``"boundary"`` — all exterior faces
                  * ``"x=0.5"`` — arbitrary fixed value of a coord

                  ``value`` can be:

                  * a **float** — constant
                  * a **string** expression using coord names and params
                  * a **callable** ``f(**coords, **params) → array``

    params      : dict of named scalar parameters available in the PDE string.
    description : human-readable description (shown in plots).
    analytical  : callable ``f(**coord_arrays, **params) → dict[field_name, array]``
                  providing the exact solution. Used for L2/Linf metrics and as
                  supervised interior targets.
    bc_weight   : multiplier on the BC loss term (default 10).

    Returns
    -------
    EasyArenaProblem
        A fully configured problem with a ``.solve()`` convenience method.

    Quick example
    -------------
    ::

        import numpy as np
        from pinneaple_arena import define_problem

        prob = define_problem(
            name="my_poisson",
            coords={"x": (0.0, 1.0), "y": (0.0, 1.0)},
            fields=["u"],
            pde="u_xx + u_yy + 2 * pi**2 * sin(pi * x) * sin(pi * y)",
            bcs=[{"type": "dirichlet", "at": "boundary", "field": "u", "value": 0.0}],
            params={},
            analytical=lambda x, y: {"u": np.sin(np.pi * x) * np.sin(np.pi * y)},
        )
        prob.solve(models=["VanillaPINN", "SIREN"], epochs=4000,
                   output_dir="outputs/my_poisson/")
    """
    if params is None:
        params = {}

    coord_names = list(coords.keys())
    evaluator = _build_pde_evaluator(pde, fields, coord_names)

    problem = EasyArenaProblem(
        name=name,
        description=description or f"Custom problem '{name}'",
        coords=coords,
        fields=fields,
        pde_evaluator=evaluator,
        bcs=bcs,
        params=params,
        analytical_fn=analytical,
        bc_weight=bc_weight,
    )

    register_problem(problem)
    return problem
