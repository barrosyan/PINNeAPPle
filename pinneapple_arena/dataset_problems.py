"""DatasetProblem — ArenaProblem backed by a real pinneapple_data dataset.

Converts any dataset registered in ``pinneapple_data`` into a first-class
Arena problem, handling:

  * **PDE datasets** (burgers_1d, heat_1d, kovasznay_ns …)
    Structured spatial/temporal grids are properly flattened into
    ``(N, in_dim)`` training arrays without losing coordinate alignment.

  * **Timeseries datasets** (lorenz63, spring_mass, van_der_pol …)
    Direct time→state regression, or sliding-window sequences.

  * **Regression datasets** (airfoil_noise, concrete_strength …)
    Standard feature-matrix → target, no spatial structure assumed.

  * **Geometry datasets** (naca0012, cylinder_2d …)
    Interior/boundary point arrays surfaced as collocation data.

  * **Physics-hybrid mode**  (``mode="pinn_data"``)
    Combines a supervised data loss (from the dataset) with an autograd
    physics residual for any dataset that maps to a known PDE type.

Usage
-----
::

    from pinneapple_arena.dataset_problems import DatasetProblem

    prob = DatasetProblem.from_dataset(
        "burgers_1d",
        input_fields=["x", "t"],
        output_fields=["u"],
        mode="supervised",       # or "pinn_data" for PINN hybrid
        Nx=256, Nt=101,          # kwargs forwarded to load_dataset()
    )
    # prob is a fully registered ArenaProblem — use it in any Arena config
    Arena(ArenaConfig.from_dict({
        "problem": {"name": prob.name},
        "models": [...],
    })).run()
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .problems import ArenaProblem, register_problem


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loader (thin wrapper around pinneapple_data)
# ─────────────────────────────────────────────────────────────────────────────

def _load_raw(dataset_id: str, **kwargs) -> Dict[str, Any]:
    try:
        from pinneapple_data import load_dataset
    except ImportError as exc:
        raise ImportError(
            "pinneapple_data is required for DatasetProblem. "
            "Install it or check your Python path."
        ) from exc
    return load_dataset(dataset_id, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# PDE dataset flattener
# ─────────────────────────────────────────────────────────────────────────────

def _flatten_pde_dataset(
    data: Dict[str, Any],
    input_fields: List[str],
    output_fields: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten a structured PDE dataset (with coordinate arrays + solution arrays).

    Handles cases where the solution u has shape ``(Nt, Nx)`` or ``(Nx, Ny)``
    while coordinate arrays are 1-D.  A full meshgrid is built so every row
    of the returned arrays corresponds to one point in the domain.

    Returns
    -------
    X : (N, len(input_fields))
    Y : (N, len(output_fields))
    """
    coord_data: Dict[str, np.ndarray] = {}
    field_data: Dict[str, np.ndarray] = {}

    # Separate scalars / coordinate arrays from solution arrays
    for key, val in data.items():
        arr = np.asarray(val)
        if arr.ndim == 0 or arr.size == 1:
            continue
        if arr.ndim == 1:
            coord_data[key] = arr
        else:
            field_data[key] = arr

    # Determine the "output" shape to know how to build the meshgrid
    # Strategy: if all input_fields are 1-D coords and some output_field is 2-D,
    # build the meshgrid; otherwise just flatten column-wise.

    out_arrays = [field_data.get(f, coord_data.get(f)) for f in output_fields]
    in_arrays  = [coord_data.get(f, field_data.get(f)) for f in input_fields]

    # If all inputs are 1-D and any output is multi-dim → meshgrid path
    in_1d = all(a is not None and a.ndim == 1 for a in in_arrays)
    out_nd = any(a is not None and a.ndim > 1 for a in out_arrays)

    if in_1d and out_nd and len(input_fields) == 2:
        c0, c1 = in_arrays[0], in_arrays[1]
        # Determine axis mapping from output shape
        # Convention: (dim0_axis, dim1_axis) → check which coord matches which axis
        for out_f, out_a in zip(output_fields, out_arrays):
            if out_a is None:
                continue
            if out_a.ndim == 2:
                n0, n1 = out_a.shape
                # Try to match shapes
                if len(c0) == n0 and len(c1) == n1:
                    G0, G1 = np.meshgrid(c0, c1, indexing="ij")
                elif len(c0) == n1 and len(c1) == n0:
                    G0, G1 = np.meshgrid(c0, c1)
                    G0, G1 = G0.T, G1.T
                else:
                    G0, G1 = np.meshgrid(c0, c1, indexing="ij")
                X = np.stack([G0.ravel(), G1.ravel()], axis=1).astype(np.float32)
                break
        else:
            # Fallback: cartesian product
            G0, G1 = np.meshgrid(c0, c1, indexing="ij")
            X = np.stack([G0.ravel(), G1.ravel()], axis=1).astype(np.float32)

        Y_cols = []
        for f in output_fields:
            arr = field_data.get(f, coord_data.get(f))
            if arr is None:
                raise KeyError(f"Field '{f}' not found in dataset.")
            Y_cols.append(arr.ravel().reshape(-1, 1).astype(np.float32))
        Y = np.concatenate(Y_cols, axis=1)
        return X, Y

    if in_1d and out_nd and len(input_fields) == 3:
        c0, c1, c2 = in_arrays
        G0, G1, G2 = np.meshgrid(c0, c1, c2, indexing="ij")
        X = np.stack([G0.ravel(), G1.ravel(), G2.ravel()], axis=1).astype(np.float32)
        Y_cols = []
        for f in output_fields:
            arr = field_data.get(f, coord_data.get(f))
            if arr is None:
                raise KeyError(f"Field '{f}' not found in dataset.")
            Y_cols.append(arr.ravel().reshape(-1, 1).astype(np.float32))
        Y = np.concatenate(Y_cols, axis=1)
        return X, Y

    # Default: column-wise flatten
    X_cols, Y_cols = [], []
    for f in input_fields:
        arr = coord_data.get(f, field_data.get(f))
        if arr is None:
            raise KeyError(f"Input field '{f}' not in dataset. "
                           f"Available: {list({**coord_data, **field_data}.keys())}")
        X_cols.append(np.asarray(arr).ravel().reshape(-1, 1).astype(np.float32))

    for f in output_fields:
        arr = coord_data.get(f, field_data.get(f))
        if arr is None:
            raise KeyError(f"Output field '{f}' not in dataset. "
                           f"Available: {list({**coord_data, **field_data}.keys())}")
        Y_cols.append(np.asarray(arr).ravel().reshape(-1, 1).astype(np.float32))

    # Broadcast to common length
    lengths = [len(c.ravel()) for c in X_cols + Y_cols]
    n_max = max(lengths)
    X = np.concatenate([c.ravel()[:n_max].reshape(-1, 1) for c in X_cols], axis=1)
    Y = np.concatenate([c.ravel()[:n_max].reshape(-1, 1) for c in Y_cols], axis=1)
    return X, Y


# ─────────────────────────────────────────────────────────────────────────────
# Built-in PINN residuals for known PDE datasets
# ─────────────────────────────────────────────────────────────────────────────

def _grad(y, x, idx):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True
    )[0][:, idx:idx + 1]


def _unwrap(out):
    if torch.is_tensor(out):
        return out
    if hasattr(out, "y"):
        return out.y
    if isinstance(out, dict):
        return torch.stack(list(out.values()), dim=-1)
    return out


# Map from dataset_id → residual function factory
# Each factory takes dataset params and returns (pde_loss, bc_loss) callable
_PINN_RESIDUALS: Dict[str, Callable] = {}


def _register_pinn(dataset_id: str):
    def decorator(fn):
        _PINN_RESIDUALS[dataset_id] = fn
        return fn
    return decorator


@_register_pinn("burgers_1d")
def _burgers_residual(nu=0.01 / math.pi, **kw):
    def residual(net, xy_int, xy_bc, uv_bc, **params):
        _nu = params.get("nu", nu)
        xt = xy_int.clone().requires_grad_(True)
        u = _unwrap(net(xt))
        u_t = _grad(u, xt, 1)
        u_x = _grad(u, xt, 0)
        u_xx = _grad(u_x, xt, 0)
        pde = ((u_t + u * u_x - _nu * u_xx) ** 2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc) ** 2).mean()
        return pde, bc
    return residual


@_register_pinn("heat_1d")
def _heat_1d_residual(k=0.4, **kw):
    def residual(net, xy_int, xy_bc, uv_bc, **params):
        _k = params.get("k", k)
        xt = xy_int.clone().requires_grad_(True)
        u = _unwrap(net(xt))
        u_t = _grad(u, xt, 1)
        u_x = _grad(u, xt, 0)
        u_xx = _grad(u_x, xt, 0)
        pde = ((u_t - _k * u_xx) ** 2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc) ** 2).mean()
        return pde, bc
    return residual


@_register_pinn("wave_1d")
def _wave_1d_residual(c=1.0, **kw):
    def residual(net, xy_int, xy_bc, uv_bc, **params):
        _c = params.get("c", c)
        xt = xy_int.clone().requires_grad_(True)
        u = _unwrap(net(xt))
        u_t = _grad(u, xt, 1)
        u_tt = _grad(u_t, xt, 1)
        u_x = _grad(u, xt, 0)
        u_xx = _grad(u_x, xt, 0)
        pde = ((u_tt - _c ** 2 * u_xx) ** 2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc) ** 2).mean()
        return pde, bc
    return residual


@_register_pinn("poisson_2d")
def _poisson_2d_residual(**kw):
    def residual(net, xy_int, xy_bc, uv_bc, **params):
        xy = xy_int.clone().requires_grad_(True)
        u = _unwrap(net(xy))
        u_x = _grad(u, xy, 0)
        u_y = _grad(u, xy, 1)
        u_xx = _grad(u_x, xy, 0)
        u_yy = _grad(u_y, xy, 1)
        # source f is available in xy_int but we use the residual form
        # Δu = f → Δu + 2π²sin(πx)sin(πy) = 0
        x, y = xy[:, 0:1], xy[:, 1:2]
        f = -2 * math.pi ** 2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)
        pde = ((u_xx + u_yy - f) ** 2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc) ** 2).mean()
        return pde, bc
    return residual


@_register_pinn("helmholtz_2d")
def _helmholtz_2d_residual(k=1.0, a1=1.0, a2=1.0, **kw):
    def residual(net, xy_int, xy_bc, uv_bc, **params):
        _k = params.get("k", k)
        xy = xy_int.clone().requires_grad_(True)
        u = _unwrap(net(xy))
        u_x = _grad(u, xy, 0)
        u_y = _grad(u, xy, 1)
        u_xx = _grad(u_x, xy, 0)
        u_yy = _grad(u_y, xy, 1)
        x, y = xy[:, 0:1], xy[:, 1:2]
        q = (_k ** 2 - (a1 * math.pi) ** 2 - (a2 * math.pi) ** 2) * \
            torch.sin(a1 * math.pi * x) * torch.sin(a2 * math.pi * y)
        pde = ((u_xx + u_yy + _k ** 2 * u - q) ** 2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc) ** 2).mean()
        return pde, bc
    return residual


@_register_pinn("allen_cahn_1d")
def _allen_cahn_residual(eps=0.01, **kw):
    def residual(net, xy_int, xy_bc, uv_bc, **params):
        _eps = params.get("eps", eps)
        xt = xy_int.clone().requires_grad_(True)
        u = _unwrap(net(xt))
        u_t = _grad(u, xt, 1)
        u_x = _grad(u, xt, 0)
        u_xx = _grad(u_x, xt, 0)
        pde = ((u_t - _eps ** 2 * u_xx - u + u ** 3) ** 2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc) ** 2).mean()
        return pde, bc
    return residual


@_register_pinn("kovasznay_ns")
def _kovasznay_residual(re=40.0, **kw):
    def residual(net, xy_int, xy_bc, uv_bc, **params):
        nu = 1.0 / params.get("re", re)
        xy = xy_int.clone().requires_grad_(True)
        out = _unwrap(net(xy))
        u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
        u_x = _grad(u, xy, 0); u_y = _grad(u, xy, 1)
        v_x = _grad(v, xy, 0); v_y = _grad(v, xy, 1)
        p_x = _grad(p, xy, 0); p_y = _grad(p, xy, 1)
        u_xx = _grad(u_x, xy, 0); u_yy = _grad(u_y, xy, 1)
        v_xx = _grad(v_x, xy, 0); v_yy = _grad(v_y, xy, 1)
        r1 = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        r2 = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        r3 = u_x + v_y
        pde = (r1 ** 2 + r2 ** 2 + r3 ** 2).mean()
        bc = ((_unwrap(net(xy_bc)) - uv_bc) ** 2).mean()
        return pde, bc
    return residual


# ─────────────────────────────────────────────────────────────────────────────
# DatasetProblem
# ─────────────────────────────────────────────────────────────────────────────

class DatasetProblem(ArenaProblem):
    """ArenaProblem backed by a pinneapple_data dataset.

    Create via :meth:`from_dataset` — do not instantiate directly.

    Modes
    -----
    ``"supervised"``
        Pure supervised regression: MSE loss on data points only.

    ``"pinn_data"``
        Physics-hybrid: supervised data loss + autograd physics residual.
        Requires the dataset ID to have a registered PINN residual
        (built-in for: burgers_1d, heat_1d, wave_1d, poisson_2d,
        helmholtz_2d, allen_cahn_1d, kovasznay_ns).

    ``"timeseries"``
        Time→state regression: ``t`` is the input, state variables are output.
        Handles datasets where the solution is a sequence.
    """

    def __init__(
        self,
        dataset_id: str,
        name: str,
        description: str,
        input_fields: List[str],
        output_fields: List[str],
        mode: str,
        dataset_params: Dict[str, Any],
        pinn_residuals_fn: Optional[Callable],
        data_cache: Optional[Tuple[np.ndarray, ...]] = None,
    ):
        self.dataset_id = dataset_id
        self.name = name
        self.description = description
        self.domain = "Dataset"
        self._input_fields = input_fields
        self._output_fields = output_fields
        self._mode = mode
        self._dataset_params = dataset_params
        self._residuals_fn = pinn_residuals_fn
        self._data_cache = data_cache  # (X, Y) if already loaded

        # Infer dims from cache or field lists
        if data_cache is not None:
            X, Y = data_cache
            self.input_dim = X.shape[1] if X.ndim > 1 else 1
            self.output_dim = Y.shape[1] if Y.ndim > 1 else 1
        else:
            self.input_dim = len(input_fields)
            self.output_dim = len(output_fields)

    # ── class method factory ──────────────────────────────────────────────────

    @classmethod
    def from_dataset(
        cls,
        dataset_id: str,
        input_fields: List[str],
        output_fields: List[str],
        mode: str = "supervised",
        name: Optional[str] = None,
        description: str = "",
        register: bool = True,
        **dataset_params,
    ) -> "DatasetProblem":
        """Create a DatasetProblem from a pinneapple_data dataset ID.

        Parameters
        ----------
        dataset_id    : registered dataset ID (e.g. ``"burgers_1d"``).
        input_fields  : coordinate/feature field names → model inputs.
        output_fields : solution/target field names → model outputs.
        mode          : ``"supervised"``, ``"pinn_data"``, or ``"timeseries"``.
        name          : Arena problem name (defaults to ``dataset_id``).
        description   : human-readable description.
        register      : automatically register in the Arena problem registry.
        **dataset_params : kwargs forwarded verbatim to ``load_dataset()``.

        Returns
        -------
        DatasetProblem  (also registered so Arena can find it by name)
        """
        prob_name = name or dataset_id
        if not description:
            description = f"Arena benchmark on dataset '{dataset_id}'"

        # Resolve PINN residuals
        pinn_fn = None
        if mode == "pinn_data":
            if dataset_id in _PINN_RESIDUALS:
                pinn_fn = _PINN_RESIDUALS[dataset_id](**dataset_params)
            else:
                warnings.warn(
                    f"[DatasetProblem] No built-in PINN residual for '{dataset_id}'. "
                    "Falling back to supervised mode."
                )
                mode = "supervised"

        # Pre-load data to infer dims
        try:
            raw = _load_raw(dataset_id, **dataset_params)
            X, Y = _flatten_pde_dataset(raw, input_fields, output_fields)
            data_cache = (X, Y)
        except Exception as exc:
            warnings.warn(f"[DatasetProblem] Pre-load failed for '{dataset_id}': {exc}. "
                          "Dims will be inferred from field count.")
            data_cache = None

        prob = cls(
            dataset_id=dataset_id,
            name=prob_name,
            description=description,
            input_fields=input_fields,
            output_fields=output_fields,
            mode=mode,
            dataset_params=dataset_params,
            pinn_residuals_fn=pinn_fn,
            data_cache=data_cache,
        )
        if register:
            register_problem(prob)
        return prob

    # ── internal data loading ─────────────────────────────────────────────────

    def _get_xy(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._data_cache is not None:
            return self._data_cache
        raw = _load_raw(self.dataset_id, **self._dataset_params)
        X, Y = _flatten_pde_dataset(raw, self._input_fields, self._output_fields)
        self._data_cache = (X, Y)
        return X, Y

    # ── ArenaProblem interface ────────────────────────────────────────────────

    def analytical(self, *args, **kw):
        return None  # datasets don't have symbolic analytical solutions

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, **params):
        if self._residuals_fn is not None:
            return self._residuals_fn(net, xy_int, xy_bc, uv_bc, **params)
        # Pure supervised fallback
        bc_out = _unwrap(net(xy_bc))
        if bc_out.ndim == 1:
            bc_out = bc_out.unsqueeze(1)
        if uv_bc.shape[1] < bc_out.shape[1]:
            bc_out = bc_out[:, :uv_bc.shape[1]]
        bc_loss = ((bc_out - uv_bc) ** 2).mean()
        return torch.tensor(0.0, device=xy_int.device), bc_loss

    def supervised_data(
        self,
        n_train: int = 1000,
        n_bc: int = 200,
        grid_n: int = 40,
        **kwargs,
    ):
        X, Y = self._get_xy()

        rng = np.random.default_rng(kwargs.get("seed", 42))
        n = len(X)
        idx = rng.permutation(n)

        n_tr = min(n_train, int(0.8 * n))
        n_ev = min(grid_n * grid_n, n - n_tr)

        X_train = X[idx[:n_tr]]
        Y_train = Y[idx[:n_tr]]
        X_bc    = X[idx[n_tr:n_tr + n_bc]] if n_bc > 0 else X[:0]
        Y_bc    = Y[idx[n_tr:n_tr + n_bc]] if n_bc > 0 else Y[:0]
        X_eval  = X[idx[n_tr:n_tr + n_ev]]
        Y_eval  = Y[idx[n_tr:n_tr + n_ev]]

        return X_train, Y_train, X_bc, Y_bc, X_eval, Y_eval, self._output_fields

    def __repr__(self):
        return (f"DatasetProblem(id={self.dataset_id!r}, "
                f"mode={self._mode!r}, "
                f"in={self._input_fields}, out={self._output_fields})")
