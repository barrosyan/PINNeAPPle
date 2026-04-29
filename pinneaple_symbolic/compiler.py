"""Symbolic PDE compiler for PINNeAPPle.

Converts SymPy PDE residual expressions into callable PyTorch autograd functions.

Usage example::

    import sympy as sp
    from pinneaple_symbolic import SymbolicPDE

    x, y = sp.symbols("x y")
    u = sp.Function("u")
    pi = sp.pi

    # Poisson: u_xx + u_yy + 2*pi^2*sin(pi*x)*sin(pi*y) = 0
    expr = u(x, y).diff(x, 2) + u(x, y).diff(y, 2) + 2 * pi**2 * sp.sin(pi * x) * sp.sin(pi * y)
    pde = SymbolicPDE(expr, coord_syms=[x, y], field_syms=[u])
    residual_fn = pde.to_residual_fn(model)
    # residual_fn(coords_tensor) -> (N,1) tensor
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import sympy as sp
from sympy.core.function import AppliedUndef as _SpAppliedUndef
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _DerivativeOp:
    """Represents a single partial derivative ∂^n u / ∂x_i1 ... ∂x_in."""

    def __init__(self, field_name: str, wrt: List[str]) -> None:
        self.field_name = field_name  # e.g. "u"
        self.wrt = wrt               # e.g. ["x", "x"] for u_xx

    def __repr__(self) -> str:  # pragma: no cover
        return f"D({self.field_name}, {self.wrt})"


def _extract_derivative_ops(expr: sp.Expr, field_syms: List[sp.Function]) -> List[_DerivativeOp]:
    """Walk a SymPy expression tree and collect all Derivative sub-expressions."""
    ops: List[_DerivativeOp] = []
    field_names = {f.__name__ if hasattr(f, "__name__") else str(f): f for f in field_syms}

    for sub in sp.preorder_traversal(expr):
        if not isinstance(sub, sp.Derivative):
            continue
        expr_inner = sub.args[0]
        # expr_inner is e.g. u(x,y); get function name
        if isinstance(expr_inner, _SpAppliedUndef):
            fname = expr_inner.func.__name__
        elif hasattr(expr_inner, "name"):
            fname = expr_inner.name
        else:
            fname = str(expr_inner.func)

        # derivative variables (each entry is (sym, order))
        wrt: List[str] = []
        for sym, order in sub.variable_count:
            wrt.extend([str(sym)] * order)

        ops.append(_DerivativeOp(fname, wrt))

    return ops


def _compute_derivative(
    field_tensor: torch.Tensor,
    coords_tensor: torch.Tensor,
    wrt: List[str],
    coord_names: List[str],
) -> torch.Tensor:
    """Compute an arbitrary-order derivative using autograd.

    Parameters
    ----------
    field_tensor : (N, 1) tensor — the field value.
    coords_tensor : (N, D) tensor with grad enabled.
    wrt : list of coordinate names in differentiation order.
    coord_names : list of coordinate names (columns of coords_tensor).
    """
    current = field_tensor
    for coord_name in wrt:
        idx = coord_names.index(coord_name)
        g = torch.autograd.grad(
            outputs=current,
            inputs=coords_tensor,
            grad_outputs=torch.ones_like(current),
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]  # (N, D)
        current = g[:, idx : idx + 1]
    return current


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SymbolicPDE:
    """Symbolic PDE residual — compile to a PyTorch autograd callable.

    Parameters
    ----------
    expr : sympy expression equal to 0 (the PDE residual).
    coord_syms : list of SymPy symbols for independent variables (e.g. [x, y, t]).
    field_syms : list of SymPy *Function* objects for dependent variables (e.g. [u, v]).
    param_syms : optional list of SymPy symbols for parameters.
    """

    def __init__(
        self,
        expr: sp.Expr,
        coord_syms: List[sp.Symbol],
        field_syms: List[sp.Function],
        param_syms: Optional[List[sp.Symbol]] = None,
    ) -> None:
        self.expr = expr
        self.coord_syms = coord_syms
        self.field_syms = field_syms
        self.param_syms = param_syms or []
        self._compiled_fn: Optional[Callable] = None
        self._coord_names: List[str] = [str(s) for s in coord_syms]
        self._field_names: List[str] = [
            f.__name__ if hasattr(f, "__name__") else str(f) for f in field_syms
        ]
        self._deriv_ops: List[_DerivativeOp] = _extract_derivative_ops(expr, field_syms)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def to_residual_fn(self, model: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return a callable that computes the PDE residual using autograd.

        Parameters
        ----------
        model : nn.Module — takes (N, D) coords tensor, returns (N, F) fields tensor.

        Returns
        -------
        residual_fn(coords_tensor) -> (N, 1) residual tensor.
        """
        coord_names = self._coord_names
        field_names = self._field_names
        expr = self.expr

        def residual_fn(coords: torch.Tensor) -> torch.Tensor:
            coords = coords.clone().requires_grad_(True)
            raw = model(coords)
            if raw.ndim == 1:
                raw = raw[:, None]

            # Build dict: field_name -> (N,1) tensor
            if raw.shape[1] != len(field_names):
                raise ValueError(
                    f"Model output dim {raw.shape[1]} != number of fields {len(field_names)}"
                )
            fields: Dict[str, torch.Tensor] = {
                fname: raw[:, i : i + 1] for i, fname in enumerate(field_names)
            }

            # Build dict: derivative key -> (N,1) tensor
            # Key: (field_name, tuple(wrt))
            deriv_cache: Dict[Tuple[str, Tuple[str, ...]], torch.Tensor] = {}
            for dop in self._deriv_ops:
                key = (dop.field_name, tuple(dop.wrt))
                if key in deriv_cache:
                    continue
                deriv_cache[key] = _compute_derivative(
                    fields[dop.field_name], coords, dop.wrt, coord_names
                )

            # Evaluate the symbolic expression numerically
            return _eval_expr_torch(expr, coords, fields, deriv_cache, coord_names)

        return residual_fn

    def compile(self, backend: str = "torch") -> "SymbolicPDE":
        """Compile the symbolic expression (currently a no-op for 'torch' backend).

        Sets an internal flag; future backends (e.g. JAX, C++) can be added here.
        """
        if backend != "torch":
            raise ValueError(f"Unsupported backend '{backend}'. Only 'torch' is supported.")
        # Mark as compiled; the heavy work happens lazily in to_residual_fn.
        self._compiled_fn = True  # type: ignore[assignment]
        return self

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SymbolicPDE(coords={self._coord_names}, fields={self._field_names}, "
            f"expr={self.expr})"
        )


# ---------------------------------------------------------------------------
# Expression evaluator: SymPy -> torch, walking the expression tree
# ---------------------------------------------------------------------------

def _eval_expr_torch(
    expr: sp.Expr,
    coords: torch.Tensor,
    fields: Dict[str, torch.Tensor],
    deriv_cache: Dict[Tuple[str, Tuple[str, ...]], torch.Tensor],
    coord_names: List[str],
) -> torch.Tensor:
    """Recursively evaluate a SymPy expression as a PyTorch tensor.

    Supports: Add, Mul, Pow, Number, Symbol (coords), AppliedUndef (fields),
    Derivative, sin, cos, exp, log, Abs, sign, and basic constants.
    """
    N = coords.shape[0]
    device = coords.device
    dtype = coords.dtype

    def _const(v: float) -> torch.Tensor:
        return torch.full((N, 1), v, device=device, dtype=dtype)

    def _eval(e: sp.Expr) -> torch.Tensor:
        # ---- constants ----
        if isinstance(e, sp.Number):
            return _const(float(e))

        if e is sp.pi:
            return _const(float(sp.pi))

        if e is sp.E:
            return _const(float(sp.E))

        # ---- coordinate symbol ----
        if isinstance(e, sp.Symbol):
            s = str(e)
            if s in coord_names:
                idx = coord_names.index(s)
                return coords[:, idx : idx + 1]
            raise ValueError(f"Unknown symbol '{s}' — not in coords {coord_names}")

        # ---- applied function (field value) ----
        if isinstance(e, _SpAppliedUndef):
            fname = e.func.__name__
            if fname not in fields:
                raise ValueError(f"Field '{fname}' not in model outputs {list(fields.keys())}")
            return fields[fname]

        # ---- derivative ----
        if isinstance(e, sp.Derivative):
            inner = e.args[0]
            if isinstance(inner, _SpAppliedUndef):
                fname = inner.func.__name__
            else:
                fname = str(inner.func)
            wrt: List[str] = []
            for sym, order in e.variable_count:
                wrt.extend([str(sym)] * order)
            key = (fname, tuple(wrt))
            if key in deriv_cache:
                return deriv_cache[key]
            # Compute on-the-fly if not pre-cached
            result = _compute_derivative(fields[fname], coords, wrt, coord_names)
            deriv_cache[key] = result
            return result

        # ---- compound expressions ----
        if isinstance(e, sp.Add):
            result = _eval(e.args[0])
            for arg in e.args[1:]:
                result = result + _eval(arg)
            return result

        if isinstance(e, sp.Mul):
            result = _eval(e.args[0])
            for arg in e.args[1:]:
                result = result * _eval(arg)
            return result

        if isinstance(e, sp.Pow):
            base, exp_ = e.args
            b = _eval(base)
            # Constant integer exponents — use torch.pow for graph efficiency
            if isinstance(exp_, sp.Integer):
                n = int(exp_)
                if n == 2:
                    return b * b
                if n == 3:
                    return b * b * b
                return torch.pow(b, n)
            if isinstance(exp_, sp.Number):
                return torch.pow(b, float(exp_))
            return torch.pow(b, _eval(exp_))

        # ---- elementary functions ----
        if isinstance(e, sp.sin):
            return torch.sin(_eval(e.args[0]))

        if isinstance(e, sp.cos):
            return torch.cos(_eval(e.args[0]))

        if isinstance(e, sp.tan):
            return torch.tan(_eval(e.args[0]))

        if isinstance(e, sp.exp):
            return torch.exp(_eval(e.args[0]))

        if isinstance(e, sp.log):
            return torch.log(_eval(e.args[0]))

        if isinstance(e, sp.Abs):
            return torch.abs(_eval(e.args[0]))

        if isinstance(e, sp.sign):
            return torch.sign(_eval(e.args[0]))

        if isinstance(e, sp.sqrt):
            return torch.sqrt(_eval(e.args[0]))

        if isinstance(e, sp.tanh):
            return torch.tanh(_eval(e.args[0]))

        if isinstance(e, sp.sinh):
            return torch.sinh(_eval(e.args[0]))

        if isinstance(e, sp.cosh):
            return torch.cosh(_eval(e.args[0]))

        raise NotImplementedError(
            f"Cannot evaluate SymPy expression of type {type(e).__name__}: {e}"
        )

    return _eval(expr)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def pde_from_sympy(
    expr: sp.Expr,
    coords: List[sp.Symbol],
    fields: List[sp.Function],
    params: Optional[List[sp.Symbol]] = None,
) -> SymbolicPDE:
    """Create a SymbolicPDE from a SymPy residual expression.

    Parameters
    ----------
    expr : sympy expression equal to 0 (the PDE residual).
    coords : list of SymPy symbols for independent variables.
    fields : list of SymPy Function objects for dependent variables.
    params : optional list of parameter symbols.

    Returns
    -------
    SymbolicPDE instance.

    Example::

        x, y = sp.symbols("x y")
        u = sp.Function("u")
        expr = u(x, y).diff(x, 2) + u(x, y).diff(y, 2)  # Laplace
        pde = pde_from_sympy(expr, [x, y], [u])
    """
    return SymbolicPDE(expr, coords, fields, params)


def auto_residual(
    model: nn.Module,
    coords_tensor: torch.Tensor,
    derivative_fns: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Compute field values and named derivative quantities via autograd.

    This is a lower-level helper intended for users who prefer to wire
    derivative operators manually rather than through SymPy expressions.

    Parameters
    ----------
    model : nn.Module taking (N, D) -> (N, F) tensors.
    coords_tensor : (N, D) input coordinates tensor (grad will be enabled).
    derivative_fns : dict mapping name -> callable(field, coords) -> tensor.
        Each callable receives the raw model output and the coords tensor
        (with grad enabled) and should return a (N, *) tensor.

    Returns
    -------
    dict with keys:
        "fields" : (N, F) raw model output
        + one key per entry in derivative_fns.

    Example::

        from pinneaple_pinn.compiler.autograd_ops import laplacian

        result = auto_residual(
            model, x_col,
            {"laplacian_u": lambda u, x: laplacian(u, x)}
        )
        res = result["laplacian_u"] - f_source
    """
    coords = coords_tensor.clone().requires_grad_(True)
    raw = model(coords)
    if raw.ndim == 1:
        raw = raw[:, None]

    out: Dict[str, torch.Tensor] = {"fields": raw}
    for name, fn in derivative_fns.items():
        out[name] = fn(raw, coords)
    return out
