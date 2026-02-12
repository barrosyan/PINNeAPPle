from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

import sympy as sp


# -----------------------------
# Compiled equation container
# -----------------------------
@dataclass(frozen=True)
class CompiledEquation:
    """
    A compiled symbolic equation.

    - expr: SymPy expression
    - derivatives: set of SymPy Derivative atoms appearing in expr
    - call: torch-compatible callable from sympy.lambdify
    - args_order: stable argument order for call(...)
    """
    expr: sp.Expr
    derivatives: Set[sp.Derivative]
    call: Callable[..., Any]
    args_order: List[sp.Basic]


# -----------------------------
# Low-level compiler
# -----------------------------
class SympyTorchCompiler:
    """
    Compiles equation strings into torch-callable lambdas using SymPy.
    """

    def __init__(
        self,
        independent_vars: List[str],
        dependent_vars: List[str],
        inverse_params: Optional[List[str]] = None,
        extra_namespace: Optional[Dict[str, Any]] = None,
    ):
        self.ind_vars_str = list(independent_vars)
        self.dep_vars_str = list(dependent_vars)
        self.inv_vars_str = list(inverse_params or [])

        self.ind_symbols = self._create_symbols(self.ind_vars_str)
        self.inv_symbols = self._create_symbols(self.inv_vars_str)

        self.dep_func_classes = {v: sp.Function(v) for v in self.dep_vars_str}
        self.dep_symbols = {v: self.dep_func_classes[v](*self.ind_symbols) for v in self.dep_vars_str}

        # Locals namespace for sympify
        self.namespace: Dict[str, Any] = {}
        self.namespace.update({s.name: s for s in self.ind_symbols + self.inv_symbols})
        self.namespace.update(self.dep_func_classes)
        self.namespace.update(
            {
                "sin": sp.sin,
                "cos": sp.cos,
                "exp": sp.exp,
                "pi": sp.pi,
                "sqrt": sp.sqrt,
                "log": sp.log,
                "Abs": sp.Abs,
            }
        )
        if extra_namespace:
            self.namespace.update(extra_namespace)

    @staticmethod
    def _create_symbols(names: List[str]) -> List[sp.Symbol]:
        if not names:
            return []
        syms = sp.symbols(" ".join(names))
        if isinstance(syms, tuple):
            return list(syms)
        return [syms]

    def compile(self, eq_str: str) -> CompiledEquation:
        expr = sp.sympify(eq_str, locals=self.namespace, evaluate=False)
        def _expand_numeric_pow(e):
            if isinstance(e, sp.Pow) and e.base.is_Number and e.exp.is_Integer and int(e.exp) >= 0:
                n = int(e.exp)
                if n == 0:
                    return sp.Integer(1)
                return sp.Mul(*([e.base] * n), evaluate=False)
            return e

        expr = expr.replace(lambda e: isinstance(e, sp.Pow) and e.base.is_Number and e.exp.is_Integer, _expand_numeric_pow)

        derivatives = set(expr.atoms(sp.Derivative))

        # Stable ordering:
        # 1) indep symbols
        # 2) dependent symbols (u(t,x), v(t,x), ...)
        # 3) inverse params
        # 4) derivatives (sorted)
        deriv_sorted = sorted(list(derivatives), key=str)
        args_order: List[sp.Basic] = [
            *self.ind_symbols,
            *self.dep_symbols.values(),
            *self.inv_symbols,
            *deriv_sorted,
        ]

        call = sp.lambdify(args_order, expr, "torch")
        return CompiledEquation(expr=expr, derivatives=derivatives, call=call, args_order=args_order)


class SympyBackend:
    """
    Compatibility wrapper.

    Your current PINNFactory uses SympyTorchCompiler directly.
    Some examples may import SympyBackend; this class simply exposes a
    make_compiler(...) method returning SympyTorchCompiler.
    """

    def __init__(self, *, extra_namespace: Optional[Dict[str, Any]] = None):
        self.extra_namespace = extra_namespace or {}

    def make_compiler(
        self,
        *,
        independent_vars: List[str],
        dependent_vars: List[str],
        inverse_params: Optional[List[str]] = None,
    ) -> SympyTorchCompiler:
        return SympyTorchCompiler(
            independent_vars=independent_vars,
            dependent_vars=dependent_vars,
            inverse_params=inverse_params or [],
            extra_namespace=self.extra_namespace,
        )
