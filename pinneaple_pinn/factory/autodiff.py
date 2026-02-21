"""Autograd-based derivative computation for PINN residual evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

import torch
import sympy as sp


Tensor = torch.Tensor

def _safe_grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Compute gradient of outputs w.r.t. inputs; returns zeros if outputs don't depend on inputs."""
    # If outputs doesn't depend on inputs, derivative is zero.
    if (not outputs.requires_grad) or (outputs.grad_fn is None):
        return torch.zeros_like(inputs)

    grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]

    if grad is None:
        return torch.zeros_like(inputs)
    return grad

def ensure_requires_grad(inputs: Sequence[Tensor]) -> None:
    """
    Ensures all tensors in `inputs` have requires_grad=True (in-place).
    """
    for i, t in enumerate(inputs):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"Input at index {i} is not a torch.Tensor")
        if not t.requires_grad:
            t.requires_grad_(True)
        if not t.requires_grad:
            raise RuntimeError(f"Input tensor at index {i} does not require grad.")


@dataclass(frozen=True)
class DerivativeKey:
    """
    A stable key for caching derivative computations.

    Example:
      - base: "u"
      - wrt: [("x", 1), ("t", 2)] means d^3 u / dx dt^2
    """
    base: str
    wrt: Tuple[Tuple[str, int], ...]


class DerivativeComputer:
    """
    Computes and caches torch autograd derivatives required by SymPy Derivative atoms.

    How it works:
      - model outputs are split into named dependent variables
      - for each needed derivative, autograd.grad is applied repeatedly
      - results cached by DerivativeKey for reuse across equations/conditions
    """

    def __init__(
        self,
        independent_vars: List[str],
        dependent_vars: List[str],
        device: torch.device,
    ) -> None:
        self.independent_vars = list(independent_vars)
        self.dependent_vars = list(dependent_vars)
        self.device = device

        self._ind_index = {name: i for i, name in enumerate(self.independent_vars)}

    def sympy_derivative_to_key(self, d: sp.Derivative) -> DerivativeKey:
        """
        Converts SymPy Derivative(u(x,t), x, t, t) into DerivativeKey("u", (("x",1),("t",2)))
        """
        # d.args: (expr, (var, order), (var, order), ...)
        base_expr = d.args[0]
        # base_expr is usually u(x,t) a sympy.Function applied to ind vars
        base_name = str(base_expr.func)

        counts: Dict[str, int] = {}
        for var, order in d.args[1:]:
            v = var.name if hasattr(var, "name") else str(var)
            counts[v] = counts.get(v, 0) + int(order)

        wrt = tuple(sorted(counts.items(), key=lambda x: x[0]))
        return DerivativeKey(base=base_name, wrt=wrt)

    def compute(
        self,
        model: torch.nn.Module,
        inputs: Tuple[Tensor, ...],
        required_derivatives: Sequence[sp.Derivative],
        dep_symbols: Dict[str, sp.Basic],
    ) -> Dict[Union[sp.Basic, sp.Derivative], Tensor]:
        """
        Returns mapping:
          - dep symbol (e.g. u(x,t)) -> predicted tensor (N,1)
          - derivative atoms -> tensor (N,1)

        dep_symbols: mapping from dep var name to sympy "u(x,t)" object
        """
        ensure_requires_grad(inputs)

        cache: Dict[Union[sp.Basic, sp.Derivative], Tensor] = {}

        # Forward pass once
        pred = model(*inputs)  # shape (N, n_dep)
        if hasattr(pred, "y"):   # PINNOutput
            pred = pred.y

        if pred.ndim != 2 or pred.shape[1] != len(self.dependent_vars):
            raise RuntimeError(
                f"Model output expected shape (N, {len(self.dependent_vars)}), got {tuple(pred.shape)}"
            )

        pred_split = torch.split(pred, 1, dim=1)
        for i, name in enumerate(self.dependent_vars):
            cache[dep_symbols[name]] = pred_split[i]

        # Build derivative graph with caching by DerivativeKey
        torch_cache: Dict[DerivativeKey, Tensor] = {}

        # Ensure deterministic order: by string repr
        for d in sorted(list(required_derivatives), key=str):
            key = self.sympy_derivative_to_key(d)
            if key in torch_cache:
                cache[d] = torch_cache[key]
                continue

            # Start from base dep var
            base_sym = dep_symbols[key.base]
            g = cache[base_sym]

            # Apply gradients in a deterministic order: alphabetical by var name
            for var_name, order in key.wrt:
                idx = self._ind_index.get(var_name)
                if idx is None:
                    raise KeyError(f"Independent var '{var_name}' not found in {self.independent_vars}")
                x = inputs[idx]
                for _ in range(order):
                    g = _safe_grad(g, x)

            torch_cache[key] = g
            cache[d] = g

        return cache
