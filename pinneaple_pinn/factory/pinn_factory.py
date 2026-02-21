"""PINN model factory: compiles symbolic equations and builds unified loss function."""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import sympy as sp

from .sympy_backend import SympyTorchCompiler, CompiledEquation
from .autodiff import DerivativeComputer


Tensor = torch.Tensor


class NeuralNetwork(nn.Module):
    """
    A simple fully-connected MLP.

    forward(*inputs): concatenates along dim=1
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_layers: int,
        num_neurons: int,
        activation: nn.Module = nn.Tanh(),
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers: List[nn.Module] = [nn.Linear(num_inputs, num_neurons), activation]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(num_neurons, num_neurons), activation])
        layers.append(nn.Linear(num_neurons, num_outputs))
        self.layers = nn.Sequential(*layers)

    def forward(self, *inputs: Tensor) -> Tensor:
        """Concatenate inputs along dim=1 and pass through MLP."""
        x = torch.cat(inputs, dim=1)
        return self.layers(x)


class PINN(nn.Module):
    """
    Wraps a neural network plus trainable inverse parameters.
    """

    def __init__(
        self,
        neural_network: NeuralNetwork,
        inverse_params_names: Optional[List[str]] = None,
        initial_guesses: Optional[Dict[str, float]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.net = neural_network
        self.inverse_params = nn.ParameterDict()
        if inverse_params_names:
            initial_guesses = initial_guesses or {}
            for name in inverse_params_names:
                init = float(initial_guesses.get(name, 0.1))
                self.inverse_params[name] = nn.Parameter(torch.tensor(init, dtype=dtype))

    def forward(self, *inputs: Tensor) -> Tensor:
        """Forward pass through wrapped network."""
        return self.net(*inputs)


@dataclass
class PINNProblemSpec:
    """
    A model-agnostic PINN specification.

    pde_residuals: list of equations that should be ~0 on collocation points
    conditions: list of dicts like:
      {
        "name": "ic_u",
        "equation": "u - sin(pi*x)",
        "weight": 1.0
      }
    independent_vars: e.g. ["t","x"] or ["t","lat","lon","lev"]
    dependent_vars: e.g. ["u"] or ["T","U","V"]
    inverse_params: optional list of trainable scalars used in equations
    loss_weights: weights for loss buckets: {"pde":1.0,"conditions":1.0,"data":1.0}
    """
    pde_residuals: List[str]
    conditions: List[Dict[str, Any]]
    independent_vars: List[str]
    dependent_vars: List[str]
    inverse_params: List[str] = field(default_factory=list)
    loss_weights: Dict[str, float] = field(default_factory=dict)
    verbose: bool = False


class PINNFactory:
    """
    Factory that compiles symbolic equations (SymPy) into torch-callable residuals,
    then creates a unified loss function.

    Data contract for the generated loss(model, batch):
      batch can contain:
        - "collocation": Tuple[Tensor,...]  # for PDE residuals
        - "conditions": List[Tuple[Tensor,...]]  # one per condition equation (same length as spec.conditions)
        - "data": (Tuple[Tensor,...], Tensor)  # supervised data pairs
    """

    def __init__(self, spec: PINNProblemSpec):
        self.spec = spec
        self.loss_weights = collections.defaultdict(lambda: 1.0)
        if spec.loss_weights:
            self.loss_weights.update(spec.loss_weights)

        self.compiler = SympyTorchCompiler(
            independent_vars=spec.independent_vars,
            dependent_vars=spec.dependent_vars,
            inverse_params=spec.inverse_params,
        )

        self.compiled_pdes: List[CompiledEquation] = [self.compiler.compile(s) for s in spec.pde_residuals]
        self.compiled_conditions: List[CompiledEquation] = [self.compiler.compile(c["equation"]) for c in spec.conditions]

        # Union of all derivatives needed across PDEs and conditions
        self.all_derivatives: set[sp.Derivative] = set()
        for ce in self.compiled_pdes + self.compiled_conditions:
            self.all_derivatives.update(ce.derivatives)

        if self.spec.verbose:
            self._print_summary()

    def _print_summary(self) -> None:
        """Print problem summary (vars, derivatives) to stdout."""
        print("--- PINN Problem Summary ---")
        print("  Independent Vars:", self.spec.independent_vars)
        print("  Dependent Vars:", self.spec.dependent_vars)
        if self.spec.inverse_params:
            print("  Inverse Params:", self.spec.inverse_params)
        if self.all_derivatives:
            print("  Required Derivatives:", {str(d) for d in sorted(self.all_derivatives, key=str)})
        else:
            print("  Required Derivatives: None")
        print("---------------------------")

    def generate_loss_function(
        self,
    ) -> Callable[[PINN, Dict[str, Any]], Tuple[Tensor, Dict[str, float]]]:
        """
        Returns (loss_fn) where:
          loss_fn(model, batch) -> (total_loss, components_dict)
        """

        ind_vars = self.spec.independent_vars
        dep_vars = self.spec.dependent_vars
        inv_vars = self.spec.inverse_params
        dep_symbols = self.compiler.dep_symbols

        def loss_fn(model: PINN, batch: Dict[str, Any]) -> Tuple[Tensor, Dict[str, float]]:
            try:
                p = next(model.parameters())
                device = p.device
                dtype = p.dtype
            except StopIteration:
                # model has no parameters (e.g., constant/analytic baseline)
                # infer from batch tensors
                device = None
                dtype = None
                for v in batch.values():
                    if v is None:
                        continue
                    if isinstance(v, torch.Tensor):
                        device, dtype = v.device, v.dtype
                        break
                    if isinstance(v, (tuple, list)):
                        for t in v:
                            if isinstance(t, torch.Tensor):
                                device, dtype = t.device, t.dtype
                                break
                    if device is not None:
                        break
                if device is None:
                    device = torch.device("cpu")
                if dtype is None:
                    dtype = torch.float32

            losses = collections.defaultdict(lambda: torch.tensor(0.0, device=device, dtype=dtype))

            # inverse parameters in fixed order
            inv_vals: List[Tensor] = []
            for name in inv_vars:
                if name not in model.inverse_params:
                    raise KeyError(f"Model is missing inverse param '{name}'")
                inv_vals.append(model.inverse_params[name])

            deriv_comp = DerivativeComputer(independent_vars=ind_vars, dependent_vars=dep_vars, device=device)

            # ---- PDE Residual Loss (collocation) ----
            if "collocation" in batch and batch["collocation"] is not None:
                inputs = tuple(t.to(device=device, dtype=dtype) for t in batch["collocation"])
                computed = deriv_comp.compute(model=model, inputs=inputs, required_derivatives=self.all_derivatives, dep_symbols=dep_symbols)

                for ce in self.compiled_pdes:
                    dep_args = [computed[dep_symbols[v]] for v in dep_vars]
                    deriv_args = [computed[d] for d in sorted(list(ce.derivatives), key=str)]
                    args = [*inputs, *dep_args, *inv_vals, *deriv_args]
                    r = ce.call(*args)
                    losses["pde"] = losses["pde"] + torch.mean(r**2)

            # ---- Conditions Loss ----
            if "conditions" in batch and batch["conditions"] is not None:
                cond_batches = batch["conditions"]
                if len(cond_batches) != len(self.compiled_conditions):
                    raise ValueError(
                        f"Expected {len(self.compiled_conditions)} condition batches, got {len(cond_batches)}"
                    )

                for i, ce in enumerate(self.compiled_conditions):
                    inputs = tuple(t.to(device=device, dtype=dtype) for t in cond_batches[i])
                    computed = deriv_comp.compute(model=model, inputs=inputs, required_derivatives=self.all_derivatives, dep_symbols=dep_symbols)

                    dep_args = [computed[dep_symbols[v]] for v in dep_vars]
                    deriv_args = [computed[d] for d in sorted(list(ce.derivatives), key=str)]
                    args = [*inputs, *dep_args, *inv_vals, *deriv_args]
                    r = ce.call(*args)

                    w = float(self.spec.conditions[i].get("weight", 1.0))
                    losses["conditions"] = losses["conditions"] + w * torch.mean(r**2)

            # ---- Supervised Data Loss ----
            if "data" in batch and batch["data"] is not None:
                inputs, y_true = batch["data"]
                inputs = tuple(t.to(device=device, dtype=dtype) for t in inputs)
                y_true = y_true.to(device=device, dtype=dtype)
                y_pred = model(*inputs)
                losses["data"] = losses["data"] + torch.mean((y_pred - y_true) ** 2)

            total = torch.tensor(0.0, device=device, dtype=dtype)
            for k, v in losses.items():
                total = total + float(self.loss_weights[k]) * v

            components = {k: float(v.detach().cpu().item()) for k, v in losses.items()}
            components["total"] = float(total.detach().cpu().item())
            return total, components

        return loss_fn
