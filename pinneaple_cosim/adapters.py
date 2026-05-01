"""Adapters connecting pinneaple_cosim to PINNeAPPle's model and problem ecosystems.

Four adapter node types:

PINNeProblemNode
    The most powerful adapter. Wraps any model from ModelRegistry together
    with a compiled physics loss from a ProblemSpec.  The PDE is enforced
    automatically — no physics residual needs to be coded by hand.

PINNeAPPleModelNode
    Wraps any BaseModel (DeepONet, FNO, GNN, surrogate, etc.) as a co-sim
    node.  No physics loss — pure data-driven prediction.

SymbolicPDENode
    Wraps a SymbolicPDE (free-form SymPy expression) + any differentiable
    model.  The residual is compiled to autograd at runtime.

TimeSeriesCoSimNode
    Wraps a TSModelBase (LSTM, N-BEATS, TFT, FFT+LSTM, etc.) so that a
    time-series forecaster can participate in a co-simulation graph.

All adapters honour the CoSimNode step() / physics_loss() contract and
degrade gracefully (ImportError) if the required PINNeAPPle sub-package
is not available.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .node import CoSimNode, PINNNode, TorchNode


# ---------------------------------------------------------------------------
# Internal: model factory
# ---------------------------------------------------------------------------

# Catalogue of known model names → (module_path, class_name)
_MODEL_CATALOGUE: Dict[str, Tuple[str, str]] = {
    "vanilla_pinn":  ("pinneaple_models.pinns.vanilla",   "VanillaPINN"),
    "siren":         ("pinneaple_models.siren",            "SIREN"),
    "modified_mlp":  ("pinneaple_models.modified_mlp",     "ModifiedMLP"),
    "hash_grid_mlp": ("pinneaple_models.hash_grid",        "HashGridMLP"),
    "afno":          ("pinneaple_models.afno",             "AFNO"),
    "mgn":           ("pinneaple_models.mesh_graph_net",   "MeshGraphNet"),
}


def _build_model(model_name: str, kwargs: Dict[str, Any]) -> nn.Module:
    """Build a model by name.  Tries ModelRegistry first, then the catalogue."""
    # 1. Try ModelRegistry (populated when model modules are imported)
    try:
        from pinneaple_models import ModelRegistry
        if model_name in ModelRegistry.list():
            return ModelRegistry.build(model_name, **kwargs)
    except Exception:
        pass

    # 2. Fall back to direct import via catalogue
    if model_name in _MODEL_CATALOGUE:
        mod_path, cls_name = _MODEL_CATALOGUE[model_name]
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            return cls(**kwargs)
        except Exception as exc:
            raise ImportError(
                f"Could not import {cls_name} from {mod_path}: {exc}"
            ) from exc

    raise ValueError(
        f"Unknown model {model_name!r}. "
        f"Known names: {list(_MODEL_CATALOGUE.keys())}. "
        "Or register your model in ModelRegistry before calling from_spec()."
    )


# ---------------------------------------------------------------------------
# PINNeProblemNode
# ---------------------------------------------------------------------------

class PINNeProblemNode(CoSimNode):
    """Co-sim node backed by a PINNeAPPle ProblemSpec + compiled physics loss.

    The physics residual is built automatically via ``compile_problem(spec)``.
    Supports any model family from ModelRegistry that accepts pointwise
    coordinate inputs.

    Args:
        name:            unique node name in the graph.
        model:           differentiable PyTorch model (``nn.Module``).
        spec:            ``ProblemSpec`` describing the PDE, BCs, and domain.
        compiled_fn:     pre-compiled physics loss callable
                         ``(model, y_hat, batch) -> Dict[str, Tensor]``.
        coord_ports:     input port names that carry coordinate tensors
                         (concatenated in order before the forward pass).
        field_ports:     output port names that carry field predictions
                         (split equally from the model output).
        extra_input_ports: additional input ports forwarded into the batch
                           under their own names (e.g., ``x_bc``, ``y_bc``).
        physics_weight:  scalar multiplier for the physics residual.
    """

    def __init__(
        self,
        name: str,
        model: nn.Module,
        spec: Any,
        compiled_fn: Callable,
        coord_ports: List[str],
        field_ports: List[str],
        extra_input_ports: Optional[List[str]] = None,
        physics_weight: float = 1.0,
    ) -> None:
        all_input_ports = coord_ports + (extra_input_ports or [])
        super().__init__(name, all_input_ports, field_ports)
        self.model = model
        self.spec = spec
        self._compiled_fn = compiled_fn
        self.coord_ports = coord_ports
        self.field_ports = field_ports
        self.extra_input_ports = extra_input_ports or []
        self.physics_weight = physics_weight
        self._last_inputs: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------
    @classmethod
    def from_spec(
        cls,
        name: str,
        spec: Any,
        *,
        model_name: str = "vanilla_pinn",
        model_kwargs: Optional[Dict[str, Any]] = None,
        coord_ports: Optional[List[str]] = None,
        field_ports: Optional[List[str]] = None,
        extra_input_ports: Optional[List[str]] = None,
        physics_weight: float = 1.0,
        loss_weights: Optional[Any] = None,
    ) -> "PINNeProblemNode":
        """Build a ``PINNeProblemNode`` from a ``ProblemSpec``.

        Args:
            name:           node name.
            spec:           ``ProblemSpec`` from ``pinneaple_environment``.
            model_name:     key in ``ModelRegistry`` (default ``"vanilla_pinn"``).
            model_kwargs:   extra kwargs forwarded to ``ModelRegistry.build()``.
                            ``in_dim`` and ``out_dim`` are inferred from *spec*
                            if not provided.
            coord_ports:    defaults to ``list(spec.coords)``.
            field_ports:    defaults to ``list(spec.fields)``.
            extra_input_ports: extra ports forwarded as batch keys (e.g. BCs).
            physics_weight: multiplier for physics residual.
            loss_weights:   optional ``LossWeights`` forwarded to
                            ``compile_problem``.
        """
        try:
            from pinneaple_pinn.compiler.compile import compile_problem
        except ImportError as exc:
            raise ImportError(
                "PINNeProblemNode.from_spec requires pinneaple_pinn."
            ) from exc

        kw = dict(model_kwargs or {})
        kw.setdefault("in_dim",  len(spec.coords))
        kw.setdefault("out_dim", len(spec.fields))

        model = _build_model(model_name, kw)
        compiled_fn = compile_problem(spec, weights=loss_weights)

        return cls(
            name=name,
            model=model,
            spec=spec,
            compiled_fn=compiled_fn,
            coord_ports=list(coord_ports or spec.coords),
            field_ports=list(field_ports or spec.fields),
            extra_input_ports=list(extra_input_ports or []),
            physics_weight=physics_weight,
        )

    # ------------------------------------------------------------------
    def step(
        self,
        inputs: Dict[str, torch.Tensor],
        t: float,
        dt: float,
    ) -> Dict[str, torch.Tensor]:
        self._last_inputs = {k: v for k, v in inputs.items()}
        # Concatenate coordinate inputs for the forward pass
        x = torch.cat([inputs[p] for p in self.coord_ports if p in inputs], dim=-1)
        raw = self.model(x)
        # Handle PINNOutput, ModelOutput, or plain tensor
        pred = raw.y if hasattr(raw, "y") else raw
        n = len(self.field_ports)
        if n == 1:
            return {self.field_ports[0]: pred}
        chunks = torch.chunk(pred, n, dim=-1)
        return {p: c for p, c in zip(self.field_ports, chunks)}

    def physics_loss(self) -> Optional[torch.Tensor]:
        if self._last_inputs is None:
            return None
        batch = {k: v for k, v in self._last_inputs.items()}
        if "x_col" not in batch and self.coord_ports:
            batch["x_col"] = torch.cat(
                [self._last_inputs[p] for p in self.coord_ports if p in self._last_inputs],
                dim=-1,
            )
        result = self._compiled_fn(self.model, None, batch)
        total = result["total"] if isinstance(result, dict) else result
        return self.physics_weight * total

    def parameters(self):
        return self.model.parameters()

    def reset(self) -> None:
        super().reset()
        self._last_inputs = None


# ---------------------------------------------------------------------------
# PINNeAPPleModelNode
# ---------------------------------------------------------------------------

class PINNeAPPleModelNode(TorchNode):
    """Wraps any ``BaseModel`` from ModelRegistry as a co-sim node.

    No physics loss — pure data-driven surrogate or neural operator.
    Supports ``forward_batch(batch)`` when the model implements it.

    Usage::

        node = PINNeAPPleModelNode.from_registry(
            "flow_surrogate",
            model_name="deeponet",
            input_ports=["u_branch", "x_query"],
            output_ports=["u_pred"],
            model_kwargs={"branch_dim": 64, "trunk_dim": 2},
        )
    """

    def __init__(
        self,
        name: str,
        model: nn.Module,
        input_ports: List[str],
        output_ports: List[str],
        batch_port_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Args:
            batch_port_map: optional mapping from port names to batch dict
                            keys (e.g. ``{"u_branch": "u_branch"}``).
                            When set, ``forward_batch(batch)`` is called
                            instead of concatenating inputs.
        """
        super().__init__(name, model, input_ports, output_ports)
        self.batch_port_map = batch_port_map

    @classmethod
    def from_registry(
        cls,
        name: str,
        model_name: str,
        input_ports: List[str],
        output_ports: List[str],
        model_kwargs: Optional[Dict[str, Any]] = None,
        batch_port_map: Optional[Dict[str, str]] = None,
    ) -> "PINNeAPPleModelNode":
        model = _build_model(model_name, model_kwargs or {})
        return cls(name, model, input_ports, output_ports, batch_port_map)

    def step(
        self,
        inputs: Dict[str, torch.Tensor],
        t: float,
        dt: float,
    ) -> Dict[str, torch.Tensor]:
        if self.batch_port_map and hasattr(self.model, "forward_batch"):
            batch = {self.batch_port_map.get(p, p): inputs[p] for p in self.input_ports if p in inputs}
            raw = self.model.forward_batch(batch)
        else:
            x = torch.cat([inputs[p] for p in self.input_ports if p in inputs], dim=-1)
            raw = self.model(x)
        pred = raw.y if hasattr(raw, "y") else raw
        n = len(self.output_ports)
        if n == 1:
            return {self.output_ports[0]: pred}
        chunks = torch.chunk(pred, n, dim=-1)
        return {p: c for p, c in zip(self.output_ports, chunks)}


# ---------------------------------------------------------------------------
# SymbolicPDENode
# ---------------------------------------------------------------------------

class SymbolicPDENode(CoSimNode):
    """Co-sim node whose physics residual is defined by a SymPy expression.

    The ``SymbolicPDE`` is compiled to a PyTorch autograd function at
    construction time.  Any differentiable model can be attached.

    Usage::

        import sympy as sp
        x, y = sp.symbols("x y")
        u = sp.Function("u")
        laplace = u(x,y).diff(x,2) + u(x,y).diff(y,2)

        node = SymbolicPDENode(
            "pde_node",
            model=my_model,
            symbolic_pde=SymbolicPDE(laplace, [x,y], [u]),
            coord_ports=["x_col"],
            field_ports=["u"],
        )
    """

    def __init__(
        self,
        name: str,
        model: nn.Module,
        symbolic_pde: Any,
        coord_ports: List[str],
        field_ports: List[str],
        physics_weight: float = 1.0,
    ) -> None:
        super().__init__(name, coord_ports, field_ports)
        self.model = model
        self._symbolic_pde = symbolic_pde
        self._residual_fn: Optional[Callable] = None
        self.coord_ports = coord_ports
        self.field_ports = field_ports
        self.physics_weight = physics_weight
        self._last_x: Optional[torch.Tensor] = None
        self._compile()

    def _compile(self) -> None:
        try:
            self._residual_fn = self._symbolic_pde.to_residual_fn(self.model)
        except Exception:
            # Will be compiled lazily on first step
            self._residual_fn = None

    def step(
        self,
        inputs: Dict[str, torch.Tensor],
        t: float,
        dt: float,
    ) -> Dict[str, torch.Tensor]:
        x = torch.cat([inputs[p] for p in self.coord_ports if p in inputs], dim=-1)
        self._last_x = x
        if self._residual_fn is None:
            self._residual_fn = self._symbolic_pde.to_residual_fn(self.model)
        pred = self.model(x)
        pred = pred.y if hasattr(pred, "y") else pred
        n = len(self.field_ports)
        if n == 1:
            return {self.field_ports[0]: pred}
        chunks = torch.chunk(pred, n, dim=-1)
        return {p: c for p, c in zip(self.field_ports, chunks)}

    def physics_loss(self) -> Optional[torch.Tensor]:
        if self._last_x is None or self._residual_fn is None:
            return None
        residual = self._residual_fn(self._last_x)
        return self.physics_weight * residual.pow(2).mean()

    def parameters(self):
        return self.model.parameters()

    def reset(self) -> None:
        super().reset()
        self._last_x = None


# ---------------------------------------------------------------------------
# TimeSeriesCoSimNode
# ---------------------------------------------------------------------------

class TimeSeriesCoSimNode(CoSimNode):
    """Wraps a PINNeAPPle ``TSModelBase`` as a co-simulation node.

    At each time step the node slides a context window forward and returns
    a multi-step forecast.  This lets time-series models (LSTM, N-BEATS,
    FFT+LSTM, HHT+LSTM, TFT, …) exchange information with physics solvers
    inside the same co-simulation graph.

    Port convention:
        Input  ``"context"`` — (1, input_len, n_features) rolling context window.
        Output ``"forecast"`` — (1, horizon,  n_features) forecast tensor.
        Output ``"context_next"`` — updated context (shift-by-one) for feedback.

    Usage::

        lstm = LSTMForecaster(...)
        node = TimeSeriesCoSimNode("demand", lstm, input_len=96, horizon=24)
        graph.connect("node_a.signal", "demand.context")
        graph.connect("demand.forecast", "node_b.load")
    """

    def __init__(
        self,
        name: str,
        model: nn.Module,
        input_len: int,
        horizon: int,
        extra_input_ports: Optional[List[str]] = None,
        extra_output_ports: Optional[List[str]] = None,
    ) -> None:
        in_ports  = ["context"] + (extra_input_ports or [])
        out_ports = ["forecast", "context_next"] + (extra_output_ports or [])
        super().__init__(name, in_ports, out_ports)
        self.model = model
        self.input_len = input_len
        self.horizon = horizon

    def step(
        self,
        inputs: Dict[str, torch.Tensor],
        t: float,
        dt: float,
    ) -> Dict[str, torch.Tensor]:
        context = inputs["context"]  # (1, input_len, F)
        raw = self.model(context)
        # TSModelBase may return TSOutput or plain Tensor
        if hasattr(raw, "y_hat"):
            forecast = raw.y_hat          # (1, H, F) or (1, H)
        elif hasattr(raw, "y"):
            forecast = raw.y
        else:
            forecast = raw
        # Slide context window forward by appending first forecast step
        first = forecast[:, :1, :] if forecast.dim() == 3 else forecast[:, :1].unsqueeze(-1)
        context_next = torch.cat([context[:, 1:, :], first], dim=1)
        return {"forecast": forecast, "context_next": context_next}

    def parameters(self):
        return self.model.parameters()
