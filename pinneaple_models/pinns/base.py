from __future__ import annotations
"""Base classes and PINNOutput for physics-informed neural networks.

Single source of truth — all PINN variants import from here.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union, Tuple

import torch
import torch.nn as nn


@dataclass
class PINNOutput:
    """Standard output for all PINN-family models."""
    y: Union[torch.Tensor, List[torch.Tensor], Any]
    losses: Dict[str, torch.Tensor]
    extras: Dict[str, Any] = field(default_factory=dict)


class PINNBase(nn.Module):
    """
    Base class for all PINN-family models.

    Contract
    --------
    - ``forward(*inputs, **kwargs) -> PINNOutput``
    - ``predict(*inputs, **kwargs) -> Tensor``  (returns ``output.y``)
    - ``physics_loss(physics_fn, physics_data) -> dict[str, Tensor]``

    Utilities
    ---------
    - ``_ref_tensor(*maybe)``  — finds a reference tensor for device/dtype
    - ``_zeros(like)``         — zero scalar on the same device/dtype
    - ``_to_scalar_tensor(v, ref)`` — casts a value to a scalar tensor
    - ``ensure_requires_grad(x)``   — enables grad if not already set
    - ``grad(y, x)``                — autograd gradient
    - ``normal_derivative(y, x, n_hat)`` — boundary flux
    """

    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------
    # Device / tensor helpers
    # ------------------------------------------------------------------

    def _ref_tensor(self, *maybe: Any) -> torch.Tensor:
        """Return a reference tensor for device / dtype inference.

        Checks (in order): model parameters, positional ``maybe`` args
        (traversing dicts and lists recursively), fallback CPU float32.
        """
        try:
            return next(self.parameters())
        except StopIteration:
            pass

        def _find(obj: Any) -> Optional[torch.Tensor]:
            if torch.is_tensor(obj):
                return obj
            if isinstance(obj, dict):
                for v in obj.values():
                    t = _find(v)
                    if t is not None:
                        return t
            if isinstance(obj, (list, tuple)):
                for v in obj:
                    t = _find(v)
                    if t is not None:
                        return t
            return None

        for obj in maybe:
            t = _find(obj)
            if t is not None:
                return t

        return torch.empty((), device="cpu", dtype=torch.float32)

    def _zeros(self, like: Any = None) -> torch.Tensor:
        """Zero scalar tensor on the same device/dtype as *like*."""
        ref = self._ref_tensor(like)
        return ref.new_zeros(())

    def _to_scalar_tensor(self, v: Any, ref: torch.Tensor) -> torch.Tensor:
        """Cast *v* to a scalar tensor on the same device/dtype as *ref*."""
        if torch.is_tensor(v):
            t = v.to(device=ref.device, dtype=ref.dtype)
        else:
            t = torch.tensor(float(v), device=ref.device, dtype=ref.dtype)
        if t.ndim != 0:
            t = t.sum()
        return t

    # ------------------------------------------------------------------
    # Differentiation utilities
    # ------------------------------------------------------------------

    @staticmethod
    def ensure_requires_grad(x: torch.Tensor) -> torch.Tensor:
        """Return x with requires_grad=True (detaches if needed)."""
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        return x

    @staticmethod
    def grad(
        y: torch.Tensor,
        x: torch.Tensor,
        *,
        retain_graph: bool = True,
        create_graph: bool = True,
    ) -> torch.Tensor:
        """∂y/∂x via autograd (sums y to scalar first)."""
        y_ = y if y.ndim == 0 else y.sum()
        (g,) = torch.autograd.grad(
            y_,
            x,
            retain_graph=retain_graph,
            create_graph=create_graph,
            allow_unused=False,
        )
        return g

    @classmethod
    def normal_derivative(
        cls, y: torch.Tensor, x: torch.Tensor, n_hat: torch.Tensor
    ) -> torch.Tensor:
        """Compute ∂y/∂n = (∇y) · n̂ for each output component."""
        if n_hat.ndim == 1:
            n_hat = n_hat.unsqueeze(0).expand(x.shape[0], -1)
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        outs = []
        for k in range(y.shape[1]):
            gk = cls.grad(y[:, k], x)
            dn = (gk * n_hat).sum(dim=1, keepdim=True)
            outs.append(dn)
        return torch.cat(outs, dim=1)

    # ------------------------------------------------------------------
    # Standard interface
    # ------------------------------------------------------------------

    def predict(self, *inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run forward and return y only (discards losses and extras)."""
        out = self.forward(*inputs, **kwargs)
        if isinstance(out, PINNOutput):
            return out.y
        return out

    def physics_loss(
        self,
        *,
        physics_fn: Optional[Callable[..., Any]] = None,
        physics_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute physics loss via an external *physics_fn*.

        *physics_fn* is called as ``physics_fn(model, physics_data, **kwargs)``
        and should return one of:
          - ``(total_loss, components_dict)``
          - ``dict[str, tensor | float]``

        The output always contains the key ``"physics"`` with the scalar total.
        Sub-components are stored as ``"physics/<name>"``.

        If *physics_fn* is ``None`` returns ``{"physics": zero}``.
        """
        if physics_fn is None or physics_data is None:
            return {"physics": self._zeros()}

        ref = self._ref_tensor(physics_data)
        z0 = ref.new_zeros(())

        # Thin adapter so factory loss functions (which expect model.forward(x)
        # to return a Tensor) can be used without modification.
        class _Adapter(nn.Module):
            def __init__(self, base: "PINNBase"):
                super().__init__()
                self.base = base
                inv = getattr(base, "inverse_params", None)
                self.inverse_params = inv if isinstance(inv, nn.ParameterDict) else nn.ParameterDict()

            def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
                y = self.base.predict(*inputs)
                if isinstance(y, (list, tuple)):
                    raise TypeError("_Adapter: factory expects a Tensor, got list/tuple.")
                if not torch.is_tensor(y):
                    raise TypeError(f"_Adapter: expected Tensor, got {type(y)}.")
                return y

        if isinstance(physics_data, dict) and not kwargs:
            res = physics_fn(_Adapter(self), physics_data)
        else:
            res = physics_fn(self, physics_data, **kwargs)

        if isinstance(res, tuple) and len(res) == 2:
            total, comps = res
            if not torch.is_tensor(total):
                total = torch.tensor(float(total), device=ref.device, dtype=ref.dtype)
            out: Dict[str, torch.Tensor] = {"physics": total}
            if isinstance(comps, dict):
                for k, v in comps.items():
                    out[f"physics/{k}"] = (
                        v if torch.is_tensor(v)
                        else torch.tensor(float(v), device=total.device, dtype=total.dtype)
                    )
            return out

        if isinstance(res, dict):
            out = {}
            for k, v in res.items():
                if torch.is_tensor(v):
                    out[k] = v
                else:
                    try:
                        out[k] = torch.tensor(float(v), device=ref.device, dtype=ref.dtype)
                    except Exception:
                        pass
            if "physics" not in out:
                total = None
                for k in ("total", "loss", "pde", "weak"):
                    if k in out and torch.is_tensor(out[k]):
                        total = out[k]
                        break
                out["physics"] = total if total is not None else z0
            return out

        return {"physics": z0}

    # ------------------------------------------------------------------
    # Model export
    # ------------------------------------------------------------------

    def export_torchscript(self, path: str, example_input: Optional[torch.Tensor] = None) -> str:
        """Export model to TorchScript (.pt).

        Parameters
        ----------
        path : output file path (should end with .pt)
        example_input : example input tensor for tracing (required for trace mode)

        Returns path where file was saved.
        """
        import os
        self.eval()
        if example_input is not None:
            try:
                scripted = torch.jit.trace(self, example_input)
            except Exception:
                scripted = torch.jit.script(self)
        else:
            scripted = torch.jit.script(self)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        scripted.save(path)
        return path

    def export_onnx(
        self,
        path: str,
        example_input: torch.Tensor,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        opset_version: int = 17,
        dynamic_axes: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Export model to ONNX format for deployment in C++, MATLAB, etc.

        Parameters
        ----------
        path : output file path (should end with .onnx)
        example_input : required for ONNX tracing
        input_names : names for input tensors (default: ["input"])
        output_names : names for output tensors (default: ["output"])
        opset_version : ONNX opset (default: 17)
        dynamic_axes : dict for dynamic batch axes (default: batch dim 0 dynamic)

        Returns path where file was saved.
        """
        import os
        import torch.onnx
        self.eval()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        _input_names = input_names or ["input"]
        _output_names = output_names or ["output"]
        if dynamic_axes is None:
            dynamic_axes = {n: {0: "batch"} for n in _input_names + _output_names}
        torch.onnx.export(
            self,
            example_input,
            path,
            input_names=_input_names,
            output_names=_output_names,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
        )
        return path

    def save_checkpoint(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save model weights + metadata as a pinneaple checkpoint.

        The checkpoint dict contains:
          - "state_dict": model weights
          - "class_name": fully qualified class name
          - "metadata": user-provided dict

        Returns path where file was saved.
        """
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        ckpt = {
            "state_dict": self.state_dict(),
            "class_name": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "metadata": metadata or {},
        }
        torch.save(ckpt, path)
        return path

    @classmethod
    def load_checkpoint(cls, path: str, **init_kwargs) -> "PINNBase":
        """Load a model from a pinneaple checkpoint.

        Usage::
            model = VanillaPINN.load_checkpoint("model.pt", in_dim=2, out_dim=1, hidden=[64,64,64])
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**init_kwargs)
        model.load_state_dict(ckpt["state_dict"])
        return model
