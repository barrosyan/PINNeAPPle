"""HybridModel — classical neural network + variational quantum circuit."""
from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

from .quantum_model import QuantumModel


class HybridModel(nn.Module):
    """
    Classical–quantum hybrid model that combines an ``nn.Module`` with a
    :class:`QuantumModel` into a single differentiable computational graph.

    Three composition modes are supported:

    ``"classical_first"`` (default)
        x → classical_net → latent → quantum_circuit → output

    ``"quantum_first"``
        x → quantum_circuit → classical_net → output

    ``"parallel"``
        x → classical_net → classical_out  ┐
        x → quantum_circuit → quantum_out  ┤ cat → linear → output
                                            ┘

    Parameters
    ----------
    classical_net : nn.Module
        Any PyTorch module. Its output dimension must be compatible with the
        quantum circuit's input (for ``"classical_first"``), or with a final
        linear projection (for ``"parallel"``).
    quantum_model : QuantumModel
        The variational quantum circuit component.
    mode : str
        Composition mode. One of ``"classical_first"``, ``"quantum_first"``,
        ``"parallel"``.
    out_dim : int, optional
        Output dimension. Required for ``"parallel"`` mode to define the
        fusion linear layer. Ignored for sequential modes.

    Examples
    --------
    >>> import torch.nn as nn
    >>> from pinneaple_quantum import QuantumModel, HybridModel, QuantumCircuitConfig
    >>> mlp = nn.Sequential(nn.Linear(2, 4), nn.Tanh())
    >>> qcfg = QuantumCircuitConfig(n_qubits=4, depth=2, n_observables=1)
    >>> qmodel = QuantumModel(qcfg)
    >>> model = HybridModel(mlp, qmodel, mode="classical_first")
    >>> out = model(torch.randn(8, 2))  # (8, 1)
    """

    def __init__(
        self,
        classical_net: nn.Module,
        quantum_model: QuantumModel,
        mode: Literal["classical_first", "quantum_first", "parallel"] = "classical_first",
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        self.classical_net  = classical_net
        self.quantum_model  = quantum_model
        self.mode           = mode

        if mode == "parallel":
            if out_dim is None:
                raise ValueError("out_dim is required for mode='parallel'")
            # Infer classical output dim by a probe forward pass
            with torch.no_grad():
                dummy = torch.zeros(1, next(classical_net.parameters()).shape[-1]
                                    if list(classical_net.parameters()) else 1)
                try:
                    c_out = classical_net(dummy).shape[-1]
                except Exception:
                    c_out = 1
            q_out = quantum_model.circuit_config.n_observables
            self.fusion = nn.Linear(c_out + q_out, out_dim)
        else:
            self.fusion = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape ``(batch, input_dim)``

        Returns
        -------
        Tensor of shape ``(batch, output_dim)``
        """
        if self.mode == "classical_first":
            latent = self.classical_net(x)
            return self.quantum_model(latent)

        elif self.mode == "quantum_first":
            q_out = self.quantum_model(x)
            return self.classical_net(q_out)

        elif self.mode == "parallel":
            c_out = self.classical_net(x)
            q_out = self.quantum_model(x)
            merged = torch.cat([c_out, q_out], dim=-1)
            return self.fusion(merged)

        raise ValueError(f"Unknown mode {self.mode!r}")

    def extra_repr(self) -> str:
        return f"mode={self.mode!r}"
