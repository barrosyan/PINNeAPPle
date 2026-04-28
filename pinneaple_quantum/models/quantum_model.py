"""QuantumModel — variational quantum circuit as a PyTorch nn.Module."""
from __future__ import annotations

from typing import Optional, Literal

import torch
import torch.nn as nn

from pinneaple_quantum.circuits.base import QuantumCircuitConfig, ClassicalVQC, build_pennylane_circuit
from pinneaple_quantum.backends.backend import BackendConfig, get_backend


class QuantumModel(nn.Module):
    """
    A variational quantum circuit (VQC) exposed as a standard PyTorch ``nn.Module``.

    When PennyLane (or Qiskit) is installed, executes on the specified quantum
    backend with exact or shot-based evaluation. Falls back automatically to a
    classical state-vector simulation when no quantum packages are present.

    Parameters
    ----------
    circuit_config : QuantumCircuitConfig
        Specifies the number of qubits, depth, ansatz, encoding, and number
        of measured observables.
    backend_config : BackendConfig, optional
        Specifies the execution backend. Defaults to PennyLane ``default.qubit``
        with parameter-shift gradients.
    output_scale : float
        Linear scale applied to the raw circuit outputs (Pauli-Z ∈ [−1, 1]).
        Default ``1.0``.

    Examples
    --------
    >>> from pinneaple_quantum import QuantumModel, QuantumCircuitConfig
    >>> cfg = QuantumCircuitConfig(n_qubits=4, depth=2, n_observables=1)
    >>> model = QuantumModel(cfg)
    >>> x = torch.randn(16, 4)   # batch of 16, 4 features
    >>> out = model(x)           # → (16, 1)
    """

    def __init__(
        self,
        circuit_config: QuantumCircuitConfig,
        backend_config: Optional[BackendConfig] = None,
        output_scale: float = 1.0,
    ):
        super().__init__()
        self.circuit_config = circuit_config
        self.backend_config = backend_config or BackendConfig()
        self.output_scale   = output_scale

        self._backend = get_backend(self.backend_config)
        self._use_pennylane = self._backend.__class__.__name__ == "PennyLaneBackend"
        self._use_qiskit    = self._backend.__class__.__name__ == "QiskitBackend"

        if self._use_pennylane or self._use_qiskit:
            self._qnode   = build_pennylane_circuit(circuit_config, self._backend)
            self._weights = nn.Parameter(
                torch.randn(circuit_config.n_params) * 0.1
            )
            self._classical_vqc = None
        else:
            self._qnode         = None
            self._weights       = None
            self._classical_vqc = ClassicalVQC(circuit_config)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_qubits(self) -> int:
        return self.circuit_config.n_qubits

    @property
    def n_params(self) -> int:
        return self.circuit_config.n_params

    @property
    def backend_name(self) -> str:
        return self._backend.name

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the VQC on input batch ``x``.

        Parameters
        ----------
        x : Tensor of shape ``(batch, input_dim)``
            Input features. For angle encoding, at most ``n_qubits`` features
            are used per sample.

        Returns
        -------
        Tensor of shape ``(batch, n_observables)``
            Expectation values of Pauli-Z observables, scaled by ``output_scale``.
        """
        if self._classical_vqc is not None:
            return self._classical_vqc(x) * self.output_scale

        # PennyLane / Qiskit path — run circuit per sample (batching via vmap
        # or explicit loop depending on PennyLane version).
        try:
            import pennylane as qml
            batched = qml.batch_input(self._qnode, argnum=0)
            out = batched(x, self._weights)
        except (AttributeError, TypeError):
            # Fallback: loop over batch
            outs = [
                torch.stack(self._qnode(x[i], self._weights))
                for i in range(x.shape[0])
            ]
            out = torch.stack(outs, dim=0)

        if out.ndim == 1:
            out = out.unsqueeze(-1)
        return out * self.output_scale

    def extra_repr(self) -> str:
        c = self.circuit_config
        return (
            f"n_qubits={c.n_qubits}, depth={c.depth}, ansatz={c.ansatz!r}, "
            f"encoding={c.encoding!r}, n_observables={c.n_observables}, "
            f"backend={self.backend_name!r}"
        )
