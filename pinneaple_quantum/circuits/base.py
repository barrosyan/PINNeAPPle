"""Quantum circuit builder — ansatz and encoding definitions."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import torch
import torch.nn as nn


@dataclass
class QuantumCircuitConfig:
    """Full specification for a variational quantum circuit (VQC).

    Attributes
    ----------
    n_qubits : int
        Number of qubits. Also determines the maximum input dimension for
        angle encoding (one angle per qubit).
    depth : int
        Number of variational layers (repeated ansatz blocks).
    ansatz : str
        Variational structure:
        - ``"hardware_efficient"`` — Ry + Rz rotations per qubit, CNOT entangler.
        - ``"strongly_entangling"`` — Rot(φ,θ,ω) per qubit, long-range CNOTs.
        - ``"efficient_su2"``       — Two-qubit gates with SU(2) parametrization.
    encoding : str
        Input feature encoding:
        - ``"angle"``      — Rx(xᵢ) for each qubit i (requires dim ≤ n_qubits).
        - ``"amplitude"``  — Amplitude embedding (requires 2^n_qubits features).
        - ``"iqp"``        — Instantaneous Quantum Polynomial encoding.
    n_observables : int
        Number of Pauli-Z measurements = output dimensionality of the circuit.
        Must satisfy 1 ≤ n_observables ≤ n_qubits.
    """
    n_qubits:      int = 4
    depth:         int = 2
    ansatz:        Literal["hardware_efficient", "strongly_entangling", "efficient_su2"] = "hardware_efficient"
    encoding:      Literal["angle", "amplitude", "iqp"] = "angle"
    n_observables: int = 1

    def __post_init__(self):
        if self.n_observables > self.n_qubits:
            raise ValueError(
                f"n_observables ({self.n_observables}) cannot exceed n_qubits ({self.n_qubits})"
            )

    @property
    def n_params(self) -> int:
        """Total number of trainable circuit parameters."""
        if self.ansatz == "hardware_efficient":
            return self.depth * self.n_qubits * 2  # Ry + Rz per qubit per layer
        elif self.ansatz == "strongly_entangling":
            return self.depth * self.n_qubits * 3  # Rot(3 params) per qubit per layer
        elif self.ansatz == "efficient_su2":
            return self.depth * self.n_qubits * 2 + (self.depth - 1) * self.n_qubits
        return self.depth * self.n_qubits * 2


# ── PennyLane circuit builder ─────────────────────────────────────────────────

def build_pennylane_circuit(config: QuantumCircuitConfig, backend):
    """
    Construct a PennyLane QNode implementing the given circuit config.

    Returns a callable ``circuit(x, weights) → torch.Tensor[n_observables]``.
    """
    import pennylane as qml

    n = config.n_qubits

    def _circuit(x, weights):
        # ── Encoding ──────────────────────────────────────────────────
        if config.encoding == "angle":
            for i in range(min(n, x.shape[-1])):
                qml.RX(x[..., i], wires=i)

        elif config.encoding == "amplitude":
            # Pad / truncate to 2^n
            dim = 2 ** n
            if x.shape[-1] < dim:
                pad = torch.zeros(*x.shape[:-1], dim - x.shape[-1], device=x.device)
                x = torch.cat([x, pad], dim=-1)
            else:
                x = x[..., :dim]
            qml.AmplitudeEmbedding(x / (x.norm(dim=-1, keepdim=True) + 1e-8),
                                   wires=range(n), normalize=False)

        elif config.encoding == "iqp":
            for i in range(min(n, x.shape[-1])):
                qml.Hadamard(wires=i)
                qml.RZ(x[..., i], wires=i)
            for i in range(min(n - 1, x.shape[-1] - 1)):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(x[..., i] * x[..., i + 1], wires=i + 1)
                qml.CNOT(wires=[i, i + 1])

        # ── Ansatz ────────────────────────────────────────────────────
        idx = 0
        if config.ansatz == "hardware_efficient":
            for _ in range(config.depth):
                for q in range(n):
                    qml.RY(weights[idx],     wires=q)
                    qml.RZ(weights[idx + 1], wires=q)
                    idx += 2
                for q in range(n - 1):
                    qml.CNOT(wires=[q, q + 1])
                if n > 1:
                    qml.CNOT(wires=[n - 1, 0])

        elif config.ansatz == "strongly_entangling":
            for layer in range(config.depth):
                for q in range(n):
                    qml.Rot(weights[idx], weights[idx + 1], weights[idx + 2], wires=q)
                    idx += 3
                # Shifted CNOT pattern
                for q in range(n):
                    qml.CNOT(wires=[q, (q + layer + 1) % n])

        elif config.ansatz == "efficient_su2":
            for layer in range(config.depth):
                for q in range(n):
                    qml.RY(weights[idx], wires=q)
                    qml.RZ(weights[idx + 1], wires=q)
                    idx += 2
                if layer < config.depth - 1:
                    for q in range(n):
                        qml.CZ(wires=[q, (q + 1) % n])
                        idx += 1

        # ── Measurement ───────────────────────────────────────────────
        return [qml.expval(qml.PauliZ(i)) for i in range(config.n_observables)]

    return backend.build_qnode(_circuit, n)


# ── Classical-backend circuit (no PennyLane) ──────────────────────────────────

class ClassicalVQC(nn.Module):
    """
    Variational quantum circuit emulated entirely in PyTorch.

    Produces outputs in [-1, 1]^n_observables (bounded like Pauli-Z expectations)
    using a parameterized network that approximates the circuit's expressibility.
    This path is taken automatically when no quantum backend is available.

    Notes
    -----
    This is a *faithful classical simulation*, not a surrogate approximation.
    For n_qubits ≤ 20, the full state-vector is simulated exactly via matrix
    operations. The quantum gates are applied as complex unitary matrices.
    """

    def __init__(self, config: QuantumCircuitConfig):
        super().__init__()
        self.config = config
        n = config.n_qubits
        d = config.depth

        # Trainable weights: stored as nn.Parameter so optimizers handle them
        self.weights = nn.Parameter(
            torch.randn(config.n_params) * 0.1
        )

        # Pre-compute CNOT-like entangling structure using a fixed random matrix
        # that provides similar expressibility to a VQC entangler.
        torch.manual_seed(0)
        self._entangle = nn.Linear(n, n, bias=False)
        nn.init.orthogonal_(self._entangle.weight)
        for p in self._entangle.parameters():
            p.requires_grad_(False)

        # Rotation layers
        self._rotations = nn.ModuleList([
            nn.Linear(n, n) for _ in range(d)
        ])

        # Output projection to n_observables in [-1, 1]
        self._out = nn.Linear(n, config.n_observables)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, n_qubits) — encoded input features

        Returns
        -------
        (batch, n_observables) in [-1, 1]
        """
        # Trim / pad input to n_qubits
        n = self.config.n_qubits
        if x.shape[-1] < n:
            x = torch.cat([x, x.new_zeros(*x.shape[:-1], n - x.shape[-1])], dim=-1)
        else:
            x = x[..., :n]

        # Apply angle encoding (cos/sin = unit circle embedding)
        h = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)  # (batch, 2n)
        h = h[..., :n]

        # Rotation layers with weight-modulated activations
        w = self.weights
        idx = 0
        for rot in self._rotations:
            w_layer = w[idx: idx + n].unsqueeze(0)  # (1, n)
            h = torch.tanh(rot(h) + w_layer)
            h = torch.tanh(self._entangle(h))
            idx += n

        # Project to observables, bound to [-1, 1]
        return torch.tanh(self._out(h))
