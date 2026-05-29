"""Quantum backend adapter — PennyLane, Qiskit, or classical simulator."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch


@dataclass
class BackendConfig:
    """Configuration for the quantum execution backend.

    Attributes
    ----------
    provider : str
        Backend provider. One of ``"pennylane"``, ``"qiskit"``, ``"classical"``.
        ``"classical"`` uses a CPU-based state-vector simulation that requires
        no additional quantum packages — useful for prototyping.
    device_name : str
        Provider-specific device string.
        - PennyLane: ``"default.qubit"`` (CPU), ``"lightning.qubit"`` (fast CPU),
          ``"default.mixed"`` (noise).
        - Qiskit Aer: ``"statevector_simulator"``, ``"qasm_simulator"``.
    diff_method : str
        Gradient computation strategy for quantum parameters.
        - ``"parameter-shift"``: exact analytic gradient via ±π/2 shift.
        - ``"adjoint"``: fast adjoint differentiation (PennyLane only).
        - ``"backprop"``: standard PyTorch autograd through the circuit (simulator only).
    shots : Optional[int]
        Number of measurement shots. ``None`` → exact (analytic) expectation values.
    """
    provider:    Literal["pennylane", "qiskit", "classical"] = "pennylane"
    device_name: str = "default.qubit"
    diff_method: str = "parameter-shift"
    shots:       Optional[int] = None


def get_backend(config: Optional[BackendConfig] = None) -> "QuantumBackend":
    """Instantiate the correct backend from a config (or defaults).

    Falls back automatically:
    - pennylane requested but not installed → classical
    - qiskit requested but not installed   → classical
    """
    cfg = config or BackendConfig()

    if cfg.provider == "pennylane":
        try:
            import pennylane  # noqa: F401
            return PennyLaneBackend(cfg)
        except ImportError:
            return ClassicalBackend(cfg)

    if cfg.provider == "qiskit":
        try:
            from qiskit_aer import AerSimulator  # noqa: F401
            return QiskitBackend(cfg)
        except ImportError:
            return ClassicalBackend(cfg)

    return ClassicalBackend(cfg)


# ── Backend classes ───────────────────────────────────────────────────────────

class QuantumBackend:
    """Abstract base for quantum backends."""

    def __init__(self, config: BackendConfig):
        self.config = config

    @property
    def name(self) -> str:
        raise NotImplementedError

    def build_qnode(self, circuit_fn, n_qubits: int):
        """Wrap a circuit function into an executable QNode."""
        raise NotImplementedError

    def info(self) -> dict:
        return {"provider": self.config.provider, "device": self.config.device_name}


class PennyLaneBackend(QuantumBackend):
    """PennyLane backend (supports CPU/GPU simulators and real QPUs)."""

    @property
    def name(self) -> str:
        return f"pennylane:{self.config.device_name}"

    def build_qnode(self, circuit_fn, n_qubits: int):
        import pennylane as qml
        dev = qml.device(
            self.config.device_name,
            wires=n_qubits,
            shots=self.config.shots,
        )
        return qml.QNode(
            circuit_fn,
            dev,
            interface="torch",
            diff_method=self.config.diff_method,
        )

    def info(self) -> dict:
        import pennylane as qml
        return {
            "provider":    "pennylane",
            "device":      self.config.device_name,
            "version":     qml.__version__,
            "diff_method": self.config.diff_method,
            "shots":       self.config.shots,
        }


class QiskitBackend(QuantumBackend):
    """Qiskit / Aer backend."""

    @property
    def name(self) -> str:
        return f"qiskit:{self.config.device_name}"

    def build_qnode(self, circuit_fn, n_qubits: int):
        # Use PennyLane-Qiskit plugin if available
        try:
            import pennylane as qml
            dev = qml.device(
                "qiskit.aer",
                wires=n_qubits,
                backend=self.config.device_name,
                shots=self.config.shots,
            )
            return qml.QNode(circuit_fn, dev, interface="torch",
                             diff_method=self.config.diff_method)
        except Exception:
            # Fallback to classical if Qiskit plugin unavailable
            return ClassicalBackend(self.config).build_qnode(circuit_fn, n_qubits)

    def info(self) -> dict:
        return {"provider": "qiskit", "device": self.config.device_name}


class ClassicalBackend(QuantumBackend):
    """
    Classical state-vector simulator.

    Implements a minimal qubit state-vector engine using NumPy/PyTorch.
    No quantum packages required. Supports Rx, Ry, Rz, CNOT, and Pauli-Z
    expectation values. Gradient flows through torch.autograd.

    This is intentionally a faithful classical *simulation* (not an approximation):
    for ≤ ~20 qubits it reproduces exact quantum circuit results.
    """

    @property
    def name(self) -> str:
        return "classical:statevector"

    def build_qnode(self, circuit_fn, n_qubits: int):
        """Return a ClassicalQNode wrapper that calls circuit_fn on our emulator."""
        return ClassicalQNode(circuit_fn, n_qubits)

    def info(self) -> dict:
        return {"provider": "classical", "device": "statevector", "max_qubits": 20}


# ── Classical state-vector engine ─────────────────────────────────────────────

def _rx(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([torch.stack([c, -1j * s]), torch.stack([-1j * s, c])])


def _ry(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def _rz(theta: torch.Tensor) -> torch.Tensor:
    e_neg = torch.exp(-1j * theta / 2)
    e_pos = torch.exp(1j * theta / 2)
    z = torch.zeros_like(e_neg)
    return torch.stack([torch.stack([e_neg, z]), torch.stack([z, e_pos])])


def _apply_single(state: torch.Tensor, gate: torch.Tensor, qubit: int, n: int) -> torch.Tensor:
    """Apply a 2×2 gate to a single qubit of the full n-qubit state."""
    shape = [2] * n
    state = state.reshape(shape)
    # Move target qubit to axis 0
    state = state.permute([qubit] + [i for i in range(n) if i != qubit])
    flat = state.reshape(2, -1)
    state = (gate.to(flat.dtype) @ flat).reshape([2] + [2] * (n - 1))
    # Move qubit back
    inv_perm = [0] * n
    inv_perm[qubit] = 0
    j = 1
    for i in range(n):
        if i != qubit:
            inv_perm[i] = j
            j += 1
    state = state.permute(inv_perm)
    return state.reshape(2 ** n)


def _apply_cnot(state: torch.Tensor, ctrl: int, tgt: int, n: int) -> torch.Tensor:
    """Apply CNOT gate."""
    shape = [2] * n
    st = state.reshape(shape)
    idx_ctrl1 = [slice(None)] * n
    idx_ctrl1[ctrl] = 1
    sub = st[tuple(idx_ctrl1)]  # shape: (2,) * (n-1) leaving tgt axis
    # flip tgt
    sub_flip = sub.flip(dims=[tgt if tgt < ctrl else tgt - 1])
    st2 = st.clone()
    st2[tuple(idx_ctrl1)] = sub_flip
    return st2.reshape(2 ** n)


def _pauli_z_expectation(state: torch.Tensor, qubit: int, n: int) -> torch.Tensor:
    """Compute ⟨Z⟩ for a single qubit."""
    shape = [2] * n
    probs = state.abs() ** 2
    probs = probs.reshape(shape)
    idx0 = [slice(None)] * n; idx0[qubit] = 0
    idx1 = [slice(None)] * n; idx1[qubit] = 1
    return (probs[tuple(idx0)] - probs[tuple(idx1)]).sum()


class ClassicalQNode:
    """
    Callable that executes a quantum circuit description on the classical engine.

    The ``circuit_fn`` receives a mock ``ops`` recorder instead of a real device.
    When called, we execute the recorded operations on a state vector.
    """

    def __init__(self, circuit_fn, n_qubits: int):
        self._circuit_fn = circuit_fn
        self.n_qubits = n_qubits

    def __call__(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        n = self.n_qubits
        # Run the circuit description to collect operations
        recorder = _CircuitRecorder(n)
        self._circuit_fn(x, weights, recorder=recorder)
        # Execute on state vector |0...0⟩
        state = torch.zeros(2 ** n, dtype=torch.complex64)
        state[0] = 1.0
        for op in recorder.ops:
            state = op(state, n)
        # Measure: return Pauli-Z expectation for each observable
        return torch.stack([
            _pauli_z_expectation(state, q, n)
            for q in recorder.observables
        ]).real


class _CircuitRecorder:
    """Collects gate operations for replay on the classical engine."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.ops = []
        self.observables = []

    def rx(self, theta, wires):
        q = wires if isinstance(wires, int) else wires[0]
        theta_t = theta if isinstance(theta, torch.Tensor) else torch.tensor(float(theta))
        self.ops.append(lambda s, n, q=q, t=theta_t: _apply_single(s, _rx(t), q, n))

    def ry(self, theta, wires):
        q = wires if isinstance(wires, int) else wires[0]
        theta_t = theta if isinstance(theta, torch.Tensor) else torch.tensor(float(theta))
        self.ops.append(lambda s, n, q=q, t=theta_t: _apply_single(s, _ry(t), q, n))

    def rz(self, theta, wires):
        q = wires if isinstance(wires, int) else wires[0]
        theta_t = theta if isinstance(theta, torch.Tensor) else torch.tensor(float(theta))
        self.ops.append(lambda s, n, q=q, t=theta_t: _apply_single(s, _rz(t), q, n))

    def cnot(self, wires):
        ctrl, tgt = wires
        self.ops.append(lambda s, n, c=ctrl, t=tgt: _apply_cnot(s, c, t, n))

    def measure_z(self, qubit):
        self.observables.append(qubit)
