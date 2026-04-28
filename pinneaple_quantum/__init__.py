"""pinneaple_quantum — PINNeAPPle Quantum Module (PQM).

Official extension for hybrid classical–quantum physics-informed machine
learning. Combines variational quantum circuits (VQCs) with PDE-constrained
loss functions to solve quantum mechanical problems on both simulators and
real quantum hardware.

Supported workflows
-------------------
VQ-PINN (Variational Quantum PINN)
    Variational quantum circuit as wavefunction ansatz ψ(x).
    Physics loss = Rayleigh quotient ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩ (VQE objective).
    Solves: Schrödinger equation (1D / 2D), ground state + excited states.

Hybrid PINN
    Classical neural network pre-encodes features → quantum circuit.
    Useful for multi-scale / high-dimensional problems.

Hamiltonian Learning
    Discover the Hamiltonian operator from energy measurement data.

Quantum Kernel PINN
    Quantum-enhanced feature map K(xᵢ, xⱼ) = |⟨φ(xᵢ)|φ(xⱼ)⟩|² for
    kernel-based surrogate modelling.

Backend support
---------------
PennyLane   — CPU/GPU simulators, parameter-shift gradients, real QPUs.
Qiskit/Aer  — Aer statevector / QASM simulators.
Classical   — Built-in PyTorch state-vector simulation (no quantum deps).

The module falls back automatically to the classical engine if PennyLane or
Qiskit are not installed, so the full API is always available.

Quick start
-----------
>>> import pinneaple_quantum as pqm

>>> # Circuit and model
>>> cfg = pqm.QuantumCircuitConfig(n_qubits=4, depth=3)
>>> model = pqm.QuantumModel(cfg)

>>> # Physics loss
>>> x = torch.linspace(-3, 3, 500).unsqueeze(-1)
>>> energy, norm = pqm.loss.energy_loss(model, x, "harmonic_oscillator")

>>> # One-line solve
>>> result = pqm.solve(
...     problem="quantum",
...     equation="schrodinger",
...     potential="harmonic",
...     method="vq_pinn",
...     n_qubits=4, depth=3, epochs=2000,
... )
>>> print(f"Ground state energy: {result['energy_est']:.4f}")
>>> print(f"Exact E₀:            {result['E_exact']:.4f}")

>>> # Hybrid model
>>> import torch.nn as nn
>>> mlp = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 4))
>>> hybrid = pqm.HybridModel(mlp, model, mode="classical_first")
"""
from __future__ import annotations

__version__ = "0.1.0"

# ── Core types ────────────────────────────────────────────────────────────────
from pinneaple_quantum.circuits.base   import QuantumCircuitConfig
from pinneaple_quantum.backends.backend import BackendConfig, get_backend
from pinneaple_quantum.models.quantum_model import QuantumModel
from pinneaple_quantum.models.hybrid_model  import HybridModel

# ── Physics losses ────────────────────────────────────────────────────────────
from pinneaple_quantum.loss.schrodinger import (
    energy_loss,
    pde_residual_loss,
    time_dependent_loss,
    normalization_loss,
    hamiltonian_expectation,
    get_potential,
    harmonic_potential,
    coulomb_potential,
    double_well_potential,
    POTENTIALS,
)

# ── Training ──────────────────────────────────────────────────────────────────
from pinneaple_quantum.training.trainer import (
    QTrainer,
    QTrainerConfig,
    QTrainHistory,
    parameter_shift_gradient,
)

# ── Data ──────────────────────────────────────────────────────────────────────
from pinneaple_quantum.data.quantum_dataset import (
    QuantumState,
    QuantumObservable,
    QuantumCollocationDataset,
    SpinChainDataset,
    sample_domain_1d,
    sample_domain_2d,
    exact_eigenstates_1d_harmonic,
)

# ── Pipeline presets ──────────────────────────────────────────────────────────
from pinneaple_quantum.pipeline.presets import (
    vq_pinn,
    hybrid_pinn,
    learn_hamiltonian,
    quantum_kernel,
    solve,
)

# ── Submodule references (allows pqm.loss.energy_loss, pqm.pipeline.vq_pinn)
from pinneaple_quantum import loss, training, data, pipeline, backends, circuits, models

# ── Build convenience helper ──────────────────────────────────────────────────

def build_circuit(
    n_qubits: int = 4,
    depth: int = 2,
    ansatz: str = "hardware_efficient",
    encoding: str = "angle",
    n_observables: int = 1,
    backend: str = "pennylane",
) -> QuantumModel:
    """
    Shortcut to build a :class:`QuantumModel` from flat arguments.

    Parameters
    ----------
    n_qubits : int
    depth : int
    ansatz : str — "hardware_efficient" | "strongly_entangling" | "efficient_su2"
    encoding : str — "angle" | "amplitude" | "iqp"
    n_observables : int
    backend : str — "pennylane" | "qiskit" | "classical"

    Returns
    -------
    QuantumModel
    """
    cfg    = QuantumCircuitConfig(
        n_qubits=n_qubits, depth=depth,
        ansatz=ansatz, encoding=encoding,
        n_observables=n_observables,
    )
    bc_cfg = BackendConfig(provider=backend)
    return QuantumModel(cfg, bc_cfg)


def backend_info(provider: str = "pennylane") -> dict:
    """Return info dict for the specified backend."""
    return get_backend(BackendConfig(provider=provider)).info()


__all__ = [
    "__version__",
    # Circuit config
    "QuantumCircuitConfig",
    # Backend
    "BackendConfig",
    "get_backend",
    "backend_info",
    # Models
    "QuantumModel",
    "HybridModel",
    # Convenience builder
    "build_circuit",
    # Physics losses
    "energy_loss",
    "pde_residual_loss",
    "time_dependent_loss",
    "normalization_loss",
    "hamiltonian_expectation",
    "get_potential",
    "harmonic_potential",
    "coulomb_potential",
    "double_well_potential",
    "POTENTIALS",
    # Training
    "QTrainer",
    "QTrainerConfig",
    "QTrainHistory",
    "parameter_shift_gradient",
    # Data
    "QuantumState",
    "QuantumObservable",
    "QuantumCollocationDataset",
    "SpinChainDataset",
    "sample_domain_1d",
    "sample_domain_2d",
    "exact_eigenstates_1d_harmonic",
    # Pipelines
    "vq_pinn",
    "hybrid_pinn",
    "learn_hamiltonian",
    "quantum_kernel",
    "solve",
    # Submodules
    "loss",
    "training",
    "data",
    "pipeline",
    "backends",
    "circuits",
    "models",
]
