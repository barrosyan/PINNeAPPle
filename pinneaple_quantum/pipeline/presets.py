"""High-level pipeline presets for common quantum ML workflows."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from pinneaple_quantum.circuits.base import QuantumCircuitConfig
from pinneaple_quantum.backends.backend import BackendConfig
from pinneaple_quantum.models.quantum_model import QuantumModel
from pinneaple_quantum.models.hybrid_model import HybridModel
from pinneaple_quantum.loss.schrodinger import (
    energy_loss, pde_residual_loss, get_potential, normalization_loss,
)
from pinneaple_quantum.training.trainer import QTrainer, QTrainerConfig
from pinneaple_quantum.data.quantum_dataset import (
    sample_domain_1d, sample_domain_2d, exact_eigenstates_1d_harmonic,
)


# ── VQ-PINN preset ────────────────────────────────────────────────────────────

def vq_pinn(
    equation: str = "schrodinger_1d",
    potential: str = "harmonic_oscillator",
    n_qubits: int = 4,
    depth: int = 3,
    ansatz: str = "hardware_efficient",
    encoding: str = "angle",
    epochs: int = 3000,
    lr: float = 0.01,
    n_col: int = 1000,
    n_bc: int = 100,
    hbar: float = 1.0,
    mass: float = 1.0,
    lambda_norm: float = 0.5,
    lambda_bc: float = 5.0,
    backend: str = "pennylane",
    device: str = "cpu",
    seed: int = 42,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Build and train a Variational Quantum PINN (VQ-PINN).

    Solves a quantum PDE (e.g. Schrödinger equation) using a variational
    quantum circuit as the wavefunction ansatz ψ(x). The energy is minimised
    via the Rayleigh quotient. Boundary conditions are enforced via a soft
    penalty loss.

    Parameters
    ----------
    equation : str
        Quantum equation preset. Currently supported: ``"schrodinger_1d"``,
        ``"schrodinger_2d"``.
    potential : str
        Built-in potential name. One of: ``"harmonic_oscillator"``,
        ``"infinite_well"``, ``"coulomb"``, ``"double_well"``.
    n_qubits : int
        Number of qubits in the VQC ansatz. Also sets the spatial input
        dimension (angle encoding).
    depth : int
        Number of variational layers.
    ansatz : str
        VQC structure: ``"hardware_efficient"``, ``"strongly_entangling"``.
    encoding : str
        Input encoding: ``"angle"``, ``"amplitude"``, ``"iqp"``.
    epochs : int
        Training iterations.
    lr : float
        Learning rate for the Adam optimizer.
    n_col : int
        Number of interior collocation points.
    n_bc : int
        Number of boundary condition points.
    hbar, mass : float
        Physical constants (natural units: both 1.0).
    lambda_norm : float
        Weight of the normalization constraint.
    lambda_bc : float
        Weight of the Dirichlet boundary loss.
    backend : str
        Quantum backend provider: ``"pennylane"``, ``"qiskit"``, ``"classical"``.
    device : str
        PyTorch device.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        - ``"model"``        — trained QuantumModel
        - ``"history"``      — QTrainHistory
        - ``"x_col"``        — collocation points used
        - ``"x_bc"``         — boundary points used
        - ``"psi_pred"``     — ψ(x_col) after training
        - ``"energy_est"``   — final energy estimate
        - ``"psi_exact"``    — analytic ψ₀(x) for harmonic potential (or None)
        - ``"E_exact"``      — analytic ground state energy (or None)

    Examples
    --------
    >>> result = vq_pinn(
    ...     equation="schrodinger_1d",
    ...     potential="harmonic_oscillator",
    ...     n_qubits=4, depth=3, epochs=2000,
    ... )
    >>> print(f"Energy estimate: {result['energy_est']:.4f}")
    >>> print(f"Exact E₀: {result['E_exact']:.4f}")
    """
    torch.manual_seed(seed)

    # ── Domain
    dim = 1 if "1d" in equation else 2
    x_min, x_max = kwargs.get("x_min", -3.0), kwargs.get("x_max", 3.0)

    if dim == 1:
        x_col, x_bc = sample_domain_1d(x_min, x_max, n_col, n_bc, device=device)
    else:
        x_col, x_bc = sample_domain_2d(x_min, x_max, x_min, x_max, n_col, n_bc, device=device)

    # ── Model
    cfg = QuantumCircuitConfig(
        n_qubits=n_qubits,
        depth=depth,
        ansatz=ansatz,
        encoding=encoding,
        n_observables=1,
    )
    bc_cfg = BackendConfig(provider=backend)
    model  = QuantumModel(cfg, bc_cfg)

    # ── Boundary loss (Dirichlet: ψ = 0 at boundary)
    def bc_loss_fn(m, xb):
        psi_bc = m(xb)
        if psi_bc.ndim > 1:
            psi_bc = psi_bc.squeeze(-1)
        return psi_bc.pow(2).mean()

    # ── Physics loss: Rayleigh quotient + normalization
    V_fn = get_potential(potential)

    def physics_loss_fn(m, x):
        energy, norm_sq = energy_loss(
            m, x, potential=V_fn,
            hbar=hbar, mass=mass,
            **{k: v for k, v in kwargs.items() if k in ("omega", "charge", "a", "b")},
        )
        norm_pen = normalization_loss(m, x)
        total = energy + lambda_norm * norm_pen
        return total, {"energy": energy, "norm_sq": norm_sq}

    # ── Train
    train_cfg = QTrainerConfig(
        epochs=epochs,
        lr=lr,
        optimizer="adam",
        lambda_bc=lambda_bc,
        device=device,
    )
    trainer = QTrainer(model, physics_loss_fn, bc_loss_fn=bc_loss_fn, config=train_cfg)
    history = trainer.train(x_col, x_bc)

    # ── Evaluate
    with torch.no_grad():
        psi_pred = model(x_col).squeeze(-1)

    # energy_loss uses autograd (Laplacian) — must run outside no_grad
    energy_est_t, _ = energy_loss(model, x_col, potential=V_fn, hbar=hbar, mass=mass)
    energy_est = float(energy_est_t.detach())

    # ── Analytic reference (harmonic only, 1D)
    psi_exact, E_exact = None, None
    if potential in ("harmonic_oscillator", "harmonic") and dim == 1:
        psi_exact, E_exact = exact_eigenstates_1d_harmonic(
            x_col, n_state=0,
            omega=kwargs.get("omega", 1.0),
            hbar=hbar, mass=mass,
        )

    return {
        "model":       model,
        "history":     history,
        "x_col":       x_col,
        "x_bc":        x_bc,
        "psi_pred":    psi_pred,
        "energy_est":  energy_est,
        "psi_exact":   psi_exact,
        "E_exact":     E_exact,
    }


# ── Hybrid PINN preset ────────────────────────────────────────────────────────

def hybrid_pinn(
    classical_in: int = 1,
    classical_hidden: int = 32,
    n_qubits: int = 4,
    depth: int = 2,
    potential: str = "harmonic_oscillator",
    epochs: int = 3000,
    lr: float = 0.005,
    n_col: int = 1000,
    n_bc: int = 100,
    hbar: float = 1.0,
    mass: float = 1.0,
    backend: str = "pennylane",
    device: str = "cpu",
    seed: int = 42,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Hybrid PINN: classical MLP pre-encodes features → quantum circuit → output.

    Useful for multi-scale or high-dimensional problems where the classical
    network extracts relevant features before the quantum circuit applies
    expressible variational ansatz.

    Parameters
    ----------
    classical_in : int
        Dimensionality of the input space.
    classical_hidden : int
        Width of the classical pre-encoding MLP.
    n_qubits : int
        Number of qubits in the quantum layer.
    depth, potential, epochs, lr, n_col, n_bc, hbar, mass, backend, device, seed
        As in :func:`vq_pinn`.

    Returns
    -------
    dict with same keys as :func:`vq_pinn`.
    """
    torch.manual_seed(seed)

    x_min, x_max = kwargs.get("x_min", -3.0), kwargs.get("x_max", 3.0)
    x_col, x_bc  = sample_domain_1d(x_min, x_max, n_col, n_bc, device=device)

    # Classical encoder: R → R^n_qubits
    classical_net = nn.Sequential(
        nn.Linear(classical_in, classical_hidden),
        nn.Tanh(),
        nn.Linear(classical_hidden, n_qubits),
        nn.Tanh(),
    )

    qcfg   = QuantumCircuitConfig(n_qubits=n_qubits, depth=depth, n_observables=1)
    qmodel = QuantumModel(qcfg, BackendConfig(provider=backend))
    model  = HybridModel(classical_net, qmodel, mode="classical_first")

    V_fn = get_potential(potential)

    def bc_loss_fn(m, xb):
        psi_bc = m(xb)
        if psi_bc.ndim > 1:
            psi_bc = psi_bc.squeeze(-1)
        return psi_bc.pow(2).mean()

    def physics_loss_fn(m, x):
        energy, norm_sq = energy_loss(m, x, potential=V_fn, hbar=hbar, mass=mass)
        norm_pen = normalization_loss(m, x)
        total = energy + 0.5 * norm_pen
        return total, {"energy": energy, "norm_sq": norm_sq}

    train_cfg = QTrainerConfig(epochs=epochs, lr=lr, device=device)
    trainer   = QTrainer(model, physics_loss_fn, bc_loss_fn=bc_loss_fn, config=train_cfg)
    history   = trainer.train(x_col, x_bc)

    with torch.no_grad():
        psi_pred = model(x_col).squeeze(-1)

    energy_est_t, _ = energy_loss(model, x_col, potential=V_fn, hbar=hbar, mass=mass)

    psi_exact, E_exact = None, None
    if potential in ("harmonic_oscillator", "harmonic"):
        psi_exact, E_exact = exact_eigenstates_1d_harmonic(
            x_col, n_state=0, hbar=hbar, mass=mass
        )

    return {
        "model":       model,
        "history":     history,
        "x_col":       x_col,
        "x_bc":        x_bc,
        "psi_pred":    psi_pred,
        "energy_est":  float(energy_est_t.detach()),
        "psi_exact":   psi_exact,
        "E_exact":     E_exact,
    }


# ── Hamiltonian learning preset ───────────────────────────────────────────────

def learn_hamiltonian(
    data: torch.Tensor,
    labels: torch.Tensor,
    n_spins: int = 4,
    epochs: int = 2000,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Learn a matrix Hamiltonian from energy measurement data.

    Fits a symmetric (Hermitian) matrix H such that ⟨ψ|H|ψ⟩ ≈ label
    for each input state ψ.

    Parameters
    ----------
    data : Tensor (N, d)
        State vectors ψ (rows).
    labels : Tensor (N,)
        Corresponding energy measurements ⟨E⟩.
    n_spins : int
        Matrix dimension d = 2^n_spins (Hilbert space dim).
    epochs, lr, device, seed : as usual.

    Returns
    -------
    dict with keys ``"H_learned"``, ``"loss_history"``, ``"predictions"``.
    """
    torch.manual_seed(seed)
    d = data.shape[-1]
    data   = data.to(device)
    labels = labels.to(device)

    # Parameterise H as a symmetric matrix: H = L + Lᵀ (L lower-triangular)
    L = nn.Parameter(torch.randn(d, d, device=device) * 0.1)
    opt = torch.optim.Adam([L], lr=lr)
    loss_history = []

    for epoch in range(epochs):
        opt.zero_grad()
        H = L + L.T  # symmetric
        psi_norm = data / (data.norm(dim=-1, keepdim=True) + 1e-8)
        H_psi = psi_norm @ H.T
        e_pred = (psi_norm * H_psi).sum(dim=-1)
        loss = (e_pred - labels).pow(2).mean()
        loss.backward()
        opt.step()
        loss_history.append(float(loss.detach()))

    with torch.no_grad():
        H_final = (L + L.T).detach()
        psi_norm = data / (data.norm(dim=-1, keepdim=True) + 1e-8)
        preds = (psi_norm @ H_final.T * psi_norm).sum(dim=-1)

    return {
        "H_learned":     H_final,
        "loss_history":  loss_history,
        "predictions":   preds,
    }


# ── Quantum kernel PINN ───────────────────────────────────────────────────────

def quantum_kernel(
    x_train: torch.Tensor,
    x_test:  Optional[torch.Tensor] = None,
    n_qubits: int = 4,
    encoding: str = "angle",
    backend: str = "pennylane",
) -> torch.Tensor:
    """
    Compute the quantum kernel matrix K(xᵢ, xⱼ) = |⟨φ(xᵢ)|φ(xⱼ)⟩|².

    The feature map φ(x) is defined by the quantum encoding circuit. This
    kernel can be used with classical kernel methods (SVM, GP regression)
    in an elevated feature space.

    For the classical fallback backend, uses the RBF kernel as an approximation:
    K(xᵢ, xⱼ) = exp(−‖xᵢ − xⱼ‖² / 2n_qubits).

    Parameters
    ----------
    x_train : Tensor (N, dim)
    x_test  : Tensor (M, dim), optional. If None, K is (N, N).
    n_qubits, encoding, backend : as usual.

    Returns
    -------
    K : Tensor (N, M) or (N, N)
    """
    qcfg = QuantumCircuitConfig(
        n_qubits=n_qubits, depth=1, encoding=encoding, n_observables=n_qubits
    )
    qmodel = QuantumModel(qcfg, BackendConfig(provider=backend))
    qmodel.eval()

    x_test_actual = x_test if x_test is not None else x_train

    with torch.no_grad():
        phi_train = qmodel(x_train)   # (N, n_qubits)
        phi_test  = qmodel(x_test_actual)   # (M, n_qubits)

    # Kernel: cosine similarity in feature space → bounded to [0, 1]
    phi_train_n = phi_train / (phi_train.norm(dim=-1, keepdim=True) + 1e-8)
    phi_test_n  = phi_test  / (phi_test.norm(dim=-1,  keepdim=True) + 1e-8)
    K = (phi_train_n @ phi_test_n.T).clamp(0.0, 1.0) ** 2
    return K


# ── Convenience high-level solve function ────────────────────────────────────

def solve(
    problem: str = "quantum",
    equation: str = "schrodinger",
    potential: str = "harmonic",
    method: str = "vq_pinn",
    backend: str = "simulator",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    High-level entry-point that mirrors the PINNeAPPle ``solve()`` API.

    Parameters
    ----------
    problem : str
        Must be ``"quantum"`` (other problems route to the main ``pinneaple`` solve).
    equation : str
        Equation name: ``"schrodinger"``.
    potential : str
        Potential name: ``"harmonic"``, ``"coulomb"``, ``"double_well"``.
    method : str
        Solution method: ``"vq_pinn"``, ``"hybrid_pinn"``.
    backend : str
        ``"simulator"`` → PennyLane default.qubit,
        ``"quantum_device"`` → real device (requires configured provider).
    **kwargs
        Forwarded to the selected pipeline function.

    Returns
    -------
    dict — result of the selected pipeline.

    Examples
    --------
    >>> from pinneaple_quantum import solve
    >>> result = solve(
    ...     problem="quantum",
    ...     equation="schrodinger",
    ...     potential="harmonic",
    ...     method="vq_pinn",
    ...     backend="simulator",
    ... )
    """
    if problem != "quantum":
        raise ValueError(
            f"pinneaple_quantum.solve only handles problem='quantum'. "
            f"Got {problem!r}. Use pinneaple.solve for classical problems."
        )

    # Map user-friendly backend name
    backend_provider = "pennylane" if backend in ("simulator", "pennylane") else backend

    # Build equation key for preset
    has_2d = any(k in equation for k in ("2d",)) or kwargs.get("dim", 1) == 2
    eq_key = f"{equation.replace('schrodinger', 'schrodinger')}"
    if not eq_key.endswith("_1d") and not eq_key.endswith("_2d"):
        eq_key += "_2d" if has_2d else "_1d"

    pot_key = potential.replace(" ", "_")

    kwargs.setdefault("backend", backend_provider)

    if method == "vq_pinn":
        return vq_pinn(equation=eq_key, potential=pot_key, **kwargs)
    elif method == "hybrid_pinn":
        return hybrid_pinn(potential=pot_key, **kwargs)
    else:
        raise ValueError(
            f"Unknown method {method!r}. Available: 'vq_pinn', 'hybrid_pinn'."
        )
