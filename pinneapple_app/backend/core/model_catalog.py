"""Model catalog: lists available models, enriches with metadata, and recommends."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ModelEntry:
    name: str
    family: str
    description: str
    supports_physics_loss: bool
    tags: List[str]
    recommended_for: List[str]   # problem families


# ── Extended model descriptions ──────────────────────────────────────────
_MODEL_META: Dict[str, Dict[str, Any]] = {
    "vanilla_pinn": {
        "family": "pinns",
        "description": "Standard PINN: fully-connected MLP + physics residual loss.",
        "recommended_for": ["fluid", "thermal", "structural", "wave", "diffusion", "generic"],
        "tags": ["pinn", "mlp", "general-purpose"],
    },
    "siren": {
        "family": "group_b",
        "description": "SIREN (sinusoidal activations) — excellent for oscillatory PDEs.",
        "recommended_for": ["wave", "fluid", "thermal", "structural"],
        "tags": ["siren", "periodic", "pinn"],
    },
    "modified_mlp": {
        "family": "group_b",
        "description": "Modified MLP with Fourier features — better spectral bias for smooth fields.",
        "recommended_for": ["thermal", "structural", "diffusion"],
        "tags": ["mlp", "fourier", "pinn"],
    },
    "hash_grid_mlp": {
        "family": "group_b",
        "description": "Hash-grid MLP (Instant-NGP style) — fast, high-resolution fields.",
        "recommended_for": ["thermal", "fluid"],
        "tags": ["hash-grid", "fast", "pinn"],
    },
    "fno": {
        "family": "neural_operators",
        "description": "Fourier Neural Operator — learns solution operators over grids.",
        "recommended_for": ["fluid", "thermal", "wave"],
        "tags": ["operator", "fourier", "grid"],
    },
    "deeponet": {
        "family": "neural_operators",
        "description": "DeepONet — branch/trunk architecture for solution operators.",
        "recommended_for": ["fluid", "thermal", "generic"],
        "tags": ["operator", "branch-trunk"],
    },
    "mesh_graph_net": {
        "family": "graphnn",
        "description": "MeshGraphNet — GNN on simulation mesh for complex geometries.",
        "recommended_for": ["fluid", "structural"],
        "tags": ["gnn", "mesh", "geometry"],
    },
    "afno": {
        "family": "group_b",
        "description": "AFNO (Adaptive Fourier Neural Operator) — transformer + spectral mixing.",
        "recommended_for": ["fluid", "wave"],
        "tags": ["transformer", "fourier", "operator"],
    },
    "pinnsformer": {
        "family": "transformers",
        "description": "PINNsFormer — transformer-based PINN for time-dependent problems.",
        "recommended_for": ["fluid", "wave", "thermal"],
        "tags": ["transformer", "time-dependent", "pinn"],
    },
    "pino": {
        "family": "neural_operators",
        "description": "Physics-Informed Neural Operator — FNO + physics constraints.",
        "recommended_for": ["fluid", "thermal", "wave"],
        "tags": ["operator", "physics-informed", "fourier"],
    },
    "xpinn": {
        "family": "pinns",
        "description": "XPINN — extended PINN with domain decomposition.",
        "recommended_for": ["fluid", "structural", "generic"],
        "tags": ["pinn", "domain-decomposition"],
    },
    "vpinn": {
        "family": "pinns",
        "description": "VPINN — variational PINN for better integration of BCs.",
        "recommended_for": ["structural", "diffusion"],
        "tags": ["pinn", "variational"],
    },
    "neural_ode": {
        "family": "continuous",
        "description": "Neural ODE — continuous-depth model, ideal for ODE systems.",
        "recommended_for": ["biological", "fluid"],
        "tags": ["ode", "time-dependent", "continuous"],
    },
    "lstm": {
        "family": "recurrent",
        "description": "LSTM — recurrent network for time-series / sequential PDE data.",
        "recommended_for": ["biological", "finance"],
        "tags": ["recurrent", "time-series"],
    },
    "tft": {
        "family": "transformers",
        "description": "Temporal Fusion Transformer — state-of-art time-series forecasting.",
        "recommended_for": ["biological", "finance"],
        "tags": ["transformer", "time-series"],
    },
    "inverse_pinn": {
        "family": "pinns",
        "description": "Inverse PINN — simultaneously infers field and unknown parameters.",
        "recommended_for": ["fluid", "thermal", "structural"],
        "tags": ["pinn", "inverse", "parameter-estimation"],
    },
    "hamiltonian_nn": {
        "family": "physics_aware",
        "description": "Hamiltonian NN — energy-conserving architecture for Hamiltonian systems.",
        "recommended_for": ["wave", "fluid"],
        "tags": ["physics-aware", "conservation", "symplectic"],
    },
    "pielm": {
        "family": "pinns",
        "description": "PI-ELM — physics-informed extreme learning machine (fast training).",
        "recommended_for": ["thermal", "diffusion"],
        "tags": ["pinn", "elm", "fast"],
    },
}

# ── Default metrics ───────────────────────────────────────────────────────
AVAILABLE_METRICS = {
    "l2_relative":     "Relative L2 error vs. reference",
    "mse":             "Mean Squared Error",
    "mae":             "Mean Absolute Error",
    "max_error":       "Maximum pointwise error",
    "pde_residual":    "Mean PDE residual norm",
    "bc_residual":     "Mean BC residual norm",
    "r2":              "R² coefficient of determination",
    "train_time_s":    "Wall-clock training time (seconds)",
    "n_params":        "Number of trainable parameters",
    "convergence_epoch": "Epoch at which loss < 1e-3",
}

DEFAULT_METRICS = ["l2_relative", "mse", "pde_residual", "bc_residual", "train_time_s", "n_params"]


def list_models(family: Optional[str] = None) -> List[ModelEntry]:
    """Return all available models, optionally filtered by family."""
    try:
        from pinneapple_neural.architectures import ModelRegistry
        all_names = ModelRegistry.list()
    except Exception:
        all_names = list(_MODEL_META.keys())

    result = []
    for name in sorted(all_names):
        try:
            from pinneapple_neural.architectures import ModelRegistry
            spec = ModelRegistry.spec(name)
            fam = spec.family
            supports_physics = getattr(spec, "supports_physics_loss", False)
            tags = list(getattr(spec, "tags", []))
        except Exception:
            fam = "unknown"
            supports_physics = False
            tags = []

        meta = _MODEL_META.get(name, {})
        entry = ModelEntry(
            name=name,
            family=fam,
            description=meta.get("description", ""),
            supports_physics_loss=supports_physics,
            tags=meta.get("tags", tags),
            recommended_for=meta.get("recommended_for", []),
        )
        if family is None or entry.family == family:
            result.append(entry)
    return result


def recommend_for_problem(problem_family: str, n: int = 5) -> List[str]:
    """Return top-N recommended model names for a given problem family."""
    candidates = [
        entry.name for entry in list_models()
        if problem_family in entry.recommended_for
    ]
    if not candidates:
        candidates = ["vanilla_pinn", "siren", "modified_mlp"]
    return candidates[:n]


def get_model_info(name: str) -> Optional[ModelEntry]:
    entries = [e for e in list_models() if e.name == name]
    return entries[0] if entries else None
