"""Build any Arena-supported model from a ModelConfig.

Uses ModelRegistry.build() (+ instantiate normalization) from pinneaple_neural
to access ALL registered models (~80+). Falls back to hardcoded builders for the
most common types when the registry is unavailable.

All 80+ model keys registered in ModelRegistry are supported:
  vanilla_pinn, inverse_pinn, siren, modified_mlp, pielm, pinnsformer, vpinn,
  xpinn, xtfc, fno, fno2d, deeponet, pino, gno, uno, meshgraphnet, gnn,
  equivariant_gnn, graphcast, neural_ode, latent_ode, neural_cde, hnn,
  transformer, informer, tft, lstm, gru, conv1d, conv2d, esn, koopman,
  pod, dmd, sindy, vae, dense_ae, ... and all Noether variants
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any, Dict, Optional

from .config import ModelConfig, NetworkConfig


# ── lightweight fallback MLP ───────────────────────────────────────────────────

class _MLP(nn.Module):
    """Simple fully-connected network used as universal fallback."""
    def __init__(self, in_dim: int, out_dim: int, hidden: list, activation: str = "tanh"):
        super().__init__()
        act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU,
                  "sigmoid": nn.Sigmoid, "silu": nn.SiLU}.get(activation.lower(), nn.Tanh)
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), act_fn()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── kwargs builder ────────────────────────────────────────────────────────────

def _build_kwargs(net_cfg: NetworkConfig, in_dim: int, out_dim: int,
                  edge_in_dim: int = 0) -> Dict[str, Any]:
    """Build a flat kwargs dict covering ALL model families.

    instantiate() from pinneaple_neural will filter and normalize these,
    so it's safe to pass everything — unsupported kwargs are silently dropped.
    """
    modes2 = net_cfg.modes2 if net_cfg.modes2 > 0 else net_cfg.modes
    edge_in = edge_in_dim if edge_in_dim > 0 else (net_cfg.edge_in_dim if net_cfg.edge_in_dim > 0 else in_dim)
    branch = net_cfg.branch_dim if net_cfg.branch_dim > 0 else in_dim
    trunk  = net_cfg.trunk_dim  if net_cfg.trunk_dim  > 0 else in_dim
    state  = net_cfg.state_dim  if net_cfg.state_dim  > 0 else in_dim

    kw: Dict[str, Any] = {
        # ── dimension (all aliases, instantiate normalizes) ──────────────
        "in_dim":       in_dim,
        "input_dim":    in_dim,
        "in_channels":  in_dim,
        "dim_in":       in_dim,
        "out_dim":      out_dim,
        "output_dim":   out_dim,
        "out_channels": out_dim,
        "dim_out":      out_dim,

        # ── PINN / MLP ───────────────────────────────────────────────────
        "hidden":           net_cfg.hidden,
        "hidden_layers":    net_cfg.hidden,
        "activation":       net_cfg.activation,

        # ── FNO ─────────────────────────────────────────────────────────
        "width":    net_cfg.width,
        "modes":    net_cfg.modes,
        "modes1":   net_cfg.modes,
        "modes2":   modes2,
        "n_layers": net_cfg.layers,
        "layers":   net_cfg.layers,
        "use_grid": net_cfg.use_grid,

        # ── MeshGraphNet / GNN ───────────────────────────────────────────
        "node_in_dim":       in_dim,
        "edge_in_dim":       edge_in,
        "hidden_dim":        net_cfg.hidden_dim,
        "n_message_passing": net_cfg.n_message_passing,
        "dropout":           net_cfg.dropout,

        # ── SIREN ────────────────────────────────────────────────────────
        "omega_0": net_cfg.omega_0,

        # ── Transformer ──────────────────────────────────────────────────
        "d_model":         net_cfg.d_model,
        "nhead":           net_cfg.nhead,
        "num_layers":      net_cfg.num_layers,
        "dim_feedforward": net_cfg.dim_feedforward,

        # ── DeepONet / Neural Operators ───────────────────────────────────
        "branch_dim":   branch,
        "branch_input_dim": branch,
        "trunk_dim":    trunk,
        "trunk_input_dim":  trunk,

        # ── NeuralODE / continuous ────────────────────────────────────────
        "state_dim": state,

        # ── Autoencoder ───────────────────────────────────────────────────
        "latent_dim": net_cfg.hidden_dim,
        "input_dim_ae": in_dim,

        # ── extra (pass-through) ─────────────────────────────────────────
        **net_cfg.extra,
    }
    return kw


# ── graph model type check (needed by arena.py) ───────────────────────────────

_GRAPH_TYPES = {
    "meshgraphnet", "gnn", "mesh_graph_net", "mgn",
    "equivariant_gnn", "egnn", "graph_neural_network",
    "spatiotemporal_gnn", "stgnn", "graphcast",
    "graph_neural_ode", "gnn_ode",
}

_PINN_TYPES = {
    "vanilla_pinn", "pinn", "vanilla",
    "siren", "modified_mlp",
    "pielm", "pi_elm",
    "pinn_lstm",
    "pinnsformer", "pinn_former",
    "vpinn", "variational_pinn",
    "xpinn",
    "xtfc", "extreme_tfc",
    "physics_aware_neural_network", "pann",
    "structure_preserving_network", "spn",
}

_INVERSE_PINN_TYPES = {
    "inverse_pinn", "inv_pinn",
}

_OPERATOR_TYPES = {
    "fno2d", "fno", "fourier", "fourier_neural_operator",
    "deeponet", "multiscale_deeponet",
    "pino", "physics_informed_neural_operator",
    "gno", "galerkin_neural_operator",
    "uno", "universal_operator_network",
}


def is_graph_model(cfg: ModelConfig) -> bool:
    return cfg.type.lower() in _GRAPH_TYPES


def is_pinn_model(cfg: ModelConfig) -> bool:
    return cfg.type.lower() in _PINN_TYPES or cfg.type.lower() in _INVERSE_PINN_TYPES


def is_inverse_model(cfg: ModelConfig) -> bool:
    return cfg.type.lower() in _INVERSE_PINN_TYPES


def is_operator_model(cfg: ModelConfig) -> bool:
    mtype = cfg.type.lower()
    return mtype in _OPERATOR_TYPES or any(
        mtype.startswith(p) for p in ("noether_", "upt", "ab_upt", "transolver", "aero_")
    )


# ── main build function ────────────────────────────────────────────────────────

def build_model(cfg: ModelConfig, in_dim: int = 2, out_dim: int = 1,
                edge_in_dim: int = 0) -> nn.Module:
    """Instantiate any pinneaple model from a ModelConfig.

    Uses ModelRegistry + instantiate() for automatic kwarg normalization
    and filtering. Supports all 80+ registered models.

    Parameters
    ----------
    cfg          : ModelConfig — type key + NetworkConfig
    in_dim       : physical/node feature input dimension
    out_dim      : number of output fields
    edge_in_dim  : edge attribute dimension (graph models only)
    """
    mtype = cfg.type.lower().strip()
    kwargs = _build_kwargs(cfg.network, in_dim, out_dim, edge_in_dim)

    # ── canonical alias map (user-facing → registry key) ─────────────────
    _ALIASES = {
        "fno2d": "fno", "fourier": "fno", "fourier_neural_operator": "fno",
        "pinn": "vanilla_pinn", "vanilla": "vanilla_pinn",
        "inv_pinn": "inverse_pinn",
        "gnn": "gnn",
        "mgn": "mesh_graph_net",
        "meshgraphnet": "mesh_graph_net",
    }
    registry_key = _ALIASES.get(mtype, mtype)

    # ── 1. Try ModelRegistry (covers all ~80+ models) ────────────────────
    try:
        from pinneaple_neural.architectures import ModelRegistry
        return ModelRegistry.build(registry_key, **kwargs)
    except KeyError:
        pass  # not in registry — try hardcoded builders below
    except Exception as e:
        _warn(f"ModelRegistry.build('{registry_key}') failed ({e}), trying fallback builders.")

    # ── 2. Hardcoded fallbacks for the most common types ─────────────────
    return _fallback_build(cfg, in_dim, out_dim, edge_in_dim)


def _fallback_build(cfg: ModelConfig, in_dim: int, out_dim: int,
                    edge_in_dim: int) -> nn.Module:
    mtype = cfg.type.lower().strip()
    net = cfg.network

    if mtype in ("vanilla_pinn", "pinn", "vanilla", "inverse_pinn", "inv_pinn"):
        return _build_vanilla_pinn(net, in_dim, out_dim)
    if mtype in ("siren",):
        return _build_siren(net, in_dim, out_dim)
    if mtype in ("modified_mlp",):
        return _build_modified_mlp(net, in_dim, out_dim)
    if mtype in ("fno2d", "fno", "fourier", "fourier_neural_operator"):
        return _build_fno2d(net, in_dim, out_dim)
    if mtype in ("deeponet",):
        return _build_deeponet(net, in_dim, out_dim)
    if mtype in _GRAPH_TYPES:
        return _build_meshgraphnet(net, in_dim, out_dim, edge_in_dim)

    # noether models
    if any(mtype.startswith(p) for p in ("noether_", "upt", "ab_upt", "transolver", "aero_")):
        return _build_noether(mtype, net, in_dim, out_dim)

    # ultimate fallback: generic MLP
    import warnings
    warnings.warn(
        f"Unknown model type '{mtype}'. Using generic MLP as fallback. "
        "Check that pinneaple_neural is installed and the model key is correct."
    )
    return _MLP(in_dim, out_dim, net.hidden, net.activation)


# ── individual fallback builders ───────────────────────────────────────────────

def _build_vanilla_pinn(net_cfg: NetworkConfig, in_dim: int, out_dim: int) -> nn.Module:
    try:
        from pinneaple_neural.architectures.pinns.vanilla import VanillaPINN
        return VanillaPINN(in_dim=in_dim, out_dim=out_dim,
                           hidden=net_cfg.hidden, activation=net_cfg.activation,
                           **net_cfg.extra)
    except Exception:
        return _MLP(in_dim, out_dim, net_cfg.hidden, net_cfg.activation)


def _build_siren(net_cfg: NetworkConfig, in_dim: int, out_dim: int) -> nn.Module:
    try:
        from pinneaple_neural.architectures.siren import SIREN
        return SIREN(in_dim=in_dim, out_dim=out_dim,
                     hidden=net_cfg.hidden, omega_0=net_cfg.omega_0,
                     **net_cfg.extra)
    except Exception:
        return _MLP(in_dim, out_dim, net_cfg.hidden, net_cfg.activation)


def _build_modified_mlp(net_cfg: NetworkConfig, in_dim: int, out_dim: int) -> nn.Module:
    try:
        from pinneaple_neural.architectures.modified_mlp import ModifiedMLP
        return ModifiedMLP(in_dim=in_dim, out_dim=out_dim,
                           hidden=net_cfg.hidden, activation=net_cfg.activation,
                           **net_cfg.extra)
    except Exception:
        return _MLP(in_dim, out_dim, net_cfg.hidden, net_cfg.activation)


def _build_fno2d(net_cfg: NetworkConfig, in_dim: int, out_dim: int) -> nn.Module:
    modes2 = net_cfg.modes2 if net_cfg.modes2 > 0 else net_cfg.modes
    try:
        from pinneaple_neural.architectures.neural_operators.fno import FNO2d
        return FNO2d(in_channels=in_dim, out_channels=out_dim,
                     width=net_cfg.width, modes1=net_cfg.modes, modes2=modes2,
                     n_layers=net_cfg.layers, **net_cfg.extra)
    except Exception:
        return _MLP(in_dim, out_dim, net_cfg.hidden, net_cfg.activation)


def _build_deeponet(net_cfg: NetworkConfig, in_dim: int, out_dim: int) -> nn.Module:
    branch = net_cfg.branch_dim if net_cfg.branch_dim > 0 else in_dim
    trunk  = net_cfg.trunk_dim  if net_cfg.trunk_dim  > 0 else in_dim
    try:
        from pinneaple_neural.architectures.neural_operators.deeponet import DeepONet
        return DeepONet(branch_dim=branch, trunk_dim=trunk, out_dim=out_dim,
                        hidden=net_cfg.hidden, **net_cfg.extra)
    except Exception:
        return _MLP(in_dim, out_dim, net_cfg.hidden, net_cfg.activation)


def _build_meshgraphnet(net_cfg: NetworkConfig, in_dim: int, out_dim: int,
                         edge_in_dim: int = 0) -> nn.Module:
    edge_in = net_cfg.edge_in_dim if net_cfg.edge_in_dim > 0 else (edge_in_dim if edge_in_dim > 0 else in_dim)
    try:
        from pinneaple_neural.architectures.graphnn.mesh_graph_net import MeshGraphNet
        return MeshGraphNet(node_in_dim=in_dim, edge_in_dim=edge_in, out_dim=out_dim,
                            hidden_dim=net_cfg.hidden_dim, n_layers=net_cfg.n_layers,
                            n_message_passing=net_cfg.n_message_passing,
                            dropout=net_cfg.dropout, **net_cfg.extra)
    except Exception:
        return _MLP(in_dim, out_dim, net_cfg.hidden, net_cfg.activation)


def _build_noether(model_type: str, net_cfg: NetworkConfig, in_dim: int, out_dim: int
                   ) -> nn.Module:
    try:
        from pinneaple_neural.architectures.neural_operators.noether_bridge import NOETHER_REGISTRY
        cls = NOETHER_REGISTRY[model_type]
        return cls(**net_cfg.extra)
    except Exception as e:
        raise ImportError(
            f"Noether model '{model_type}' not available. "
            "Install with: pip install pinneaple[noether]"
        ) from e


def _warn(msg: str):
    import warnings
    warnings.warn(f"[Arena model_factory] {msg}", stacklevel=3)
