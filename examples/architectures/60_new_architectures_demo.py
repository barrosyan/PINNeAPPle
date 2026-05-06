"""Demo: Group B neural architectures — forward pass shapes.

Runs a single forward pass through each new architecture with random dummy
data and prints the output shape.  No training is performed.

Architectures demonstrated
--------------------------
1. SIREN              — Sinusoidal Representation Network
2. ModifiedMLP        — Fourier-feature embedding + highway U/V gating
3. HashGridMLP        — Instant-NGP style multi-resolution hash encoding
4. MeshGraphNet       — GNN for unstructured FEM-style meshes
5. AFNO               — Adaptive Fourier Neural Operator

Run
---
    python examples/pinneaple_models/60_new_architectures_demo.py
"""

import torch

# ---------------------------------------------------------------------------
# 1. SIREN
# ---------------------------------------------------------------------------
from pinneaple_models.siren import SIREN

print("=" * 60)
print("1. SIREN (Sinusoidal Representation Network)")
print("=" * 60)

model_siren = SIREN(
    in_dim=2,       # (t, x)
    out_dim=1,      # u(t, x)
    hidden_dim=256,
    n_layers=5,
    omega_0=30.0,
    outermost_linear=True,
)
print(model_siren)

x_siren = torch.rand(512, 2)          # 512 collocation points, 2-D input
out_siren = model_siren(x_siren)
print(f"  Input  shape : {tuple(x_siren.shape)}")
print(f"  Output shape : {tuple(out_siren.y.shape)}")
print()

# Verify gradient flow (important for PINN usage)
x_grad = x_siren.requires_grad_(True)
out_grad = model_siren(x_grad)
loss = out_grad.y.sum()
loss.backward()
print(f"  Gradient norm of input: {x_grad.grad.norm().item():.4f}  (should be non-zero)")
print()


# ---------------------------------------------------------------------------
# 2. Modified MLP
# ---------------------------------------------------------------------------
from pinneaple_models.modified_mlp import ModifiedMLP, FourierFeatureEmbedding

print("=" * 60)
print("2. ModifiedMLP (Fourier features + highway U/V gating)")
print("=" * 60)

model_mlp = ModifiedMLP(
    in_dim=3,        # (t, x, y)
    out_dim=2,       # (u, v) velocity components
    hidden_dim=128,
    n_layers=6,
    n_fourier=32,
    sigma=1.0,
)
print(model_mlp)

x_mlp = torch.rand(1024, 3)
out_mlp = model_mlp(x_mlp)
print(f"  Input  shape : {tuple(x_mlp.shape)}")
print(f"  Output shape : {tuple(out_mlp.y.shape)}")
print()

# Also demonstrate the embedding standalone
embed = FourierFeatureEmbedding(in_dim=3, n_fourier=32)
emb_out = embed(x_mlp)
print(f"  Fourier embedding output shape: {tuple(emb_out.shape)}")
print()


# ---------------------------------------------------------------------------
# 3. HashGridMLP
# ---------------------------------------------------------------------------
from pinneaple_models.hash_grid import HashGridMLP, HashGridEncoding

print("=" * 60)
print("3. HashGridMLP (Instant-NGP style multi-resolution hash encoding)")
print("=" * 60)

model_hash = HashGridMLP(
    in_dim=3,                    # 3-D spatial coordinates
    out_dim=1,                   # scalar field (e.g., pressure)
    hidden_dim=64,
    n_hidden=2,
    n_levels=8,
    n_features_per_level=2,
    log2_hashmap_size=16,
    base_resolution=16,
    finest_resolution=128,
)
print(model_hash)

# Coordinates must be in [0, 1]
x_hash = torch.rand(256, 3)
out_hash = model_hash(x_hash)
print(f"  Input  shape : {tuple(x_hash.shape)}")
print(f"  Output shape : {tuple(out_hash.y.shape)}")
print()

# Also demonstrate the encoder alone
encoder = HashGridEncoding(in_dim=3, n_levels=8, n_features_per_level=2)
enc_out = encoder(x_hash)
print(f"  HashGrid encoding output shape: {tuple(enc_out.shape)}")
print(f"  (n_levels={encoder.n_levels} x n_features_per_level={encoder.n_features_per_level} = {encoder.out_dim})")
print()


# ---------------------------------------------------------------------------
# 4. MeshGraphNet
# ---------------------------------------------------------------------------
from pinneaple_models.graphnn.mesh_graph_net import MeshGraphNet
from pinneaple_models.graphnn.base import GraphBatch

print("=" * 60)
print("4. MeshGraphNet (GNN for unstructured FEM meshes)")
print("=" * 60)

B_graphs = 2     # batch size (graphs processed in parallel)
N_nodes  = 200   # nodes per graph
N_edges  = 800   # edges per graph (shared topology across batch)

# ── 4a. Standard usage: raw edge features, no positions ────────────────────
model_mgn = MeshGraphNet(
    node_in_dim=5,        # e.g. (x, y, u, v, p) node attributes
    out_dim=2,            # predict (u_next, v_next)
    edge_in_dim=3,        # e.g. (dx, dy, |d|) edge displacement + distance
    hidden_dim=128,
    n_message_passing=6,
)
print(model_mgn)

node_feats = torch.rand(B_graphs, N_nodes, 5)    # (B, N, node_in_dim)
src_idx    = torch.randint(0, N_nodes, (N_edges,))
dst_idx    = torch.randint(0, N_nodes, (N_edges,))
edge_index = torch.stack([src_idx, dst_idx], dim=0)  # (2, E) — shared topology
edge_feats = torch.rand(B_graphs, N_edges, 3)    # (B, E, edge_in_dim)

g = GraphBatch(x=node_feats, edge_index=edge_index, edge_attr=edge_feats)
out_mgn = model_mgn(g)
print(f"  [GraphBatch] node shape  : {tuple(node_feats.shape)}")
print(f"  [GraphBatch] edge_index  : {tuple(edge_index.shape)}")
print(f"  [GraphBatch] edge shape  : {tuple(edge_feats.shape)}")
print(f"  [GraphBatch] output y    : {tuple(out_mgn.y.shape)}")  # (B, N, out_dim)
print()

# ── 4b. Dict-based forward_batch interface (used by Arena / GNNAdapter) ────
batch_mgn = {
    "x":          node_feats,
    "edge_index": edge_index,
    "edge_attr":  edge_feats,
}
out_batch = model_mgn.forward_batch(batch_mgn)
print(f"  [forward_batch] output y : {tuple(out_batch.y.shape)}")
print()

# ── 4c. With node positions and return_loss ─────────────────────────────────
pos_dim = 2  # 2-D mesh
model_mgn_pos = MeshGraphNet(
    node_in_dim=3,        # e.g. (rho, u, v) — velocity + density
    out_dim=3,
    edge_in_dim=0,        # no explicit edge features; build from positions
    pos_dim=pos_dim,      # must be > 0 when use_pos=True
    hidden_dim=64,
    n_message_passing=4,
    use_pos=True,         # auto-compute relative displacement + distance edges
)
node_feats_pos = torch.rand(B_graphs, N_nodes, 3)       # (B, N, 3)
positions      = torch.rand(B_graphs, N_nodes, pos_dim)  # (B, N, 2)
y_true         = torch.rand(B_graphs, N_nodes, 3)        # (B, N, 3)

g_pos = GraphBatch(x=node_feats_pos, edge_index=edge_index, pos=positions)
out_pos = model_mgn_pos(g_pos, y_true=y_true, return_loss=True)
print(f"  [use_pos] output y       : {tuple(out_pos.y.shape)}")
print(f"  [use_pos] MSE loss       : {out_pos.losses['mse'].item():.4f}")
print()


# ---------------------------------------------------------------------------
# 5. AFNO
# ---------------------------------------------------------------------------
from pinneaple_models.afno import AFNO, AFNOLayer

print("=" * 60)
print("5. AFNO (Adaptive Fourier Neural Operator)")
print("=" * 60)

B, H, W, C_in = 4, 64, 64, 20    # batch, height, width, channels
C_out = 20                         # predict same number of channels (e.g. next step)

model_afno = AFNO(
    in_channels=C_in,
    out_channels=C_out,
    hidden_dim=64,
    n_layers=4,
    n_modes_h=12,
    n_modes_w=12,
    mlp_ratio=4.0,
    dropout=0.0,
)
print(model_afno)

x_afno = torch.rand(B, H, W, C_in)   # (B, H, W, C)
out_afno = model_afno(x_afno)
print(f"  Input  shape : {tuple(x_afno.shape)}")
print(f"  Output shape : {tuple(out_afno.y.shape)}")
print()

# Single AFNO layer standalone
layer = AFNOLayer(hidden_dim=64, n_modes_h=12, n_modes_w=12)
x_layer = torch.rand(2, 32, 32, 64)
y_layer = layer(x_layer)
print(f"  AFNOLayer standalone: {tuple(x_layer.shape)} -> {tuple(y_layer.shape)}")
print()


# ---------------------------------------------------------------------------
# 6. Registry check
# ---------------------------------------------------------------------------
from pinneaple_models.register_all import register_all
from pinneaple_models.registry import ModelRegistry

print("=" * 60)
print("6. ModelRegistry — Group B models")
print("=" * 60)

register_all()

group_b_models = ModelRegistry.list(family="group_b")
print(f"  Registered Group-B models ({len(group_b_models)}):")
for name in group_b_models:
    spec = ModelRegistry.spec(name)
    print(f"    {name:<45} input_kind={spec.input_kind}")
print()

print("All Group B architecture demos passed successfully.")
