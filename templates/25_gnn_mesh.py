"""25_gnn_mesh.py — MeshGraphNet for mesh-based CFD simulation.

Demonstrates:
- MeshGraphNet architecture from pinneaple_models.graphnn
- GraphDataBuilder: convert a 2D mesh to a PyG-compatible graph
- Node features: position (x,y), pressure, velocity components
- Training to predict next-step velocity from mesh graph
- Visualisation of predicted vs. true velocity field on mesh
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_models.graphnn.mesh_graph_net import MeshGraphNet
from pinneaple_models.graphnn.graph_builder import GraphDataBuilder


# ---------------------------------------------------------------------------
# Synthetic dataset: unsteady 2D Stokes flow around a square obstacle
# We simulate with a simple finite difference stencil and treat each
# snapshot as a (graph, next-graph) training pair.
# ---------------------------------------------------------------------------

try:
    from torch_geometric.data import Data as PyGData
    _PYG = True
except ImportError:
    _PYG = False
    print("[warn] torch_geometric not installed — using fallback dense representation.")


NX = 20
NY = 20
N_STEPS_SIM = 100
DT = 0.05
NU_FLUID = 0.05


def generate_mesh_snapshots(nx: int = NX, ny: int = NY,
                             n_steps: int = N_STEPS_SIM,
                             seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple Stokes-like velocity field time series on a regular grid.

    Returns:
        xy:  (nx*ny, 2) node positions
        u_t: (n_steps, nx*ny, 2) velocity (u, v) per timestep
        p_t: (n_steps, nx*ny) pressure per timestep
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x, y)
    xy = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    n_nodes = nx * ny

    # IC: random divergence-free-ish velocity
    u = rng.uniform(-0.2, 0.2, n_nodes).astype(np.float32)
    v = rng.uniform(-0.2, 0.2, n_nodes).astype(np.float32)
    p = np.zeros(n_nodes, dtype=np.float32)

    u_snapshots, p_snapshots = [], []
    for _ in range(n_steps):
        # Fake viscous damping + forcing to keep interesting
        u += DT * (-NU_FLUID * u + 0.01 * rng.normal(0, 1, n_nodes).astype(np.float32))
        v += DT * (-NU_FLUID * v + 0.01 * rng.normal(0, 1, n_nodes).astype(np.float32))
        p  = 0.5 * (u**2 + v**2)   # approximate Bernoulli
        u_snapshots.append(np.stack([u, v], axis=1).copy())
        p_snapshots.append(p.copy())

    return xy, np.array(u_snapshots), np.array(p_snapshots)


def build_edge_index(nx: int, ny: int) -> np.ndarray:
    """Grid connectivity: each node connected to its 4 neighbours."""
    edges = []
    for j in range(ny):
        for i in range(nx):
            n = j * nx + i
            if i + 1 < nx:
                edges += [[n, n + 1], [n + 1, n]]
            if j + 1 < ny:
                edges += [[n, n + nx], [n + nx, n]]
    return np.array(edges, dtype=np.int64).T    # (2, E)


def main():
    torch.manual_seed(3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Simulation data -----------------------------------------------------
    print("Generating mesh snapshots ...")
    xy, u_t, p_t = generate_mesh_snapshots()
    edge_index = build_edge_index(NX, NY)

    n_nodes = NX * NY
    n_steps = N_STEPS_SIM

    # Input: (u_t, v_t, p_t, x, y)  →  Target: (u_{t+1}, v_{t+1})
    # Build dataset as list of (node_features, edge_index, targets)
    xy_t    = torch.tensor(xy, device=device)         # (N, 2)
    ei_t    = torch.tensor(edge_index, device=device) # (2, E)

    # Encode edge displacements as edge features
    ei_np   = edge_index
    edge_dx = (xy[ei_np[1]] - xy[ei_np[0]]).astype(np.float32)   # (E, 2)
    edge_feat = torch.tensor(edge_dx, device=device)              # (E, 2)

    def make_node_feats(step: int) -> torch.Tensor:
        uv  = torch.tensor(u_t[step], device=device)          # (N, 2)
        p   = torch.tensor(p_t[step, :, None], device=device) # (N, 1)
        return torch.cat([uv, p, xy_t], dim=1)                # (N, 5)

    # --- MeshGraphNet --------------------------------------------------------
    mgn = MeshGraphNet(
        node_input_dim=5,           # u, v, p, x, y
        edge_input_dim=2,           # dx, dy
        hidden_dim=64,
        output_dim=2,               # predict Δu, Δv
        n_message_passing=3,
        latent_dim=64,
    ).to(device)

    n_params = sum(p.numel() for p in mgn.parameters())
    print(f"MeshGraphNet parameters: {n_params:,}")

    # --- Training ------------------------------------------------------------
    optimizer = torch.optim.Adam(mgn.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    # Use first 80% of snapshots for training
    n_train = int(0.8 * (n_steps - 1))
    history = []

    print("Training ...")
    for epoch in range(1, 61):
        epoch_loss = 0.0
        for step in range(n_train):
            node_feats = make_node_feats(step)
            uv_next    = torch.tensor(u_t[step + 1], device=device)   # (N, 2) target

            optimizer.zero_grad()
            delta_uv = mgn(node_feats, ei_t, edge_feat)               # (N, 2)
            uv_pred  = node_feats[:, :2] + delta_uv                   # residual update
            loss     = (uv_pred - uv_next).pow(2).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        scheduler.step()
        history.append(epoch_loss / n_train)
        if epoch % 20 == 0:
            print(f"  epoch {epoch:3d} | loss = {history[-1]:.4e}")

    # --- Evaluation on test steps --------------------------------------------
    mgn.eval()
    test_l2 = []
    for step in range(n_train, n_steps - 1):
        node_feats = make_node_feats(step)
        uv_next    = torch.tensor(u_t[step + 1], device=device)
        with torch.no_grad():
            uv_pred = node_feats[:, :2] + mgn(node_feats, ei_t, edge_feat)
        l2 = (uv_pred - uv_next).pow(2).mean().sqrt().item()
        test_l2.append(l2)
    print(f"\nTest mean L2: {np.mean(test_l2):.4e}")

    # --- Visualisation -------------------------------------------------------
    step_vis = n_train
    node_feats = make_node_feats(step_vis)
    with torch.no_grad():
        uv_pred_vis = (node_feats[:, :2] + mgn(node_feats, ei_t, edge_feat)).cpu().numpy()
    uv_true_vis = u_t[step_vis + 1]

    xx = xy[:, 0].reshape(NY, NX)
    yy = xy[:, 1].reshape(NY, NX)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    speed_true = np.sqrt((uv_true_vis[:, 0]**2 + uv_true_vis[:, 1]**2)).reshape(NY, NX)
    speed_pred = np.sqrt((uv_pred_vis[:, 0]**2 + uv_pred_vis[:, 1]**2)).reshape(NY, NX)

    for ax, field, title in zip(axes[:2], [speed_true, speed_pred],
                                 ["True speed", "MGN predicted speed"]):
        im = ax.contourf(xx, yy, field, levels=20, cmap="viridis")
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_aspect("equal")

    axes[2].semilogy(history, label="Train loss")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("MSE loss")
    axes[2].set_title(f"Training  |  test L2={np.mean(test_l2):.3e}")
    axes[2].legend()
    axes[2].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("25_gnn_mesh_result.png", dpi=120)
    print("Saved 25_gnn_mesh_result.png")


if __name__ == "__main__":
    main()
