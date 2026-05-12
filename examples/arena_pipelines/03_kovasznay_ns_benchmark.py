"""
PINNeAPPle Arena — 2D Kovasznay Navier-Stokes Benchmark
=========================================================
Comparing three physics-AI paradigms on the classical 2D Kovasznay flow
(analytical solution at Re = 40):

  VanillaPINN   — meshfree physics residual, ZERO training data
  FNO-2D        — Fourier Neural Operator, data-driven grid surrogate
  MeshGraphNet  — graph neural network, data-driven mesh surrogate

Each model learns the velocity field (u, v) in its own paradigm.
The benchmark produces a publication-quality figure ready for LinkedIn.

Usage
-----
    python examples/arena_pipelines/03_kovasznay_ns_benchmark.py

Output
------
    outputs/kovasznay_benchmark.png   — multi-panel comparison figure
    outputs/kovasznay_streams.png     — streamline overlay figure
    (results printed to stdout as a metrics table)
"""
from __future__ import annotations

import math
import os
import sys
import time
from typing import Dict, List, Tuple

# Ensure the project root is on the path when running from examples/
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import numpy as np
import torch
import torch.nn as nn

# ── PINNeAPPle imports ────────────────────────────────────────────────────────
from pinneaple_neural.architectures.pinns.vanilla import VanillaPINN
from pinneaple_neural.architectures.neural_operators.fno import FNO2d
from pinneaple_neural.architectures.graphnn.mesh_graph_net import MeshGraphNet
from pinneaple_neural.architectures.graphnn.base import GraphBatch
from pinneaple_tools.visualization.fields import plot_streamlines, plot_scalar

# ── Configuration ─────────────────────────────────────────────────────────────
RE           = 40.0          # Reynolds number
GRID_N       = 40            # evaluation grid resolution (N×N)
N_COL        = 2_000         # PINN interior collocation points
N_BC         = 500           # PINN boundary collocation points
EPOCHS_PINN  = 2_000         # VanillaPINN training epochs
LR_PINN      = 5e-4          # VanillaPINN learning rate
EPOCHS_FNO   = 600           # FNO-2D training epochs
LR_FNO       = 1e-3          # FNO-2D learning rate
EPOCHS_MGN   = 600           # MeshGraphNet training epochs
LR_MGN       = 1e-3          # MeshGraphNet learning rate
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR      = "outputs"

os.makedirs(OUT_DIR, exist_ok=True)

print(f"[PINNeAPPle Arena] Kovasznay NS Benchmark  |  Re={RE}  |  device={DEVICE}")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# 1. ANALYTICAL SOLUTION — Kovasznay 2D NS
# ══════════════════════════════════════════════════════════════════════════════
# Reference: Kovasznay (1948).  Analytical solution for a steady 2D flow behind
# a periodic array of cylinders. Domain: x ∈ [-0.5, 1.5], y ∈ [0, 2].
#
#   λ  = Re/2 - sqrt(Re²/4 + 4π²)
#   u  = 1 - exp(λ·x)·cos(2π·y)
#   v  = λ/(2π)·exp(λ·x)·sin(2π·y)
#   p  = (1 - exp(2λ·x)) / 2

LAMBDA_K: float = RE / 2.0 - math.sqrt((RE / 2.0) ** 2 + 4.0 * math.pi ** 2)
X_MIN, X_MAX = -0.5, 1.5
Y_MIN, Y_MAX =  0.0, 2.0


def kovasznay_uv(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    u = 1.0 - np.exp(LAMBDA_K * x) * np.cos(2.0 * math.pi * y)
    v = (LAMBDA_K / (2.0 * math.pi)) * np.exp(LAMBDA_K * x) * np.sin(2.0 * math.pi * y)
    return u, v


def kovasznay_p(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 - np.exp(2.0 * LAMBDA_K * x))


# ── Reference grid ────────────────────────────────────────────────────────────
xs = np.linspace(X_MIN, X_MAX, GRID_N)
ys = np.linspace(Y_MIN, Y_MAX, GRID_N)
XX, YY = np.meshgrid(xs, ys, indexing="ij")           # (N, N)
U_ref, V_ref = kovasznay_uv(XX.ravel(), YY.ravel())   # (N²,)
P_ref = kovasznay_p(XX.ravel(), YY.ravel())


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _t(a, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(a, dtype=dtype)


# ── PINN: interior & boundary collocation ─────────────────────────────────────
rng = np.random.default_rng(42)

# Interior uniform random
x_int = rng.uniform(X_MIN, X_MAX, N_COL).astype(np.float32)
y_int = rng.uniform(Y_MIN, Y_MAX, N_COL).astype(np.float32)
XY_int = torch.from_numpy(np.column_stack([x_int, y_int]))  # (N_COL, 2)

# Boundary: four sides
_nb = N_BC // 4
def _bc_side(fixed_axis, fixed_val, free_axis_range, n):
    t = np.linspace(free_axis_range[0], free_axis_range[1], n, dtype=np.float32)
    if fixed_axis == 0:
        return np.column_stack([np.full(n, fixed_val, dtype=np.float32), t])
    return np.column_stack([t, np.full(n, fixed_val, dtype=np.float32)])

bc_pts = np.vstack([
    _bc_side(0, X_MIN, (Y_MIN, Y_MAX), _nb),
    _bc_side(0, X_MAX, (Y_MIN, Y_MAX), _nb),
    _bc_side(1, Y_MIN, (X_MIN, X_MAX), _nb),
    _bc_side(1, Y_MAX, (X_MIN, X_MAX), _nb),
])
XY_bc = torch.from_numpy(bc_pts)
u_bc_np, v_bc_np = kovasznay_uv(bc_pts[:, 0], bc_pts[:, 1])
UV_bc = torch.from_numpy(np.column_stack([u_bc_np, v_bc_np]).astype(np.float32))

# ── FNO-2D: supervised grid dataset ───────────────────────────────────────────
# Input : (1, GRID_N, GRID_N) — coarse/noisy u observation
# Target: (2, GRID_N, GRID_N) — clean (u, v) field
# We synthesise 200 training instances by randomly masking 60% of the grid
# and asking FNO to recover the full field (compressed-sensing-style task).

N_FNO_TRAIN = 200
MASK_RATIO   = 0.50      # fraction of grid points randomly zeroed in input

U_grid = _t(U_ref.reshape(GRID_N, GRID_N))   # (H, W)
V_grid = _t(V_ref.reshape(GRID_N, GRID_N))
UV_full = torch.stack([U_grid, V_grid], dim=0)  # (2, H, W)

def make_fno_batch(n: int, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create (inputs, targets) for FNO training.
    Input: (n, 1, H, W) masked u field.  Target: (n, 2, H, W) full (u, v).
    """
    mask = (torch.rand(n, 1, GRID_N, GRID_N) > mask_ratio).float()
    x_in = U_grid.unsqueeze(0).unsqueeze(0).expand(n, -1, -1, -1) * mask
    y_out = UV_full.unsqueeze(0).expand(n, -1, -1, -1)
    return x_in, y_out

X_fno_train, Y_fno_train = make_fno_batch(N_FNO_TRAIN, MASK_RATIO)
X_fno_val,   Y_fno_val   = make_fno_batch(40, MASK_RATIO)

# ── MeshGraphNet: mesh dataset ────────────────────────────────────────────────
# Nodes: random points in domain.  Features: (x, y).  Target: (u, v).
# Graph topology: Delaunay triangulation → edges from triangles.

N_NODES = 600
N_MGN_SNAPSHOTS = 1   # single flow field, augmented with noise

from scipy.spatial import Delaunay

node_xy = rng.uniform([X_MIN, Y_MIN], [X_MAX, Y_MAX], (N_NODES, 2)).astype(np.float32)
u_nodes, v_nodes = kovasznay_uv(node_xy[:, 0], node_xy[:, 1])

tri = Delaunay(node_xy)
edges_set = set()
for simplex in tri.simplices:
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = int(simplex[i]), int(simplex[j])
            edges_set.add((a, b))
            edges_set.add((b, a))
edge_index = torch.tensor(list(edges_set), dtype=torch.long).T  # (2, E)

NODE_XY_T   = torch.from_numpy(node_xy).unsqueeze(0)   # (1, N, 2)
UV_nodes_T  = torch.from_numpy(
    np.column_stack([u_nodes, v_nodes]).astype(np.float32)
).unsqueeze(0)                                          # (1, N, 2)


# ══════════════════════════════════════════════════════════════════════════════
# 3. PHYSICS RESIDUALS FOR VANILLA PINN (NS 2D)
# ══════════════════════════════════════════════════════════════════════════════

def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]


def ns_residuals(
    net: nn.Module,          # model.net (raw MLP, no physics callback)
    xy_int: torch.Tensor,    # (N, 2)
    xy_bc: torch.Tensor,     # (M, 2)
    uv_bc: torch.Tensor,     # (M, 2)
    nu: float,               # kinematic viscosity = 1/Re
) -> Tuple[torch.Tensor, Dict[str, float]]:

    xy = xy_int.clone().requires_grad_(True)
    out = net(xy)                    # (N, 3): [u, v, p]
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    u_x = _grad(u, xy)[:, 0:1];  u_y = _grad(u, xy)[:, 1:2]
    v_x = _grad(v, xy)[:, 0:1];  v_y = _grad(v, xy)[:, 1:2]
    p_x = _grad(p, xy)[:, 0:1];  p_y = _grad(p, xy)[:, 1:2]

    u_xx = _grad(u_x, xy)[:, 0:1];  u_yy = _grad(u_y, xy)[:, 1:2]
    v_xx = _grad(v_x, xy)[:, 0:1];  v_yy = _grad(v_y, xy)[:, 1:2]

    res_u    = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    res_v    = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    res_cont = u_x + v_y

    pde_loss = (res_u ** 2 + res_v ** 2 + res_cont ** 2).mean()

    # Boundary condition
    out_bc = net(xy_bc)
    bc_loss = ((out_bc[:, 0:1] - uv_bc[:, 0:1]) ** 2
             + (out_bc[:, 1:2] - uv_bc[:, 1:2]) ** 2).mean()

    total = pde_loss + 10.0 * bc_loss
    return total, {"pde": pde_loss.item(), "bc": bc_loss.item()}


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODEL INSTANTIATION
# ══════════════════════════════════════════════════════════════════════════════

pinn = VanillaPINN(
    in_dim=2, out_dim=3,
    hidden=[128, 128, 128, 128],
    activation="tanh",
).to(DEVICE)

fno = FNO2d(
    in_channels=1,    # masked u observation
    out_channels=2,   # predict (u, v)
    width=32,
    modes1=12,
    modes2=12,
    layers=4,
    use_grid=True,    # append (x, y) coordinate grids
).to(DEVICE)

mgn = MeshGraphNet(
    node_in_dim=2,    # (x, y)
    out_dim=2,        # (u, v)
    edge_in_dim=0,
    pos_dim=2,
    use_pos=True,
    hidden_dim=128,
    n_layers=2,
    n_message_passing=6,
).to(DEVICE)

def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

print(f"  VanillaPINN  params: {param_count(pinn):>9,}")
print(f"  FNO-2D       params: {param_count(fno):>9,}")
print(f"  MeshGraphNet params: {param_count(mgn):>9,}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING
# ══════════════════════════════════════════════════════════════════════════════

# ── 5.1 VanillaPINN ───────────────────────────────────────────────────────────
print("[1/3] Training VanillaPINN (physics residuals, no data)...")
opt_pinn  = torch.optim.Adam(pinn.parameters(), lr=LR_PINN)
sched_pinn = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pinn, T_max=EPOCHS_PINN)

xy_int_d = XY_int.to(DEVICE)
xy_bc_d  = XY_bc.to(DEVICE)
uv_bc_d  = UV_bc.to(DEVICE)
nu_val   = 1.0 / RE

pinn_history: List[Dict] = []
t0 = time.perf_counter()

for epoch in range(EPOCHS_PINN):
    pinn.train()
    opt_pinn.zero_grad()
    loss, comps = ns_residuals(pinn.net, xy_int_d, xy_bc_d, uv_bc_d, nu_val)
    loss.backward()
    nn.utils.clip_grad_norm_(pinn.parameters(), 1.0)
    opt_pinn.step()
    sched_pinn.step()

    pinn_history.append({"epoch": epoch, "total": loss.item(), **comps})
    if (epoch + 1) % 500 == 0:
        print(f"  epoch {epoch+1:5d}/{EPOCHS_PINN}  "
              f"loss={loss.item():.3e}  pde={comps['pde']:.3e}  bc={comps['bc']:.3e}")

pinn_time = time.perf_counter() - t0
print(f"  Done in {pinn_time:.1f}s\n")


# ── 5.2 FNO-2D ────────────────────────────────────────────────────────────────
print("[2/3] Training FNO-2D (data-driven grid operator)...")
opt_fno = torch.optim.Adam(fno.parameters(), lr=LR_FNO)
sched_fno = torch.optim.lr_scheduler.CosineAnnealingLR(opt_fno, T_max=EPOCHS_FNO)

BS_FNO = 16
fno_history: List[Dict] = []
t0 = time.perf_counter()

for epoch in range(EPOCHS_FNO):
    fno.train()
    idx = torch.randperm(N_FNO_TRAIN)[:BS_FNO]
    xb = X_fno_train[idx].to(DEVICE)
    yb = Y_fno_train[idx].to(DEVICE)

    opt_fno.zero_grad()
    out = fno(xb, y_true=yb, return_loss=True)
    loss = out.losses["total"]
    loss.backward()
    opt_fno.step()
    sched_fno.step()

    fno_history.append({"epoch": epoch, "total": loss.item()})
    if (epoch + 1) % 100 == 0:
        print(f"  epoch {epoch+1:4d}/{EPOCHS_FNO}  loss={loss.item():.3e}")

fno_time = time.perf_counter() - t0
print(f"  Done in {fno_time:.1f}s\n")


# ── 5.3 MeshGraphNet ──────────────────────────────────────────────────────────
print("[3/3] Training MeshGraphNet (data-driven mesh surrogate)...")
opt_mgn = torch.optim.Adam(mgn.parameters(), lr=LR_MGN)
sched_mgn = torch.optim.lr_scheduler.CosineAnnealingLR(opt_mgn, T_max=EPOCHS_MGN)

node_x_d  = NODE_XY_T.to(DEVICE)          # (1, N, 2) — also the input features (x,y)
uv_tgt_d  = UV_nodes_T.to(DEVICE)         # (1, N, 2)
eidx_d    = edge_index.to(DEVICE)          # (2, E)

# Augment with Gaussian noise to avoid memorisation
NOISE_STD = 0.05

mgn_history: List[Dict] = []
t0 = time.perf_counter()

for epoch in range(EPOCHS_MGN):
    mgn.train()
    noise = NOISE_STD * torch.randn_like(node_x_d)
    x_noisy = node_x_d + noise              # perturb coordinates slightly

    g = GraphBatch(x=x_noisy, pos=node_x_d, edge_index=eidx_d)
    out = mgn(g, y_true=uv_tgt_d, return_loss=True)
    loss = out.losses["total"]

    opt_mgn.zero_grad()
    loss.backward()
    opt_mgn.step()
    sched_mgn.step()

    mgn_history.append({"epoch": epoch, "total": loss.item()})
    if (epoch + 1) % 100 == 0:
        print(f"  epoch {epoch+1:4d}/{EPOCHS_MGN}  loss={loss.item():.3e}")

mgn_time = time.perf_counter() - t0
print(f"  Done in {mgn_time:.1f}s\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6. EVALUATION ON THE REFERENCE GRID
# ══════════════════════════════════════════════════════════════════════════════

def l2_rel(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-12))

def linf_rel(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(pred - ref)) / (np.max(np.abs(ref)) + 1e-12))


# — VanillaPINN ----------------------------------------------------------------
pinn.eval()
with torch.no_grad():
    xy_test = torch.from_numpy(
        np.column_stack([XX.ravel(), YY.ravel()]).astype(np.float32)
    ).to(DEVICE)
    pinn_out = pinn.net(xy_test).cpu().numpy()
U_pinn = pinn_out[:, 0]
V_pinn = pinn_out[:, 1]


# — FNO-2D ---------------------------------------------------------------------
fno.eval()
with torch.no_grad():
    # Full (clean) input for evaluation
    fno_in = U_grid.unsqueeze(0).unsqueeze(0).to(DEVICE)   # (1, 1, H, W)
    fno_out = fno(fno_in).y.squeeze(0).cpu().numpy()       # (2, H, W)
U_fno = fno_out[0].ravel()
V_fno = fno_out[1].ravel()


# — MeshGraphNet ---------------------------------------------------------------
# Predict on the evaluation grid nodes via interpolation from mesh nodes
mgn.eval()
with torch.no_grad():
    g_eval = GraphBatch(x=node_x_d, pos=node_x_d, edge_index=eidx_d)
    mgn_out = mgn(g_eval).y.squeeze(0).cpu().numpy()  # (N_nodes, 2)

# Interpolate from mesh nodes to evaluation grid
from scipy.interpolate import LinearNDInterpolator

mgn_interp_u = LinearNDInterpolator(node_xy, mgn_out[:, 0])(
    np.column_stack([XX.ravel(), YY.ravel()])
)
mgn_interp_v = LinearNDInterpolator(node_xy, mgn_out[:, 1])(
    np.column_stack([XX.ravel(), YY.ravel()])
)
# Handle NaN at extrapolation boundaries with nearest fallback
from scipy.interpolate import NearestNDInterpolator
nn_u = NearestNDInterpolator(node_xy, mgn_out[:, 0])
nn_v = NearestNDInterpolator(node_xy, mgn_out[:, 1])
nan_mask = np.isnan(mgn_interp_u)
mgn_interp_u[nan_mask] = nn_u(np.column_stack([XX.ravel(), YY.ravel()])[nan_mask])
mgn_interp_v[nan_mask] = nn_v(np.column_stack([XX.ravel(), YY.ravel()])[nan_mask])
U_mgn, V_mgn = mgn_interp_u, mgn_interp_v


# — Metrics --------------------------------------------------------------------
models_eval = {
    "VanillaPINN": (U_pinn, V_pinn),
    "FNO-2D":      (U_fno,  V_fno),
    "MeshGraphNet":(U_mgn,  V_mgn),
}
train_times = {
    "VanillaPINN": pinn_time,
    "FNO-2D":      fno_time,
    "MeshGraphNet":mgn_time,
}
params = {
    "VanillaPINN": param_count(pinn),
    "FNO-2D":      param_count(fno),
    "MeshGraphNet":param_count(mgn),
}

print("=" * 70)
print(f"{'Model':<15} {'L2-rel u':>10} {'L2-rel v':>10} {'Linf u':>10} {'Time(s)':>10} {'Params':>10}")
print("-" * 70)
for name, (up, vp) in models_eval.items():
    l2u = l2_rel(up, U_ref); l2v = l2_rel(vp, V_ref)
    liu = linf_rel(up, U_ref)
    print(f"{name:<15} {l2u:>10.4f} {l2v:>10.4f} {liu:>10.4f} "
          f"{train_times[name]:>10.1f} {params[name]:>10,}")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# 7. VISUALISATION — Main benchmark figure
# ══════════════════════════════════════════════════════════════════════════════

CMAP_FIELD = "RdBu_r"
CMAP_ERR   = "hot_r"
MODEL_LABELS = ["VanillaPINN", "FNO-2D", "MeshGraphNet"]
PRED_UV = [(U_pinn, V_pinn), (U_fno, V_fno), (U_mgn, V_mgn)]
COLORS  = ["#2196F3", "#FF9800", "#4CAF50"]

xx_flat = XX.ravel();  yy_flat = YY.ravel()

fig = plt.figure(figsize=(22, 14))
fig.patch.set_facecolor("#0d1117")

gs_main = gridspec.GridSpec(
    3, 5,
    figure=fig,
    left=0.04, right=0.98,
    top=0.91, bottom=0.08,
    wspace=0.06, hspace=0.06,
)

def _imshow(ax, field, cmap, vmin=None, vmax=None, title="", xlabel=False, ylabel=False):
    im = ax.imshow(
        field.reshape(GRID_N, GRID_N).T,
        origin="lower",
        extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    if title:
        ax.set_title(title, color="white", fontsize=9, pad=3)
    if xlabel:
        ax.set_xlabel("x", color="#aaa", fontsize=8)
    if ylabel:
        ax.set_ylabel("y", color="#aaa", fontsize=8)
    return im

u_vmin, u_vmax = U_ref.min(), U_ref.max()
v_vmin, v_vmax = V_ref.min(), V_ref.max()

# — Row 0: u-velocity ----------------------------------------------------------
# Column 0: Reference
ax = fig.add_subplot(gs_main[0, 0])
im = _imshow(ax, U_ref, CMAP_FIELD, u_vmin, u_vmax,
             title="Reference  u", ylabel=True)
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01, format="%.2f").ax.yaxis.set_tick_params(colors="white")

# Columns 1-3: Model predictions
for col, (name, (up, vp), color) in enumerate(zip(MODEL_LABELS, PRED_UV, COLORS)):
    ax = fig.add_subplot(gs_main[0, col + 1])
    im = _imshow(ax, up, CMAP_FIELD, u_vmin, u_vmax, title=f"{name}  u")
    l2u = l2_rel(up, U_ref)
    ax.text(0.97, 0.04, f"L₂={l2u:.3f}", transform=ax.transAxes,
            ha="right", va="bottom", color=color, fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="#111", ec=color, lw=0.8))
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01, format="%.2f").ax.yaxis.set_tick_params(colors="white")

# Column 4: Error bar chart for u (L2)
ax_bar = fig.add_subplot(gs_main[0, 4])
ax_bar.set_facecolor("#0d1117")
l2_u_vals = [l2_rel(up, U_ref) for up, _ in PRED_UV]
bars = ax_bar.barh(MODEL_LABELS[::-1], l2_u_vals[::-1], color=COLORS[::-1], alpha=0.85, height=0.5)
ax_bar.set_xlabel("L₂ relative error  (u)", color="#aaa", fontsize=8)
ax_bar.tick_params(colors="white", labelsize=8)
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)
for sp in ["bottom", "left"]:
    ax_bar.spines[sp].set_edgecolor("#444")
for bar, val in zip(bars, l2_u_vals[::-1]):
    ax_bar.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", color="white", fontsize=8)
ax_bar.set_title("L₂ error — u", color="white", fontsize=9, pad=3)


# — Row 1: v-velocity ----------------------------------------------------------
ax = fig.add_subplot(gs_main[1, 0])
im = _imshow(ax, V_ref, CMAP_FIELD, v_vmin, v_vmax,
             title="Reference  v", ylabel=True)
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01, format="%.2f").ax.yaxis.set_tick_params(colors="white")

for col, (name, (up, vp), color) in enumerate(zip(MODEL_LABELS, PRED_UV, COLORS)):
    ax = fig.add_subplot(gs_main[1, col + 1])
    im = _imshow(ax, vp, CMAP_FIELD, v_vmin, v_vmax, title=f"{name}  v")
    l2v = l2_rel(vp, V_ref)
    ax.text(0.97, 0.04, f"L₂={l2v:.3f}", transform=ax.transAxes,
            ha="right", va="bottom", color=color, fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="#111", ec=color, lw=0.8))
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01, format="%.2f").ax.yaxis.set_tick_params(colors="white")

# Column 4: Error bar for v
ax_bar2 = fig.add_subplot(gs_main[1, 4])
ax_bar2.set_facecolor("#0d1117")
l2_v_vals = [l2_rel(vp, V_ref) for _, vp in PRED_UV]
bars2 = ax_bar2.barh(MODEL_LABELS[::-1], l2_v_vals[::-1], color=COLORS[::-1], alpha=0.85, height=0.5)
ax_bar2.set_xlabel("L₂ relative error  (v)", color="#aaa", fontsize=8)
ax_bar2.tick_params(colors="white", labelsize=8)
ax_bar2.spines["top"].set_visible(False)
ax_bar2.spines["right"].set_visible(False)
for sp in ["bottom", "left"]:
    ax_bar2.spines[sp].set_edgecolor("#444")
for bar, val in zip(bars2, l2_v_vals[::-1]):
    ax_bar2.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", color="white", fontsize=8)
ax_bar2.set_title("L₂ error — v", color="white", fontsize=9, pad=3)


# — Row 2: Relative error maps + training curves ──────────────────────────────
ax = fig.add_subplot(gs_main[2, 0])
mag_ref = np.sqrt(U_ref ** 2 + V_ref ** 2)
im = _imshow(ax, mag_ref, "plasma", title="|v|  reference", xlabel=True, ylabel=True)
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01, format="%.2f").ax.yaxis.set_tick_params(colors="white")

for col, (name, (up, vp), color) in enumerate(zip(MODEL_LABELS, PRED_UV, COLORS)):
    ax = fig.add_subplot(gs_main[2, col + 1])
    err_u = np.abs(up - U_ref) / (np.abs(U_ref).max() + 1e-12)
    err_v = np.abs(vp - V_ref) / (np.abs(V_ref).max() + 1e-12)
    err = 0.5 * (err_u + err_v)
    im = _imshow(ax, err, CMAP_ERR, 0, None, title=f"{name}  rel. error", xlabel=True)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01, format="%.3f").ax.yaxis.set_tick_params(colors="white")

# Column 4: Training curves
ax_loss = fig.add_subplot(gs_main[2, 4])
ax_loss.set_facecolor("#0d1117")

def _smooth(vals, w=20):
    kernel = np.ones(w) / w
    return np.convolve(vals, kernel, mode="same")

ep_pinn = [h["epoch"] for h in pinn_history]
ls_pinn = [h["total"] for h in pinn_history]
ep_fno  = [h["epoch"] * (EPOCHS_PINN // EPOCHS_FNO) for h in fno_history]
ls_fno  = [h["total"] for h in fno_history]
ep_mgn  = [h["epoch"] * (EPOCHS_PINN // EPOCHS_MGN) for h in mgn_history]
ls_mgn  = [h["total"] for h in mgn_history]

ax_loss.semilogy(ep_pinn, _smooth(ls_pinn, 30), color=COLORS[0], lw=1.5, label="VanillaPINN", alpha=0.9)
ax_loss.semilogy(ep_fno,  _smooth(ls_fno, 15),  color=COLORS[1], lw=1.5, label="FNO-2D",     alpha=0.9)
ax_loss.semilogy(ep_mgn,  _smooth(ls_mgn, 15),  color=COLORS[2], lw=1.5, label="MeshGraphNet",alpha=0.9)
ax_loss.set_xlabel("Equivalent epoch", color="#aaa", fontsize=8)
ax_loss.set_ylabel("Training loss", color="#aaa", fontsize=8)
ax_loss.set_title("Training curves", color="white", fontsize=9, pad=3)
ax_loss.tick_params(colors="white", labelsize=7)
ax_loss.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")
ax_loss.spines["top"].set_visible(False)
ax_loss.spines["right"].set_visible(False)
for sp in ["bottom", "left"]:
    ax_loss.spines[sp].set_edgecolor("#444")
ax_loss.grid(True, which="both", alpha=0.1, color="white")

# — Title & metadata label ─────────────────────────────────────────────────────
fig.suptitle(
    "PINNeAPPle Arena  ·  2D Kovasznay Navier–Stokes  (Re = 40)\n"
    "VanillaPINN  ·  FNO-2D  ·  MeshGraphNet",
    color="white", fontsize=14, fontweight="bold", y=0.975,
)
fig.text(
    0.99, 0.005,
    f"Params — PINN:{param_count(pinn):,}  FNO:{param_count(fno):,}  MGN:{param_count(mgn):,}  "
    f"|  Train time — PINN:{pinn_time:.0f}s  FNO:{fno_time:.0f}s  MGN:{mgn_time:.0f}s",
    ha="right", va="bottom", color="#888", fontsize=7,
)

out_path = os.path.join(OUT_DIR, "kovasznay_benchmark.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\n[saved] {out_path}")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 8. BONUS — Streamline figure (very visual for social media)
# ══════════════════════════════════════════════════════════════════════════════

fig2, axes2 = plt.subplots(1, 4, figsize=(22, 5))
fig2.patch.set_facecolor("#0d1117")

ALL_NAMES = ["Reference"] + MODEL_LABELS
ALL_UV    = [(U_ref, V_ref)] + [(u, v) for u, v in PRED_UV]
ALL_COLORS_TITLE = ["white"] + COLORS
mag_vmax = np.sqrt(U_ref**2 + V_ref**2).max()

import scipy.interpolate as si

def _streamax(ax, u, v, name, col):
    ax.set_facecolor("#0d1117")
    mag = np.sqrt(u**2 + v**2).reshape(GRID_N, GRID_N)

    # Background: velocity magnitude
    im = ax.imshow(
        mag.T, origin="lower", extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
        cmap="plasma", vmin=0, vmax=mag_vmax, aspect="auto", interpolation="bilinear"
    )
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.01, label="|v|"
                 ).ax.yaxis.set_tick_params(colors="white")

    # Streamlines on regular grid
    grid_xi = np.linspace(X_MIN, X_MAX, 60)
    grid_yi = np.linspace(Y_MIN, Y_MAX, 80)
    pts = np.column_stack([xx_flat, yy_flat])
    Ui  = si.griddata(pts, u, (*np.meshgrid(grid_xi, grid_yi, indexing="ij"),), method="linear")
    Vi  = si.griddata(pts, v, (*np.meshgrid(grid_xi, grid_yi, indexing="ij"),), method="linear")
    Ui  = np.nan_to_num(Ui)
    Vi  = np.nan_to_num(Vi)
    mag_i = np.sqrt(Ui**2 + Vi**2)
    lw = 1.5 * mag_i / (mag_i.max() + 1e-8) + 0.3

    try:
        ax.streamplot(
            grid_xi, grid_yi, Ui.T, Vi.T,
            color="white", linewidth=lw.T,
            density=1.8, arrowsize=0.8, alpha=0.6,
        )
    except Exception:
        pass  # graceful skip if streamplot fails (e.g., all-zero field)

    ax.set_title(name, color=col, fontsize=12, fontweight="bold", pad=4)
    ax.set_xlabel("x", color="#aaa", fontsize=9)
    ax.set_ylabel("y", color="#aaa", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")

    if name != "Reference":
        l2u = l2_rel(u, U_ref)
        ax.text(0.97, 0.04, f"L₂={l2u:.3f}", transform=ax.transAxes,
                ha="right", va="bottom", color=col, fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="#0d1117cc", ec=col, lw=1.0))

for ax, name, (u, v), col in zip(axes2, ALL_NAMES, ALL_UV, ALL_COLORS_TITLE):
    _streamax(ax, u, v, name, col)

fig2.suptitle(
    "PINNeAPPle Arena  ·  Kovasznay NS Flow  —  Velocity Streamlines",
    color="white", fontsize=13, fontweight="bold", y=1.01,
)
fig2.tight_layout()

stream_path = os.path.join(OUT_DIR, "kovasznay_streams.png")
fig2.savefig(stream_path, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
print(f"[saved] {stream_path}")
plt.close(fig2)

print("\nBenchmark complete.")
print(f"  >> {out_path}")
print(f"  >> {stream_path}")
