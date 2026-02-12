import numpy as np

import torch
import torch.optim as optim

import matplotlib.pyplot as plt

from pinneaple_pinn.factory.pinn_factory import PINNProblemSpec, PINNFactory, PINN
from pinneaple_models.pinns.inverse import InversePINN
from pinneaple_models.pinns.pielm import PIELM, PIELMFactoryAdapter
from pinneaple_models.pinns.pinn_lstm import PINNLSTM
from pinneaple_models.pinns.pinnsformer import PINNsFormer
from pinneaple_models.pinns.vanilla import VanillaPINN
from pinneaple_models.pinns.vpinn import VPINN, pinn_factory_adapter
from pinneaple_models.pinns.xpinn import XPINN
from pinneaple_models.pinns.xtfc import XTFC, XTFCFactoryModel, XTFCConfig

from typing import Dict, Any, List, Tuple

# Inverse PINN

spec = PINNProblemSpec(
    pde_residuals=[
        "Derivative(u(t,x), t) - k*Derivative(u(t,x), x, 2)"
    ],
    conditions=[
        {"name": "ic",  "equation": "u(t,x) - sin(pi*x)", "weight": 10.0},  # u(0,x)=sin(pi x)
        {"name": "bc0", "equation": "u(t,x)",             "weight": 5.0},   # u(t,0)=0
        {"name": "bc1", "equation": "u(t,x)",             "weight": 5.0},   # u(t,1)=0
    ],
    independent_vars=["t", "x"],
    dependent_vars=["u"],
    inverse_params=["k"],
    loss_weights={"pde": 1.0, "conditions": 1.0, "data": 0.0},
    verbose=True,
)

factory = PINNFactory(spec)
loss_fn = factory.generate_loss_function()

core = InversePINN(
    in_dim=2,
    out_dim=1,
    hidden=[128, 128, 128],
    activation="tanh",
    inverse_params=["k"],
    initial_guesses={"k": 0.2},
)

model = core.as_factory_model(independent_vars=spec.independent_vars)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

opt = optim.Adam(model.parameters(), lr=1e-3)

def sample_collocation(n: int):
    t = torch.rand(n, 1, device=device)
    x = torch.rand(n, 1, device=device)
    # IMPORTANT: order must match spec.independent_vars == ["t","x"]
    return (t, x)

def sample_ic(n: int):
    t0 = torch.zeros(n, 1, device=device)
    x = torch.rand(n, 1, device=device)
    return (t0, x)

def sample_bc0(n: int):
    t = torch.rand(n, 1, device=device)
    x0 = torch.zeros(n, 1, device=device)
    return (t, x0)

def sample_bc1(n: int):
    t = torch.rand(n, 1, device=device)
    x1 = torch.ones(n, 1, device=device)
    return (t, x1)

for step in range(1, 2001):
    batch = {
        "collocation": sample_collocation(2048),
        "conditions": [
            sample_ic(512),   # matches conditions[0]
            sample_bc0(512),  # matches conditions[1]
            sample_bc1(512),  # matches conditions[2]
        ],
        "data": None,
    }

    total, comps = loss_fn(model, batch)

    opt.zero_grad(set_to_none=True)
    total.backward()
    opt.step()

    if step % 200 == 0:
        k_val = float(model.inverse_params["k"].detach().cpu().item())
        print(f"step={step:4d} total={comps['total']:.4e} pde={comps.get('pde',0):.4e} "
              f"cond={comps.get('conditions',0):.4e}  k={k_val:.6f}")

# PIELM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

T_MAX = 1.0
X_MIN, X_MAX = -1.0, 1.0

# "ground-truth" (MMS) solution
def u_exact_np(t, x):
    # t: (Nt,), x: (Nx,) -> (Nt,Nx)
    T, X = np.meshgrid(t, x, indexing="ij")
    return np.exp(-T) * np.sin(np.pi * X)

def u_exact_torch(t, x):
    # t,x: torch (N,1)
    return torch.exp(-t) * torch.sin(torch.pi * x)


pde_mms = (
    "diff(u(t,x), t) + u(t,x)*diff(u(t,x), x) - nu*diff(u(t,x), x, 2)"
    " - ( -exp(-t)*sin(pi*x)"
    "     + (exp(-t)*sin(pi*x))*(exp(-t)*pi*cos(pi*x))"
    "     - nu*(-exp(-t)*(pi*pi)*sin(pi*x))"
    "   )"
)

spec = PINNProblemSpec(
    independent_vars=["t", "x"],
    dependent_vars=["u"],
    inverse_params=["nu"],
    pde_residuals=[pde_mms],
    conditions=[
        {"name": "ic", "equation": "u(t,x) - sin(pi*x)", "weight": 1.0},
        {"name": "bc_left", "equation": "u(t,x)", "weight": 1.0},
        {"name": "bc_right", "equation": "u(t,x)", "weight": 1.0},
    ],
   loss_weights={"pde": 1.0, "conditions": 1.0, "data": 10.0},
    verbose=True,
)

factory = PINNFactory(spec)
loss_fn = factory.generate_loss_function()

pielm = PIELM(in_dim=2, out_dim=1, hidden_dim=1024, activation="tanh", freeze_random=True).to(device=device)
model = PIELMFactoryAdapter(
    pielm,
    inverse_params_names=spec.inverse_params,
    initial_guesses={"nu": 0.01},
    dtype=dtype,
).to(device=device)

opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

def sample_collocation(N: int):
    t = torch.rand(N, 1, device=device, dtype=dtype) * T_MAX
    x = (X_MIN + (X_MAX - X_MIN) * torch.rand(N, 1, device=device, dtype=dtype))
    t.requires_grad_(True)
    x.requires_grad_(True)
    return (t, x)

def sample_ic(N: int):
    t = torch.zeros(N, 1, device=device, dtype=dtype)
    x = (X_MIN + (X_MAX - X_MIN) * torch.rand(N, 1, device=device, dtype=dtype))
    t.requires_grad_(True)
    x.requires_grad_(True)
    return (t, x)

def sample_bc_left(N: int):
    t = torch.rand(N, 1, device=device, dtype=dtype) * T_MAX
    x = torch.full((N, 1), X_MIN, device=device, dtype=dtype)
    t.requires_grad_(True)
    x.requires_grad_(True)
    return (t, x)

def sample_bc_right(N: int):
    t = torch.rand(N, 1, device=device, dtype=dtype) * T_MAX
    x = torch.full((N, 1), X_MAX, device=device, dtype=dtype)
    t.requires_grad_(True)
    x.requires_grad_(True)
    return (t, x)

def sample_data(N: int):
    t = torch.rand(N, 1, device=device, dtype=dtype) * T_MAX
    x = (X_MIN + (X_MAX - X_MIN) * torch.rand(N, 1, device=device, dtype=dtype))
    y = u_exact_torch(t, x)
    return (t, x), y

for step in range(1, 501):
    (t_obs, x_obs), u_obs = sample_data(2048)

    batch = {
        "collocation": sample_collocation(2048),
        "conditions": [
            sample_ic(512),
            sample_bc_left(512),
            sample_bc_right(512),
        ],
        "data": ((t_obs, x_obs), u_obs),
    }

    opt.zero_grad(set_to_none=True)
    total, comps = loss_fn(model, batch)
    total.backward()
    opt.step()

    if step % 50 == 0:
        nu_val = float(model.inverse_params["nu"].detach().cpu())
        print(
            f"step={step:04d} total={comps['total']:.3e} "
            f"pde={comps['pde']:.3e} cond={comps['conditions']:.3e} data={comps['data']:.3e} "
            f"nu={nu_val:.5f}"
        )

def burgers_fdm_forced(nu=0.01, Nx=256, Nt=4000, t_max=1.0, x_min=-1.0, x_max=1.0):
    x = np.linspace(x_min, x_max, Nx)
    dx = x[1] - x[0]
    dt = t_max / Nt

    # IC
    u = np.sin(np.pi * x).astype(np.float64)
    u[0] = 0.0
    u[-1] = 0.0

    # snapshots
    snap_times = np.linspace(0, t_max, 201)
    snaps = []
    t = 0.0
    k_snap = 0

    for n in range(Nt):
        if k_snap < len(snap_times) and t >= snap_times[k_snap] - 1e-12:
            snaps.append(u.copy())
            k_snap += 1

        un = u.copy()

        dudx_upwind = (un[1:-1] - un[:-2]) / dx
        d2udx2 = (un[2:] - 2.0 * un[1:-1] + un[:-2]) / (dx * dx)

        # f = u_t + u*u_x - nu*u_xx  com u_ex
        # u_ex(t,x) = exp(-t)*sin(pi x)
        uex = np.exp(-t) * np.sin(np.pi * x[1:-1])
        ut = -np.exp(-t) * np.sin(np.pi * x[1:-1])
        ux = np.exp(-t) * np.pi * np.cos(np.pi * x[1:-1])
        uxx = -np.exp(-t) * (np.pi**2) * np.sin(np.pi * x[1:-1])
        f = ut + uex * ux - nu * uxx

        u[1:-1] = un[1:-1] - dt * (un[1:-1] * dudx_upwind) + dt * nu * d2udx2 + dt * f

        # BCs
        u[0] = 0.0
        u[-1] = 0.0
        t += dt

    while k_snap < len(snap_times):
        snaps.append(u.copy())
        k_snap += 1

    U = np.stack(snaps, axis=0)     # (Nt_snap, Nx)
    t_grid = snap_times[:U.shape[0]]
    return t_grid, x, U

nu_used = float(model.inverse_params["nu"].detach().cpu())
t_fdm, x_fdm, U_fdm = burgers_fdm_forced(nu=nu_used, Nx=256, Nt=6000, t_max=T_MAX, x_min=X_MIN, x_max=X_MAX)

def pinn_on_grid(model, t_grid, x_grid, device):
    model.eval()
    Nt = len(t_grid)
    Nx = len(x_grid)
    T, X = np.meshgrid(t_grid, x_grid, indexing="ij")
    t_flat = torch.tensor(T.reshape(-1, 1), dtype=torch.float32, device=device)
    x_flat = torch.tensor(X.reshape(-1, 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        U = model(t_flat, x_flat).reshape(Nt, Nx).cpu().numpy()
    return U

U_pinn = pinn_on_grid(model, t_fdm, x_fdm, device)
U_true = u_exact_np(t_fdm, x_fdm)

def rel_l2(A, B):
    return np.linalg.norm(A - B) / (np.linalg.norm(B) + 1e-12)

print("Rel L2: PINN vs TRUE =", rel_l2(U_pinn, U_true))
print("Rel L2: FDM  vs TRUE =", rel_l2(U_fdm,  U_true))
print("Rel L2: PINN vs FDM  =", rel_l2(U_pinn, U_fdm))

def show_heatmap(U, title):
    plt.figure(figsize=(8, 4))
    plt.imshow(
        U,
        extent=[x_fdm.min(), x_fdm.max(), t_fdm.min(), t_fdm.max()],
        origin="lower",
        aspect="auto",
    )
    plt.colorbar(label="u")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(title)
    plt.show()

show_heatmap(U_true, "TRUE (MMS)  u(t,x) = exp(-t) sin(pi x)")
show_heatmap(U_pinn, "PINN (PI-ELM)")
show_heatmap(U_fdm,  "FDM (forced Burgers)")

show_heatmap(U_pinn - U_true, "Error: PINN - TRUE")
show_heatmap(U_fdm  - U_true, "Error: FDM - TRUE")

plt.figure(figsize=(8, 5))
for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
    k = int(round(t_val / T_MAX * (len(t_fdm)-1)))
    plt.plot(x_fdm, U_true[k], label=f"TRUE t={t_fdm[k]:.2f}")
    plt.plot(x_fdm, U_pinn[k], "--", label=f"PINN t={t_fdm[k]:.2f}")
    plt.plot(x_fdm, U_fdm[k], ":", label=f"FDM  t={t_fdm[k]:.2f}")
plt.xlabel("x")
plt.ylabel("u")
plt.title("Spatial slices")
plt.legend(ncol=2, fontsize=8)
plt.show()

# PINN LSTM

def make_bt(B: int, T: int, device, *, requires_grad: bool = True):
    t = torch.rand(B, T, 1, device=device, requires_grad=requires_grad)
    x = torch.rand(B, T, 1, device=device, requires_grad=requires_grad)
    return t, x

def flatten_bt_to_n1(a_bt1: torch.Tensor) -> torch.Tensor:
    # (B,T,1) -> (N,1)
    B, T, C = a_bt1.shape
    assert C == 1
    return a_bt1.reshape(B * T, 1)

spec = PINNProblemSpec(
    pde_residuals=["Derivative(u(t,x), t) + u(t,x)"],
    conditions=[{"name": "ic", "equation": "u(t,x) - sin(pi*x)", "weight": 1.0}],
    independent_vars=["t", "x"],
    dependent_vars=["u"],
    inverse_params=[],
    loss_weights={"pde": 1.0, "conditions": 1.0, "data": 0.0},
    verbose=True,
)

factory = PINNFactory(spec)
loss_fn = factory.generate_loss_function()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

B, T = 8, 64
model = PINNLSTM(in_dim=2, out_dim=1, seq_len=T, hidden_dim=64, num_layers=2, dropout=0.0).to(device)

t_col_bt, x_col_bt = make_bt(B, T, device, requires_grad=True)
t_col = flatten_bt_to_n1(t_col_bt)
x_col = flatten_bt_to_n1(x_col_bt)

t_ic_bt = torch.zeros(B, T, 1, device=device, requires_grad=True)
x_ic_bt = torch.rand(B, T, 1, device=device, requires_grad=True)
t_ic = flatten_bt_to_n1(t_ic_bt)
x_ic = flatten_bt_to_n1(x_ic_bt)

batch = {
    "collocation": (t_col, x_col),
    "conditions": [(t_ic, x_ic)],
}

total, comps = loss_fn(model, batch)
print("total:", float(total.detach().cpu()))
print("components:", comps)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    opt.zero_grad(set_to_none=True)
    total, comps = loss_fn(model, batch)
    total.backward()
    opt.step()

    if step % 50 == 0:
        print(step, comps)

# PINNSFormer

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(False)

spec = PINNProblemSpec(
    independent_vars=["t", "x"],     # in_dim=2
    dependent_vars=["u"],            # out_dim=1
    pde_residuals=[
        "Derivative(u(t,x), t) + u(t,x)*Derivative(u(t,x), x)"
    ],
    conditions=[
        {"name": "ic", "equation": "u(t,x) - sin(pi*x)", "weight": 1.0}
    ],
    inverse_params=[],
    loss_weights={"pde": 1.0, "conditions": 1.0, "data": 0.0},
    verbose=True,
)

factory = PINNFactory(spec)
loss_fn = factory.generate_loss_function()

seq_len = 64
net = PINNsFormer(
    in_dim=2,
    out_dim=1,
    seq_len=seq_len,
    d_model=128,
    nhead=4,
    num_layers=4,
    dim_feedforward=256,
    dropout=0.0,
    max_len=512,
    learnable_pos_emb=True,
)

model = PINN(neural_network=net, inverse_params_names=None, initial_guesses=None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

B = 8
T = seq_len
N = B * T

t = torch.rand(N, 1, device=device, requires_grad=True)
x = torch.rand(N, 1, device=device, requires_grad=True)

t_ic = torch.zeros(N, 1, device=device, requires_grad=True)
x_ic = torch.rand(N, 1, device=device, requires_grad=True)

batch = {
    "collocation": (t, x),
    "conditions": [(t_ic, x_ic)],
    "data": None,
}

opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    opt.zero_grad(set_to_none=True)
    total, comps = loss_fn(model, batch)
    total.backward()
    opt.step()

    if step % 50 == 0:
        print(step, comps)

# Vanilla PINN

spec = PINNProblemSpec(
    pde_residuals=[
        "Derivative(u(t), t) + u(t)",
    ],
    conditions=[
        {"name": "ic_u", "equation": "u(t) - 1.0", "weight": 1.0},  # u(0)=1
    ],
    independent_vars=["t"],
    dependent_vars=["u"],
    inverse_params=[],
    loss_weights={"pde": 1.0, "conditions": 10.0, "data": 0.0},
    verbose=True,
)

factory = PINNFactory(spec)
loss_fn = factory.generate_loss_function()

model = VanillaPINN(in_dim=1, out_dim=1, hidden=[64, 64, 64], activation="tanh")
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def sample_collocation(n: int, device="cpu"):
    t = torch.rand(n, 1, device=device)  # (N,1) em [0,1]
    return (t,)

def sample_ic(n: int, device="cpu"):
    t0 = torch.zeros(n, 1, device=device)
    return (t0,)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

for step in range(2000):
    batch = {
        "collocation": sample_collocation(1024, device=device),
        "conditions": [sample_ic(256, device=device)],
        "data": None,
    }

    total, comps = loss_fn(model, batch)

    opt.zero_grad()
    total.backward()
    opt.step()

    if step % 200 == 0:
        print(step, comps)

t_test = torch.linspace(0, 1, 200, device=device)[:, None]
with torch.no_grad():
    out = model(t_test)
    u_pred = out.y               # (200,1)

u_true = torch.exp(-t_test)
mse = torch.mean((u_pred - u_true) ** 2).item()
print("MSE:", mse)

# VPINN

spec = PINNProblemSpec(
    pde_residuals=[ "Derivative(u(t,x),t) + u(t,x)*Derivative(u(t,x),x) - nu*Derivative(Derivative(u(t,x),x),x)" ],
    conditions=[
        {"name": "ic_u", "equation": "u(t,x) - sin(pi*x)", "weight": 1.0},
    ],
    independent_vars=["t", "x"],
    dependent_vars=["u"],
    inverse_params=["nu"],
    loss_weights={"pde": 1.0, "conditions": 10.0, "data": 1.0},
    verbose=True,
)

factory = PINNFactory(spec)
loss_fn = factory.generate_loss_function()

model = VPINN(
    in_dim=2,
    out_dim=1,
    inverse_params_names=spec.inverse_params,
    initial_guesses={"nu": 0.01},
)

physics_fn = pinn_factory_adapter(loss_fn)

N = 4096
t = torch.rand(N, 1)
x = torch.rand(N, 1)

Nc = 1024
t0 = torch.zeros(Nc, 1)
x0 = torch.rand(Nc, 1)

batch = {
    "collocation": (t, x),
    "conditions": [(t0, x0)],
    "data": None,
}

out = model(t, x, physics_fn=physics_fn, physics_data=batch)
loss = out.losses["total"]

loss.backward()

# XPINN

def sample_uniform(n: int, lo: float, hi: float, device: torch.device) -> torch.Tensor:
    return (lo + (hi - lo) * torch.rand(n, 1, device=device)).requires_grad_(True)

def make_heat_batches_for_subdomain(*,
    n_col: int,
    n_ic: int,
    n_bc: int,
    t_lo: float,
    t_hi: float,
    x_lo: float,
    x_hi: float,
    device: torch.device,
) -> Dict[str, Any]:
    # collocation
    t_c = sample_uniform(n_col, t_lo, t_hi, device)
    x_c = sample_uniform(n_col, x_lo, x_hi, device)

    # IC: t=0, x in [x_lo, x_hi]
    t_ic = torch.zeros(n_ic, 1, device=device, requires_grad=True)
    x_ic = sample_uniform(n_ic, x_lo, x_hi, device)

    t_b0 = sample_uniform(n_bc, t_lo, t_hi, device)
    x_b0 = torch.zeros(n_bc, 1, device=device, requires_grad=True)

    t_b1 = sample_uniform(n_bc, t_lo, t_hi, device)
    x_b1 = torch.ones(n_bc, 1, device=device, requires_grad=True)

    batch = {
        "collocation": (t_c, x_c),
        "conditions": [
            (t_ic, x_ic),  # ic_u
            (t_b0, x_b0),  # bc_x0
            (t_b1, x_b1),  # bc_x1
        ],
        "data": None,
    }
    return batch


def make_interface_pair(
    *,
    n_iface: int,
    x_iface: float,
    t_lo: float,
    t_hi: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    t = sample_uniform(n_iface, t_lo, t_hi, device)
    x = torch.full((n_iface, 1), float(x_iface), device=device, requires_grad=True)
    xi = torch.cat([t, x], dim=1)
    xj = torch.cat([t, x], dim=1)

    ni = torch.tensor([0.0, +1.0], device=device)
    nj = torch.tensor([0.0, -1.0], device=device)
    return xi, xj, ni, nj

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

nu = 0.1
spec = PINNProblemSpec(
    independent_vars=["t", "x"],
    dependent_vars=["u"],
    inverse_params=[],
    pde_residuals=[
        f"Derivative(u(t,x),t) - {nu}*Derivative(u(t,x),x,2)",
    ],
    conditions=[
        {"name": "ic_u", "equation": "u(t,x) - sin(pi*x)", "weight": 1.0},
        {"name": "bc_x0", "equation": "u(t,x)", "weight": 1.0},
        {"name": "bc_x1", "equation": "u(t,x)", "weight": 1.0},
    ],
    loss_weights={"pde": 1.0, "conditions": 10.0, "data": 0.0},
    verbose=True,
)

factory = PINNFactory(spec)
loss_fn = factory.generate_loss_function()

xpinn = XPINN(
    n_subdomains=2,
    in_dim=2,          # (t,x)
    out_dim=1,         # u
    hidden=(128, 128, 128),
    activation="tanh",
    interface_weight=1.0,
    interface_flux_weight=1.0,
    physics_weight=1.0,
).to(device)

opt = torch.optim.Adam(xpinn.parameters(), lr=1e-3)

doms = [(0.0, 0.5), (0.5, 1.0)]

steps = 2000
for it in range(1, steps + 1):
    xpinn.train()
    opt.zero_grad(set_to_none=True)

    physics_data_list: List[Dict[str, Any]] = []
    x_list: List[torch.Tensor] = []

    for (x_lo, x_hi) in doms:
        batch = make_heat_batches_for_subdomain(
            n_col=2048,
            n_ic=512,
            n_bc=512,
            t_lo=0.0,
            t_hi=1.0,
            x_lo=x_lo,
            x_hi=x_hi,
            device=device,
        )
        physics_data_list.append(batch)

        t_vis = sample_uniform(256, 0.0, 1.0, device)
        x_vis = sample_uniform(256, x_lo, x_hi, device)
        x_list.append(torch.cat([t_vis, x_vis], dim=1))

    xi, xj, ni, nj = make_interface_pair(
        n_iface=512,
        x_iface=0.5,
        t_lo=0.0,
        t_hi=1.0,
        device=device,
    )
    interface_pairs = [(0, 1, xi, xj, ni, nj)]

    out = xpinn(
        x_list,
        interface_pairs=interface_pairs,
        physics_fn=loss_fn,
        physics_data_list=physics_data_list
    )

    loss = out.losses["total"]
    loss.backward()
    opt.step()

    if it % 200 == 0 or it == 1:
        pde = out.losses.get("physics/pde", torch.tensor(0.0)).item()
        cond = out.losses.get("physics/conditions", torch.tensor(0.0)).item()
        iface = out.losses.get("interface", torch.tensor(0.0)).item()
        iflux = out.losses.get("interface_flux", torch.tensor(0.0)).item()
        print(f"[{it:4d}] total={loss.item():.4e}  pde={pde:.3e}  cond={cond:.3e}  iface={iface:.3e}  iflux={iflux:.3e}")

xpinn.eval()
with torch.no_grad():
    x = torch.linspace(0, 1, 200, device=device).unsqueeze(1)
    t = torch.ones_like(x)
    X = torch.cat([t, x], dim=1)

    mask = (x[:, 0] <= 0.5)
    u = torch.empty_like(x)
    u[mask] = xpinn.subnets[0].predict(X[mask])
    u[~mask] = xpinn.subnets[1].predict(X[~mask])
    print("u(t=1,x) sample:", u[:5, 0].detach().cpu().numpy())

# XTFC

spec = PINNProblemSpec(
    independent_vars=["t", "x"],
    dependent_vars=["u"],
    inverse_params=["nu"],
    pde_residuals=[
        "Derivative(u(t,x), t) + u(t,x)*Derivative(u(t,x), x) - nu*Derivative(u(t,x), x, 2)"
    ],
    conditions=[
        {"name": "ic", "equation": "u(t,x) - sin(pi*x)", "weight": 1.0},
        {"name": "bc0", "equation": "u(t,x)", "weight": 1.0},
        {"name": "bc1", "equation": "u(t,x)", "weight": 1.0},
    ],
    loss_weights={"pde": 1.0, "conditions": 10.0, "data": 1.0},
    verbose=True,
)

factory = PINNFactory(spec)
loss_fn = factory.generate_loss_function()

# 2) Hard-constraint pieces (example)
def g_fn(x_cat: torch.Tensor) -> torch.Tensor:
    # x_cat = [t,x]
    x = x_cat[:, 1:2]
    return torch.sin(torch.pi * x)  # ensures IC at t=0

def B_fn(x_cat: torch.Tensor) -> torch.Tensor:
    t = x_cat[:, 0:1]
    x = x_cat[:, 1:2]
    return t * (1.0 - x**2)         # zeros at t=0 and x=±1

cfg = XTFCConfig(
    in_dim=2,
    out_dim=1,
    rf_dim=2048,
    rf_kind="rff",
    rff_sigma=1.0,
    freeze_random=True,
)

xtfc = XTFC(
    in_dim=2,
    out_dim=1,
    g_fn=g_fn,
    B_fn=B_fn,
    config=cfg,
)

model = XTFCFactoryModel(xtfc, inverse_params_names=["nu"], initial_guesses={"nu": 0.01})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 3) Batch
Nf = 4096
t_f = torch.rand(Nf, 1, device=device, requires_grad=True)                 # [0,1]
x_f = (2.0 * torch.rand(Nf, 1, device=device) - 1.0).requires_grad_(True)  # [-1,1]

Nc = 1024
# IC: t=0
t_ic = torch.zeros(Nc, 1, device=device, requires_grad=True)
x_ic = (2.0 * torch.rand(Nc, 1, device=device) - 1.0).requires_grad_(True)
# BC
t_bc = torch.rand(Nc, 1, device=device, requires_grad=True)
x_bc0 = (-1.0 * torch.ones(Nc, 1, device=device)).requires_grad_(True)
x_bc1 = (+1.0 * torch.ones(Nc, 1, device=device)).requires_grad_(True)

batch = {
    "collocation": (t_f, x_f),
    "conditions": [
        (t_ic, x_ic),    # ic
        (t_bc, x_bc0),   # bc0
        (t_bc, x_bc1),   # bc1
    ],
    "data": None,
}

# 4) Training
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

for step in range(1, 1001):
    opt.zero_grad(set_to_none=True)
    total, comps = loss_fn(model, batch)
    total.backward()
    opt.step()

    if step % 200 == 0:
        nu_val = float(model.inverse_params["nu"].detach().cpu())
        print(
            f"step={step} total={comps['total']:.3e} "
            f"pde={comps.get('pde',0):.3e} cond={comps.get('conditions',0):.3e} nu={nu_val:.5f}"
        )

