import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from pinneaple_models.continuous.bayesian_rnn import BayesianRNN
from pinneaple_models.continuous.deep_state_space import DeepStateSpaceModel
from pinneaple_models.continuous.hamiltonian import HamiltonianNeuralNetwork
from pinneaple_models.continuous.latent_ode import LatentODE
from pinneaple_models.continuous.neural_cde import NeuralCDE
from pinneaple_models.continuous.neural_gp import NeuralGaussianProcess
from pinneaple_models.continuous.neural_ode import NeuralODE
from pinneaple_models.continuous.neural_sde import NeuralSDE
from pinneaple_models.continuous.ode_rnn import ODERNN
from pinneaple_models.continuous.symplectic_ode import SymplecticODENet
from pinneaple_models.continuous.symplectic_rnn import SymplecticRNN

# Bayesian RNN

def make_sine_dataset(B=32, T=50, input_dim=2, out_dim=1, device="cpu"):
    t = torch.linspace(0, 1, T, device=device).view(1, T, 1).repeat(B, 1, 1)  # (B,T,1)
    a = torch.empty(B, 1, 1, device=device).uniform_(0.5, 1.5).repeat(1, T, 1) # (B,T,1)

    x = torch.cat([t, a], dim=-1)  # (B,T,2)
    y = a * torch.sin(2.0 * torch.pi * t)  # (B,T,1)

    y = y + 0.02 * torch.randn_like(y)
    return x, y

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(7)

model = BayesianRNN(
    input_dim=2,
    out_dim=1,
    hidden_dim=128,
    num_layers=2,
    dropout=0.1,
    cell="gru",
).to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train(True)
for step in range(1, 801):
    x, y = make_sine_dataset(B=64, T=60, device=device)

    out = model(x, y_true=y, return_loss=True)
    loss = out.losses["total"]

    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % 100 == 0:
        print(f"step={step:04d}  mse={out.losses['mse'].item():.6f}")

model.eval()
x_test, y_test = make_sine_dataset(B=8, T=80, device=device)

# Option 1
out_det = model(x_test)  # mc_samples=0
y_det = out_det.y
out_mc = model.predict_mc(x_test, mc_samples=64, return_samples=False)
y_mu = out_mc.y
y_logvar = out_mc.extras["logvar"]
y_std = torch.exp(0.5 * y_logvar)

print("\nShapes:")
print("x_test   :", tuple(x_test.shape))
print("y_test   :", tuple(y_test.shape))
print("y_det    :", tuple(y_det.shape))
print("y_mu     :", tuple(y_mu.shape))
print("y_std    :", tuple(y_std.shape))

mae_det = (y_det - y_test).abs().mean().item()
mae_mc = (y_mu - y_test).abs().mean().item()
avg_std = y_std.mean().item()
print("\nMetrics:")
print(f"MAE(det) = {mae_det:.6f}")
print(f"MAE(MC ) = {mae_mc:.6f}")
print(f"mean(std)= {avg_std:.6f}")

# Option 2
model.train(True)
out_mc2 = model(x_test, mc_samples=16, return_samples=True)  # grad-friendly
print("\nforward(mc) samples shape:", tuple(out_mc2.extras["samples"].shape))

# Deep State Space

def make_synth_ssm(B=64, T=60, input_dim=1, out_dim=1, latent_dim=2, device="cpu"):
    A = torch.tensor([[0.85, 0.05],
                      [0.00, 0.80]], device=device)
    U = torch.randn(latent_dim, input_dim, device=device) * 0.3
    C = torch.randn(out_dim, latent_dim, device=device) * 1.0

    x = torch.randn(B, T, input_dim, device=device) * 0.8

    z = torch.zeros(B, T, latent_dim, device=device)
    y = torch.zeros(B, T, out_dim, device=device)

    z0 = torch.randn(B, latent_dim, device=device) * 0.5
    z[:, 0] = z0

    q_std = 0.10
    r_std = 0.15

    for t in range(1, T):
        eps = torch.randn(B, latent_dim, device=device) * q_std
        z[:, t] = (z[:, t-1] @ A.T) + (x[:, t] @ U.T) + eps

    y = (z @ C.T) + torch.randn(B, T, out_dim, device=device) * r_std
    return x, y

device="cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)

input_dim = 1
out_dim = 1

model = DeepStateSpaceModel(
    input_dim=input_dim,
    out_dim=out_dim,
    latent_dim=8,
    hidden_dim=128,
    mlp_depth=2,
    beta_kl=0.1,
    min_logvar=-10.0,
    max_logvar=2.0,
).to(device)

opt = torch.optim.Adam(model.parameters(), lr=2e-3)

x_train, y_train = make_synth_ssm(B=256, T=80, input_dim=input_dim, out_dim=out_dim, device=device)
x_val, y_val     = make_synth_ssm(B=64,  T=80, input_dim=input_dim, out_dim=out_dim, device=device)

batch_size = 64
steps = 1200

for step in range(1, steps + 1):
    idx = torch.randint(0, x_train.size(0), (batch_size,), device=device)
    xb = x_train[idx]
    yb = y_train[idx]

    out = model(xb, y_true=yb, return_loss=True)  # ELBO
    loss = out.losses["total"]

    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % 200 == 0:
        with torch.no_grad():
            out_val = model(x_val, y_true=y_val, return_loss=True)
            print(
                f"step={step:04d} "
                f"train_total={out.losses['total'].item():.4f} "
                f"(nll={out.losses['nll'].item():.4f}, kl={out.losses['kl'].item():.4f}) | "
                f"val_total={out_val.losses['total'].item():.4f} "
                f"(nll={out_val.losses['nll'].item():.4f}, kl={out_val.losses['kl'].item():.4f})"
            )

model.eval()
with torch.no_grad():
    x0 = x_val[:1]
    y0 = y_val[:1]

    pred = model(x0, y_true=y0, return_loss=False, sample=False)
    y_mu = pred.y[0, :, 0].cpu()

    pred_s = model(x0, y_true=y0, return_loss=False, sample=True)
    y_samp = pred_s.y[0, :, 0].cpu()

    y_true = y0[0, :, 0].cpu()

plt.figure()
plt.plot(y_true.numpy(), label="y_true")
plt.plot(y_mu.numpy(), label="y_pred_mu")
plt.plot(y_samp.numpy(), label="y_sample", alpha=0.7)
plt.title("DSSM: real vs predição (média e amostra)")
plt.legend()
plt.show()


# Hamiltonian Neural Network

def true_dynamics(z: torch.Tensor) -> torch.Tensor:
    """
    z = [q, p]
    dq/dt = p
    dp/dt = -q
    """
    q = z[:, 0:1]
    p = z[:, 1:2]
    qdot = p
    pdot = -q
    return torch.cat([qdot, pdot], dim=-1)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = HamiltonianNeuralNetwork(dim_q=1, hidden=128, num_layers=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.train()
for step in range(2000):
    z = torch.randn(256, 2, device=device)      # (B,2) = [q,p]
    y_true = true_dynamics(z)                   # (B,2) = [qdot,pdot]

    out = model(z, y_true=y_true, return_loss=True)
    loss = out.losses["total"]

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"Step {step:04d} | Loss: {loss.item():.6f}")

model.eval()

q = torch.linspace(-2, 2, 20, device=device)
p = torch.linspace(-2, 2, 20, device=device)
Q, P = torch.meshgrid(q, p, indexing="ij")

z_grid = torch.stack([Q.flatten(), P.flatten()], dim=-1)  # (400,2)

pred = model(z_grid).y.detach().cpu()  # (400,2)

pred_qdot = pred[:, 0].reshape(Q.shape).numpy()
pred_pdot = pred[:, 1].reshape(P.shape).numpy()

plt.figure(figsize=(6, 6))
plt.quiver(Q.detach().cpu().numpy(), P.detach().cpu().numpy(), pred_qdot, pred_pdot)
plt.xlabel("q")
plt.ylabel("p")
plt.title("Vector field learned by HNN")
plt.show()


z_test = torch.randn(1000, 2, device=device)

out = model(z_test)
H_pred = out.extras["H"].detach().cpu().squeeze(-1)  # (1000,)

plt.hist(H_pred.numpy(), bins=50)
plt.title("Energy distribution learned (H per sample)")
plt.show()

# Latent ODE

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

def make_irregular_times(T: int, t0: float = 0.0, t1: float = 4.0, jitter: float = 0.15):
    t = torch.linspace(t0, t1, T)
    if T > 2:
        noise = (torch.rand(T) - 0.5) * (2.0 * jitter)  # [-jitter, +jitter]
        noise[0] = 0.0
        noise[-1] = 0.0
        t = t + noise
    t = torch.sort(t).values
    return t

def make_batch(B: int, T: int, obs_dim: int):
    t = make_irregular_times(T)  # (T,)

    A     = 0.8 + 0.4 * torch.rand(B, 1, 1)
    w     = 1.0 + 0.5 * torch.rand(B, 1, 1)
    phi   = 2.0 * math.pi * torch.rand(B, 1, 1)
    decay = 0.2 + 0.2 * torch.rand(B, 1, 1)

    tt = t.view(1, T, 1)  # (1,T,1)
    base = A * torch.sin(w * tt + phi) * torch.exp(-decay * tt)  # (B,T,1)

    y_true = base.repeat(1, 1, obs_dim)                          # (B,T,obs_dim)
    y_true = y_true + 0.05 * torch.randn_like(y_true)

    x = y_true.clone()
    return x, t, y_true

obs_dim = 3
model = LatentODE(
    obs_dim=obs_dim,
    latent_dim=32,
    hidden=128,
    ode_method="dopri5",
    enc_method="dopri5",
).to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def beta_schedule(step, warmup=1000, beta_max=1.0):
    return beta_max * min(1.0, step / float(warmup))

B, T = 64, 30
steps = 1500

for step in range(1, steps + 1):
    x, t, y_true = make_batch(B, T, obs_dim)
    x = x.to(device)
    t = t.to(device)
    y_true = y_true.to(device)

    beta = beta_schedule(step, warmup=800, beta_max=1.0)

    out = model(x, t, y_true=y_true, beta_kl=beta, return_loss=True)

    opt.zero_grad(set_to_none=True)
    out.losses["total"].backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % 200 == 0:
        rec = float(out.losses["rec"].detach().cpu())
        kl = float(out.losses["kl"].detach().cpu())
        tot = float(out.losses["total"].detach().cpu())
        print(f"step={step:04d} total={tot:.4f} rec={rec:.4f} kl={kl:.4f} beta={beta:.3f}")

model.eval()
with torch.no_grad():
    x, t, y_true = make_batch(B=4, T=30, obs_dim=obs_dim)
    x, t, y_true = x.to(device), t.to(device), y_true.to(device)

    out = model(x, t, return_loss=False)

    y_hat = out.y                       # (B,T,obs_dim) = mu_x
    mu_x = out.extras["mu_x"]
    log_sigma_x = out.extras["log_sigma_x"]

    print("y_hat:", y_hat.shape)
    print("mu/logsigma:", mu_x.shape, log_sigma_x.shape)

    eps = torch.randn_like(mu_x)
    samples = mu_x + eps * torch.exp(log_sigma_x)  # (B,T,obs_dim)

    mae = (y_hat - y_true).abs().mean().item()
    print("MAE:", mae)

model.train()
t_obs = t.to(device)     # (30,)
x_obs = x.to(device)     # (B,30,obs_dim)

t_new = make_irregular_times(50, t0=0.0, t1=6.0).to(device)

out_new = model.forecast(x_obs, t_obs, t_new)
print(out_new.y.shape)   # (B,50,obs_dim)

def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

@torch.no_grad()
def plot_reconstruction(model, x_obs, t_obs, y_true=None, *, sample_idx=0, dim=0):
    model.eval()
    out = model(x_obs, t_obs, return_loss=False)

    t = to_np(t_obs)
    yhat = to_np(out.y)[sample_idx, :, dim]

    plt.figure()
    plt.plot(t, yhat, label="pred (mu)")

    # banda de incerteza (se existir)
    if "log_sigma_x" in out.extras:
        logs = to_np(out.extras["log_sigma_x"])[sample_idx, :, dim]
        sig = np.exp(logs)
        plt.fill_between(t, yhat - 2 * sig, yhat + 2 * sig, alpha=0.2, label="±2σ")

    if y_true is not None:
        yt = to_np(y_true)[sample_idx, :, dim]
        plt.scatter(t, yt, s=18, label="true", zorder=3)

    plt.xlabel("t")
    plt.ylabel(f"y[dim={dim}]")
    plt.title(f"Reconstruction (sample={sample_idx})")
    plt.legend()
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def plot_forecast(model, x_obs, t_obs, t_new, y_true_new=None, *, sample_idx=0, dim=0):
    model.eval()
    out_new = model.forecast(x_obs, t_obs, t_new)

    t = to_np(t_new)
    yhat = to_np(out_new.y)[sample_idx, :, dim]

    plt.figure()
    plt.plot(t, yhat, label="forecast (mu)")

    if "log_sigma_x" in out_new.extras:
        logs = to_np(out_new.extras["log_sigma_x"])[sample_idx, :, dim]
        sig = np.exp(logs)
        plt.fill_between(t, yhat - 2 * sig, yhat + 2 * sig, alpha=0.2, label="±2σ")

    if y_true_new is not None:
        yt = to_np(y_true_new)[sample_idx, :, dim]
        plt.scatter(t, yt, s=18, label="true", zorder=3)

    plt.xlabel("t")
    plt.ylabel(f"y[dim={dim}]")
    plt.title(f"Forecast (sample={sample_idx})")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_reconstruction(model, x_obs, t_obs, y_true=y_true, sample_idx=0, dim=0)

out_new = model.forecast(x_obs, t_obs, t_new)
plot_forecast(model, x_obs, t_obs, t_new, y_true_new=None, sample_idx=0, dim=0)

# Neural CDE

def make_batch(B=64, T=60, input_dim=2, out_dim=1, device="cpu"):
    t = torch.linspace(0.0, 1.0, T, device=device)  # (T,)
    tt = t[None, :, None].expand(B, T, 1)          # (B,T,1)

    noise = 0.05 * torch.randn(B, T, input_dim, device=device)
    x0 = torch.sin(2 * torch.pi * tt).squeeze(-1)
    x1 = torch.cos(2 * torch.pi * tt).squeeze(-1)
    x = torch.stack([x0, x1], dim=-1) + noise      # (B,T,2)

    y = 0.7 * torch.sin(x[..., 0:1]) + 0.3 * x[..., 1:2]  # (B,T,1)
    return x, t, y

device = "cuda" if torch.cuda.is_available() else "cpu"

# dims
input_dim = 2
hidden_dim = 64
out_dim = 1

model = NeuralCDE(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    out_dim=out_dim,
    f_hidden=128,
    g_hidden=128,
    num_layers=2,
    activation="tanh",
    solver="euler",
    check_strict_t=True,
).to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for step in range(1, 1001):
    x, t, y_true = make_batch(B=64, T=60, input_dim=input_dim, out_dim=out_dim, device=device)

    out = model(x, t, y_true=y_true, return_loss=True)
    loss = out.losses["total"]

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    if step % 200 == 0:
        print(f"step={step:04d} loss={loss.item():.6f}")

model.eval()
with torch.no_grad():
    x, t, y_true = make_batch(B=8, T=60, input_dim=input_dim, out_dim=out_dim, device=device)
    out = model(x, t)
    y_hat = out.y  # (B,T,1)

print("y_hat shape:", tuple(y_hat.shape))
print("y_true shape:", tuple(y_true.shape))
print("example (first batch, first 5 timesteps):")
print(torch.cat([y_hat[0, :5], y_true[0, :5]], dim=-1))  # [pred, true]

# Neural Gaussian Process

def make_data(n: int = 128, noise_std: float = 0.10, device="cpu"):
    x = torch.linspace(-1.0, 1.0, n, device=device).unsqueeze(-1)  # (N,1)
    y_clean = torch.sin(3.0 * x)
    y = y_clean + noise_std * torch.randn_like(y_clean)
    return x, y, y_clean

@torch.no_grad()
def eval_on_grid(model: NeuralGaussianProcess, device="cpu"):
    xg = torch.linspace(-1.5, 1.5, 400, device=device).unsqueeze(-1)  # (G,1)
    out = model(xg)
    mu = out.y  # (G,1)
    logvar = out.extras["logvar"]  # (G,1)
    std = torch.exp(0.5 * logvar)
    return xg, mu, std

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

x_train, y_train, y_clean = make_data(n=160, noise_std=0.10, device=device)

model = NeuralGaussianProcess(
    in_dim=1,
    out_dim=1,
    embed_dim=64,
    mlp_hidden=128,
    mlp_layers=2,
    rff_features=512,
    rff_lengthscale=1.0,
    freeze_rff=True,
    noise_mode="learned",   # learn sigma^2
    alpha=1.0,
    jitter=1e-6,
).to(device)

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.Adam(params, lr=1e-3)

for epoch in range(1, 401):
    model.train()

    model.condition(x_train, y_train)

    out = model(x_train, y_true=y_train, return_loss=True, use_nll=True)
    loss = out.losses["total"]

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    if epoch % 50 == 0:
        with torch.no_grad():
            if model.noise_mode == "learned":
                sigma2 = float(torch.exp(model.log_noise).detach().cpu())
            else:
                sigma2 = float(torch.exp(model.log_noise_buf).detach().cpu())
        print(f"epoch={epoch:04d} nll={float(loss.detach().cpu()):.4f}  sigma2={sigma2:.6f}")

model.eval()
model.condition(x_train, y_train)
xg, mu, std = eval_on_grid(model, device=device)

lo = mu - 2.0 * std
hi = mu + 2.0 * std

print("\nGride Samples (x, mu, std):")
for i in [0, 100, 200, 300, 399]:
    print(
        f"x={float(xg[i]): .3f}  mu={float(mu[i]): .3f}  std={float(std[i]): .3f} "
        f"  2σ=[{float(lo[i]): .3f},{float(hi[i]): .3f}]"
    )

plt.figure()
plt.scatter(x_train.detach().cpu().numpy(), y_train.detach().cpu().numpy(), s=12, label="train")
plt.plot(xg.detach().cpu().numpy(), mu.detach().cpu().numpy(), label="mean")
plt.fill_between(
    xg.squeeze(-1).detach().cpu().numpy(),
    lo.squeeze(-1).detach().cpu().numpy(),
    hi.squeeze(-1).detach().cpu().numpy(),
    alpha=0.25,
    label="±2σ",
)
plt.legend()
plt.title("NeuralGaussianProcess (RFF + Bayesian Linear Head)")
plt.show()

# Neural ODE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def true_solution(y0, t):
    # y(t) = y0 * exp(-t)
    return y0[:, None, :] * torch.exp(-t[None, :, None])

B = 64          # batch size
state_dim = 1
T = 50

t = torch.linspace(0, 2.0, T, device=device)
y0 = torch.randn(B, state_dim, device=device)
y_true = true_solution(y0, t)

model = NeuralODE(
    state_dim=state_dim,
    hidden=64,
    num_layers=3,
    method="rk4",
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

for step in range(2000):
    optimizer.zero_grad()

    out = model(
        y0,
        t,
        y_true=y_true,
        return_loss=True,
    )

    loss = out.losses["total"]
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"step={step:04d} loss={loss.item():.6f}")

model.eval()

with torch.no_grad():
    y0_test = torch.tensor([[2.0]], device=device)
    y_pred = model(y0_test, t).y
    y_exact = true_solution(y0_test, t)

plt.plot(t.cpu(), y_exact[0, :, 0].cpu(), label="Exact")
plt.plot(t.cpu(), y_pred[0, :, 0].cpu(), "--", label="NeuralODE")
plt.legend()
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("Neural ODE learning dy/dt = -y")
plt.show()

# Neural SDE
# dY = theta*(mu - Y) dt + sigma dW

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(7)

T = 101
t0, t1 = 0.0, 1.0
t = torch.linspace(t0, t1, T, device=device)

B = 256
y0 = torch.randn(B, 1, device=device) * 0.5  # (B, D=1)

theta_true = 2.0
mu_true = 0.3
sigma_true = 0.25

def simulate_ou(y0: torch.Tensor, t: torch.Tensor, theta: float, mu: float, sigma: float) -> torch.Tensor:
    B, D = y0.shape
    ys = [y0]
    y = y0
    dt = (t[1:] - t[:-1]).to(y0.dtype)
    for i in range(t.numel() - 1):
        dti = dt[i]
        dW = torch.randn_like(y) * torch.sqrt(dti)
        drift = theta * (mu - y)
        y = y + drift * dti + sigma * dW
        ys.append(y)
    return torch.stack(ys, dim=1)

with torch.no_grad():
    y_true = simulate_ou(y0, t, theta_true, mu_true, sigma_true)  # (B,T,1)

model = NeuralSDE(
    state_dim=1,
    hidden=128,
    num_layers=3,
    diffusion="diag",
    n_paths=16,
    antithetic=True,
    keep_paths=False,
).to(device)

opt = torch.optim.Adam(model.parameters(), lr=2e-3)

steps = 800
for step in range(1, steps + 1):
    out = model(y0=y0, t=t, y_true=y_true, return_loss=True)  # out.y: (B,T,1)
    loss = out.losses["total"]

    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % 100 == 0:
        print(f"step={step:04d} loss={loss.item():.6f}")

with torch.no_grad():
    y0_new = torch.tensor([[0.0], [0.5], [-0.5]], device=device)  # 3 ICs
    out_new = model(y0=y0_new, t=t, return_loss=False)
    y_pred_path = out_new.y
    print("y_pred_path shape:", tuple(y_pred_path.shape))
    print("y(t=0) =", y_pred_path[:, 0, 0].tolist())
    print("y(t=1) =", y_pred_path[:, -1, 0].tolist())

model.keep_paths = True
with torch.no_grad():
    out_mc = model(y0=y0_new, t=t, return_loss=False)
    y_paths = out_mc.y
    mean = y_paths.mean(dim=1)         # (3,T,1)
    std = y_paths.std(dim=1)           # (3,T,1)
    print("MC paths shape:", tuple(y_paths.shape))
    print("mean(t=1) =", mean[:, -1, 0].tolist())
    print("std(t=1)  =", std[:, -1, 0].tolist())

# ODERNN
def make_sine_dataset(
    B: int = 64,
    T: int = 50,
    D: int = 2,
    device: str = "cpu",
):
    dt = torch.rand(T, device=device) * 0.15 + 0.05
    t = torch.cumsum(dt, dim=0)
    t = t - t[0]

    amp = torch.rand(B, 1, 1, device=device) * 0.9 + 0.1
    phase = torch.rand(B, 1, 1, device=device) * 2.0 * torch.pi
    tt = t.view(1, T, 1).expand(B, T, D)

    x_clean = amp * torch.sin(tt + phase)
    x = x_clean + 0.05 * torch.randn_like(x_clean)

    y_true = x_clean
    return x, t, y_true

device = "cuda" if torch.cuda.is_available() else "cpu"
steps = 500
torch.manual_seed(0)

B, T, D = 64, 50, 2
model = ODERNN(input_dim=D, hidden_dim=128).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for step in range(1, steps + 1):
    x, t, y_true = make_sine_dataset(B=B, T=T, D=D, device=device)

    out = model(x, t, y_true=y_true, return_loss=True)
    loss = out.losses["total"]

    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()

    if step % 100 == 0:
        print(
            f"step={step:04d} "
            f"loss={float(loss.detach().cpu()):.4e} "
            f"mse={float(out.losses['mse'].detach().cpu()):.4e}"
        )

model.eval()
with torch.no_grad():
    x, t, y_true = make_sine_dataset(B=8, T=T, D=D, device=device)
    out = model(x, t, return_loss=False)
    y_hat = out.y          # (B,T,D)
    h_path = out.extras["h"]  # (B,T,H)

print("y_hat:", y_hat.shape, "h_path:", h_path.shape, "t:", t.shape)

# SymplecticODE

def ho_vector_field(z: torch.Tensor) -> torch.Tensor:
    q = z[:, 0:1]
    p = z[:, 1:2]
    return torch.cat([p, -q], dim=-1)

def ho_exact(z0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # z0: (N,2), t: (S,)  -> traj: (S,N,2)
    q0 = z0[:, 0:1]
    p0 = z0[:, 1:2]
    ct = torch.cos(t).view(-1, 1, 1)
    st = torch.sin(t).view(-1, 1, 1)
    q = q0.unsqueeze(0) * ct + p0.unsqueeze(0) * st
    p = p0.unsqueeze(0) * ct - q0.unsqueeze(0) * st
    return torch.cat([q, p], dim=-1)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

model = SymplecticODENet(dim_q=1, hidden=128).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def sample_batch(n: int):
    q = (2.0 * torch.rand(n, 1) - 1.0) * 2.0  # [-2,2]
    p = (2.0 * torch.rand(n, 1) - 1.0) * 2.0  # [-2,2]
    z = torch.cat([q, p], dim=-1)
    dz = ho_vector_field(z)
    return z.to(device), dz.to(device)

steps = 3000
batch_size = 512

for step in range(1, steps + 1):
    z, dz_true = sample_batch(batch_size)
    out = model(z, y_true=dz_true, return_loss=True)

    loss = out.losses["total"]
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    if step % 300 == 0:
        print(f"step={step:04d} mse={out.losses['mse'].item():.3e}")

model.eval()

N = 8
z0 = torch.tensor([[1.0, 0.0]], device=device).repeat(N, 1)
z0[:, 0] += 0.2 * torch.linspace(-1, 1, N, device=device)

dt = 0.05
S = 400  # steps
t = torch.arange(S + 1, device=device, dtype=z0.dtype) * dt

traj_pred = model.rollout(z0, dt=dt, steps=S)          # (S+1,N,2)
traj_true = ho_exact(z0, t)                            # (S+1,N,2)

mse_traj = torch.mean((traj_pred - traj_true) ** 2).item()
mse_q = torch.mean((traj_pred[..., 0] - traj_true[..., 0]) ** 2).item()
mse_p = torch.mean((traj_pred[..., 1] - traj_true[..., 1]) ** 2).item()

H_true = 0.5 * (traj_true[..., 0] ** 2 + traj_true[..., 1] ** 2)
H_pred = 0.5 * (traj_pred[..., 0] ** 2 + traj_pred[..., 1] ** 2)
energy_drift = (H_pred - H_pred[0]).abs().mean().item()

print("\n--- Rollout metrics ---")
print(f"MSE traj total: {mse_traj:.3e}")
print(f"MSE q:          {mse_q:.3e}")
print(f"MSE p:          {mse_p:.3e}")
print(f"Mean |ΔH_pred|:  {energy_drift:.3e}")

print("\nExample (first trajectory):")
for k in [0, 50, 100, 200, 400]:
    qp_t = traj_true[k, 0].detach().cpu().tolist()
    qp_p = traj_pred[k, 0].detach().cpu().tolist()
    print(f"t={float(t[k]):.2f}  true={qp_t}  pred={qp_p}")

# Symplectic RNN

def harmonic_oscillator_truth(z0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    z0: (B,2) com [q0,p0]
    t:  (T,)
    retorna: (B,T,2) com [q(t), p(t)]
    """
    q0 = z0[:, 0:1]
    p0 = z0[:, 1:2]

    tt = t.view(1, -1, 1)  # (1,T,1)
    ct = torch.cos(tt)
    st = torch.sin(tt)

    q = q0[:, None, :] * ct + p0[:, None, :] * st
    p = p0[:, None, :] * ct - q0[:, None, :] * st
    return torch.cat([q, p], dim=-1)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

Tn = 101
t = torch.linspace(0.0, 6.0, Tn, device=device)  # (T,)

B = 256
q0 = 2.0 * torch.rand(B, 1, device=device) - 1.0
p0 = 2.0 * torch.rand(B, 1, device=device) - 1.0
z0 = torch.cat([q0, p0], dim=-1)  # (B,2)

y_true = harmonic_oscillator_truth(z0, t)  # (B,T,2)

model = SymplecticRNN(dim_q=1, hidden=128, num_layers=2, activation="tanh").to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for step in range(1, 2001):
    opt.zero_grad(set_to_none=True)

    out = model(z0, t, y_true=y_true, return_loss=True)
    loss = out.losses["total"]
    loss.backward()
    opt.step()

    if step % 200 == 0:
        print(f"step={step:04d} loss={loss.item():.4e}")

model.train()
z0_test = torch.tensor([[0.7, -0.2]], device=device)  # (1,2)
pred = model(z0_test, t).y                          # (1,T,2)
true = harmonic_oscillator_truth(z0_test, t)         # (1,T,2)

mse = torch.mean((pred - true) ** 2).item()
print(f"test mse = {mse:.4e}")

q_pred = pred[0, :, 0].detach().cpu()
p_pred = pred[0, :, 1].detach().cpu()
q_true = true[0, :, 0].detach().cpu()
p_true = true[0, :, 1].detach().cpu()

print("First 5 points (q_pred, q_true):")
for i in range(5):
    print(float(q_pred[i]), float(q_true[i]))
