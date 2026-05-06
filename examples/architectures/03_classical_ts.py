import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from pinneaple_models.classical_ts.arima import ARIMA
from pinneaple_models.classical_ts.ekf import ExtendedKalmanFilter
from pinneaple_models.classical_ts.enkf import EnsembleKalmanFilter
from pinneaple_models.classical_ts.kalman import KalmanFilter
from pinneaple_models.classical_ts.tcn import TCN
from pinneaple_models.classical_ts.ukf import UnscentedKalmanFilter
from pinneaple_models.classical_ts.var import VAR

# ARIMA

torch.manual_seed(0)

T = 200
B = 1
D = 1

time = torch.arange(T).float()
trend = 0.05 * time
signal = torch.sin(0.2 * time)
noise = 0.2 * torch.randn(T)

series = trend + signal + noise
series = series.view(1, T, 1)  # (B,T,D)

model = ARIMA(
    dim=1,
    p=3,
    d=1,
    q=0,
    l2=1e-4
)

model.fit(series)
output = model(series, y_true=series, return_loss=True)

print("Loss original-space MSE:", output.losses["mse"].item())
print("Loss differenced-space MSE:", output.losses["mse_d"].item())

y_pred = output.y  # (B, T - (p+d), D)

steps = 30
forecast = model.forecast(series, steps=steps)  # (B,steps,D)

plt.figure(figsize=(10,4))

plt.plot(series[0,:,0].numpy(), label="History")
plt.plot(
    range(T - y_pred.shape[1], T),
    y_pred[0,:,0].numpy(),
    label="One-step fit"
)

plt.plot(
    range(T, T + steps),
    forecast[0,:,0].numpy(),
    label="Forecast"
)

plt.legend()
plt.title("ARIMA Example")
plt.show()

# Extended Kalman Filter

torch.manual_seed(0)
device = "cpu"
dtype = torch.float32

B, T = 4, 30
n = 2
m = 1
du = 1

def f(x: torch.Tensor, u: torch.Tensor | None) -> torch.Tensor:
    pos, vel = x[:, 0], x[:, 1]
    uu = 0.0 if u is None else u[:, 0]
    pos_next = pos + vel + 0.1 * torch.sin(pos) + 0.5 * uu
    vel_next = vel + 0.1 * torch.sin(vel)
    return torch.stack([pos_next, vel_next], dim=1)

def h(x: torch.Tensor) -> torch.Tensor:
    pos = x[:, 0]
    y = pos**2
    return y.unsqueeze(1)  # (B,1)

def F_jac(x: torch.Tensor, u: torch.Tensor | None) -> torch.Tensor:
    pos, vel = x[:, 0], x[:, 1]
    dpos_dpos = 1.0 + 0.1 * torch.cos(pos)
    dpos_dvel = torch.ones_like(vel)
    dvel_dpos = torch.zeros_like(pos)
    dvel_dvel = 1.0 + 0.1 * torch.cos(vel)

    F = torch.zeros((x.shape[0], 2, 2), device=x.device, dtype=x.dtype)
    F[:, 0, 0] = dpos_dpos
    F[:, 0, 1] = dpos_dvel
    F[:, 1, 0] = dvel_dpos
    F[:, 1, 1] = dvel_dvel
    return F

def H_jac(x: torch.Tensor) -> torch.Tensor:
    pos = x[:, 0]
    H = torch.zeros((x.shape[0], 1, 2), device=x.device, dtype=x.dtype)
    H[:, 0, 0] = 2.0 * pos  # d(pos^2)/dpos
    H[:, 0, 1] = 0.0        # d(pos^2)/dvel
    return H

Q = torch.diag(torch.tensor([1e-3, 1e-3], dtype=dtype, device=device))  # (n,n)
R = torch.diag(torch.tensor([5e-2], dtype=dtype, device=device))        # (m,m)

x_true = torch.zeros((B, T, n), device=device, dtype=dtype)
u = torch.randn((B, T, du), device=device, dtype=dtype) * 0.3

x0_true = torch.tensor([[0.5, 0.2]], device=device, dtype=dtype).expand(B, -1)
x_true[:, 0, :] = x0_true

for t in range(1, T):
    x_true[:, t, :] = f(x_true[:, t-1, :], u[:, t, :])

y_clean = torch.stack([h(x_true[:, t, :]) for t in range(T)], dim=1).squeeze(-1)  # (B,T,1)
y = y_clean + torch.randn_like(y_clean) * torch.sqrt(R[0, 0])                     # (B,T,1)

y = y.unsqueeze(-1)

ekf = ExtendedKalmanFilter(
    f=f,
    h=h,
    Q=Q,
    R=R,
    F_jac=F_jac,
    H_jac=H_jac,
).to(device)

x0 = torch.zeros((B, n), device=device, dtype=dtype)
P0 = torch.eye(n, device=device, dtype=dtype) * 1.0

out = ekf(
    y=y,          # (B,T,m)
    u=u,          # (B,T,du)  (optional)
    x0=x0,        # (B,n)     (optional)
    P0=P0,        # (n,n) ou (B,n,n) (optional)
    return_gain=True,
)

x_filt = out.y                 # (B,T,n)
P_filt = out.extras["P"]       # (B,T,n,n)
K = out.extras["K"]            # (B,T,n,m)

print("x_filt:", x_filt.shape)
print("P_filt:", P_filt.shape)
print("K:", K.shape)

mse = torch.mean((x_filt - x_true) ** 2).item()
print("MSE filter state vs truth:", mse)

# Ensemble Kalman Filter

dt = 0.1
A = torch.tensor([[1.0, dt],
                  [0.0, 1.0]])
H = torch.tensor([[1.0, 0.0]])

def f(x: torch.Tensor, u=None):
    # x: (B*M, n)
    return x @ A.T

def h(x: torch.Tensor):
    # x: (B*M, n)
    return x @ H.T

Q = 0.01 * torch.eye(2)
R = 0.05 * torch.eye(1)

T = 100
B = 1  # batch size
n = 2
m = 1

x_true = torch.zeros(B, T, n)
y_obs = torch.zeros(B, T, m)

x = torch.tensor([[0.0, 1.0]])

Lq = torch.linalg.cholesky(Q)
Lr = torch.linalg.cholesky(R)

for t in range(T):
    w = torch.randn(1, n) @ Lq.T
    x = f(x) + w
    v = torch.randn(1, m) @ Lr.T
    y = h(x) + v

    x_true[:, t, :] = x
    y_obs[:, t, :] = y

enkf = EnsembleKalmanFilter(
    f=f,
    h=h,
    Q=Q,
    R=R,
    ensemble_size=100,
)

out = enkf(y_obs)

x_est = out.y  # (B,T,n)

plt.figure(figsize=(10,5))
plt.plot(x_true[0,:,0], label="True Position")
plt.plot(x_est[0,:,0], label="Estimated Position")
plt.scatter(range(T), y_obs[0,:,0], label="Observations", s=10)
plt.legend()
plt.title("Ensemble Kalman Filter Example")
plt.show()

# Kalman filter

dt = 0.1
n = 2
m = 1

A = torch.tensor([[1.0, dt],
                  [0.0, 1.0]])

H = torch.tensor([[1.0, 0.0]])

Q = 0.01 * torch.eye(n)
R = 0.1 * torch.eye(m)

kf = KalmanFilter(A=A, H=H, Q=Q, R=R)

B = 1        # batch size
T = 100      # time

x_true = torch.zeros(B, T, n)
y_obs = torch.zeros(B, T, m)

x = torch.tensor([[0.0, 1.0]])

for t in range(T):
    w = torch.randn(B, n) @ torch.linalg.cholesky(Q)
    x = (A @ x.unsqueeze(-1)).squeeze(-1) + w
    x_true[:, t, :] = x

    v = torch.randn(B, m) @ torch.linalg.cholesky(R)
    y = (H @ x.unsqueeze(-1)).squeeze(-1) + v
    y_obs[:, t, :] = y

output = kf(y_obs)

x_filt = output.y            # (B,T,n)
P_filt = output.extras["P"]

plt.figure(figsize=(10,6))

plt.plot(x_true[0,:,0], label="Posição verdadeira")
plt.plot(y_obs[0,:,0], label="Observação ruidosa", alpha=0.5)
plt.plot(x_filt[0,:,0], label="Posição filtrada")

plt.legend()
plt.title("Kalman Filter - Estimação de posição")
plt.show()

# TCN

def generate_batch(batch_size=32, T=100):
    t = torch.linspace(0, 8*np.pi, T)

    x1 = torch.sin(t)
    x2 = torch.cos(t)

    x = torch.stack([x1, x2], dim=-1)  # (T,2)
    x = x.unsqueeze(0).repeat(batch_size, 1, 1)  # (B,T,2)

    y = torch.roll(x, shifts=-1, dims=1)

    return x.float(), y.float()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TCN(
    in_dim=2,
    out_dim=2,
    channels=[32, 32, 32],
    kernel=3,
    dropout=0.1,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(200):
    x, y = generate_batch(batch_size=64, T=100)
    x, y = x.to(device), y.to(device)

    out = model(x, y_true=y, return_loss=True)
    loss = out.losses["total"]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

model.eval()

x_test, _ = generate_batch(batch_size=1, T=100)
x_test = x_test.to(device)

with torch.no_grad():
    out = model(x_test)
    y_pred = out.y  # (1,T,2)

print("Prediction shape:", y_pred.shape)

# Unscented Kalman Filter

def f_pendulum(x: torch.Tensor, u: torch.Tensor | None):
    """
    x: (B,2)  -> [theta, omega]
    u: (B,1)  -> torque (optional)
    """
    dt = 0.05
    g = 9.81
    L = 1.0

    theta = x[:, 0]
    omega = x[:, 1]

    torque = 0.0 if u is None else u[:, 0]

    theta_next = theta + dt * omega
    omega_next = omega + dt * (-(g/L) * torch.sin(theta) + torque)

    return torch.stack([theta_next, omega_next], dim=1)

def h_pendulum(x: torch.Tensor):
    theta = x[:, 0]
    return theta.unsqueeze(-1)

torch.manual_seed(0)

B = 1       # batch size
T = 200     # time
n = 2       # state
m = 1       # measure

x_true = torch.zeros(B, T, n)
y_obs = torch.zeros(B, T, m)

x = torch.tensor([[1.0, 0.0]])

for t in range(T):
    x = f_pendulum(x, None)
    x_true[:, t] = x

    noise = 0.05 * torch.randn_like(x[:, :1])
    y_obs[:, t] = h_pendulum(x) + noise


Q = 1e-4 * torch.eye(n)
R = 5e-3 * torch.eye(m)

ukf = UnscentedKalmanFilter(
    f=f_pendulum,
    h=h_pendulum,
    Q=Q,
    R=R,
    alpha=1e-3,
    beta=2.0,
    kappa=0.0,
)

output = ukf(
    y=y_obs,
    x0=torch.tensor([0.5, 0.0]),
    P0=0.5 * torch.eye(n),
    return_gain=True,
)

x_est = output.y          # (B,T,n)
P_est = output.extras["P"]
K_est = output.extras["K"]

# VAR

torch.manual_seed(42)

B = 16
T = 200
D = 2
p = 2

A1 = torch.tensor([[0.6,  0.1],
                   [-0.2, 0.5]])

A2 = torch.tensor([[0.2,  0.0],
                   [0.1,  0.3]])

c = torch.tensor([0.1, -0.05])

x = torch.zeros(B, T, D)

x[:, 0, :] = torch.randn(B, D)
x[:, 1, :] = torch.randn(B, D)

for t in range(2, T):
    noise = 0.05 * torch.randn(B, D)
    x[:, t, :] = (
        c
        + x[:, t-1, :] @ A1.T
        + x[:, t-2, :] @ A2.T
        + noise
    )

model = VAR(dim=D, p=p, l2=1e-4, use_bias=True)
model.fit(x)

print("Fitted:", model._fitted)

out = model(x, y_true=x, return_loss=True)

print("Prediction shape:", out.y.shape)  # (B, T-p, D)
print("Train MSE:", out.losses["mse"].item())

steps = 20
x_hist = x[:, -p:, :]
preds = model.forecast(x_hist, steps=steps)

print("Forecast shape:", preds.shape)  # (B, steps, D)

seq = 0

plt.figure(figsize=(10,4))
plt.plot(range(T), x[seq, :, 0].numpy(), label="Real")
plt.plot(range(T, T+steps), preds[seq, :, 0].numpy(), "--", label="Forecast")
plt.legend()
plt.title("VAR Forecast Example")
plt.show()
