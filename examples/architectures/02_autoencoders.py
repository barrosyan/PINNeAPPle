import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

import matplotlib.pyplot as plt
import numpy as np

from pinneaple_models.autoencoders.ae_2d import Autoencoder2D
from pinneaple_models.autoencoders.ae_rom_hybrid import AEROMHybrid
from pinneaple_models.autoencoders.dense_ae import DenseAutoencoder
from pinneaple_models.autoencoders.kae import KAEAutoencoder
from pinneaple_models.autoencoders.koopman_pi_ae import PhysicsInformedKoopmanAutoencoder
from pinneaple_models.autoencoders.vae import VariationalAutoencoder

# Autoencoder 2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

H, W = 64, 64
in_channels = 1
latent_dim = 32
batch_size = 16
n_epochs = 10

def generate_field(batch_size, H, W):
    x = torch.linspace(-1, 1, W)
    y = torch.linspace(-1, 1, H)
    X, Y = torch.meshgrid(x, y, indexing="xy")

    fields = []
    for _ in range(batch_size):
        cx = torch.randn(1) * 0.5
        cy = torch.randn(1) * 0.5
        sigma = torch.rand(1) * 0.3 + 0.1

        field = torch.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
        fields.append(field.unsqueeze(0))  # add channel dim

    return torch.stack(fields)  # (B,1,H,W)

model = Autoencoder2D(
    in_channels=in_channels,
    latent_dim=latent_dim,
    img_size=(H, W),
    base_channels=32,
    norm="group",
    output_activation="sigmoid"
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.train()

for epoch in range(n_epochs):
    x = generate_field(batch_size, H, W).to(device)

    optimizer.zero_grad()

    z = model.encode(x)
    x_rec = model.decode(z)

    loss = criterion(x_rec, x)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{n_epochs} | Loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    x = generate_field(1, H, W).to(device)
    z = model.encode(x)
    x_rec = model.decode(z)

x = x.cpu().squeeze().numpy()
x_rec = x_rec.cpu().squeeze().numpy()

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(x, cmap="viridis")
plt.colorbar()

plt.subplot(1,2,2)
plt.title("Reconstruction")
plt.imshow(x_rec, cmap="viridis")
plt.colorbar()

plt.tight_layout()
plt.show()

# AEROMHybrid

class ToyDynDataset(Dataset):
    def __init__(self, N=2000, input_dim=64, K=5, dt=0.1, noise=0.01, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x0 = torch.randn(N, input_dim, generator=g)
        self.dt = torch.full((N,), float(dt))
        self.K = K
        self.noise = noise

        W = torch.randn(input_dim, input_dim, generator=g) / (input_dim ** 0.5)
        self.W = W

        xs = []
        x = self.x0.clone()
        for _ in range(K):
            dx = torch.tanh(x @ self.W.t())
            x = x + dt * dx + noise * torch.randn_like(x, generator=g)
            xs.append(x.clone())
        self.x_seq = torch.stack(xs, dim=1)

    def __len__(self):
        return self.x0.shape[0]

    def __getitem__(self, i):
        return {
            "x": self.x0[i],                 # (D,)
            "x_next": self.x_seq[i, 0],      # (D,)
            "x_seq": self.x_seq[i],          # (K,D)
            "dt": self.dt[i],                # ()
            "dt_seq": self.dt[i].repeat(self.K),  # (K,)
        }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 64
latent_dim = 16
K_rollout = 5

ds = ToyDynDataset(N=3000, input_dim=input_dim, K=K_rollout, dt=0.1)
dl = DataLoader(ds, batch_size=128, shuffle=True, drop_last=True)

model = AEROMHybrid(
    input_dim=input_dim,
    latent_dim=latent_dim,
    hidden=[128, 128],
    activation="gelu",
    rom_weight=1.0,
    pred_weight=1.0,
    rollout_latent_weight=0.5,
    rollout_x_weight=0.5,
    stability_weight=1e-3,
    stability_margin=0.0,
    ls_ridge=1e-6,
    ls_update_in_forward=False,
    detach_ls_targets=True,
).to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(1, 6):
    total = 0.0
    for batch in dl:
        x = batch["x"].to(device)
        x_next = batch["x_next"].to(device)
        x_seq = batch["x_seq"].to(device)          # (B,K,D)
        dt = batch["dt"].to(device)                # (B,)
        dt_seq = batch["dt_seq"].to(device)        # (B,K) or (K,)

        opt.zero_grad(set_to_none=True)

        out_ae = model(x)
        loss_ae = out_ae.losses["total"]

        out_1 = model(x, x_next=x_next, dt=dt)
        loss_1 = out_1.losses["total"]

        out_r = model(x, x_seq=x_seq, dt_seq=dt_seq)
        loss_r = out_r.losses["total"]

        loss = 0.2 * loss_ae + 0.4 * loss_1 + 0.4 * loss_r

        loss.backward()
        opt.step()

        total += float(loss.item())

    print(f"epoch {epoch} | loss {total/len(dl):.6f} "
            f"| recon {float(out_ae.losses['recon']):.6f}")

model.eval()
batch = next(iter(dl))
with torch.no_grad():
    x = batch["x"].to(device)
    x_next = batch["x_next"].to(device)
    dt = batch["dt"].to(device)

    z = model.encode(x)
    z_next = model.encode(x_next)

    stats = model.fit_rom_ls(z=z, z_next=z_next, dt=dt, ridge=1e-6)
    print("LS updated K,c. K norm:", stats["K"].norm().item(), "c norm:", stats["c"].norm().item())

model.eval()
with torch.no_grad():
    x0 = torch.randn(4, input_dim, device=device)
    out = model(x0)
    z0 = out.z
    x0_hat = out.x_hat

    z1 = model.latent_step(z0, dt=0.1)
    x1 = model.decode(z1)

    K = 10
    z = z0
    xs = []
    for _ in range(K):
        z = model.latent_step(z, dt=0.1)
        xs.append(model.decode(z))
    x_roll = torch.stack(xs, dim=1)  # (B,K,D)

print("ok:", x0_hat.shape, x1.shape, x_roll.shape)

# DenseAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 100
latent_dim = 10
hidden_layers = [256, 128]
batch_size = 64
epochs = 20
lr = 1e-3

N = 2000
X = torch.randn(N, input_dim)

dataset = TensorDataset(X)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = DenseAutoencoder(
    input_dim=input_dim,
    latent_dim=latent_dim,
    hidden=hidden_layers,
    activation="gelu",
).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

model.train()
for epoch in range(epochs):
    total_loss = 0.0

    for (batch,) in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        z = model.encode(batch)
        recon = model.decode(z)

        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.6f}")

model.eval()
with torch.no_grad():
    sample = X[:5].to(device)
    z_sample = model.encode(sample)
    recon_sample = model.decode(z_sample)

print("\nLatent shape:", z_sample.shape)
print("Recon shape:", recon_sample.shape)

# KAEAutoencoder

torch.manual_seed(42)

N = 5000
input_dim = 20
latent_dim = 5

X = torch.randn(N, input_dim)
X = X @ torch.randn(input_dim, input_dim) * 0.2

dataset = TensorDataset(X)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = KAEAutoencoder(
    input_dim=input_dim,
    latent_dim=latent_dim,
    hidden=[128, 64],
    activation="gelu",
    mmd_weight=10.0,
    mmd_sigmas=[0.1, 0.5, 1.0],
    prior="normal",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 20

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for (x,) in loader:
        x = x.to(device)

        z = model.encode(x)
        x_hat = model.decode(z)

        losses = model.loss_from_parts(
            x_hat=x_hat,
            z=z,
            x=x,
        )

        loss = losses["total"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1:03d} | Loss: {total_loss/len(loader):.6f} "
          f"| Recon: {losses['recon'].item():.6f} "
          f"| MMD: {losses['mmd'].item():.6f}")

model.eval()

with torch.no_grad():
    x_sample = X[:10].to(device)
    z_sample = model.encode(x_sample)
    x_recon = model.decode(z_sample)

print("Latent sample shape:", z_sample.shape)
print("Reconstruction shape:", x_recon.shape)


# Physics Informed Koopman Autoencoder


class LinearLatentToy(Dataset):
    def __init__(self, N=5000, D=32, H=5, noise=0.01, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.N, self.D, self.H = N, D, H
        M = torch.randn(D, D, generator=g) * 0.05
        M = M - 0.10 * torch.eye(D)
        self.M = M
        self.noise = noise

        self.x0 = torch.randn(N, D, generator=g)

        self.dt = 0.8 + 0.4 * torch.rand(N, H, generator=g)  # (N,H)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = self.x0[idx].clone()
        H = self.H
        xs = []
        for k in range(H):
            x = x @ self.M.t() + self.noise * torch.randn_like(x)
            xs.append(x.clone())
        x_next = torch.stack(xs, dim=0)      # (H, D)
        dt = self.dt[idx].clone()           # (H,)
        return x_next[0] * 0 + self.x0[idx], x_next, dt

device="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

D = 32          # input_dim
zdim = 8        # latent_dim
H = 5           # rollout horizon
batch_size = 128

ds = LinearLatentToy(N=8000, D=D, H=H, noise=0.01, seed=123)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

model = PhysicsInformedKoopmanAutoencoder(
    input_dim=D,
    latent_dim=zdim,
    hidden=[128, 128],
    activation="gelu",
    koopman_weight=1.0,
    pred_weight=1.0,
    rollout_weight=0.5,
    stability_weight=0.1,
    use_affine=True,
    use_generator_A=False,  # se True, usa A e matrix_exp(A*dt)
    ls_update=False,        # se True, faz LS no K em batches (EDMD-like)
    ls_ridge=1e-6,
    power_iters=10,
).to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(5):
    running = {}
    for x, x_next, dt in dl:
        x = x.to(device)                 # (B, D)
        x_next = x_next.to(device)       # (B, H, D)
        dt = dt.to(device)               # (B, H)

        out = model(x, x_next=x_next, dt=dt)
        loss = out.losses["total"]

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # logging simples
        for k, v in out.losses.items():
            running[k] = running.get(k, 0.0) + float(v.detach().cpu())

    n = len(dl)
    msg = " | ".join([f"{k}={running[k]/n:.4f}" for k in ["total","recon","koopman","pred_x","rollout_x","stability"] if k in running])
    print(f"epoch {epoch}: {msg}")


model.eval()

B, D = 4, model.input_dim
x = torch.randn(B, D, device=device)

out_recon = model(x)
print("recon x_hat:", out_recon.x_hat.shape, "z:", out_recon.z.shape)

dt = torch.tensor(1.0, device=device)
K_eff = model._effective_operator(dt)  # (zdim,zdim)
z = model.encode(x)
z1 = model._apply_K(z, K_eff)
x1_pred = model.decode(z1)
print("1-step pred:", x1_pred.shape)

H = 5
dts = torch.ones(H, device=device)  # (H,) or (B,H)
z_roll = z
x_preds = []
for k in range(H):
    Kk = model._effective_operator(dts[k])
    z_roll = model._apply_K(z_roll, Kk)
    x_preds.append(model.decode(z_roll))
x_rollout = torch.stack(x_preds, dim=1)  # (B,H,D)
print("rollout pred:", x_rollout.shape)

# Variational Autoencoder

input_dim = 20
n_samples = 5000

X = np.random.randn(n_samples, input_dim)
X = X @ np.random.randn(input_dim, input_dim) * 0.1
X = torch.tensor(X, dtype=torch.float32)

dataset = TensorDataset(X)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = VariationalAutoencoder(
    input_dim=input_dim,
    latent_dim=4,
    hidden=[128, 64],
    beta=1.0,
    activation="gelu",
    recon_reduction="sum",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 30

for epoch in range(n_epochs):
    model.train()
    total_loss = 0.0

    for (batch,) in loader:
        batch = batch.to(device)

        out = model(batch)
        loss = out.losses["total"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1:03d} | Loss: {total_loss/len(loader):.6f} | "
          f"Recon: {out.losses['recon'].item():.6f} | "
          f"KL: {out.losses['kl'].item():.6f}")

model.eval()
with torch.no_grad():
    x_sample = X[:5].to(device)
    out = model(x_sample)

    print("Reconstrução shape:", out.x_hat.shape)
    print("Latente shape:", out.z.shape)
    print("Mu shape:", out.extras["mu"].shape)
