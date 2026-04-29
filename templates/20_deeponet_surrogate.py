"""20_deeponet_surrogate.py — DeepONet surrogate for parametric ODEs.

Demonstrates:
- DeepONet architecture (branch + trunk networks) from pinneaple_models
- Operator learning: map forcing function f → solution u
- Query-point evaluation: trunk inputs are the spatial coordinates
- Mean/std normalisation of branch inputs via InputNormaliser
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_models.neural_operators.deeponet import DeepONet
from pinneaple_models.utils import InputNormaliser


# ---------------------------------------------------------------------------
# Problem: 1D linear ODE  u'' = f(x),  x∈[0,1],  u(0)=u'(0)=0
# Exact solution via double integration: u(x) = ∫₀ˣ ∫₀ˢ f(t) dt ds
# Branch input: discrete forcing f evaluated at M_branch sensor points
# Trunk input:  query coordinate x  (scalar in [0,1])
# ---------------------------------------------------------------------------

M_BRANCH = 50    # branch sensor resolution
N_TR     = 2000  # training samples
N_TE     = 200   # test samples
N_QUERY  = 60    # query points per sample during training


def double_integral(f_vals: np.ndarray, x_query: np.ndarray,
                    x_sensors: np.ndarray) -> np.ndarray:
    """Compute ∫₀ˣ ∫₀ˢ f(t) dt ds via trapezoidal rule."""
    results = []
    for x in x_query:
        # Inner integral  ∫₀ˢ f(t) dt   for s ∈ [0, x]
        s_pts = x_sensors[x_sensors <= x]
        if len(s_pts) < 2:
            results.append(0.0)
            continue
        # Interpolate f to s_pts
        f_interp = np.interp(s_pts, x_sensors, f_vals)
        inner = np.trapz(f_interp, s_pts)           # ∫₀ˣ f dt  (single value)
        # Actually we need ∫₀ˣ ∫₀ˢ ... ds; build running integral
        inner_arr = []
        for s in s_pts:
            t_pts = s_pts[s_pts <= s]
            if len(t_pts) < 2:
                inner_arr.append(0.0)
            else:
                f_t = np.interp(t_pts, x_sensors, f_vals)
                inner_arr.append(np.trapz(f_t, t_pts))
        results.append(np.trapz(np.array(inner_arr), s_pts))
    return np.array(results, dtype=np.float32)


def generate_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x_sensors = np.linspace(0, 1, M_BRANCH)
    x_query   = np.linspace(0, 1, N_QUERY)

    branch_inputs = []
    trunk_inputs  = []
    targets       = []

    for _ in range(n):
        # Random forcing: sum of sinusoids
        f_vals = np.zeros(M_BRANCH)
        for _ in range(rng.integers(1, 4)):
            freq  = rng.uniform(1, 5)
            amp   = rng.uniform(-2, 2)
            phase = rng.uniform(0, 2 * np.pi)
            f_vals += amp * np.sin(freq * np.pi * x_sensors + phase)

        u_vals = double_integral(f_vals, x_query, x_sensors)

        # Stack: branch is (M_branch,), trunk is (N_query, 1), target is (N_query,)
        branch_inputs.append(f_vals.astype(np.float32))
        trunk_inputs.append(x_query[:, None].astype(np.float32))
        targets.append(u_vals.astype(np.float32))

    return (np.array(branch_inputs),     # (n, M_branch)
            np.array(trunk_inputs),       # (n, N_query, 1)
            np.array(targets))            # (n, N_query)


def main():
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Dataset -------------------------------------------------------------
    print("Generating DeepONet dataset ...")
    b_tr, t_tr, y_tr = generate_dataset(N_TR, seed=0)
    b_te, t_te, y_te = generate_dataset(N_TE, seed=1)

    # Convert
    B_tr = torch.tensor(b_tr, device=device)   # (N, M)
    T_tr = torch.tensor(t_tr, device=device)   # (N, Q, 1)
    Y_tr = torch.tensor(y_tr, device=device)   # (N, Q)

    B_te = torch.tensor(b_te, device=device)
    T_te = torch.tensor(t_te, device=device)
    Y_te = torch.tensor(y_te, device=device)

    # --- Normalise branch inputs ---------------------------------------------
    normaliser = InputNormaliser()
    normaliser.fit(B_tr)
    B_tr_n = normaliser.transform(B_tr)
    B_te_n = normaliser.transform(B_te)

    # --- DeepONet ------------------------------------------------------------
    model = DeepONet(
        branch_input_dim=M_BRANCH,
        trunk_input_dim=1,
        hidden_dim=64,
        n_basis=32,
        branch_layers=4,
        trunk_layers=4,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"DeepONet parameters: {n_params:,}")

    # --- Training ------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    batch_size = 64
    n_epochs   = 300
    history    = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        idx = torch.randperm(N_TR, device=device)
        epoch_loss = 0.0
        n_batches  = 0
        for i in range(0, N_TR, batch_size):
            bi = idx[i: i + batch_size]
            b  = B_tr_n[bi]                       # (bs, M)
            t  = T_tr[bi].reshape(-1, 1)           # (bs*Q, 1)
            y  = Y_tr[bi].reshape(-1)              # (bs*Q,)

            # Repeat branch for each query point
            bs_actual = b.shape[0]
            b_rep = b.unsqueeze(1).expand(-1, N_QUERY, -1).reshape(-1, M_BRANCH)

            optimizer.zero_grad()
            y_hat = model(b_rep, t).squeeze(-1)    # (bs*Q,)
            loss  = (y_hat - y).pow(2).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches  += 1
        scheduler.step()
        history.append(epoch_loss / n_batches)

        if epoch % 60 == 0:
            print(f"  epoch {epoch:3d} | loss = {history[-1]:.4e}")

    # --- Evaluation ----------------------------------------------------------
    model.eval()
    with torch.no_grad():
        b_rep_te = B_te_n.unsqueeze(1).expand(-1, N_QUERY, -1).reshape(-1, M_BRANCH)
        t_flat   = T_te.reshape(-1, 1)
        y_hat_te = model(b_rep_te, t_flat).squeeze(-1).reshape(N_TE, N_QUERY).cpu().numpy()
    y_true_te = Y_te.cpu().numpy()

    rel_l2 = np.sqrt(((y_hat_te - y_true_te) ** 2).sum(1)) / \
             (np.sqrt((y_true_te ** 2).sum(1)) + 1e-8)
    print(f"\nTest relative L2: {rel_l2.mean():.4e} ± {rel_l2.std():.4e}")

    # --- Visualisation -------------------------------------------------------
    x_q = np.linspace(0, 1, N_QUERY)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, idx_ex, label in zip(
        axes[:2],
        [np.argmin(rel_l2), np.argmax(rel_l2)],
        ["Best", "Worst"],
    ):
        ax.plot(x_q, y_true_te[idx_ex], "k-",  label="True u(x)")
        ax.plot(x_q, y_hat_te[idx_ex],  "r--", label="DeepONet")
        ax.set_title(f"{label} (L2={rel_l2[idx_ex]:.3e})")
        ax.set_xlabel("x")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[2].semilogy(history)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("MSE loss")
    axes[2].set_title(f"Training  |  mean L2={rel_l2.mean():.3e}")
    axes[2].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("20_deeponet_surrogate_result.png", dpi=120)
    print("Saved 20_deeponet_surrogate_result.png")


if __name__ == "__main__":
    main()
