"""
Training utilities — wraps PyTorch training loops for the UI.
Returns history dicts that can be plotted by the app.
"""
from __future__ import annotations
from typing import Any, Callable, Dict, Generator, List, Optional
import numpy as np


def build_pinn_loss(problem: Dict) -> Callable:
    """Return a simple physics residual loss function for a known problem."""
    import torch
    import torch.nn as nn

    pde_kind = problem.get("_preset_key", "")

    def burgers_residual(model, pts):
        pts = pts.requires_grad_(True)
        u   = model(pts)
        grads = torch.autograd.grad(u.sum(), pts, create_graph=True)[0]
        u_t = grads[:, 1:2]; u_x = grads[:, 0:1]
        nu  = float(problem.get("params", {}).get("nu", 0.01))
        u_xx = torch.autograd.grad(u_x.sum(), pts, create_graph=True)[0][:, 0:1]
        return (u_t + u * u_x - nu * u_xx).pow(2).mean()

    def generic_residual(model, pts):
        pts = pts.requires_grad_(True)
        u = model(pts)
        return u.pow(2).mean() * 0.0  # placeholder — zero loss

    return burgers_residual if "burgers" in pde_kind else generic_residual


def train_mlp(
    problem:       Dict,
    n_epochs:      int        = 500,
    lr:            float      = 1e-3,
    hidden:        int        = 64,
    n_layers:      int        = 4,
    n_interior:    int        = 1000,
    callback:      Optional[Callable[[int, float], None]] = None,
) -> Dict:
    """
    Train a simple fully-connected PINN on a given problem.

    Returns:
        dict with keys: model, history (list of {epoch, loss}), final_loss
    """
    import torch
    import torch.nn as nn
    from .problem_library import generate_collocation_points

    # --- Build MLP ---
    dim = len(problem.get("domain", {"x": (0,1), "y": (0,1)}))
    layers = [nn.Linear(dim, hidden), nn.Tanh()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.Tanh()]
    layers.append(nn.Linear(hidden, 1))
    model = nn.Sequential(*layers)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Collocation points ---
    col = generate_collocation_points(problem, n_interior=n_interior, n_boundary=200)
    pts = torch.tensor(col["interior"], dtype=torch.float32)
    bnd = torch.tensor(col["boundary"], dtype=torch.float32)

    residual_fn = build_pinn_loss(problem)

    history = []
    for epoch in range(1, n_epochs + 1):
        opt.zero_grad()
        # PDE residual
        pde_loss = residual_fn(model, pts)
        # BC loss (u = 0 on boundary — simplification)
        bc_loss  = model(bnd).pow(2).mean()
        loss     = pde_loss + bc_loss
        loss.backward()
        opt.step()

        if epoch % max(1, n_epochs // 100) == 0:
            val = float(loss.item())
            history.append({"epoch": epoch, "loss": val})
            if callback:
                callback(epoch, val)

    return {
        "model":      model,
        "history":    history,
        "final_loss": history[-1]["loss"] if history else float("nan"),
        "type":       "pinn_mlp",
        "config":     {"hidden": hidden, "n_layers": n_layers, "lr": lr},
    }


def train_timeseries(
    y:          np.ndarray,
    model_type: str = "tcn",
    input_len:  int = 32,
    horizon:    int = 16,
    epochs:     int = 50,
    lr:         float = 1e-3,
) -> Dict:
    """Train a timeseries forecasting model."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from pinneaple_timeseries.features.engineering import window_features
    from pinneaple_timeseries.models import TCNForecaster, LSTMForecaster, TFTForecaster

    X, Y = window_features(y, input_len, horizon)
    X_t = torch.tensor(X[:, :, None], dtype=torch.float32)  # (N, L, 1)
    Y_t = torch.tensor(Y[:, :, None], dtype=torch.float32)  # (N, H, 1)
    ds  = TensorDataset(X_t, Y_t)
    dl  = DataLoader(ds, batch_size=32, shuffle=True)

    mdl_map = {"tcn": TCNForecaster, "lstm": LSTMForecaster, "tft": TFTForecaster}
    cls  = mdl_map.get(model_type, TCNForecaster)
    model = cls(input_len=input_len, horizon=horizon, n_features=1)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for ep in range(1, epochs + 1):
        model.train(); ep_loss = 0.0
        for xb, yb in dl:
            pred = model(xb).y_hat
            loss = torch.nn.functional.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        avg = ep_loss / len(dl)
        if ep % max(1, epochs // 20) == 0:
            history.append({"epoch": ep, "loss": avg})

    return {"model": model, "history": history, "type": model_type,
            "input_len": input_len, "horizon": horizon}
