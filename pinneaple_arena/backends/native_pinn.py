from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from pinneaple_arena.bundle.loader import BundleData


Tensor = torch.Tensor


def _act(name: str) -> nn.Module:
    """
    Return a PyTorch activation module given its name.

    Supported activations:
        - tanh
        - relu
        - gelu
        - silu

    Defaults to Tanh if the provided name is unknown or None.
    """
    name = (name or "tanh").lower()
    return {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}.get(name, nn.Tanh())


class MLP(nn.Module):
    """
    Fully connected multilayer perceptron (MLP).

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output feature dimension.
    hidden : Tuple[int, ...]
        Tuple containing hidden layer sizes.
    activation : str
        Activation function name.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: Tuple[int, ...], activation: str = "tanh"):
        super().__init__()
        act = _act(activation)
        dims = [int(in_dim), *[int(h) for h in hidden], int(out_dim)]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act)
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the MLP.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N, in_dim).

        Returns
        -------
        Tensor
            Output tensor of shape (N, out_dim).
        """
        return self.net(x)


def _grad(y: Tensor, x: Tensor) -> Tensor:
    """
    Compute first-order gradients of y with respect to x.

    Parameters
    ----------
    y : Tensor
        Output tensor.
    x : Tensor
        Input tensor with requires_grad=True.

    Returns
    -------
    Tensor
        Gradient tensor dy/dx.
    """
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True, allow_unused=False
    )[0]


def _ns2d_residuals(xy: Tensor, uvp: Tensor, nu: float) -> Dict[str, Tensor]:
    """
    Compute 2D steady Navier-Stokes residuals.

    Parameters
    ----------
    xy : Tensor
        Input spatial coordinates (x, y).
    uvp : Tensor
        Model outputs containing velocity components (u, v) and pressure (p).
    nu : float
        Kinematic viscosity.

    Returns
    -------
    Dict[str, Tensor]
        Dictionary with momentum and continuity residuals:
            - mom_u
            - mom_v
            - cont
    """
    u = uvp[:, 0:1]
    v = uvp[:, 1:2]
    p = uvp[:, 2:3]

    gu = _grad(u, xy)
    gv = _grad(v, xy)
    gp = _grad(p, xy)

    u_x, u_y = gu[:, 0:1], gu[:, 1:2]
    v_x, v_y = gv[:, 0:1], gv[:, 1:2]
    p_x, p_y = gp[:, 0:1], gp[:, 1:2]

    u_xx = _grad(u_x, xy)[:, 0:1]
    u_yy = _grad(u_y, xy)[:, 1:2]
    v_xx = _grad(v_x, xy)[:, 0:1]
    v_yy = _grad(v_y, xy)[:, 1:2]

    mom_u = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    mom_v = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    cont = u_x + v_y
    return {"mom_u": mom_u, "mom_v": mom_v, "cont": cont}


class NativePINNBackend:
    """
    Native PINN training backend for steady 2D Navier-Stokes problems.

    This backend:
        - Samples collocation and boundary points from a BundleData object.
        - Computes PDE residual losses.
        - Enforces boundary conditions.
        - Trains an MLP using Adam optimizer.
    """

    name = "pinneaple_native"

    def train(self, bundle: BundleData, run_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the PINN model using configuration parameters.

        Parameters
        ----------
        bundle : BundleData
            Dataset bundle containing collocation and boundary points,
            as well as physical parameters (e.g., viscosity).
        run_cfg : Dict[str, Any]
            Configuration dictionary including:
                - train settings
                - model architecture
                - arena sampling parameters

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
                - device used
                - trained model
                - final training metrics
        """
        train_cfg = dict(run_cfg.get("train", {}))
        model_cfg = dict(run_cfg.get("model", {}))
        arena_cfg = dict(run_cfg.get("arena", {}))

        device = str(train_cfg.get("device", "cpu"))
        epochs = int(train_cfg.get("epochs", 5000))
        lr = float(train_cfg.get("lr", 1e-3))
        seed = int(train_cfg.get("seed", 0))
        log_every = int(train_cfg.get("log_every", 200))

        weights = dict(train_cfg.get("weights", {}))
        w_pde = float(weights.get("pde", 1.0))
        w_bc = float(weights.get("bc", 10.0))

        n_collocation = int(arena_cfg.get("n_collocation", 4096))
        n_boundary = int(arena_cfg.get("n_boundary", 2048))

        hidden = tuple(int(x) for x in model_cfg.get("hidden", [256, 256, 256, 256]))
        activation = str(model_cfg.get("activation", "tanh"))

        torch.manual_seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)

        dev = torch.device(device)
        nu = float(bundle.manifest["nu"])

        model = MLP(in_dim=2, out_dim=3, hidden=hidden, activation=activation).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        # Samplers from parquet pools
        col_df = bundle.points_collocation
        bnd_df = bundle.points_boundary

        def sample_collocation(n: int):
            """
            Randomly sample collocation points from dataset.
            """
            return col_df.sample(n=min(n, len(col_df)), replace=(len(col_df) < n), random_state=None)[["x", "y"]]

        def sample_boundary(n: int):
            """
            Randomly sample boundary points from dataset.
            """
            return bnd_df.sample(n=min(n, len(bnd_df)), replace=(len(bnd_df) < n), random_state=None)[["x", "y", "region"]]

        model.train()
        last = {"total": float("nan"), "pde": float("nan"), "bc": float("nan")}

        for step in range(epochs):
            opt.zero_grad(set_to_none=True)

            # PDE
            col = sample_collocation(n_collocation)
            xy_col = torch.tensor(col.to_numpy(), dtype=torch.float32, device=dev)
            xy_col.requires_grad_(True)
            uvp_col = model(xy_col)
            res = _ns2d_residuals(xy_col, uvp_col, nu=nu)
            pde_loss = (res["mom_u"].pow(2).mean() + res["mom_v"].pow(2).mean() + res["cont"].pow(2).mean())

            # BC
            bnd = sample_boundary(n_boundary)
            xy_b = torch.tensor(bnd[["x", "y"]].to_numpy(), dtype=torch.float32, device=dev)
            uvp_b = model(xy_b)
            u_b = uvp_b[:, 0:1]
            v_b = uvp_b[:, 1:2]
            p_b = uvp_b[:, 2:3]
            regions = bnd["region"].astype(str).to_numpy()

            bc_terms = []

            # inlet u=1 v=0
            m_in = torch.tensor(regions == "inlet", device=dev)
            if bool(m_in.any()):
                bc_terms.append((u_b[m_in] - 1.0).pow(2).mean())
                bc_terms.append((v_b[m_in] - 0.0).pow(2).mean())

            # walls/obstacle no-slip
            for reg in ("walls", "obstacle"):
                m = torch.tensor(regions == reg, device=dev)
                if bool(m.any()):
                    bc_terms.append((u_b[m] - 0.0).pow(2).mean())
                    bc_terms.append((v_b[m] - 0.0).pow(2).mean())

            # outlet p=0
            m_out = torch.tensor(regions == "outlet", device=dev)
            if bool(m_out.any()):
                bc_terms.append((p_b[m_out] - 0.0).pow(2).mean())

            bc_loss = torch.stack(bc_terms).mean() if bc_terms else torch.tensor(0.0, device=dev)

            loss = w_pde * pde_loss + w_bc * bc_loss
            loss.backward()
            opt.step()

            if step % log_every == 0 or step == epochs - 1:
                last = {
                    "total": float(loss.detach().cpu()),
                    "pde": float(pde_loss.detach().cpu()),
                    "bc": float(bc_loss.detach().cpu()),
                }
                print(f"[native] step={step:06d} total={last['total']:.3e} pde={last['pde']:.3e} bc={last['bc']:.3e}")

        return {
            "device": device,
            "model": model,
            "metrics": {
                "train_total": last["total"],
                "train_pde": last["pde"],
                "train_bc": last["bc"],
            },
        }