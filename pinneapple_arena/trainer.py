"""Unified training engine for Arena benchmarks.

Handles four paradigms:
  1. PINN          — physics residuals (own autograd OR pinneapple_physics compiled losses)
  2. Supervised    — MSE on (input, target) pairs
  3. Graph/GNN     — supervised on GraphBatch objects
  4. Inverse       — pinneapple_analysis.InverseProblemSolver

Post-training:
  - UQ analysis    — pinneapple_analysis.uq_predict
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import ModelConfig, TrainingConfig, InverseConfig, UQConfig


# ── result containers ─────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    name: str
    model: nn.Module
    train_losses: List[float]
    train_time: float
    metrics: Dict[str, float] = field(default_factory=dict)
    uq_result: Optional[Any] = None
    inverse_result: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# ── optimiser / scheduler factory ─────────────────────────────────────────────

def _make_optimizer(model: nn.Module, cfg: TrainingConfig) -> optim.Optimizer:
    name = cfg.optimizer.lower()
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9)
    return optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def _make_scheduler(opt: optim.Optimizer, cfg: TrainingConfig, epochs: int):
    name = cfg.scheduler.lower()
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if name == "step":
        return optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 3), gamma=0.5)
    return None


# ── PINN training loop ─────────────────────────────────────────────────────────

def train_pinn(
    model: nn.Module,
    cfg: ModelConfig,
    pinn_residuals_fn: Callable,
    xy_int: np.ndarray,
    xy_bc: np.ndarray,
    uv_bc: np.ndarray,
    problem_params: Dict[str, Any],
    device: str = "cpu",
    log_interval: int = 200,
    compiled_losses: Optional[Dict[str, Callable]] = None,
) -> TrainResult:
    """Train a PINN model.

    If ``compiled_losses`` is provided (from pinneapple_physics.compile_physics),
    those are used instead of the fallback ``pinn_residuals_fn``.
    """
    tc = cfg.training
    if tc.seed is not None:
        torch.manual_seed(tc.seed)

    model = model.to(device)
    opt = _make_optimizer(model, tc)
    sched = _make_scheduler(opt, tc, tc.epochs)

    t_int = torch.tensor(xy_int, dtype=torch.float32, device=device)
    t_bc  = torch.tensor(xy_bc,  dtype=torch.float32, device=device)
    t_ubc = torch.tensor(uv_bc,  dtype=torch.float32, device=device)

    losses = []
    t0 = time.time()

    if compiled_losses is not None:
        _train_pinn_compiled(model, opt, sched, tc, compiled_losses,
                             t_int, t_bc, t_ubc, losses, log_interval, cfg.name)
    else:
        _train_pinn_autograd(model, opt, sched, tc, pinn_residuals_fn,
                             t_int, t_bc, t_ubc, problem_params, losses, log_interval, cfg.name)

    return TrainResult(name=cfg.name, model=model,
                       train_losses=losses, train_time=time.time() - t0)


def _train_pinn_autograd(model, opt, sched, tc, residuals_fn,
                         t_int, t_bc, t_ubc, params, losses, log_interval, name):
    for ep in range(1, tc.epochs + 1):
        model.train()
        opt.zero_grad()
        r_loss, bc_loss = residuals_fn(model, t_int, t_bc, t_ubc, **params)
        loss = r_loss + 10.0 * bc_loss
        loss.backward()
        if tc.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        if sched:
            sched.step()
        losses.append(loss.item())
        if ep % log_interval == 0 or ep == 1:
            print(f"  [{name}] epoch {ep:5d}/{tc.epochs}  "
                  f"res={r_loss.item():.3e}  bc={bc_loss.item():.3e}")


def _train_pinn_compiled(model, opt, sched, tc, compiled_losses,
                         t_int, t_bc, t_ubc, losses, log_interval, name):
    """Training loop using losses compiled by pinneapple_physics.compile_physics."""
    for ep in range(1, tc.epochs + 1):
        model.train()
        opt.zero_grad()
        total = torch.tensor(0.0, device=t_int.device)
        try:
            # compiled_losses is Dict[str, callable]; each callable may accept
            # (model, x_col) or (model, x_col, x_bc, y_bc) — try both
            for lname, lfn in compiled_losses.items():
                try:
                    lval = lfn(model, t_int, t_bc, t_ubc)
                except TypeError:
                    try:
                        lval = lfn(model, t_int)
                    except Exception:
                        continue
                if torch.is_tensor(lval):
                    total = total + lval
        except Exception as e:
            # compiled losses failed mid-epoch — warn once and skip
            warnings.warn(f"[Arena] compiled_losses failed at epoch {ep}: {e}. "
                          "Switching to zero loss this step.")
        total.backward()
        if tc.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        if sched:
            sched.step()
        losses.append(total.item())
        if ep % log_interval == 0 or ep == 1:
            print(f"  [{name}] epoch {ep:5d}/{tc.epochs}  loss={total.item():.3e}  "
                  f"(pinneapple_physics compiled losses)")


# ── Supervised training loop ───────────────────────────────────────────────────

def train_supervised(
    model: nn.Module,
    cfg: ModelConfig,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    device: str = "cpu",
    log_interval: int = 100,
) -> TrainResult:
    tc = cfg.training
    if tc.seed is not None:
        torch.manual_seed(tc.seed)

    model = model.to(device)
    opt = _make_optimizer(model, tc)
    sched = _make_scheduler(opt, tc, tc.epochs)

    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y_train, dtype=torch.float32, device=device)

    n = len(Xt)
    bs = min(tc.batch_size, n)

    losses = []
    t0 = time.time()
    for ep in range(1, tc.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0; steps = 0
        for i in range(0, n, bs):
            idx = perm[i: i + bs]
            xb, yb = Xt[idx], Yt[idx]
            opt.zero_grad()
            pred = _forward_supervised(model, xb, cfg)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            if tc.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()
            ep_loss += loss.item(); steps += 1
        if sched:
            sched.step()
        ep_loss /= max(steps, 1)
        losses.append(ep_loss)
        if ep % log_interval == 0 or ep == 1:
            print(f"  [{cfg.name}] epoch {ep:5d}/{tc.epochs}  loss={ep_loss:.3e}")

    return TrainResult(name=cfg.name, model=model,
                       train_losses=losses, train_time=time.time() - t0)


def _unwrap_output(out) -> torch.Tensor:
    """Unwrap PINNOutput / OperatorOutput / dict to a plain tensor."""
    if torch.is_tensor(out):
        return out
    if hasattr(out, "y") and torch.is_tensor(out.y):
        return out.y
    if hasattr(out, "x") and torch.is_tensor(out.x):
        return out.x
    if isinstance(out, dict):
        return torch.stack(list(out.values()), dim=-1)
    return out


def _forward_supervised(model: nn.Module, x: torch.Tensor, cfg: ModelConfig
                        ) -> torch.Tensor:
    mtype = cfg.type.lower()
    if mtype in ("fno2d", "fno", "fourier", "fourier_neural_operator"):
        return _fno_forward(model, x)
    return _unwrap_output(model(x))


def _fno_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Forward pass for FNO-family models.

    Tries input shapes in order:
      1. (1, C, H, W)  — 2D grid (if N is a perfect square)
      2. (1, C, N)     — 1D sequence
      3. (N, C)        — pointwise (fallback; most FNOs reject this)
    Always returns (N, out_dim).
    """
    n, c = x.shape
    # Try 2D grid
    side = int(n ** 0.5)
    if side * side == n:
        try:
            xg = x.T.reshape(1, c, side, side)  # (1, C, H, W)
            out = _unwrap_output(model(xg))      # (1, out, H, W)
            return out.reshape(n, -1)
        except Exception:
            pass
    # Try 1D sequence
    try:
        xg = x.T.unsqueeze(0)          # (1, C, N)
        out = _unwrap_output(model(xg))  # (1, out, N)
        return out.squeeze(0).T         # (N, out)
    except Exception:
        pass
    # Pointwise fallback
    return _unwrap_output(model(x))


# ── Graph / MeshGraphNet training loop ────────────────────────────────────────

def train_graph(
    model: nn.Module,
    cfg: ModelConfig,
    node_feats: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    node_targets: np.ndarray,
    device: str = "cpu",
    log_interval: int = 100,
) -> TrainResult:
    tc = cfg.training
    if tc.seed is not None:
        torch.manual_seed(tc.seed)

    model = model.to(device)
    opt = _make_optimizer(model, tc)
    sched = _make_scheduler(opt, tc, tc.epochs)

    try:
        from pinneapple_neural.architectures.graphnn.base import GraphBatch
        x_t   = torch.tensor(node_feats,   dtype=torch.float32, device=device).unsqueeze(0)
        ea_t  = torch.tensor(edge_attr,    dtype=torch.float32, device=device).unsqueeze(0)
        ei_t  = torch.tensor(edge_index,   dtype=torch.long,    device=device)
        pos_t = torch.tensor(node_feats[:, :2], dtype=torch.float32, device=device).unsqueeze(0)
        g = GraphBatch(x=x_t, edge_index=ei_t, edge_attr=ea_t, pos=pos_t)
        Y = torch.tensor(node_targets, dtype=torch.float32, device=device).unsqueeze(0)
    except Exception as e:
        raise RuntimeError(f"GraphBatch construction failed: {e}") from e

    losses = []
    t0 = time.time()
    for ep in range(1, tc.epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(g)
        pred = out.y if hasattr(out, "y") else (out.x if hasattr(out, "x") else out)
        loss = nn.functional.mse_loss(pred, Y)
        loss.backward()
        if tc.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        if sched:
            sched.step()
        losses.append(loss.item())
        if ep % log_interval == 0 or ep == 1:
            print(f"  [{cfg.name}] epoch {ep:5d}/{tc.epochs}  loss={loss.item():.3e}")

    return TrainResult(name=cfg.name, model=model,
                       train_losses=losses, train_time=time.time() - t0)


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    result: TrainResult,
    cfg: ModelConfig,
    xy_eval: np.ndarray,
    Y_ref: np.ndarray,
    field_names: List[str],
    device: str = "cpu",
    node_positions: Optional[np.ndarray] = None,
    edge_index: Optional[np.ndarray] = None,
    edge_attr: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    from .model_factory import is_graph_model
    model = result.model.eval()

    if is_graph_model(cfg) and node_positions is not None:
        pred = _eval_graph(model, xy_eval, node_positions, edge_index, edge_attr, device)
    else:
        pred = _eval_dense(model, cfg, xy_eval, device)

    metrics = {}
    for i, fname in enumerate(field_names):
        ref_i = Y_ref[:, i] if Y_ref.ndim > 1 else Y_ref.ravel()
        pred_i = pred[:, i] if pred.ndim > 1 and pred.shape[1] > i else pred.ravel()
        l2   = float(np.sqrt(np.mean((pred_i - ref_i) ** 2)))
        linf = float(np.max(np.abs(pred_i - ref_i)))
        rel  = l2 / (float(np.sqrt(np.mean(ref_i ** 2))) + 1e-8)
        metrics[f"L2_{fname}"]   = l2
        metrics[f"Linf_{fname}"] = linf
        metrics[f"rel_{fname}"]  = rel

    result.metrics = metrics
    return {"pred": pred, "ref": Y_ref, "metrics": metrics}


def _eval_dense(model, cfg, xy_eval, device):
    x = torch.tensor(xy_eval, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = _forward_supervised(model, x, cfg)
    pred = out.cpu().numpy()
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    return pred


def _eval_graph(model, xy_eval, node_positions, edge_index, edge_attr, device):
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    try:
        from pinneapple_neural.architectures.graphnn.base import GraphBatch
        x_t   = torch.tensor(node_positions, dtype=torch.float32, device=device).unsqueeze(0)
        ea_t  = torch.tensor(edge_attr,      dtype=torch.float32, device=device).unsqueeze(0)
        ei_t  = torch.tensor(edge_index,     dtype=torch.long,    device=device)
        pos_t = torch.tensor(node_positions[:, :2], dtype=torch.float32, device=device).unsqueeze(0)
        g = GraphBatch(x=x_t, edge_index=ei_t, edge_attr=ea_t, pos=pos_t)
        with torch.no_grad():
            out = model(g)
        raw = out.y if hasattr(out, "y") else (out.x if hasattr(out, "x") else out)
        pred_nodes = raw.squeeze(0).cpu().numpy()
    except Exception as e:
        raise RuntimeError(f"Graph eval failed: {e}") from e

    pts_mesh = node_positions[:, :2]
    n_out = pred_nodes.shape[1] if pred_nodes.ndim > 1 else 1
    interp = np.zeros((len(xy_eval), n_out))
    for j in range(n_out):
        vals = pred_nodes[:, j] if pred_nodes.ndim > 1 else pred_nodes.ravel()
        lin = LinearNDInterpolator(pts_mesh, vals)
        col = lin(xy_eval[:, 0], xy_eval[:, 1])
        nan_mask = np.isnan(col)
        if nan_mask.any():
            col[nan_mask] = NearestNDInterpolator(pts_mesh, vals)(
                xy_eval[nan_mask, 0], xy_eval[nan_mask, 1])
        interp[:, j] = col
    return interp


# ── Uncertainty Quantification ────────────────────────────────────────────────

def run_uq(
    result: TrainResult,
    xy_eval: np.ndarray,
    uq_cfg: UQConfig,
    device: str = "cpu",
) -> Optional[Any]:
    """Run UQ analysis using pinneapple_analysis.uq_predict."""
    if not uq_cfg.enabled:
        return None
    try:
        from pinneapple_analysis import uq_predict
    except ImportError:
        warnings.warn("[Arena] pinneapple_analysis not available; skipping UQ.")
        return None

    x = torch.tensor(xy_eval, dtype=torch.float32, device=device)
    model = result.model

    try:
        uq_result = uq_predict(
            model, x,
            method=uq_cfg.method,
            n_samples=uq_cfg.n_samples,
            dropout_rate=uq_cfg.dropout_rate,
        )
        result.uq_result = uq_result
        print(f"  [{result.name}] UQ ({uq_cfg.method}): "
              f"mean_std={float(uq_result.std.mean()):.3e}" if hasattr(uq_result, "std") else
              f"  [{result.name}] UQ done.")
        return uq_result
    except Exception as e:
        warnings.warn(f"[Arena] UQ failed for {result.name}: {e}")
        return None


# ── Inverse problem ───────────────────────────────────────────────────────────

def run_inverse(
    result: TrainResult,
    xy_eval: np.ndarray,
    Y_ref: np.ndarray,
    inv_cfg: InverseConfig,
    device: str = "cpu",
) -> Optional[Any]:
    """Run inverse problem using pinneapple_analysis.invert."""
    if not inv_cfg.enabled:
        return None
    try:
        from pinneapple_analysis import invert
    except ImportError:
        warnings.warn("[Arena] pinneapple_analysis not available; skipping inverse problem.")
        return None

    try:
        rng = np.random.default_rng(0)
        n_obs = min(inv_cfg.n_obs, len(xy_eval))
        idx = rng.choice(len(xy_eval), n_obs, replace=False)
        sensor_locs = torch.tensor(xy_eval[idx], dtype=torch.float32, device=device)
        y_obs = torch.tensor(
            Y_ref[idx, 0] if Y_ref.ndim > 1 else Y_ref[idx],
            dtype=torch.float32, device=device
        )

        inv_result = invert(
            result.model, y_obs, sensor_locs,
            noise_std=inv_cfg.noise_std,
            lambda_reg=inv_cfg.lambda_reg,
            method=inv_cfg.method,
            n_iters=inv_cfg.n_iters,
        )
        result.inverse_result = inv_result
        print(f"  [{result.name}] Inverse: final_misfit={inv_result.final_misfit:.3e}"
              if hasattr(inv_result, "final_misfit") else
              f"  [{result.name}] Inverse problem solved.")
        return inv_result
    except Exception as e:
        warnings.warn(f"[Arena] Inverse problem failed for {result.name}: {e}")
        return None


# ── Dataset loading (pinneapple_data) ──────────────────────────────────────────

def load_pinneapple_dataset(dataset_id: str,
                            input_fields: List[str],
                            output_fields: List[str],
                            n_train: int = 1000,
                            n_val: int = 200,
                            split_seed: int = 42,
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load a pinneapple_data dataset and split into train/val arrays.

    Returns
    -------
    X_train, Y_train, X_val, Y_val, field_names
    """
    try:
        from pinneapple_data import load_dataset
    except ImportError as e:
        raise ImportError("pinneapple_data not available.") from e

    data = load_dataset(dataset_id)

    if not input_fields:
        raise ValueError(f"DatasetConfig.input_fields must be set for dataset '{dataset_id}'.")
    if not output_fields:
        raise ValueError(f"DatasetConfig.output_fields must be set for dataset '{dataset_id}'.")

    # build X from input_fields
    X_parts = []
    for f in input_fields:
        if f not in data:
            raise KeyError(f"Field '{f}' not in dataset '{dataset_id}'. "
                           f"Available: {list(data.keys())}")
        arr = np.asarray(data[f]).reshape(-1, 1) if np.asarray(data[f]).ndim == 1 \
            else np.asarray(data[f])
        X_parts.append(arr)
    X = np.concatenate(X_parts, axis=1)

    Y_parts = []
    for f in output_fields:
        if f not in data:
            raise KeyError(f"Field '{f}' not in dataset '{dataset_id}'. "
                           f"Available: {list(data.keys())}")
        arr = np.asarray(data[f]).reshape(-1, 1) if np.asarray(data[f]).ndim == 1 \
            else np.asarray(data[f])
        Y_parts.append(arr)
    Y = np.concatenate(Y_parts, axis=1)

    n = len(X)
    rng = np.random.default_rng(split_seed)
    idx = rng.permutation(n)
    n_tr = min(n_train, int(0.8 * n))
    n_va = min(n_val, n - n_tr)

    X_train = X[idx[:n_tr]]
    Y_train = Y[idx[:n_tr]]
    X_val   = X[idx[n_tr: n_tr + n_va]]
    Y_val   = Y[idx[n_tr: n_tr + n_va]]

    return X_train, Y_train, X_val, Y_val, output_fields
