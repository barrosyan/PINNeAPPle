"""Gray-box PINN training (AI-Aristotle style): a main network learns the
state variables while one or more small auxiliary networks learn an unknown
functional term in an otherwise-known governing equation.

Convention: the caller supplies a `known_residual_fn(main_model, coords,
learnable, known, graybox_nets) -> Tensor` that assembles whatever residual
their governing equation needs, calling into `graybox_nets[name](coords)`
wherever an unknown term belongs — the same "supply the physics, we handle
the training loop" pattern used elsewhere in this package (e.g. the
`reaction_kinetics_network` PDE kind's `rate_fn` convention). No equation is
hardcoded here, so this works for any PDE/ODE with one or more unknown terms,
not a fixed catalog of problem types.

Typical workflow
-----------------
>>> from pinneapple_neural.trainer.graybox import GrayBoxNet, train_graybox
>>> def known_residual_fn(model, coords, learnable, known, graybox_nets):
...     # e.g. heat equation with an unknown source: dT/dt - alpha*d2T/dx2 = h(x,t)
...     ...
...     h = graybox_nets["source"](coords.detach())
...     return known_part - h
>>> graybox_nets = {"source": GrayBoxNet(n_in=2, n_out=1)}
>>> main_model, graybox_nets, learned_params, history = train_graybox(
...     main_model, graybox_nets, known_residual_fn,
...     obs_coords, obs_values, learnable_params={"alpha": 0.1},
... )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class GrayBoxNet(nn.Module):
    """Small network approximating an unknown functional term in a PDE/ODE.
    Kept shallow/narrow by design: if the term is later distilled into a
    closed-form expression (e.g. via symbolic regression), a simple network
    is much easier to distill accurately than a large one."""

    def __init__(self, n_in: int = 1, n_out: int = 1, width: int = 32, depth: int = 3):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(n_in, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, n_out))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


KnownResidualFn = Callable[
    [nn.Module, torch.Tensor, Dict[str, torch.Tensor], Dict[str, float], Dict[str, nn.Module]],
    torch.Tensor,
]
CollocationSamplerFn = Callable[[], torch.Tensor]


@dataclass
class GrayBoxConfig:
    epochs_supervised: int = 500
    epochs_joint: int = 2000
    lr_main: float = 1e-3
    lr_graybox: float = 1e-3
    lr_params: float = 1e-2
    w_data: float = 10.0
    w_pde: float = 1.0
    pde_warmup_frac: float = 0.10
    batch_size: int = 512
    grad_clip: float = 1.0
    ema_alpha: float = 0.98
    log_every: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_graybox(
    main_model: nn.Module,
    graybox_nets: Dict[str, GrayBoxNet],
    known_residual_fn: KnownResidualFn,
    obs_coords: torch.Tensor,
    obs_values: torch.Tensor,
    collocation_sampler: CollocationSamplerFn,
    learnable_params: Optional[Dict[str, float]] = None,
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    known_params: Optional[Dict[str, float]] = None,
    config: Optional[GrayBoxConfig] = None,
    progress_cb: Optional[Callable[[int, int, Dict[str, float], Dict[str, float]], None]] = None,
) -> Tuple[nn.Module, Dict[str, GrayBoxNet], Dict[str, float], List[Dict[str, Any]]]:
    """Two-phase gray-box training: Phase 1 fits `main_model` to
    (obs_coords, obs_values) alone; Phase 2 jointly trains `main_model`, the
    gray-box networks, and any unknown scalar parameters against a weighted
    sum of the data loss and `known_residual_fn`'s PDE residual, with the PDE
    weight auto-balanced via an EMA loss-magnitude ratio (not a fixed
    constant — prevents whichever term is larger from dominating training).

    `learnable_params`: {name: initial_value} — becomes `nn.Parameter`s
    passed to `known_residual_fn` as its `learnable` dict, clamped every step
    to `param_bounds[name]` (default (1e-4, 100.0) if unspecified).
    `known_params`: {name: value} — fixed (non-trainable) values passed
    through as `known_residual_fn`'s `known` dict.
    `collocation_sampler`: called with no arguments each epoch (after the PDE
    warmup) to draw a fresh batch of PDE collocation points; return shape
    should match what `known_residual_fn` expects for `coords`.

    Returns (main_model, graybox_nets, estimated_params, convergence_history).
    """
    cfg = config or GrayBoxConfig()
    device = torch.device(cfg.device)
    main_model = main_model.to(device)
    for net in graybox_nets.values():
        net.to(device)

    obs_coords = obs_coords.to(device)
    obs_values = obs_values.to(device)

    coord_min = obs_coords.min(0).values
    coord_max = obs_coords.max(0).values
    coord_range = (coord_max - coord_min).clamp(min=1e-6)
    obs_norm = (obs_coords - coord_min) / coord_range

    val_mean = obs_values.mean(0, keepdim=True)
    val_std = obs_values.std(0, keepdim=True).clamp(min=1e-6)
    obs_norm_y = (obs_values - val_mean) / val_std

    learnable: Dict[str, nn.Parameter] = {
        name: nn.Parameter(torch.tensor([val], dtype=torch.float32, device=device))
        for name, val in (learnable_params or {}).items()
    }
    bounds = dict(param_bounds or {})
    for name in learnable:
        bounds.setdefault(name, (1e-4, 100.0))
    known = dict(known_params or {})

    batch = min(cfg.batch_size, len(obs_norm))
    mse = nn.MSELoss()
    all_params = (
        list(main_model.parameters())
        + [p for net in graybox_nets.values() for p in net.parameters()]
        + list(learnable.values())
    )

    convergence: List[Dict[str, Any]] = []
    log_every_sup = cfg.log_every or max(1, cfg.epochs_supervised // 100)

    # ── Phase 1: supervised on observations only ─────────────────────────────
    opt1 = torch.optim.Adam(main_model.parameters(), lr=cfg.lr_main)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, cfg.epochs_supervised, eta_min=cfg.lr_main * 0.01)

    for epoch in range(cfg.epochs_supervised):
        main_model.train()
        idx = torch.randperm(len(obs_norm), device=device)[:batch]
        pred = main_model(obs_norm[idx])
        loss = mse(pred, obs_norm_y[idx])
        opt1.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(main_model.parameters(), cfg.grad_clip)
        opt1.step()
        sch1.step()
        if progress_cb and epoch % log_every_sup == 0:
            progress_cb(epoch, cfg.epochs_supervised + cfg.epochs_joint, {"sup_data": float(loss)}, {})

    # ── Phase 2: joint main + gray-box nets + learnable params ───────────────
    opt2 = torch.optim.Adam([
        {"params": main_model.parameters(), "lr": cfg.lr_main},
        *[{"params": net.parameters(), "lr": cfg.lr_graybox} for net in graybox_nets.values()],
        {"params": list(learnable.values()), "lr": cfg.lr_params},
    ])
    t0 = max(cfg.epochs_joint // 4, 100)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt2, t0, T_mult=2, eta_min=cfg.lr_main * 0.01)
    pde_warmup = int(cfg.pde_warmup_frac * cfg.epochs_joint)
    log_every = cfg.log_every or max(1, cfg.epochs_joint // 200)
    ema_data, ema_pde = 1.0, 1.0

    for epoch in range(cfg.epochs_joint):
        main_model.train()
        for net in graybox_nets.values():
            net.train()

        idx = torch.randperm(len(obs_norm), device=device)[:batch]
        pred = main_model(obs_norm[idx])
        l_data = mse(pred, obs_norm_y[idx])

        l_pde = torch.tensor(0.0, device=device)
        if epoch >= pde_warmup:
            coll = collocation_sampler().to(device)
            res = known_residual_fn(main_model, coll, dict(learnable), known, dict(graybox_nets))
            l_pde = res.pow(2).mean()

        with torch.no_grad():
            ema_data = cfg.ema_alpha * ema_data + (1 - cfg.ema_alpha) * float(l_data + 1e-12)
            ema_pde = cfg.ema_alpha * ema_pde + (1 - cfg.ema_alpha) * float(l_pde + 1e-12)
            w_pde_eff = cfg.w_pde * min(max(ema_data / (ema_pde + 1e-12), 0.01), 100.0)

        loss = cfg.w_data * l_data + w_pde_eff * l_pde
        opt2.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
        opt2.step()
        sch2.step(epoch)

        for name, param in learnable.items():
            lo, hi = bounds[name]
            with torch.no_grad():
                param.data.clamp_(lo, hi)

        if epoch % log_every == 0 or epoch == cfg.epochs_joint - 1:
            current = {k: float(v.item()) for k, v in learnable.items()}
            convergence.append({
                "epoch": cfg.epochs_supervised + epoch,
                "loss_data": float(l_data), "loss_pde": float(l_pde), **current,
            })
            if progress_cb:
                progress_cb(cfg.epochs_supervised + epoch, cfg.epochs_supervised + cfg.epochs_joint,
                            {"data": float(l_data), "pde": float(l_pde)}, current)

    estimated = {k: float(v.item()) for k, v in learnable.items()}
    norm_meta = {
        "coord_min": coord_min.cpu().numpy().tolist(),
        "coord_range": coord_range.cpu().numpy().tolist(),
        "val_mean": val_mean.squeeze(0).cpu().numpy().tolist(),
        "val_std": val_std.squeeze(0).cpu().numpy().tolist(),
    }
    main_model._norm = norm_meta  # type: ignore[attr-defined]
    for net in graybox_nets.values():
        net._norm = norm_meta  # type: ignore[attr-defined]

    return main_model, graybox_nets, estimated, convergence


def distill_to_symbolic(
    graybox_net: GrayBoxNet,
    coord_bounds: List[Tuple[float, float]],
    n_samples: int = 2000,
    var_names: Optional[List[str]] = None,
    n_iterations: int = 40,
) -> Dict[str, Any]:
    """Sample a trained gray-box network on its input domain and run PySR
    symbolic regression to recover a closed-form analytical expression for
    the unknown term it learned. Requires the optional `pysr` package."""
    try:
        from pysr import PySRRegressor
    except ImportError:
        return {"available": False, "expression": None,
                "message": "PySR not installed (pip install pysr)."}

    import numpy as np

    n_in = len(coord_bounds)
    rng = np.random.default_rng(0)
    x_np = np.column_stack([rng.uniform(lo, hi, n_samples) for lo, hi in coord_bounds]).astype(np.float32)

    x_t = torch.tensor(x_np, dtype=torch.float32)
    graybox_net.eval()
    with torch.no_grad():
        y_np = graybox_net(x_t).numpy()

    names = var_names or [f"x{i}" for i in range(n_in)]
    results = []
    for col in range(y_np.shape[1]):
        sr = PySRRegressor(
            niterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["exp", "sin", "cos", "sqrt"],
            verbosity=0,
        )
        sr.fit(x_np, y_np[:, col], variable_names=names)
        results.append({
            "output_index": col,
            "expression": str(sr.get_best()["equation"]) if hasattr(sr, "get_best") else str(sr.sympy()),
        })

    return {"available": True, "results": results}
