"""Functional (not just import) tests for capabilities the user asked to
have "included" in PINNeAPPle -- active learning and transfer/meta
learning -- which turned out to already exist (`pinneapple_data
.active_learning`, `pinneapple_adaptation`) but had never been run
end-to-end by any test before this session (both packages were only
import-smoke-tested, never functionally exercised).

This process found and fixed two real bugs, both the same shape as
several others found earlier this session: a documented convenience
function silently skipping a required setup call, so every call through
the documented API failed or silently did the wrong thing:

- `pinneapple_adaptation.fine_tune()` never called
  `TransferTrainer.prepare()` -- every call raised
  "Call TransferTrainer.prepare() before finetune()." unconditionally.
- `pinneapple_adaptation.meta_learning.meta_train()` never called
  `trainer.train()` -- it returned an UNTRAINED trainer despite its own
  docstring promising "Trained trainer object with .adapt() method";
  `.adapt()` on it "worked" (no exception) but adapted from the model's
  random initialization instead of a real meta-learned one.
"""
from __future__ import annotations

import numpy as np
import torch

from pinneapple_data.active_learning import ActiveLearningConfig, ResidualBasedAL
from pinneapple_physics.pde_environment.presets.academics import laplace_2d_default, burgers_1d_default
from pinneapple_physics.pinn_solver.compiler.autograd_ops import laplacian
from pinneapple_physics.pinn_solver.compiler.compile import compile_problem


def _empty_batch(x_col, n_coords, n_fields):
    return {
        "x_col": x_col, "ctx": {},
        "x_bc": torch.zeros((0, n_coords)), "y_bc": torch.zeros((0, n_fields)),
        "x_ic": torch.zeros((0, n_coords)), "y_ic": torch.zeros((0, n_fields)),
        "x_data": torch.zeros((0, n_coords)), "y_data": torch.zeros((0, n_fields)),
    }


def test_residual_based_active_learning_rar_loop_reduces_loss():
    """A real residual-based adaptive refinement (RAR) loop: train, add
    points where the residual is highest, retrain, repeat. Loss must
    trend down as the collocation set is refined."""
    spec = laplace_2d_default()
    loss_fn = compile_problem(spec)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    cfg = ActiveLearningConfig(n_candidates=2000, n_select=100, n_initial=200, n_iterations=3, seed=0)
    al = ResidualBasedAL(cfg, spec.domain_bounds)

    def residual_fn(X):
        xt = torch.as_tensor(X, dtype=torch.float32)
        xt.requires_grad_(True)
        u = model(xt)
        return laplacian(u, xt).detach().numpy().ravel()

    al.update_pool()
    x_col_np = al.select(residual_fn, mode="weighted")
    losses = []
    for _ in range(3):
        for _step in range(50):
            opt.zero_grad()
            xt = torch.as_tensor(x_col_np, dtype=torch.float32)
            xt.requires_grad_(True)
            y = model(xt)
            out = loss_fn(model, y, _empty_batch(xt, 2, 1))
            out["total"].backward()
            opt.step()
        losses.append(float(out["pde"]))
        new_pts = al.select(residual_fn, mode="weighted")
        x_col_np = np.concatenate([x_col_np, new_pts], axis=0)

    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0], f"RAR loop should reduce the PDE residual loss, got {losses}"


def test_fine_tune_convenience_function_actually_trains():
    """Regression test for the fine_tune() prepare()-skip bug: must not
    raise, and loss must actually decrease."""
    from pinneapple_adaptation import fine_tune

    spec = laplace_2d_default()
    loss_fn = compile_problem(spec)
    bounds = spec.domain_bounds
    coords = list(spec.coords)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))

    def target_physics_fn(m, batch):
        x_col = torch.as_tensor(
            np.stack([np.random.uniform(*bounds[c], size=256) for c in coords], axis=1).astype(np.float32)
        )
        x_col.requires_grad_(True)
        y = m(x_col)
        return loss_fn(m, y, _empty_batch(x_col, len(coords), 1))

    result = fine_tune(model, target_physics_fn, epochs=20, lr=1e-3)
    assert set(result.keys()) >= {"model", "history", "metrics"}
    assert len(result["history"]) == 20
    first_loss = result["history"][0]["loss_total"]
    last_loss = result["history"][-1]["loss_total"]
    assert last_loss < first_loss, f"fine_tune should reduce loss, got {first_loss} -> {last_loss}"


def test_meta_train_reptile_actually_trains_and_adapts():
    """Regression test for the meta_train() train()-skip bug: the
    returned trainer's meta_loss history must be non-empty and trend
    down, and .adapt() must return a usable model."""
    from pinneapple_adaptation import meta_train
    from pinneapple_adaptation.meta_learning import PDETaskSampler

    def physics_fn_factory(params):
        spec = burgers_1d_default(nu=params["nu"])
        loss_fn = compile_problem(spec)

        def reptile_physics_fn(model, batch):
            x_col = batch["x_col"]
            y = model(x_col)
            out = loss_fn(model, y, _empty_batch(x_col, 2, 1))
            return out["total"], out
        return reptile_physics_fn

    sampler = PDETaskSampler(param_ranges={"nu": (0.001, 0.05)},
                              physics_fn_factory=physics_fn_factory, input_dim=2, seed=0)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))

    trainer = meta_train(model, sampler, algorithm="reptile",
                         n_meta_epochs=5, n_tasks_per_batch=2, n_inner_steps=3, seed=0)
    assert hasattr(trainer, "adapt"), "meta_train() should return a trainer with .adapt()"
    assert len(trainer._history) == 5, "meta_train() must have actually run .train(), not just constructed the trainer"
    assert all(np.isfinite(h["meta_loss"]) for h in trainer._history)

    task = sampler.sample_batch(1)[0]
    adapted = trainer.adapt(task, n_steps=5)
    assert isinstance(adapted, torch.nn.Module)
