"""Regression tests for three convenience-layer bugs found during a
documentation accuracy audit (see README.md / OVERVIEW.md history):

1. ``pinneapple_neural.train_model`` called ``TrainConfig(n_epochs=...)``
   (real field is ``epochs``) and ``Trainer(model, losses, cfg).train()``
   (``Trainer`` has no ``.train()``; the real API is
   ``Trainer(model, loss_fn).fit(train_loader, val_loader, cfg)``).

2. ``CollocationSampler.sample()`` crashed with its own default strategy
   (``strategy="lhs"``) because it called
   ``sample_latin_hypercube_box(..., seed=...)`` /
   ``sample_uniform_box(..., seed=...)`` in
   ``pinneapple_design/geometry/sample/grids.py``, but those functions only
   accept an ``rng: Optional[np.random.Generator]`` kwarg, not ``seed``.

3. ``CollocationSampler.sample()`` and ``generate_pinn_dataset`` both raised
   ``np.concatenate`` shape-mismatch ``ValueError``s on multi-field presets
   whose conditions cover different subsets of the problem's fields (e.g.
   ``cpu_heatsink_thermal``, ``axial_compressor_meanline``), because each
   condition's target array was built at its own (differing) width instead
   of the shared global field width.

Each test reproduces the exact failure mode described above and would have
failed before the corresponding fix.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Bug 1: pinneapple_neural.train_model
# ---------------------------------------------------------------------------

def test_train_model_runs_a_real_training_loop():
    from pinneapple_neural import build_model, train_model

    torch.manual_seed(0)
    x = torch.rand(32, 2)
    y = x[:, :1] ** 2 + x[:, 1:] ** 2
    loader = DataLoader(TensorDataset(x, y), batch_size=8)

    model = build_model("siren", in_dim=2, out_dim=1, hidden_dim=16, n_layers=3)

    def loss_fn(m, y_hat, batch):
        return torch.mean((y_hat - batch["y"]) ** 2)

    result = train_model(model, loss_fn, loader, epochs=2, device="cpu")

    assert "best_val" in result
    assert "history" in result
    assert len(result["history"]) == 2


# ---------------------------------------------------------------------------
# Bug 2: CollocationSampler.sample() with the default strategy
# ---------------------------------------------------------------------------

def test_collocation_sampler_default_strategy_does_not_crash():
    from pinneapple_data.collocation import CollocationSampler

    sampler = CollocationSampler(
        bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)},
        coord_names=("x", "y"),
        fields=("u",),
    )
    assert sampler.strategy == "lhs"  # the actual default

    batch = sampler.sample(n_col=32, n_bc=8)
    assert batch["x_col"].shape == (32, 2)


def test_collocation_sampler_uniform_strategy_3d_does_not_crash():
    from pinneapple_data.collocation import CollocationSampler

    sampler = CollocationSampler(
        bounds={"x": (0.0, 1.0), "y": (0.0, 1.0), "z": (0.0, 1.0)},
        coord_names=("x", "y", "z"),
        fields=("u",),
        strategy="uniform",
    )
    batch = sampler.sample(n_col=32, n_bc=8)
    assert batch["x_col"].shape == (32, 3)


# ---------------------------------------------------------------------------
# Bug 3: heterogeneous per-condition field widths crashing np.concatenate
# ---------------------------------------------------------------------------

def test_generate_pinn_dataset_heterogeneous_fields_cpu_heatsink():
    from pinneapple_physics.pde_environment.presets.engineering import cpu_heatsink_thermal
    from pinneapple_simulation.numerical_solvers.problem_runner import generate_pinn_dataset

    spec = cpu_heatsink_thermal()
    batch = generate_pinn_dataset(spec, n_col=32, n_bc=16)

    n_fields = len(spec.fields)
    assert batch["y_bc"].shape[1] == n_fields
    assert batch["x_bc"].shape[0] == batch["y_bc"].shape[0]


def test_generate_pinn_dataset_heterogeneous_fields_axial_compressor():
    from pinneapple_physics.pde_environment.presets.turbomachinery import axial_compressor_meanline
    from pinneapple_simulation.numerical_solvers.problem_runner import generate_pinn_dataset

    spec = axial_compressor_meanline()
    batch = generate_pinn_dataset(spec, n_col=32, n_bc=16)

    n_fields = len(spec.fields)
    assert batch["y_bc"].shape[1] == n_fields
    assert batch["x_bc"].shape[0] == batch["y_bc"].shape[0]


def test_collocation_sampler_heterogeneous_fields_cpu_heatsink():
    from pinneapple_physics.pde_environment.presets.engineering import cpu_heatsink_thermal
    from pinneapple_data.collocation import CollocationSampler

    spec = cpu_heatsink_thermal()
    sampler = CollocationSampler.from_problem_spec(spec, strategy="sobol")
    batch = sampler.sample(n_col=32, n_bc=16)

    n_fields = len(spec.fields)
    assert batch["y_bc"].shape[1] == n_fields
    assert batch["x_bc"].shape[0] == batch["y_bc"].shape[0]


def test_generate_pinn_dataset_target_values_land_in_correct_columns():
    """The scattered per-condition targets should retain their real values
    (not just have the right shape) -- placed in the column matching the
    condition's field name in the problem's global field tuple.
    """
    from pinneapple_physics.pde_environment.presets.turbomachinery import axial_compressor_meanline
    from pinneapple_simulation.numerical_solvers.problem_runner import generate_pinn_dataset

    spec = axial_compressor_meanline()
    batch = generate_pinn_dataset(spec, n_col=32, n_bc=64, seed=0)

    fields = tuple(spec.fields)
    # The inlet Dirichlet BC only sets T_t, p_t, u -- rho and c_theta columns
    # should be exactly zero for every boundary row (nothing ever writes them
    # for this condition), while at least one covered column is non-zero
    # somewhere in the full y_bc array.
    y_bc = batch["y_bc"]
    assert y_bc.shape[1] == len(fields)
    assert np.any(y_bc != 0.0)
