"""Smoke + shape tests for `pinneapple_analysis.inspection` -- the NDE
(non-destructive evaluation) PINN suite.

Each of the 6 modalities gets a synthetic-data generation smoke test and
a brief training smoke test (small network, few epochs -- this checks
"does it run, do shapes line up, does the loss trend down, are there no
NaNs", not full convergence). The eddy-current test additionally checks
that its synthetic generator's (r, z, A) outputs are shape/dtype
consistent with `eddy_current_fdm.solve_axisymmetric_eddy_current`'s own
contract, since that module builds directly on top of it.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pinneapple_analysis.inspection import (
    InspectionPINN,
    generate_eddy_current_synthetic,
    train_eddy_current,
    generate_magnetic_flux_leakage_synthetic,
    train_magnetic_flux_leakage,
    generate_acoustic_synthetic,
    train_acoustic,
    generate_emat_synthetic,
    train_emat,
    generate_guided_wave_synthetic,
    train_guided_wave_ultrasonic,
    generate_phased_array_synthetic,
    train_phased_array_ultrasonic,
)
from pinneapple_simulation.numerical_solvers.eddy_current_fdm import (
    solve_axisymmetric_eddy_current,
    annular_current_source,
    MU0,
)


def _no_nan(*arrays) -> None:
    for a in arrays:
        arr = np.asarray(a)
        assert not np.isnan(arr).any(), "found NaN in array"
        assert not np.isinf(arr).any(), "found Inf in array"


def _loss_trended_down(history, key: str = "val_total") -> None:
    """Compares the mean of the first third of epochs to the mean of the
    last third -- robust to the epoch-to-epoch noise a physics-residual
    loss on a barely-trained network can show, while still being a real
    check that training moved the needle."""
    vals = [h[key] for h in history]
    assert len(vals) >= 3, "need at least 3 epochs to check a trend"
    _no_nan(vals)
    third = max(1, len(vals) // 3)
    first_third = float(np.mean(vals[:third]))
    last_third = float(np.mean(vals[-third:]))
    assert last_third < first_third, (
        f"loss did not trend down: first-third mean={first_third:.6g}, "
        f"last-third mean={last_third:.6g}"
    )


# --------------------------------------------------------------------------
# InspectionPINN base
# --------------------------------------------------------------------------

def test_inspection_pinn_forward_shape():
    model = InspectionPINN(in_dim=3, out_dim=2, hidden=[16, 16], modality="unit_test")
    x = torch.randn(11, 3)
    out = model(x)
    assert out.y.shape == (11, 2)
    _no_nan(out.y.detach().numpy())


# --------------------------------------------------------------------------
# Eddy current
# --------------------------------------------------------------------------

def test_eddy_current_synthetic_matches_solver_convention():
    data = generate_eddy_current_synthetic(
        nr=14, nz=20, n_supervised=60, n_collocation=40, scan_positions=[0.0, 0.005],
    )
    r, z = data["r"], data["z"]
    assert r.ndim == 1 and z.ndim == 1

    # Rebuild the exact same defect-free solve independently via the
    # underlying solver's own public functional API, and check
    # generate_eddy_current_synthetic's own baseline solve matches shape
    # and dtype convention (same grid, same complex dtype).
    mu = MU0 * np.ones((len(r), len(z)))
    sigma = np.zeros((len(r), len(z)))
    sigma[np.meshgrid(r, z, indexing="ij")[1] < 0.0] = data["sigma_bulk"]
    J = annular_current_source(r, z, 0.02, 0.012, 0.004, 0.004, 100.0, 1.0)
    A_ref = solve_axisymmetric_eddy_current(r, z, data["omega"], mu, sigma, J)

    assert data["A_baseline"].shape == A_ref.shape == (len(r), len(z))
    assert np.iscomplexobj(data["A_baseline"]) and np.iscomplexobj(A_ref)

    assert data["x"].shape == (60, 2)
    assert data["y"].shape == (60, 2)
    assert data["x_col"].shape == (40, 2)
    assert data["scan_signal"].shape == (2,)
    _no_nan(data["A_baseline"].real, data["A_baseline"].imag, data["x"], data["y"], data["x_col"])


def test_train_eddy_current_smoke():
    data = generate_eddy_current_synthetic(nr=14, nz=20, n_supervised=80, n_collocation=50, scan_positions=[0.0, 0.006])
    out = train_eddy_current(
        synthetic_data=data, hidden=[16, 16], epochs=9, lr=1e-3, batch_size=16,
        log_dir="examples/_out/test_inspection", run_name="eddy_current_test",
    )
    history = out["train_out"]["history"]
    _loss_trended_down(history)

    model = out["model"]
    x = torch.as_tensor(data["x"][:5], dtype=torch.float32)
    y_hat = model.predict(x)
    assert y_hat.shape == (5, 2)
    _no_nan(y_hat.detach().numpy())


# --------------------------------------------------------------------------
# Magnetic flux leakage
# --------------------------------------------------------------------------

def test_magnetic_flux_leakage_synthetic_shapes():
    data = generate_magnetic_flux_leakage_synthetic(nx=18, ny=8, n_picard=3, n_supervised=50, n_collocation=30)
    assert data["phi"].shape == (18, 8)
    assert data["mu"].shape == (18, 8)
    assert data["scan_signal"].shape == (18,)
    _no_nan(data["phi"], data["mu"], data["scan_signal"])


def test_train_magnetic_flux_leakage_smoke():
    data = generate_magnetic_flux_leakage_synthetic(nx=18, ny=8, n_picard=3, n_supervised=60, n_collocation=40)
    out = train_magnetic_flux_leakage(
        synthetic_data=data, hidden=[16, 16], epochs=25, lr=3e-3, batch_size=16,
        log_dir="examples/_out/test_inspection", run_name="mfl_test",
    )
    _loss_trended_down(out["train_out"]["history"])

    model = out["model"]
    x = torch.as_tensor(data["x"][:5], dtype=torch.float32)
    y_hat = model.predict(x)
    assert y_hat.shape == (5, 1)
    _no_nan(y_hat.detach().numpy())


# --------------------------------------------------------------------------
# Acoustic
# --------------------------------------------------------------------------

def test_acoustic_synthetic_shapes():
    data = generate_acoustic_synthetic(n_points=24, duration=0.0004, n_supervised=60, n_collocation=40)
    nt, nx = data["u_field"].shape
    assert nx == 24
    assert data["freqs"].shape == data["spectrum_baseline"].shape == data["spectrum_defect"].shape
    _no_nan(data["u_field"], data["spectrum_baseline"], data["spectrum_defect"])


def test_train_acoustic_smoke():
    data = generate_acoustic_synthetic(n_points=24, duration=0.0004, n_supervised=60, n_collocation=40)
    out = train_acoustic(
        synthetic_data=data, hidden=[16, 16], epochs=18, lr=2e-3, batch_size=16,
        log_dir="examples/_out/test_inspection", run_name="acoustic_test",
    )
    _loss_trended_down(out["train_out"]["history"])

    model = out["model"]
    x = torch.as_tensor(data["x"][:5], dtype=torch.float32)
    y_hat = model.predict(x)
    assert y_hat.shape == (5, 1)
    _no_nan(y_hat.detach().numpy())


# --------------------------------------------------------------------------
# EMAT
# --------------------------------------------------------------------------

def test_emat_synthetic_shapes():
    data = generate_emat_synthetic(n_points=24, duration=0.00002, n_supervised=60, n_collocation=40)
    nt, nx = data["u_field"].shape
    assert nx == 24
    assert data["F_field"].shape == (nt, nx)
    _no_nan(data["u_field"], data["F_field"])


def test_train_emat_smoke():
    data = generate_emat_synthetic(n_points=24, duration=0.00002, n_supervised=60, n_collocation=40)
    out = train_emat(
        synthetic_data=data, hidden=[16, 16], epochs=18, lr=1e-3, batch_size=16,
        log_dir="examples/_out/test_inspection", run_name="emat_test",
    )
    _loss_trended_down(out["train_out"]["history"])

    model = out["model"]
    x = torch.as_tensor(data["x"][:5], dtype=torch.float32)
    y_hat = model.predict(x)
    assert y_hat.shape == (5, 1)
    _no_nan(y_hat.detach().numpy())


# --------------------------------------------------------------------------
# Guided-wave ultrasonic
# --------------------------------------------------------------------------

def test_guided_wave_synthetic_shapes():
    data = generate_guided_wave_synthetic(n_points=40, duration=0.0002, n_supervised=60, n_collocation=40)
    nt, nx = data["u_field"].shape
    assert nx == 40
    assert data["beta_field"].shape == (40,)
    _no_nan(data["u_field"], data["beta_field"])


def test_train_guided_wave_ultrasonic_smoke():
    data = generate_guided_wave_synthetic(n_points=40, duration=0.0002, n_supervised=60, n_collocation=40)
    out = train_guided_wave_ultrasonic(
        synthetic_data=data, hidden=[16, 16], epochs=18, lr=1e-3, batch_size=16,
        log_dir="examples/_out/test_inspection", run_name="guided_wave_test",
    )
    _loss_trended_down(out["train_out"]["history"])

    model = out["model"]
    x = torch.as_tensor(data["x"][:5], dtype=torch.float32)
    y_hat = model.predict(x)
    assert y_hat.shape == (5, 1)
    _no_nan(y_hat.detach().numpy())


# --------------------------------------------------------------------------
# Phased-array ultrasonic
# --------------------------------------------------------------------------

def test_phased_array_synthetic_shapes():
    data = generate_phased_array_synthetic(
        n_elements=8, n_time=120, nx_image=10, nz_image=10, n_supervised=50, n_collocation=30,
    )
    assert data["image"].shape == (10, 10)
    assert data["raw_traces"].shape == (8, 120)
    _no_nan(data["image"], data["raw_traces"])


def test_train_phased_array_ultrasonic_smoke():
    data = generate_phased_array_synthetic(
        n_elements=8, n_time=120, nx_image=10, nz_image=10, n_supervised=60, n_collocation=40,
    )
    out = train_phased_array_ultrasonic(
        synthetic_data=data, hidden=[16, 16], epochs=18, lr=2e-3, batch_size=16,
        log_dir="examples/_out/test_inspection", run_name="phased_array_test",
    )
    _loss_trended_down(out["train_out"]["history"])

    model = out["model"]
    x = torch.as_tensor(data["x"][:5], dtype=torch.float32)
    y_hat = model.predict(x)
    assert y_hat.shape == (5, 1)
    _no_nan(y_hat.detach().numpy())
