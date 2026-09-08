"""Tests for pinneapple_physics.pde_environment.turbulence_selector.

get_turbulence_selector's job is pure dispatch/validation over the real
closures in turbulence_presets.py (PINN) and the real 'Cs' LBM parameter --
these tests check it returns the right *type*/*value* for every valid
(model, solver_family) combination and raises ValueError, naming the
unsupported closure, for every invalid one.
"""
import pytest

from pinneapple_physics.pde_environment.turbulence_presets import (
    KOmegaSSTResiduals,
    WALEResiduals,
)
from pinneapple_physics.pde_environment.turbulence_selector import (
    TurbulenceModel,
    get_turbulence_closure,
)


# ---------------------------------------------------------------------------
# solver_family="pinn"
# ---------------------------------------------------------------------------

def test_pinn_laminar_is_none():
    assert get_turbulence_closure(TurbulenceModel.LAMINAR, solver_family="pinn") is None


def test_pinn_laminar_accepts_string_value():
    assert get_turbulence_closure("laminar", solver_family="pinn") is None


def test_pinn_rans_k_omega_sst_returns_configured_instance():
    closure = get_turbulence_closure(
        TurbulenceModel.RANS_K_OMEGA_SST, dim=3, solver_family="pinn", nu=2e-5, rho=1.2,
    )
    assert isinstance(closure, KOmegaSSTResiduals)
    assert closure.nu == pytest.approx(2e-5)
    assert closure.rho == pytest.approx(1.2)


def test_pinn_rans_k_omega_sst_2d_also_returns_same_class():
    # KOmegaSSTResiduals dispatches 2D/3D at __call__ time (x_col.shape[1]),
    # not at construction -- dim=2 is a valid request and must still return
    # a usable KOmegaSSTResiduals.
    closure = get_turbulence_closure(
        TurbulenceModel.RANS_K_OMEGA_SST, dim=2, solver_family="pinn",
    )
    assert isinstance(closure, KOmegaSSTResiduals)


def test_pinn_les_wale_returns_configured_instance():
    closure = get_turbulence_closure(
        TurbulenceModel.LES_WALE, dim=3, solver_family="pinn", nu=1e-5, cw=0.325, delta=0.01,
    )
    assert isinstance(closure, WALEResiduals)
    assert closure.nu == pytest.approx(1e-5)
    assert closure.cw == pytest.approx(0.325)


def test_pinn_les_wale_rejects_2d():
    with pytest.raises(ValueError, match="3D"):
        get_turbulence_closure(TurbulenceModel.LES_WALE, dim=2, solver_family="pinn")


def test_pinn_les_smagorinsky_raises_lbm_native_error():
    with pytest.raises(ValueError, match="LBM"):
        get_turbulence_closure(TurbulenceModel.LES_SMAGORINSKY, solver_family="pinn")


# ---------------------------------------------------------------------------
# solver_family="lbm"
# ---------------------------------------------------------------------------

def test_lbm_laminar_is_zero():
    val = get_turbulence_closure(TurbulenceModel.LAMINAR, solver_family="lbm")
    assert val == 0.0
    assert isinstance(val, float)


def test_lbm_les_smagorinsky_default_cs():
    val = get_turbulence_closure(TurbulenceModel.LES_SMAGORINSKY, solver_family="lbm")
    assert val == pytest.approx(0.17)


def test_lbm_les_smagorinsky_custom_cs():
    val = get_turbulence_closure(TurbulenceModel.LES_SMAGORINSKY, solver_family="lbm", Cs=0.12)
    assert val == pytest.approx(0.12)


def test_lbm_rans_k_omega_sst_raises_pinn_native_error():
    with pytest.raises(ValueError, match="PINN"):
        get_turbulence_closure(TurbulenceModel.RANS_K_OMEGA_SST, solver_family="lbm")


def test_lbm_les_wale_raises_pinn_native_error():
    with pytest.raises(ValueError, match="PINN"):
        get_turbulence_closure(TurbulenceModel.LES_WALE, solver_family="lbm")


# ---------------------------------------------------------------------------
# Generic validation
# ---------------------------------------------------------------------------

def test_unknown_solver_family_raises():
    with pytest.raises(ValueError):
        get_turbulence_closure(TurbulenceModel.LAMINAR, solver_family="fem")


def test_invalid_dim_raises():
    with pytest.raises(ValueError):
        get_turbulence_closure(TurbulenceModel.LAMINAR, dim=4, solver_family="pinn")


def test_invalid_model_string_raises():
    with pytest.raises(ValueError):
        get_turbulence_closure("not_a_model", solver_family="pinn")
