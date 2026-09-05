"""Tests for ``PhysicsGuardrail``'s dimensional-analysis and conservation
checks (``pinneapple_llm/guardrail.py``), added alongside those two checks
themselves -- see ``ROADMAP_PHYSICS_AI_HUB.md`` section P3.2 for what was
and wasn't built.

Same verification discipline as ``tests/test_manufactured_solutions.py``
(look for "audit_physics" there for the style this mirrors): for every
new check, construct a KNOWN-GOOD case (dimensionally consistent
parameters; a numerically-EXACT divergence-free velocity field or a
numerically-exact harmonic/source-free steady-conduction solution) and a
DELIBERATELY-WRONG case (an inconsistent parameter combination; a field
with real, nonzero divergence or a non-harmonic temperature field
implying a hidden source) and assert the check tells them apart clearly.
A check that reports "fine" for both the good and the bad case is exactly
as broken as one that reports "broken" for both -- this file's whole
point is to rule that out for every new check added this pass.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from pinneapple_physics.pde_environment.presets.registry import get_preset
from pinneapple_physics.pde_environment.spec import PDETermSpec, ProblemSpec
from pinneapple_llm.guardrail import PhysicsGuardrail


class _ExactFn(nn.Module):
    """Wraps a plain torch-differentiable function as a fake "model" --
    same helper ``tests/test_manufactured_solutions.py`` uses for the
    same reason: ``PhysicsGuardrail``'s checks call ``model(x)`` and (for
    the residual/heat-flux checks) need real autograd through it, which a
    closed-form torch expression provides without needing an actually
    trained network."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x)


def _probe_spec(kind: str, coords, fields, params, domain_bounds=None) -> ProblemSpec:
    pde = PDETermSpec(kind=kind, fields=fields, coords=coords, params=params)
    return ProblemSpec(
        name=f"_probe_{kind}", dim=len(coords), coords=coords, fields=fields, pde=pde,
        domain_bounds=domain_bounds or {c: (0.0, 1.0) for c in coords},
    )


# ---------------------------------------------------------------------------
# Dimensional analysis: diffusion / advection-diffusion / Burgers family
# ---------------------------------------------------------------------------

def test_dimensional_analysis_diffusion_family_positive_alpha_passes():
    spec = _probe_spec("heat_equation", ("x", "y", "t"), ("T",), {"alpha": 1e-4})
    result = PhysicsGuardrail(spec)._check_dimensional_analysis()
    assert result is not None and result.name == "dimensional_analysis"
    assert result.passed, result.detail


def test_dimensional_analysis_diffusion_family_negative_alpha_fails():
    spec = _probe_spec("heat_equation", ("x", "y", "t"), ("T",), {"alpha": -1e-4})
    result = PhysicsGuardrail(spec)._check_dimensional_analysis()
    assert not result.passed
    assert "alpha" in result.detail


def test_dimensional_analysis_burgers_negative_nu_fails_but_positive_passes():
    good = _probe_spec("burgers", ("x", "t"), ("u",), {"nu": 0.01})
    bad = _probe_spec("burgers", ("x", "t"), ("u",), {"nu": -0.01})
    assert PhysicsGuardrail(good)._check_dimensional_analysis().passed
    assert not PhysicsGuardrail(bad)._check_dimensional_analysis().passed


# ---------------------------------------------------------------------------
# Dimensional analysis: incompressible Navier-Stokes family
# ---------------------------------------------------------------------------

def test_dimensional_analysis_navier_stokes_positive_Re_passes():
    spec = get_preset("channel_flow_3d")  # a real preset from this repo
    result = PhysicsGuardrail(spec)._check_dimensional_analysis()
    assert result.passed, result.detail
    assert "Re" in result.detail


def test_dimensional_analysis_navier_stokes_Re_and_inv_Re_must_be_reciprocal():
    consistent = _probe_spec(
        "navier_stokes_incompressible", ("x", "y"), ("u", "v", "p"),
        {"Re": 100.0, "inv_Re": 1.0 / 100.0},
    )
    inconsistent = _probe_spec(
        "navier_stokes_incompressible", ("x", "y"), ("u", "v", "p"),
        {"Re": 100.0, "inv_Re": 0.5},
    )
    assert PhysicsGuardrail(consistent)._check_dimensional_analysis().passed
    bad_result = PhysicsGuardrail(inconsistent)._check_dimensional_analysis()
    assert not bad_result.passed
    assert "reciprocal" in bad_result.detail


# ---------------------------------------------------------------------------
# Dimensional analysis: linear elasticity family -- the nu naming collision
# ---------------------------------------------------------------------------

def test_dimensional_analysis_elasticity_negative_poisson_ratio_is_physically_valid():
    """A real auxetic material (Poisson's ratio nu<0) is NOT unphysical --
    unlike the OLD global 'nu must be positive' heuristic, which would
    have wrongly rejected it because it conflates elasticity's nu
    (dimensionless Poisson ratio) with fluid dynamics' nu (kinematic
    viscosity, must be positive). The Lame parameter lambda computed from
    a negative nu is itself legitimately negative (checked via the bulk
    modulus lambda+2*mu/3 > 0, not lambda > 0 in isolation)."""
    E, nu = 2.1e11, -0.2
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    assert lam < 0  # sanity check on the fixture itself
    spec = _probe_spec(
        "linear_elasticity", ("x", "y"), ("ux", "uy"),
        {"E": E, "nu": nu, "lambda": lam, "mu": mu},
    )
    result = PhysicsGuardrail(spec)._check_dimensional_analysis()
    assert result.passed, result.detail


def test_dimensional_analysis_elasticity_poisson_ratio_out_of_range_fails():
    spec = _probe_spec(
        "linear_elasticity", ("x", "y"), ("ux", "uy"),
        {"E": 2.1e11, "nu": 0.9},  # outside the valid (-1, 0.5) isotropic range
    )
    result = PhysicsGuardrail(spec)._check_dimensional_analysis()
    assert not result.passed
    assert "Poisson" in result.detail


def test_dimensional_analysis_elasticity_lambda_inconsistent_with_E_nu_fails():
    """E, nu, mu are mutually consistent (mu=E/(2*(1+nu))) but lambda is a
    made-up number that does NOT satisfy the standard isotropic relation
    -- an internally-inconsistent spec the compiled residual would never
    itself catch (it just uses whatever lambda/mu it's given)."""
    E, nu = 2.1e11, 0.3
    mu = E / (2.0 * (1.0 + nu))
    spec = _probe_spec(
        "linear_elasticity", ("x", "y"), ("ux", "uy"),
        {"E": E, "nu": nu, "mu": mu, "lambda": 999.0},
    )
    result = PhysicsGuardrail(spec)._check_dimensional_analysis()
    assert not result.passed
    assert "isotropic relation" in result.detail


def test_dimensional_analysis_elasticity_real_preset_passes():
    spec = get_preset("linear_elasticity_3d")
    result = PhysicsGuardrail(spec)._check_dimensional_analysis()
    assert result.passed, result.detail


# ---------------------------------------------------------------------------
# Dimensional analysis: transient heat conduction -- alpha vs k/(rho*cp)
# ---------------------------------------------------------------------------

def test_dimensional_analysis_transient_heat_real_preset_is_self_consistent():
    """car_brake_thermal declares k, rho, cp AND a separately-derived
    alpha; the compiled heat_equation_transient residual only reads
    alpha directly (see compile.py), so this test is a real check that
    the preset's own author computed alpha = k/(rho*cp) correctly."""
    spec = get_preset("car_brake_thermal")
    assert spec.pde.kind == "heat_equation_transient"
    result = PhysicsGuardrail(spec)._check_dimensional_analysis()
    assert result.passed, result.detail
    assert "alpha==k/(rho*cp)" in result.detail


def test_dimensional_analysis_transient_heat_inconsistent_alpha_fails():
    from dataclasses import replace

    spec = get_preset("car_brake_thermal")
    bad_params = dict(spec.pde.params)
    bad_params["alpha"] = bad_params["alpha"] * 3.0  # break the k/(rho*cp) relation
    bad_spec = replace(spec, pde=replace(spec.pde, params=bad_params))
    result = PhysicsGuardrail(bad_spec)._check_dimensional_analysis()
    assert not result.passed
    assert "k/(rho*cp)" in result.detail


# ---------------------------------------------------------------------------
# Dimensional analysis: uncovered pde_kind falls back to the legacy heuristic
# ---------------------------------------------------------------------------

def test_uncovered_pde_kind_falls_back_to_legacy_positivity_heuristic():
    spec = _probe_spec("darcy", ("x", "y"), ("p",), {"k": -1.0})
    guardrail = PhysicsGuardrail(spec)
    assert guardrail._check_dimensional_analysis() is None  # not a covered family
    result = guardrail._check_parameter_sanity()
    assert result.name == "parameter_sanity"  # NOT "dimensional_analysis" -- honestly labelled
    assert not result.passed


def test_covered_pde_kind_reports_dimensional_analysis_name_not_legacy():
    spec = get_preset("cpu_heatsink_thermal")
    result = PhysicsGuardrail(spec)._check_parameter_sanity()
    assert result.name == "dimensional_analysis"


# ---------------------------------------------------------------------------
# Conservation: incompressible continuity (mass conservation)
# ---------------------------------------------------------------------------

def _curl_divergence_free_velocity(x: torch.Tensor) -> torch.Tensor:
    """U = curl(A), A = (0, 0, sin(x)*cos(y)*cos(z)) -- divergence-free
    EXACTLY, by the vector identity div(curl(A)) == 0 for any smooth A
    (verified directly below too: du/dx + dv/dy = 0 by construction)."""
    xx, yy, zz = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    u = -torch.sin(xx) * torch.sin(yy) * torch.cos(zz)   # dAz/dy
    v = -torch.cos(xx) * torch.cos(yy) * torch.cos(zz)   # -dAz/dx
    w = torch.zeros_like(u)
    p = torch.zeros_like(u)
    return torch.cat([u, v, w, p], dim=1)


def _nondivergence_free_velocity(x: torch.Tensor) -> torch.Tensor:
    """U = (x, y, z): div(U) = 3 everywhere, a clean, exactly-known,
    strongly non-conservative field for the contrast case."""
    xx, yy, zz = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    p = torch.zeros_like(xx)
    return torch.cat([xx, yy, zz, p], dim=1)


def test_conservation_mass_continuity_exact_divergence_free_field_is_near_zero():
    spec = get_preset("channel_flow_3d")  # real preset; domain x:(0,2) y:(0,1) z:(0,1)
    guardrail = PhysicsGuardrail(spec)
    result = guardrail._check_conservation_mass_continuity(_ExactFn(_curl_divergence_free_velocity))
    assert result.name == "conservation_mass_continuity"
    # Empirically-characterized MC noise floor at this n_points_per_face
    # (2048, this class's default) on this exact domain size/shape was
    # ~0.007-0.01 mean / ~0.03 max over 50 resamples (see guardrail.py's
    # class docstring) -- assert comfortably inside that, not just "< 1".
    assert result.value < 0.05, result.detail
    assert result.passed


def test_conservation_mass_continuity_nondivergence_free_field_is_clearly_flagged():
    spec = get_preset("channel_flow_3d")
    guardrail = PhysicsGuardrail(spec)
    result = guardrail._check_conservation_mass_continuity(_ExactFn(_nondivergence_free_velocity))
    assert not result.passed
    # The exact analytic imbalance ratio for this field on any box is 1.0
    # (all six faces contribute outward flux, nothing cancels) -- assert
    # clearly separated from the noise floor above, not just "not small".
    assert result.value > 0.9, result.detail


def test_conservation_mass_continuity_absent_for_uncovered_pde_kind():
    spec = get_preset("laplace_2d")
    guardrail = PhysicsGuardrail(spec)
    assert guardrail._check_conservation(_ExactFn(lambda x: x[:, 0:1])) is None


# ---------------------------------------------------------------------------
# Conservation: steady heat conduction, no source
# ---------------------------------------------------------------------------

def _harmonic_temperature(x: torch.Tensor) -> torch.Tensor:
    """T = e^x * cos(y): d2T/dx2 = e^x*cos(y), d2T/dy2 = -e^x*cos(y), and
    no z-dependence -- Laplacian is EXACTLY zero everywhere, in 3D."""
    return torch.exp(x[:, 0:1]) * torch.cos(x[:, 1:2])


def _nonharmonic_temperature(x: torch.Tensor) -> torch.Tensor:
    """T = x^2+y^2+z^2: Laplacian = 6 everywhere (implies a uniform
    volumetric source) -- NOT a steady, source-free solution."""
    return x[:, 0:1] ** 2 + x[:, 1:2] ** 2 + x[:, 2:3] ** 2


def test_conservation_heat_flux_exact_harmonic_field_is_near_zero():
    spec = get_preset("cpu_heatsink_thermal")  # real preset; k=205, domain 0.08x0.08x0.05
    guardrail = PhysicsGuardrail(spec)
    result = guardrail._check_conservation_heat_flux(_ExactFn(_harmonic_temperature))
    assert result.name == "conservation_heat_conduction"
    # Empirically this case's MC noise floor is far quieter than the
    # velocity-flux case (~1e-4 at 2048 points/face, see guardrail.py's
    # class docstring) -- still assert well inside the shared 0.15
    # threshold rather than exactly 0, since it IS a Monte-Carlo estimate.
    assert result.value < 0.01, result.detail
    assert result.passed


def test_conservation_heat_flux_nonharmonic_field_is_clearly_flagged():
    spec = get_preset("cpu_heatsink_thermal")
    guardrail = PhysicsGuardrail(spec)
    result = guardrail._check_conservation_heat_flux(_ExactFn(_nonharmonic_temperature))
    assert not result.passed
    assert result.value > 0.9, result.detail


def test_conservation_heat_flux_absent_for_uncovered_pde_kind():
    spec = get_preset("car_brake_thermal")  # heat_equation_transient, not in the steady-conduction family
    guardrail = PhysicsGuardrail(spec)
    assert guardrail._check_conservation(_ExactFn(lambda x: x[:, 0:1])) is None


def test_conservation_heat_flux_anisotropic_uses_per_axis_conductivity():
    """pcb_thermal is heat_equation_steady_anisotropic with k_x=k_y=0.3,
    k_z=0.25 -- confirm the check runs (per-axis k, not a single scalar
    k) and reports a small imbalance for an exact harmonic field."""
    spec = get_preset("pcb_thermal")
    assert spec.pde.kind == "heat_equation_steady_anisotropic"
    guardrail = PhysicsGuardrail(spec)

    def harmonic_2d(x):
        # T=x^2-y^2 is harmonic for ANY (isotropic or anisotropic-equal)
        # k_x, k_y with k_x=k_y; pcb_thermal's k_x==k_y==0.3 here.
        return x[:, 0:1] ** 2 - x[:, 1:2] ** 2

    result = guardrail._check_conservation_heat_flux(_ExactFn(harmonic_2d))
    assert result.passed, result.detail


# ---------------------------------------------------------------------------
# Full end-to-end smoke tests against real presets from this repo
# ---------------------------------------------------------------------------

def test_end_to_end_laplace_2d_exact_solution_is_trustworthy():
    """laplace_2d has no physical parameters at all (falls back to the
    legacy parameter_sanity heuristic, trivially passing on an empty
    dict) and no known conservation law in this module's covered set --
    exercises the "most checks absent" end of the spectrum."""
    spec = get_preset("laplace_2d")
    model = _ExactFn(lambda x: (x[:, 0:1] ** 2 - x[:, 1:2] ** 2))  # exact harmonic solution
    report = PhysicsGuardrail(spec).check(model)
    assert report.checked_names == ["parameter_sanity", "pde_residual"]
    assert report.trustworthy, report.summary()


def test_end_to_end_cpu_heatsink_thermal_exact_solution_is_trustworthy():
    """cpu_heatsink_thermal (heat_equation_steady, k=205) exercises the
    "everything real: dimensional_analysis + pde_residual +
    conservation_heat_conduction, all genuinely passing" end -- a full,
    real preset from this repo, an exact closed-form solution plugged in
    directly (no training), checked end-to-end through check()."""
    spec = get_preset("cpu_heatsink_thermal")
    model = _ExactFn(lambda x: (x[:, 0:1] ** 2 + x[:, 1:2] ** 2 - 2 * x[:, 2:3] ** 2))  # exact harmonic T
    report = PhysicsGuardrail(spec).check(model)
    assert set(report.checked_names) == {"dimensional_analysis", "pde_residual", "conservation_heat_conduction"}
    assert report.trustworthy, report.summary()


def test_end_to_end_navier_stokes_incompressible_runs_and_reports_conservation():
    """channel_flow_3d (navier_stokes_incompressible): the checks are
    genuinely independent of each other -- a divergence-free-but-not-a-
    real-Navier-Stokes-solution field passes conservation while still
    correctly failing the (unrelated) PDE residual check, demonstrating
    check() doesn't conflate the two signals into one pass/fail bit."""
    spec = get_preset("channel_flow_3d")
    model = _ExactFn(_curl_divergence_free_velocity)
    report = PhysicsGuardrail(spec).check(model)
    assert set(report.checked_names) == {"dimensional_analysis", "pde_residual", "conservation_mass_continuity"}
    by_name = {c.name: c for c in report.checks}
    assert by_name["dimensional_analysis"].passed
    assert by_name["conservation_mass_continuity"].passed
    assert not by_name["pde_residual"].passed  # a divergence-free field need not satisfy full NS momentum
    assert not report.trustworthy  # compound claim correctly reflects the one real failure


def test_guardrail_report_skipped_reflects_reference_data_not_supplied():
    spec = get_preset("laplace_2d")
    model = _ExactFn(lambda x: (x[:, 0:1] ** 2 - x[:, 1:2] ** 2))
    report = PhysicsGuardrail(spec).check(model)
    assert "reference_data_match" in report.skipped
    assert "reference_data_match" not in report.checked_names


# ---------------------------------------------------------------------------
# Reference-data auto-fetch from a UPD zarr store on disk
#
# Audit result (full writeup in ROADMAP_PHYSICS_AI_HUB.md section P3.2):
# pinneapple_pdb has NO named-benchmark-dataset catalog. Its
# PhysicalDatasetBuilder only ever builds datasets by querying external
# Earth-data hubs (NASA CMR / earthaccess) and writing the result to disk;
# its catalog_path parquet is an output manifest of what a given build
# produced (keyed by a content hash 'uid'), not an input registry a caller
# could resolve a friendly name against without having built it first.
# schema_templates() is the only name -> dict lookup in the package, and it
# returns physical-schema metadata (governing equations, units policy),
# never x/y data arrays. So PhysicsGuardrail's auto-fetch (below) resolves
# a real FILE PATH to the on-disk UPD zarr format
# PhysicalDatasetBuilder._write_upd actually writes for every shard (a
# plain xr.Dataset.to_zarr(...) store) -- not a name from a catalog. These
# tests build one for real with plain xarray (the identical on-disk shape
# _write_upd produces) and read it back end-to-end -- no mocks.
# ---------------------------------------------------------------------------

def _write_real_upd_zarr(path, n: int = 6):
    """Build and write a real xr.Dataset to `path` in exactly the format
    PhysicalDatasetBuilder._write_upd uses (``ds.to_zarr(path, mode="w")``)
    -- an (x, y) grid plus a data variable 'u' = x**2 - y**2, the same
    exact harmonic solution used by
    ``test_end_to_end_laplace_2d_exact_solution_is_trustworthy`` above, so
    a model implementing that exact solution should score ~0 RMSE against
    it."""
    import xarray as xr

    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    uu = xx ** 2 - yy ** 2
    ds = xr.Dataset(
        data_vars={"u": (("x", "y"), uu.astype("float32"))},
        coords={"x": xs.astype("float32"), "y": ys.astype("float32")},
    )
    ds.to_zarr(str(path), mode="w")
    return ds


def test_load_reference_from_upd_zarr_shapes_and_values_match_manual_extraction(tmp_path):
    spec = get_preset("laplace_2d")  # coords=('x','y'), fields=('u',)
    zarr_path = tmp_path / "laplace_ref.zarr"
    _write_real_upd_zarr(zarr_path)

    guardrail = PhysicsGuardrail(spec)
    ref_x, ref_y = guardrail._load_reference_from_upd_zarr(str(zarr_path))
    assert ref_x.shape == (36, 2)
    assert ref_y.shape == (36, 1)
    # y == x0^2 - x1^2 exactly, for every row (this is the exact solution
    # the fixture was built from) -- confirms columns weren't shuffled.
    assert np.allclose(ref_y[:, 0], ref_x[:, 0] ** 2 - ref_x[:, 1] ** 2, atol=1e-5)


def test_load_reference_from_upd_zarr_missing_variable_raises_keyerror(tmp_path):
    spec = get_preset("laplace_2d")
    zarr_path = tmp_path / "laplace_ref_missing.zarr"
    _write_real_upd_zarr(zarr_path)
    guardrail = PhysicsGuardrail(spec)
    with pytest.raises(KeyError):
        guardrail._load_reference_from_upd_zarr(str(zarr_path), y_vars=["not_a_real_field"])


def test_check_reference_dataset_path_matches_manual_reference_arrays_end_to_end(tmp_path):
    """The auto-fetch path (``reference_dataset_path=...``) must produce
    the SAME CheckResult (same RMSE, same pass/fail, same set of checked
    names) as manually loading the identical underlying data and passing
    ``reference_x``/``reference_y`` directly -- this is the core contract
    of an ADDITIVE alternative API: the manual path must keep working
    unchanged, and the two paths must agree when fed the same data."""
    spec = get_preset("laplace_2d")
    zarr_path = tmp_path / "laplace_ref_e2e.zarr"
    ds = _write_real_upd_zarr(zarr_path)
    model = _ExactFn(lambda x: (x[:, 0:1] ** 2 - x[:, 1:2] ** 2))

    # manual path: extract the identical arrays by hand
    df = ds.to_dataframe().reset_index()
    manual_x = df[["x", "y"]].to_numpy(dtype="float32")
    manual_y = df[["u"]].to_numpy(dtype="float32")
    manual_report = PhysicsGuardrail(spec).check(
        model, reference_x=manual_x, reference_y=manual_y, reference_rmse_threshold=1e-3,
    )

    # auto-fetch path: same UPD zarr store, resolved by file path
    auto_report = PhysicsGuardrail(spec).check(
        model, reference_dataset_path=str(zarr_path), reference_rmse_threshold=1e-3,
    )

    manual_check = next(c for c in manual_report.checks if c.name == "reference_data_match")
    auto_check = next(c for c in auto_report.checks if c.name == "reference_data_match")
    assert manual_check.passed
    assert auto_check.passed
    assert math.isclose(manual_check.value, auto_check.value, rel_tol=1e-6)
    assert set(manual_report.checked_names) == set(auto_report.checked_names)
    assert manual_report.trustworthy and auto_report.trustworthy


def test_check_reference_dataset_path_and_manual_arrays_are_mutually_exclusive(tmp_path):
    spec = get_preset("laplace_2d")
    zarr_path = tmp_path / "laplace_ref_excl.zarr"
    _write_real_upd_zarr(zarr_path)
    model = _ExactFn(lambda x: (x[:, 0:1] ** 2 - x[:, 1:2] ** 2))
    with pytest.raises(ValueError, match="mutually exclusive"):
        PhysicsGuardrail(spec).check(
            model,
            reference_dataset_path=str(zarr_path),
            reference_x=np.zeros((1, 2), dtype="float32"),
            reference_y=np.zeros((1, 1), dtype="float32"),
            reference_rmse_threshold=1e-3,
        )


def test_check_reference_dataset_path_requires_rmse_threshold(tmp_path):
    spec = get_preset("laplace_2d")
    zarr_path = tmp_path / "laplace_ref_thresh.zarr"
    _write_real_upd_zarr(zarr_path)
    model = _ExactFn(lambda x: (x[:, 0:1] ** 2 - x[:, 1:2] ** 2))
    with pytest.raises(ValueError, match="reference_rmse_threshold"):
        PhysicsGuardrail(spec).check(model, reference_dataset_path=str(zarr_path))


def test_check_reference_dataset_path_custom_x_y_vars(tmp_path):
    """``reference_x_vars``/``reference_y_vars`` override the
    ``spec.coords``/``spec.fields`` defaults -- confirm a store using
    different variable names than the spec's own coords/fields can still
    be read by naming them explicitly."""
    import xarray as xr

    spec = get_preset("laplace_2d")
    zarr_path = tmp_path / "laplace_ref_customvars.zarr"
    xs = np.linspace(0.0, 1.0, 4)
    ys = np.linspace(0.0, 1.0, 4)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    uu = xx ** 2 - yy ** 2
    ds = xr.Dataset(
        data_vars={"solution": (("lon", "lat"), uu.astype("float32"))},
        coords={"lon": xs.astype("float32"), "lat": ys.astype("float32")},
    )
    ds.to_zarr(str(zarr_path), mode="w")

    model = _ExactFn(lambda x: (x[:, 0:1] ** 2 - x[:, 1:2] ** 2))
    report = PhysicsGuardrail(spec).check(
        model,
        reference_dataset_path=str(zarr_path),
        reference_x_vars=["lon", "lat"],
        reference_y_vars=["solution"],
        reference_rmse_threshold=1e-3,
    )
    check = next(c for c in report.checks if c.name == "reference_data_match")
    assert check.passed, check.detail
