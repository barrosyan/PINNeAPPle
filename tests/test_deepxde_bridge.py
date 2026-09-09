"""Tests for pinneapple_physics.deepxde_bridge.

``deepxde`` was installed into this environment specifically to validate this
bridge against the real library (``pip install deepxde``, via
``uv pip install --python .venv/bin/python deepxde``), so the tests below
exercise the real translation *and* real DeepXDE training end to end:
``get_preset("burgers_1d")`` -> :func:`solve_with_deepxde` -> a real,
trained ``dde.Model`` whose loss actually decreases and which can be queried
with ``.predict()``.

They also always cover the core optional-dependency contract (importing the
bridge never requires deepxde) and the clear-error paths for unsupported
``pde.kind``/condition shapes, matching this repo's established convention
(see ``tests/test_physicsnemo_bridge.py``) for how an optional-dependency
bridge should be tested regardless of whether the dependency happens to be
installed in a given environment.
"""
from __future__ import annotations

import numpy as np
import pytest


def _deepxde_available() -> bool:
    try:
        import deepxde  # noqa: F401
        return True
    except ImportError:
        return False


DEEPXDE_AVAILABLE = _deepxde_available()


# ---------------------------------------------------------------------------
# Import contract
# ---------------------------------------------------------------------------

def test_import_succeeds_without_deepxde():
    """Core contract: importing the bridge never requires deepxde."""
    import pinneapple_physics.deepxde_bridge as bridge

    expected = {
        "DeepXDESolveResult",
        "SUPPORTED_PDE_KINDS",
        "is_deepxde_available",
        "require_deepxde",
        "solve_with_deepxde",
    }
    assert expected.issubset(set(bridge.__all__))


def test_is_deepxde_available_matches_real_environment():
    from pinneapple_physics.deepxde_bridge import is_deepxde_available

    assert is_deepxde_available() == DEEPXDE_AVAILABLE


@pytest.mark.skipif(
    DEEPXDE_AVAILABLE,
    reason="deepxde is installed; this test targets the missing-dependency error path",
)
def test_require_deepxde_raises_clear_error_without_deepxde():
    from pinneapple_physics.deepxde_bridge import require_deepxde

    with pytest.raises(ImportError, match="deepxde is required"):
        require_deepxde()


@pytest.mark.skipif(
    DEEPXDE_AVAILABLE,
    reason="deepxde is installed; this test targets the missing-dependency error path",
)
def test_solve_with_deepxde_raises_clear_error_without_deepxde():
    from pinneapple_physics.deepxde_bridge import solve_with_deepxde
    from pinneapple_physics.pde_environment.presets import get_preset

    spec = get_preset("burgers_1d")
    with pytest.raises(ImportError, match="deepxde is required"):
        solve_with_deepxde(spec, epochs=1)


# ---------------------------------------------------------------------------
# Translation error paths (do not require deepxde to be importable to reason
# about -- but DirichletBC/IC construction below does touch real deepxde
# objects once geometry/pde translation succeeds, so these run only when
# deepxde is installed).
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DEEPXDE_AVAILABLE, reason="requires deepxde installed")
def test_unsupported_pde_kind_raises_clear_not_implemented_error():
    from dataclasses import replace

    from pinneapple_physics.deepxde_bridge import solve_with_deepxde
    from pinneapple_physics.pde_environment.presets import get_preset

    spec = get_preset("poisson_2d")
    bogus_pde = replace(spec.pde, kind="navier_stokes_incompressible")
    bogus_spec = replace(spec, pde=bogus_pde)

    with pytest.raises(NotImplementedError, match="navier_stokes_incompressible"):
        solve_with_deepxde(bogus_spec, epochs=1)


@pytest.mark.skipif(not DEEPXDE_AVAILABLE, reason="requires deepxde installed")
def test_unsupported_condition_tag_raises_clear_not_implemented_error():
    from dataclasses import replace

    from pinneapple_physics.deepxde_bridge import solve_with_deepxde
    from pinneapple_physics.pde_environment.conditions import DirichletBC
    from pinneapple_physics.pde_environment.presets import get_preset

    spec = get_preset("poisson_2d")
    weird_bc = DirichletBC(
        "weird",
        fields=("u",),
        selector_type="tag",
        selector={"tag": "inlet"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
    )
    weird_spec = replace(spec, conditions=(weird_bc,))

    with pytest.raises(NotImplementedError, match="tag"):
        solve_with_deepxde(weird_spec, epochs=1)


# ---------------------------------------------------------------------------
# Real end-to-end training: burgers_1d
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DEEPXDE_AVAILABLE, reason="requires deepxde installed")
def test_solve_with_deepxde_burgers_1d_trains_for_real():
    import deepxde as dde

    from pinneapple_physics.deepxde_bridge import DeepXDESolveResult, solve_with_deepxde
    from pinneapple_physics.pde_environment.presets import get_preset

    spec = get_preset("burgers_1d")

    result = solve_with_deepxde(
        spec,
        n_domain=256,
        n_boundary=32,
        n_initial=32,
        layers=(20, 20, 20),
        epochs=300,
        verbose=0,
        display_every=300,
    )

    assert isinstance(result, DeepXDESolveResult)
    assert result.spec is spec
    assert isinstance(result.deepxde_model, dde.Model)
    assert np.isfinite(result.final_loss)
    assert result.final_loss >= 0.0
    assert result.train_time_s > 0.0

    # The model is real and callable: .predict() works and returns the right shape.
    x_query = np.array(
        [[0.0, 0.0], [0.5, 0.25], [-0.5, 0.75]], dtype=np.float32
    )
    pred = result.deepxde_model.predict(x_query)
    assert pred.shape == (3, 1)
    assert np.all(np.isfinite(pred))

    # The initial condition u(x, 0) = -sin(pi x) is a strong training signal
    # even after very few epochs: u(0, 0) should already be close to 0 and
    # u(-0.5, 0)/u(0.5, 0) should have opposite signs, sanity-checking that
    # real learning (not random-init noise) happened.
    ic_query = np.array([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float32)
    ic_pred = result.deepxde_model.predict(ic_query)
    assert ic_pred[0, 0] > 0.0  # -sin(-pi/2) = 1 > 0
    assert ic_pred[1, 0] < 0.0  # -sin(pi/2) = -1 < 0


@pytest.mark.skipif(not DEEPXDE_AVAILABLE, reason="requires deepxde installed")
def test_solve_with_deepxde_loss_actually_decreases():
    """Not a no-op: training for more iterations reduces the training loss."""
    from pinneapple_physics.deepxde_bridge import solve_with_deepxde
    from pinneapple_physics.pde_environment.presets import get_preset

    spec = get_preset("burgers_1d")

    short = solve_with_deepxde(
        spec,
        n_domain=256,
        n_boundary=32,
        n_initial=32,
        layers=(20, 20, 20),
        epochs=10,
        verbose=0,
        display_every=1000,
    )
    long = solve_with_deepxde(
        spec,
        n_domain=256,
        n_boundary=32,
        n_initial=32,
        layers=(20, 20, 20),
        epochs=400,
        verbose=0,
        display_every=1000,
    )

    assert long.final_loss < short.final_loss


# ---------------------------------------------------------------------------
# poisson_2d translation (a second, structurally different pde.kind)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DEEPXDE_AVAILABLE, reason="requires deepxde installed")
def test_solve_with_deepxde_poisson_2d_trains_for_real():
    import deepxde as dde

    from pinneapple_physics.deepxde_bridge import solve_with_deepxde
    from pinneapple_physics.pde_environment.presets import get_preset

    spec = get_preset("poisson_2d")

    result = solve_with_deepxde(
        spec,
        n_domain=200,
        n_boundary=40,
        layers=(20, 20),
        epochs=150,
        verbose=0,
        display_every=150,
    )

    assert isinstance(result.deepxde_model, dde.Model)
    assert np.isfinite(result.final_loss)

    x_query = np.array([[0.5, 0.5], [0.1, 0.9]], dtype=np.float32)
    pred = result.deepxde_model.predict(x_query)
    assert pred.shape == (2, 1)
    assert np.all(np.isfinite(pred))
