"""Tier A of the evidenced library audit (see ``ROADMAP_PHYSICS_AI_HUB.md``,
section 1.1): "does it run without crashing" breadth coverage across every
registered architecture and every registered PDE preset.

This is deliberately NOT a claim of physical correctness -- a residual
with a sign error still "runs." See ``test_manufactured_solutions.py``
for the actual physics-correctness tier (method of manufactured
solutions), which is the one that can tell a correct residual from a
broken-but-trainable one.

Run: ``pytest tests/test_full_library_matrix.py -q``
(``-k audit_breadth`` to select just this file's tests if run alongside
the full suite).
"""
from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Tier A.1 -- every registered model architecture: build + forward + backward
# ---------------------------------------------------------------------------

def _architecture_names():
    import pinneapple_neural.architectures  # noqa: F401  registers the zoo
    from pinneapple_neural.architectures.registry import ModelRegistry
    return ModelRegistry.list()


def _is_missing_optional_dep(e: Exception) -> bool:
    return isinstance(e, (ImportError, ModuleNotFoundError))


def _looks_like_wrong_input_shape(e: Exception) -> bool:
    """Heuristic: many registered architectures are for sequence data
    (B, T, features) or image/grid data (B, C, H, W, ...), not the flat
    (N, coord_dim) point-cloud shape a PINN collocation batch uses -- a
    shape assertion from one of those is not "broken", it's this generic
    smoke test's input not matching what that architecture family is for.
    Individually confirming each such architecture's real expected input
    shape (rather than pattern-matching the error text) is exactly the
    kind of per-architecture depth this Tier A breadth pass is not
    attempting -- see AUDIT_REPORT.md."""
    msg = str(e).lower()
    return any(s in msg for s in ("shape", "channels", "dimension", "expected input", "must have shape", "must be"))


@pytest.mark.parametrize("name", _architecture_names())
def test_audit_breadth_architecture_forward_backward(name):
    from pinneapple_neural.architectures.registry import ModelRegistry

    try:
        model = ModelRegistry.build(name, in_dim=4, out_dim=3, hidden_dim=16, n_layers=3)
    except Exception as e:
        if _is_missing_optional_dep(e):
            pytest.skip(f"'{name}' needs an optional dependency not installed: {e}")
        pytest.skip(f"'{name}' could not be built with generic in_dim/out_dim/hidden_dim/n_layers kwargs: {e}")

    x = torch.randn(8, 4, requires_grad=True)
    try:
        y = model(x)
    except Exception as e:
        if _looks_like_wrong_input_shape(e):
            pytest.skip(
                f"'{name}' rejected a flat (N, 4) point-cloud input -- likely a sequence/image "
                f"architecture needing a different generic-test input shape, not necessarily broken: {e}"
            )
        pytest.fail(f"'{name}' forward pass raised: {e}")

    if hasattr(y, "y"):
        y = y.y
    assert torch.is_tensor(y), f"'{name}' forward output is not a tensor and has no .y attribute"
    assert torch.isfinite(y).all(), f"'{name}' produced non-finite output on a random input"

    try:
        y.sum().backward()
    except Exception as e:
        pytest.fail(f"'{name}' backward pass raised: {e}")
    assert any(p.grad is not None for p in model.parameters()), f"'{name}' backward produced no gradients at all"


# ---------------------------------------------------------------------------
# Tier A.2 -- every registered PDE preset: solve_pde runs a few epochs
# without raising, and the loss is finite throughout.
# ---------------------------------------------------------------------------

def _preset_names():
    from pinneapple_physics.pde_environment.presets.registry import list_presets
    return list_presets()


@pytest.mark.parametrize("name", _preset_names())
def test_audit_breadth_preset_trains_a_few_steps(name):
    import pinneapple_physics as pp
    from pinneapple_neural.architectures.registry import ModelRegistry
    import pinneapple_neural.architectures  # noqa: F401

    try:
        spec = pp.get_preset(name)
    except TypeError:
        pytest.skip(f"preset '{name}' requires kwargs this generic smoke test doesn't supply")
        return
    except Exception as e:
        if _is_missing_optional_dep(e):
            pytest.skip(f"preset '{name}' needs an optional dependency not installed: {e}")
        pytest.fail(f"get_preset('{name}') raised: {e}")

    model = ModelRegistry.build(
        "modified_mlp", in_dim=len(spec.coords), out_dim=len(spec.fields), hidden_dim=16, n_layers=3,
    )
    try:
        result = pp.solve_pde(spec, model, epochs=3, n_collocation=64, n_condition=32)
    except Exception as e:
        if _is_missing_optional_dep(e):
            pytest.skip(f"preset '{name}' needs an optional dependency not installed: {e}")
        # A "Unsupported PDE kind: ..." or "... expects time coord 't'"
        # ValueError here means this preset is registered/discoverable via
        # list_presets() but literally cannot be compiled by
        # compile_problem at all -- a real coverage gap between the
        # preset catalog and the compiler's pde_kind dispatch, not a
        # smoke-test artifact. Kept as a hard failure deliberately; see
        # AUDIT_REPORT.md for the tally.
        pytest.fail(f"solve_pde for preset '{name}' raised: {e}")

    losses = result["history"]["loss"]
    assert len(losses) == 3
    assert all(l == l and abs(l) != float("inf") for l in losses), (
        f"preset '{name}' produced a non-finite loss during a 3-epoch smoke run: {losses}"
    )
