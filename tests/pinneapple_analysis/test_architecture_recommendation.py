"""Tests for pinneapple_analysis.verification.architecture_recommendation."""
from __future__ import annotations

import pytest

from pinneapple_analysis.verification.architecture_recommendation import (
    ARCHITECTURE_CATALOG,
    ArchitectureCandidate,
    ArchitectureRecommendation,
    recommend_architecture,
)
from pinneapple_neural.architectures.registry import ModelRegistry


def test_catalog_registry_keys_are_all_real():
    """Every catalog entry's registry_key must be a real, currently
    registered PINNeAPPle architecture -- catching a stale/typo'd key
    the same way test_physics_tools.py's module-name check catches stale
    module paths."""
    real_keys = set(ModelRegistry.list())
    for name, cand in ARCHITECTURE_CATALOG.items():
        assert cand.registry_key in real_keys, (
            f"{name}: registry_key={cand.registry_key!r} not in ModelRegistry "
            f"(real keys include e.g. {sorted(real_keys)[:5]}...)"
        )


def test_scarce_data_no_solver_recommends_pinn():
    rec = recommend_architecture(n_high_fidelity_simulations=0, has_analytical_solver=False)
    assert isinstance(rec, ArchitectureRecommendation)
    assert rec.recommended[0] == "pinn"
    assert "xpinn" in rec.recommended


def test_scarce_data_with_solver_still_recommends_pinn_base_with_hybrid_caveat():
    rec = recommend_architecture(n_high_fidelity_simulations=0, has_analytical_solver=True)
    assert rec.recommended[0] == "pinn"
    assert any("hybrid" in c.lower() for c in rec.caveats)


def test_lots_of_data_no_generalization_still_recommends_pinn():
    rec = recommend_architecture(n_high_fidelity_simulations=500, needs_parameter_generalization=False,
                                  geometry_varies=False)
    assert rec.recommended[0] == "pinn"


def test_lots_of_data_with_parameter_generalization_recommends_fno():
    rec = recommend_architecture(n_high_fidelity_simulations=500, needs_parameter_generalization=True)
    assert rec.recommended[0] == "fno"
    assert "deeponet" in rec.recommended
    assert "pino" in rec.recommended


def test_lots_of_data_with_geometry_variation_recommends_mesh_graph_net():
    rec = recommend_architecture(n_high_fidelity_simulations=500, needs_parameter_generalization=True,
                                  geometry_varies=True)
    assert rec.recommended == ["mesh_graph_net"]


def test_has_lots_of_data_explicit_override_takes_precedence_over_count():
    # 5 simulations would normally be "scarce" under the default threshold,
    # but an explicit has_lots_of_data=True must override that heuristic.
    rec = recommend_architecture(n_high_fidelity_simulations=5, has_lots_of_data=True,
                                  needs_parameter_generalization=True)
    assert rec.recommended[0] == "fno"


def test_inverse_problem_adds_inverse_pinn_regardless_of_other_axes():
    rec_scarce = recommend_architecture(n_high_fidelity_simulations=0, is_inverse_problem=True)
    rec_rich = recommend_architecture(n_high_fidelity_simulations=500, needs_parameter_generalization=True,
                                       is_inverse_problem=True)
    assert "inverse_pinn" in rec_scarce.recommended
    assert "inverse_pinn" in rec_rich.recommended


def test_decision_path_is_nonempty_and_explains_the_choice():
    rec = recommend_architecture(n_high_fidelity_simulations=0)
    assert len(rec.decision_path) >= 1
    assert all(isinstance(step, str) and len(step) > 10 for step in rec.decision_path)


def test_alternatives_never_include_the_primary_pick():
    rec = recommend_architecture(n_high_fidelity_simulations=500, needs_parameter_generalization=True)
    primary = rec.recommended[0]
    alt_names = [a["architecture"] for a in rec.alternatives]
    assert primary not in alt_names


def test_custom_catalog_override_replaces_entry_content():
    """The decision tree's internal branches resolve to fixed category
    keys ("pinn", "fno", "mesh_graph_net", ...), so a custom catalog must
    use those same keys -- overriding lets a caller replace an entry's
    CONTENT (e.g. a different registry_key/description), not rename the
    category itself. (A catalog missing a key the decision path resolves
    to raises KeyError, confirmed below -- documents the real contract
    rather than silently accepting an incompatible override.)"""
    custom = dict(ARCHITECTURE_CATALOG)
    custom["pinn"] = ArchitectureCandidate(
        name="My Custom PINN", registry_key="modified_mlp", category="PINN",
        when_to_use="overridden for this test",
    )
    rec = recommend_architecture(n_high_fidelity_simulations=0, catalog=custom)
    assert rec.recommended[0] == "pinn"
    assert "My Custom PINN" in rec.reasoning


def test_catalog_missing_a_resolvable_key_raises_keyerror():
    incompatible = {"my_arch": ArchitectureCandidate(
        name="MyArch", registry_key="modified_mlp", category="PINN", when_to_use="test",
    )}
    with pytest.raises(KeyError):
        recommend_architecture(n_high_fidelity_simulations=0, catalog=incompatible)
