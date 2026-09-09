"""Tests for the autonomy safety-boundary gate (pinneapple_problemdesign.autonomy).

The critical invariant under test: consequential actions ALWAYS require
human approval, at every AutonomyLevel including FULL_AUTONOMOUS, and the
gate fails closed (never silently auto-approves) when no approver_fn is
wired up. See pinneapple_problemdesign/autonomy.py's module docstring for
the full rationale.
"""
from __future__ import annotations

import pytest

from pinneapple_problemdesign.autonomy import (
    ApprovalDecision,
    ApprovalGate,
    ApprovalRequest,
    AutonomyLevel,
    ConsequentialActionClassifier,
)

# A real non-consequential tool actually registered by
# PhysicsToolRegistry._register_simulation_tools (pure local FDM/particle
# simulation -- no external license, no money, no network).
NON_CONSEQUENTIAL_TOOL = "run_fdm_solver"

# A real consequential action: the function name actually used by
# pinneapple_simulation.external_solvers.ansys.runner.run_fluent_case,
# which shells out to a licensed ANSYS Fluent installation to drive an
# actual paid-license solver run. Not (yet) registered in
# PhysicsToolRegistry, which is exactly why the classifier works by
# pattern, not by allowlisting only currently-registered tool names.
CONSEQUENTIAL_TOOL = "run_fluent_case"


# ---------------------------------------------------------------------------
# ConsequentialActionClassifier
# ---------------------------------------------------------------------------

class TestConsequentialActionClassifier:
    def test_real_fluent_tool_is_consequential(self):
        clf = ConsequentialActionClassifier()
        assert clf.classify(CONSEQUENTIAL_TOOL) is True

    def test_real_ansys_named_tool_is_consequential(self):
        clf = ConsequentialActionClassifier()
        assert clf.classify("ansys_solve_case") is True

    @pytest.mark.parametrize(
        "tool_name",
        [
            "run_fdm_solver",
            "simulate_trajectory",
            "simulate_batch",
            "run_sph_simulation",
            "compile_pinn",
            "train_pinn",
            "train_world_model",
            "validate_physics",
            "build_digital_twin",
            "run_cosim",
            "sindy_equation_discovery",
            "mc_dropout_uncertainty",
        ],
    )
    def test_real_registered_tools_are_non_consequential(self, tool_name):
        clf = ConsequentialActionClassifier()
        assert clf.classify(tool_name) is False

    @pytest.mark.parametrize(
        "tool_name",
        [
            "publish_results",
            "send_email_report",
            "purchase_compute_credits",
            "deploy_model_to_prod",
            "push_to_hub",
            "provision_cloud_instance",
            "order_hardware",
        ],
    )
    def test_general_consequential_patterns(self, tool_name):
        clf = ConsequentialActionClassifier()
        assert clf.classify(tool_name) is True

    def test_metadata_category_triggers_consequential(self):
        clf = ConsequentialActionClassifier()
        assert clf.classify(
            "some_new_tool", tool_metadata={"category": "procurement", "tags": []}
        ) is True

    def test_metadata_tag_triggers_consequential(self):
        clf = ConsequentialActionClassifier()
        assert clf.classify(
            "some_new_tool", tool_metadata={"category": "simulation", "tags": ["licensed"]}
        ) is True

    def test_exceptions_override_name_pattern_false_positive(self):
        clf = ConsequentialActionClassifier()
        # "postprocess_field" contains "post" but is not actually consequential.
        assert clf.classify("postprocess_field") is True  # matches pattern by default
        clf.exceptions.add("postprocess_field")
        assert clf.classify("postprocess_field") is False

    def test_ruleset_is_real_editable_data(self):
        clf = ConsequentialActionClassifier()
        assert "brand_new_consequential_action" not in clf.name_patterns
        clf.name_patterns.add("brand_new_consequential_action")
        assert clf.classify("brand_new_consequential_action") is True
        # Class-level defaults are untouched by instance mutation.
        assert "brand_new_consequential_action" not in ConsequentialActionClassifier.DEFAULT_NAME_PATTERNS

    def test_empty_tool_name_is_non_consequential(self):
        clf = ConsequentialActionClassifier()
        assert clf.classify("") is False


# ---------------------------------------------------------------------------
# ApprovalGate -- per-AutonomyLevel x consequential/non-consequential matrix
# ---------------------------------------------------------------------------

def _always_approve(request: ApprovalRequest) -> bool:
    return True


def _always_deny(request: ApprovalRequest) -> bool:
    return False


class TestApprovalGateSupervised:
    def test_non_consequential_still_requires_human(self):
        gate = ApprovalGate(AutonomyLevel.SUPERVISED, approver_fn=_always_approve)
        decision = gate.check(NON_CONSEQUENTIAL_TOOL, {})
        assert decision.requires_human is True
        assert decision.approved is True

    def test_consequential_requires_human(self):
        gate = ApprovalGate(AutonomyLevel.SUPERVISED, approver_fn=_always_approve)
        decision = gate.check(CONSEQUENTIAL_TOOL, {})
        assert decision.requires_human is True
        assert decision.approved is True

    def test_approver_denial_is_respected(self):
        gate = ApprovalGate(AutonomyLevel.SUPERVISED, approver_fn=_always_deny)
        decision = gate.check(NON_CONSEQUENTIAL_TOOL, {})
        assert decision.requires_human is True
        assert decision.approved is False


class TestApprovalGateSemiAutonomous:
    def test_non_consequential_within_bounds_auto_approves(self):
        gate = ApprovalGate(
            AutonomyLevel.SEMI_AUTONOMOUS, approver_fn=_always_deny, iteration_report_every=5
        )
        # 1st through 4th calls should not hit the periodic check-in.
        for _ in range(4):
            decision = gate.check(NON_CONSEQUENTIAL_TOOL, {})
            assert decision.requires_human is False
            assert decision.approved is True

    def test_periodic_report_every_n_calls_requires_human(self):
        gate = ApprovalGate(
            AutonomyLevel.SEMI_AUTONOMOUS, approver_fn=_always_approve, iteration_report_every=3
        )
        decisions = [gate.check(NON_CONSEQUENTIAL_TOOL, {}) for _ in range(6)]
        # Internal call counter is 1-indexed; every 3rd call requires human.
        requires_human_flags = [d.requires_human for d in decisions]
        assert requires_human_flags == [False, False, True, False, False, True]

    def test_periodic_report_uses_explicit_current_iteration_when_given(self):
        gate = ApprovalGate(
            AutonomyLevel.SEMI_AUTONOMOUS, approver_fn=_always_approve, iteration_report_every=5
        )
        decision = gate.check(NON_CONSEQUENTIAL_TOOL, {}, current_iteration=10)
        assert decision.requires_human is True
        decision2 = gate.check(NON_CONSEQUENTIAL_TOOL, {}, current_iteration=11)
        assert decision2.requires_human is False

    def test_low_confidence_requires_human(self):
        gate = ApprovalGate(
            AutonomyLevel.SEMI_AUTONOMOUS,
            approver_fn=_always_approve,
            iteration_report_every=1000,
            confidence_pause_threshold=0.5,
        )
        decision = gate.check(NON_CONSEQUENTIAL_TOOL, {}, current_confidence=0.2)
        assert decision.requires_human is True

    def test_high_confidence_does_not_require_human(self):
        gate = ApprovalGate(
            AutonomyLevel.SEMI_AUTONOMOUS,
            approver_fn=_always_approve,
            iteration_report_every=1000,
            confidence_pause_threshold=0.5,
        )
        decision = gate.check(NON_CONSEQUENTIAL_TOOL, {}, current_confidence=0.9)
        assert decision.requires_human is False

    def test_confidence_trigger_disabled_when_threshold_is_none(self):
        gate = ApprovalGate(
            AutonomyLevel.SEMI_AUTONOMOUS,
            approver_fn=_always_approve,
            iteration_report_every=1000,
            confidence_pause_threshold=None,
        )
        decision = gate.check(NON_CONSEQUENTIAL_TOOL, {}, current_confidence=0.0)
        assert decision.requires_human is False

    def test_consequential_always_requires_human_even_within_bounds(self):
        gate = ApprovalGate(
            AutonomyLevel.SEMI_AUTONOMOUS, approver_fn=_always_approve, iteration_report_every=1000
        )
        decision = gate.check(CONSEQUENTIAL_TOOL, {})
        assert decision.requires_human is True


class TestApprovalGateFullAutonomous:
    """The critical safety-invariant tests: FULL_AUTONOMOUS never gates on
    non-consequential actions, but ALWAYS gates on consequential ones."""

    def test_non_consequential_never_requires_human(self):
        # approver_fn deliberately omitted: if this ever required human
        # approval, the fail-closed path would make this test fail loudly
        # (approved would be False), catching a safety regression.
        gate = ApprovalGate(AutonomyLevel.FULL_AUTONOMOUS)
        for _ in range(20):
            decision = gate.check(NON_CONSEQUENTIAL_TOOL, {}, current_iteration=1000000)
            assert decision.requires_human is False
            assert decision.approved is True

    def test_consequential_action_always_requires_human(self):
        gate = ApprovalGate(AutonomyLevel.FULL_AUTONOMOUS, approver_fn=_always_approve)
        decision = gate.check(CONSEQUENTIAL_TOOL, {})
        assert decision.requires_human is True
        assert decision.approved is True

    def test_consequential_action_requires_human_regardless_of_iteration_or_confidence(self):
        # Even with settings that would normally suppress gating for
        # non-consequential actions, a consequential action must still gate.
        gate = ApprovalGate(AutonomyLevel.FULL_AUTONOMOUS, approver_fn=_always_approve)
        decision = gate.check(
            CONSEQUENTIAL_TOOL, {}, current_iteration=1, current_confidence=1.0
        )
        assert decision.requires_human is True

    def test_this_is_the_non_negotiable_safety_floor(self):
        """Explicit statement of the invariant this whole module exists for:
        no AutonomyLevel, no configuration of ApprovalGate, causes a
        consequential action to skip human approval."""
        for level in AutonomyLevel:
            gate = ApprovalGate(level, approver_fn=_always_approve, iteration_report_every=10**9)
            decision = gate.check(CONSEQUENTIAL_TOOL, {}, current_confidence=1.0)
            assert decision.requires_human is True, (
                f"Consequential action was not gated under AutonomyLevel.{level.name} "
                "-- this breaks the module's core safety invariant."
            )


# ---------------------------------------------------------------------------
# Fail-closed behavior
# ---------------------------------------------------------------------------

class TestFailClosed:
    def test_supervised_no_approver_fails_closed(self):
        gate = ApprovalGate(AutonomyLevel.SUPERVISED)
        decision = gate.check(NON_CONSEQUENTIAL_TOOL, {})
        assert decision.requires_human is True
        assert decision.approved is False

    def test_semi_autonomous_periodic_checkin_no_approver_fails_closed(self):
        gate = ApprovalGate(AutonomyLevel.SEMI_AUTONOMOUS, iteration_report_every=1)
        decision = gate.check(NON_CONSEQUENTIAL_TOOL, {})
        assert decision.requires_human is True
        assert decision.approved is False

    def test_full_autonomous_consequential_no_approver_fails_closed(self):
        gate = ApprovalGate(AutonomyLevel.FULL_AUTONOMOUS)
        decision = gate.check(CONSEQUENTIAL_TOOL, {})
        assert decision.requires_human is True
        assert decision.approved is False

    def test_fail_closed_never_silently_auto_approves(self):
        """Never approved=True when requires_human=True and approver_fn is None."""
        for level in AutonomyLevel:
            gate = ApprovalGate(level, iteration_report_every=1)
            decision = gate.check(CONSEQUENTIAL_TOOL, {})
            assert not (decision.requires_human and decision.approved and gate.approver_fn is None)


# ---------------------------------------------------------------------------
# ApprovalRequest wiring
# ---------------------------------------------------------------------------

class TestApprovalRequestWiring:
    def test_approver_fn_receives_populated_request(self):
        captured = {}

        def approver(request: ApprovalRequest) -> bool:
            captured["request"] = request
            return True

        gate = ApprovalGate(AutonomyLevel.SUPERVISED, approver_fn=approver)
        gate.check("run_fdm_solver", {"n_steps": 10}, tool_metadata={"category": "simulation"})

        request = captured["request"]
        assert isinstance(request, ApprovalRequest)
        assert request.tool_name == "run_fdm_solver"
        assert request.tool_args == {"n_steps": 10}
        assert request.autonomy_level == AutonomyLevel.SUPERVISED
        assert request.is_consequential is False

    def test_consequential_flag_set_correctly_on_request(self):
        captured = {}

        def approver(request: ApprovalRequest) -> bool:
            captured["request"] = request
            return True

        gate = ApprovalGate(AutonomyLevel.FULL_AUTONOMOUS, approver_fn=approver)
        gate.check(CONSEQUENTIAL_TOOL, {})
        assert captured["request"].is_consequential is True

    def test_decision_is_real_dataclass_instance(self):
        gate = ApprovalGate(AutonomyLevel.FULL_AUTONOMOUS)
        decision = gate.check(NON_CONSEQUENTIAL_TOOL, {})
        assert isinstance(decision, ApprovalDecision)
        assert isinstance(decision.reason, str) and decision.reason

    def test_iteration_report_every_must_be_positive(self):
        with pytest.raises(ValueError):
            ApprovalGate(AutonomyLevel.SEMI_AUTONOMOUS, iteration_report_every=0)


# ---------------------------------------------------------------------------
# Package export surface
# ---------------------------------------------------------------------------

class TestPackageExports:
    def test_autonomy_symbols_exported_from_package_root(self):
        import pinneapple_problemdesign as pd

        assert pd.AutonomyLevel is AutonomyLevel
        assert pd.ConsequentialActionClassifier is ConsequentialActionClassifier
        assert pd.ApprovalRequest is ApprovalRequest
        assert pd.ApprovalDecision is ApprovalDecision
        assert pd.ApprovalGate is ApprovalGate
