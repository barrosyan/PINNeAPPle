"""Autonomy safety-boundary gate for the autonomous physics research loop.

This module is the **non-negotiable safety floor** for the autonomous
research loop being built on top of :class:`~.unified_agent.UnifiedPhysicsAgent`
and ``run_agent_loop`` elsewhere in this repo. The research loop is meant to
run many iterations *without* a human approving every step -- that is the
whole point of "autonomous". But some actions cross outside pure sandboxed
simulation/computation/validation: they cost real money, invoke a licensed
external tool per-run, or would affect something outside this repo's
sandbox (an external service, an account, a filesystem/network resource
that isn't just "read a local input, write a local output"). Those actions
-- which this module calls *consequential* -- always require an explicit
human approval gate, **regardless of how autonomous the loop is otherwise
configured to be.**

This is by design, not a bug or a temporary limitation. ``FULL_AUTONOMOUS``
means "don't bother a human about ordinary simulation/compute steps" -- it
does not, and must never, mean "don't bother a human before spending money
or invoking a licensed external solver." See
:class:`ConsequentialActionClassifier` and :class:`ApprovalGate` below --
:meth:`ApprovalGate.check` enforces this floor unconditionally at every
autonomy level, and it *fails closed*: if no human approver is wired up
and human approval is required, the action is **not** approved.

Grounding
---------
The classifier's default patterns are grounded in the tools actually
registered by :class:`pinneapple_worldmodel.physics_tools.PhysicsToolRegistry`
(``simulate_trajectory``, ``run_fdm_solver``, ``train_pinn``,
``validate_physics``, ``build_digital_twin``, ``run_cosim``, ... -- all pure
local simulation/compute, correctly classified as non-consequential) and in
:func:`pinneapple_simulation.external_solvers.ansys.runner.run_fluent_case`,
a real example of a consequential action: it shells out to a licensed
ANSYS Fluent installation to drive an actual paid-license solver run. No
tool of that shape is registered in ``PhysicsToolRegistry`` today, which is
exactly why this module cannot rely on an allowlist of "the tools that
exist" -- it has to recognise the *pattern* ("this looks like it invokes a
licensed external solver / spends money / talks to the outside world") so
that a future tool registry adding a Fluent, Abaqus, cloud-provisioning, or
notification tool is caught automatically, not only after someone remembers
to update a list by hand.

Companion module
-----------------
This mirrors the house style of this package's other hard-boundary module,
:mod:`pinneapple_problemdesign.policy` (the "non-invention policy": never
silently fabricate spec details -- surface them as gaps/assumptions
instead). Same idea, different boundary: ``policy`` guards against silently
inventing facts; this module guards against silently taking consequential
actions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set


# ---------------------------------------------------------------------------
# AutonomyLevel
# ---------------------------------------------------------------------------

class AutonomyLevel(str, Enum):
    """How much of the research loop's step-by-step decision-making runs
    without a human in the loop.

    This governs only *non-consequential* actions (pure simulation,
    computation, validation, data generation, training, etc. -- the kinds
    of tools ``PhysicsToolRegistry`` actually registers today). Consequential
    actions (see :class:`ConsequentialActionClassifier`) always require
    human approval at every level below -- that floor is not configurable
    through ``AutonomyLevel`` and is enforced by :class:`ApprovalGate`
    regardless of which level is selected.

    SUPERVISED
        Every single action requires human approval before it executes,
        consequential or not. The most conservative level; use it while
        trusting a new loop/tool/model combination for the first time.

    SEMI_AUTONOMOUS
        The loop runs freely (no human approval) for ordinary
        non-consequential actions, but is required to check in with a
        human periodically -- every ``iteration_report_every`` iterations
        (:class:`ApprovalGate`'s constructor argument, not hardcoded) --
        and *also* whenever the loop's own reported confidence drops below
        ``confidence_pause_threshold`` (also constructor-configurable, and
        optional: pass ``None`` to disable the confidence trigger). This is
        the level intended for "let it run, but keep a human in the review
        loop at a sane cadence."

    FULL_AUTONOMOUS
        The loop runs to completion, or until it hits a real budget/
        iteration cap that the *loop itself* enforces (this module does
        not implement iteration/budget caps -- that belongs to the research
        loop that will be built on top of this gate), entirely without
        human approval for non-consequential actions. Consequential
        actions are still gated -- see the module docstring. This is the
        level for "trust the loop to actually finish the research task on
        its own," with the one exception this whole module exists to
        enforce.
    """

    SUPERVISED = "supervised"
    SEMI_AUTONOMOUS = "semi_autonomous"
    FULL_AUTONOMOUS = "full_autonomous"


# ---------------------------------------------------------------------------
# ConsequentialActionClassifier
# ---------------------------------------------------------------------------

class ConsequentialActionClassifier:
    """Explicit, inspectable (no ML/fuzzy matching) classifier for whether
    a tool call is "consequential": crosses outside pure sandboxed
    simulation/computation/validation because it costs real money, invokes
    a licensed external tool per-run, or would affect something outside
    this sandbox (an external account/service/network resource).

    The ruleset is a set of plain substring patterns matched against the
    tool name (case-insensitively), plus optional category/tag patterns
    matched against ``tool_metadata`` when the caller supplies it (shaped
    like :class:`pinneapple_worldmodel.physics_tools.PhysicsTool`'s
    ``category``/``tags`` fields: a ``category`` string and a ``tags``
    iterable of strings). This is deliberately simple pattern matching, not
    a model -- every classification decision can be read straight off the
    ruleset below, which is the point: a human reviewing this module's
    safety floor should be able to see exactly what triggers it.

    Extending the ruleset
    ----------------------
    The patterns are plain, mutable ``set`` instance attributes -- real,
    editable data, not something baked into unreachable branches of
    ``classify``. To extend or narrow the ruleset for a specific
    deployment::

        clf = ConsequentialActionClassifier()
        clf.name_patterns.add("slurm_submit")       # new consequential pattern
        clf.category_patterns.add("procurement")    # new consequential category
        clf.tag_patterns.add("licensed")             # new consequential tag
        clf.exceptions.add("send_to_sandbox_queue")  # false-positive override

    ``exceptions`` takes priority over every pattern: a tool name placed in
    ``exceptions`` is always classified as non-consequential, which is the
    supported way to correct a false positive (e.g. a tool that happens to
    contain a pattern substring like "post" -- ``postprocess_field`` --
    without actually being consequential) without weakening the pattern
    itself for every other tool.
    """

    #: Substrings matched against ``tool_name.lower()``. Grounded in this
    #: repo's real consequential example
    #: (``pinneapple_simulation.external_solvers.ansys.runner.run_fluent_case``,
    #: which shells out to a licensed ANSYS Fluent install -- hence
    #: "fluent"/"ansys") plus commercial-solver and general "leaves the
    #: sandbox" patterns (money, external communication, deployment,
    #: provisioning) that no tool in ``PhysicsToolRegistry`` currently
    #: matches, but that a future tool registry plausibly could.
    DEFAULT_NAME_PATTERNS: Set[str] = {
        # Licensed external solvers (real example: run_fluent_case).
        "fluent", "ansys", "abaqus", "comsol", "starccm", "nastran",
        # Money.
        "purchase", "pay", "order", "buy", "checkout", "invoice", "subscribe",
        # External communication / publication.
        "publish", "send", "email", "post", "notify", "webhook",
        "push_to_hub",
        # Deployment / provisioning of real resources outside the sandbox.
        "deploy", "provision", "terminate_instance", "delete_production",
    }

    #: ``PhysicsTool.category`` values treated as consequential outright.
    #: None of the categories ``PhysicsToolRegistry.register_all`` actually
    #: registers today (simulation, pde_solving, data_generation, training,
    #: validation, uncertainty, inverse, transfer, meta_learning,
    #: timeseries, co_simulation, geometry, inference, design_opt,
    #: digital_twin, world_model -- see that module's taxonomy docstring)
    #: are consequential; this set exists so a future category clearly
    #: outside the sandbox (e.g. a procurement or deployment category) is
    #: caught even before any specific tool name pattern is added.
    DEFAULT_CATEGORY_PATTERNS: Set[str] = {
        "external_solver", "procurement", "deployment", "finance",
        "communication",
    }

    #: ``PhysicsTool.tags`` entries treated as consequential outright.
    DEFAULT_TAG_PATTERNS: Set[str] = {
        "licensed", "paid", "external_network", "financial", "irreversible",
    }

    def __init__(
        self,
        *,
        name_patterns: Optional[Set[str]] = None,
        category_patterns: Optional[Set[str]] = None,
        tag_patterns: Optional[Set[str]] = None,
        exceptions: Optional[Set[str]] = None,
    ) -> None:
        # Copy the class-level defaults so mutating an instance's ruleset
        # (the documented extension mechanism above) never mutates the
        # shared class defaults out from under other instances.
        self.name_patterns: Set[str] = set(
            name_patterns if name_patterns is not None else self.DEFAULT_NAME_PATTERNS
        )
        self.category_patterns: Set[str] = set(
            category_patterns if category_patterns is not None else self.DEFAULT_CATEGORY_PATTERNS
        )
        self.tag_patterns: Set[str] = set(
            tag_patterns if tag_patterns is not None else self.DEFAULT_TAG_PATTERNS
        )
        self.exceptions: Set[str] = set(exceptions) if exceptions else set()

    def classify(self, tool_name: str, tool_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Return ``True`` if ``tool_name`` (optionally described further by
        ``tool_metadata``) is a consequential action.

        Parameters
        ----------
        tool_name:
            The tool/action identifier, e.g. ``"run_fluent_case"``.
        tool_metadata:
            Optional dict shaped like ``{"category": str, "tags": [str, ...]}``
            (matching :class:`~pinneapple_worldmodel.physics_tools.PhysicsTool`'s
            fields). When omitted, classification falls back to name
            matching alone.
        """
        if not tool_name:
            return False

        if tool_name in self.exceptions:
            return False

        lowered = tool_name.lower()
        if any(pattern in lowered for pattern in self.name_patterns):
            return True

        if tool_metadata:
            category = str(tool_metadata.get("category", "") or "").lower()
            if category and category in self.category_patterns:
                return True

            tags = tool_metadata.get("tags") or []
            lowered_tags = {str(t).lower() for t in tags}
            if lowered_tags & self.tag_patterns:
                return True

        return False


# ---------------------------------------------------------------------------
# ApprovalRequest / ApprovalDecision
# ---------------------------------------------------------------------------

@dataclass
class ApprovalRequest:
    """Everything a human approver needs to decide whether a gated action
    should proceed. Passed to ``approver_fn`` by :class:`ApprovalGate`."""

    tool_name: str
    tool_args: Dict[str, Any]
    reason: str
    autonomy_level: AutonomyLevel
    is_consequential: bool


@dataclass
class ApprovalDecision:
    """The outcome of :meth:`ApprovalGate.check`.

    approved:
        Whether the action may proceed. When ``requires_human`` is True and
        no ``approver_fn`` was wired up, this is always ``False`` (fail
        closed -- see :class:`ApprovalGate`).
    reason:
        Human-readable explanation of why this decision was made.
    requires_human:
        Whether this decision needed (or would have needed) a human
        approval step at all, independent of the outcome. Useful for
        logging/reporting even when ``approver_fn`` ultimately approved it.
    """

    approved: bool
    reason: str
    requires_human: bool


# ---------------------------------------------------------------------------
# ApprovalGate
# ---------------------------------------------------------------------------

class ApprovalGate:
    """The gate itself: decides, per tool call, whether it may proceed
    autonomously or needs a human's explicit sign-off.

    Safety floor (non-negotiable)
    ------------------------------
    A consequential action (per ``classifier``) **always** requires human
    approval, at *every* :class:`AutonomyLevel` including
    ``FULL_AUTONOMOUS``. This is enforced first, unconditionally, in
    :meth:`check`, before any autonomy-level-specific logic runs -- it
    cannot be configured away by ``iteration_report_every``,
    ``confidence_pause_threshold``, or the autonomy level itself. This is
    the entire reason this module exists; see the module docstring.

    Fail closed
    -----------
    When a decision requires human approval and no ``approver_fn`` was
    provided to the constructor, :meth:`check` returns
    ``approved=False, requires_human=True`` -- it never silently
    auto-approves just because nobody wired up an approver. A caller that
    wants a fully hands-off loop must explicitly supply an ``approver_fn``
    (which is free to always return ``True`` if that is genuinely the
    caller's intent -- but that is then an explicit, visible choice in the
    caller's code, not a silent default of this gate).

    Parameters
    ----------
    autonomy_level:
        The configured :class:`AutonomyLevel` for this gate.
    approver_fn:
        Optional callable invoked with an :class:`ApprovalRequest` whenever
        human approval is required; must return a real ``bool``. If
        omitted, any action requiring human approval is refused (fail
        closed) rather than silently approved.
    iteration_report_every:
        Under ``SEMI_AUTONOMOUS``, how often (in iterations) the gate
        requires a human check-in even for non-consequential actions.
        1-indexed: a check-in is required when the effective iteration
        count is a positive multiple of this value (iteration
        ``iteration_report_every``, ``2 * iteration_report_every``, ...).
        The effective iteration count is ``current_iteration`` when the
        caller supplies it to :meth:`check`, otherwise an internal
        call counter maintained by this gate (so periodic pausing still
        works even if the caller never passes ``current_iteration``).
        Must be a positive integer; pass a very large value to effectively
        disable periodic check-ins while keeping the confidence trigger.
    confidence_pause_threshold:
        Under ``SEMI_AUTONOMOUS``, if the caller supplies
        ``current_confidence`` to :meth:`check` and it is below this
        threshold, human approval is required regardless of the iteration
        count. ``None`` (the default) disables this trigger entirely.
    classifier:
        The :class:`ConsequentialActionClassifier` used to detect
        consequential actions. Defaults to a fresh instance with the
        default ruleset; pass a customised instance (see that class's
        docstring for how to extend the ruleset) to change what counts as
        consequential for this gate.
    """

    def __init__(
        self,
        autonomy_level: AutonomyLevel,
        approver_fn: Optional[Callable[[ApprovalRequest], bool]] = None,
        iteration_report_every: int = 5,
        confidence_pause_threshold: Optional[float] = None,
        classifier: Optional[ConsequentialActionClassifier] = None,
    ) -> None:
        if iteration_report_every <= 0:
            raise ValueError("iteration_report_every must be a positive integer.")
        self.autonomy_level = autonomy_level
        self.approver_fn = approver_fn
        self.iteration_report_every = iteration_report_every
        self.confidence_pause_threshold = confidence_pause_threshold
        self.classifier = classifier or ConsequentialActionClassifier()
        self._call_count = 0

    def check(
        self,
        tool_name: str,
        tool_args: Optional[Dict[str, Any]] = None,
        *,
        tool_metadata: Optional[Dict[str, Any]] = None,
        current_iteration: Optional[int] = None,
        current_confidence: Optional[float] = None,
    ) -> ApprovalDecision:
        """Decide whether ``tool_name`` may proceed right now.

        Parameters
        ----------
        tool_name, tool_args:
            The tool call being gated.
        tool_metadata:
            Optional category/tags metadata forwarded to the classifier
            (see :meth:`ConsequentialActionClassifier.classify`).
        current_iteration:
            The research loop's current iteration count, if the caller
            tracks one. Used for the ``SEMI_AUTONOMOUS`` periodic
            check-in trigger; see ``iteration_report_every`` above.
        current_confidence:
            The research loop's current confidence in its own progress
            (e.g. a validation/convergence score), if available. Used for
            the ``SEMI_AUTONOMOUS`` confidence-pause trigger.
        """
        self._call_count += 1
        tool_args = tool_args or {}

        is_consequential = self.classifier.classify(tool_name, tool_metadata)
        requires_human, reason = self._requires_human(
            tool_name, is_consequential, current_iteration, current_confidence
        )

        if not requires_human:
            return ApprovalDecision(approved=True, reason=reason, requires_human=False)

        if self.approver_fn is None:
            return ApprovalDecision(
                approved=False,
                requires_human=True,
                reason=(
                    f"{reason} No approver_fn is configured on this ApprovalGate, "
                    "so the action is refused (fail closed) rather than silently "
                    "auto-approved."
                ),
            )

        request = ApprovalRequest(
            tool_name=tool_name,
            tool_args=tool_args,
            reason=reason,
            autonomy_level=self.autonomy_level,
            is_consequential=is_consequential,
        )
        approved = bool(self.approver_fn(request))
        return ApprovalDecision(approved=approved, requires_human=True, reason=reason)

    def _requires_human(
        self,
        tool_name: str,
        is_consequential: bool,
        current_iteration: Optional[int],
        current_confidence: Optional[float],
    ) -> tuple:
        # --- Hard safety floor: not skippable at any AutonomyLevel. -------
        if is_consequential:
            return True, (
                f"Tool '{tool_name}' is classified as a consequential action "
                "(costs real money, invokes a licensed external tool, or "
                "affects something outside the sandbox). Consequential "
                "actions always require human approval regardless of the "
                f"configured autonomy level ({self.autonomy_level.value}); "
                "this is a non-negotiable safety floor, not a configurable "
                "policy."
            )

        # --- Autonomy-level-specific logic for non-consequential actions.
        if self.autonomy_level == AutonomyLevel.SUPERVISED:
            return True, (
                "SUPERVISED autonomy: every action requires human approval."
            )

        if self.autonomy_level == AutonomyLevel.SEMI_AUTONOMOUS:
            effective_iteration = (
                current_iteration if current_iteration is not None else self._call_count
            )
            if effective_iteration > 0 and effective_iteration % self.iteration_report_every == 0:
                return True, (
                    "SEMI_AUTONOMOUS: periodic human check-in "
                    f"(iteration {effective_iteration}, every "
                    f"{self.iteration_report_every})."
                )
            if (
                current_confidence is not None
                and self.confidence_pause_threshold is not None
                and current_confidence < self.confidence_pause_threshold
            ):
                return True, (
                    f"SEMI_AUTONOMOUS: confidence {current_confidence} is below "
                    f"confidence_pause_threshold {self.confidence_pause_threshold}."
                )
            return False, (
                "SEMI_AUTONOMOUS: non-consequential action within the "
                "configured iteration/confidence bounds; proceeding "
                "autonomously."
            )

        # FULL_AUTONOMOUS: non-consequential actions never gate. Consequential
        # actions are already handled above, unconditionally.
        return False, (
            "FULL_AUTONOMOUS: non-consequential action; proceeding "
            "autonomously (consequential actions still always require "
            "human approval -- see the module docstring)."
        )
