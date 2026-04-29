"""LearningPath — guides users through the three-tier Physics AI journey."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Tier descriptors
# ---------------------------------------------------------------------------

_TIERS: dict[int, dict] = {
    1: {
        "name": "Explorer",
        "tagline": "I understand the physics. I want to see what AI can do with it.",
        "prerequisites": "Basic Python · NumPy · some calculus and ODEs/PDEs",
        "milestones": [
            "Run a PINN on a simple ODE and compare to the exact solution",
            "Understand what the PDE residual loss means physically",
            "Change boundary conditions and observe how the solution changes",
            "Interpret loss curves and relative L2 error",
        ],
        "next_steps": [
            "Complete all 10 examples in examples/getting_started/",
            "Run templates/01_basic_pinn.py  through  templates/06_time_marching.py",
            "When comfortable: move to Tier 2",
        ],
        "entry_point": "pinneaple_learning/tier1_explorer/README.md",
        "key_files": [
            "examples/getting_started/01_harmonic_oscillator.py",
            "examples/getting_started/03_heat_diffusion_1d.py",
            "examples/getting_started/06_lotka_volterra.py",
            "templates/01_basic_pinn.py",
            "templates/03_navier_stokes_pinn.py",
        ],
    },
    2: {
        "name": "Experimenter",
        "tagline": "I have run PINNs before. I want to compare approaches and build experiments.",
        "prerequisites": "Tier 1 complete · PyTorch basics · understanding of at least one PDE",
        "milestones": [
            "Compare FNO, DeepONet, and PINN on the same problem using Arena",
            "Run an inverse problem and recover a physical parameter",
            "Quantify prediction uncertainty with MC Dropout or deep ensembles",
            "Validate that your model satisfies conservation laws",
            "Run active learning and observe faster convergence",
        ],
        "next_steps": [
            "Work through templates/16 – 29 (UQ, transfer, FNO, DeepONet, ROM, …)",
            "Use pinneaple_arena to benchmark your own problem",
            "Validate results with pinneaple_validate",
            "When confident in your approach: move to Tier 3",
        ],
        "entry_point": "pinneaple_learning/tier2_experimenter/README.md",
        "key_files": [
            "templates/16_uncertainty_quantification.py",
            "templates/18_inverse_problem.py",
            "templates/19_fno_neural_operator.py",
            "templates/20_deeponet_surrogate.py",
            "templates/21_active_learning.py",
            "templates/28_arena_benchmark.py",
            "templates/32_physics_validation.py",
        ],
    },
    3: {
        "name": "Builder",
        "tagline": "I have a validated approach. I want to scale it and prepare for production.",
        "prerequisites": "Tier 2 complete · understanding of distributed training · deployment basics",
        "milestones": [
            "Train a PINN on multiple GPUs with DDP",
            "Export a trained model to ONNX and benchmark inference latency",
            "Build a digital twin with live sensor assimilation",
            "Pass all physics validation checks (residual, conservation, BCs)",
            "Complete the PhysicsNeMo migration checklist",
        ],
        "next_steps": [
            "Work through templates/07, 22, 23, 24, 33, 34, 35",
            "Run examples/vs_physicsnemo/ side-by-side comparisons",
            "Complete pinneaple_learning/physicsnemo_roadmap/migration_guide.md",
            "Deploy on NVIDIA PhysicsNeMo for production",
        ],
        "entry_point": "pinneaple_learning/tier3_builder/README.md",
        "key_files": [
            "templates/07_ddp_distributed.py",
            "templates/22_model_serving.py",
            "templates/23_model_export.py",
            "templates/24_digital_twin.py",
            "templates/33_rans_turbulence.py",
            "examples/vs_physicsnemo/",
        ],
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class LearningPath:
    """A structured guide through the Physics AI learning journey.

    Parameters
    ----------
    tier : int
        User tier (1 = Explorer, 2 = Experimenter, 3 = Builder).
        If None, self-assessment is triggered.
    """

    tier: Optional[int] = None
    _info: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.tier is not None:
            self._validate_tier()
            self._info = _TIERS[self.tier]

    def _validate_tier(self):
        if self.tier not in _TIERS:
            raise ValueError(f"tier must be 1, 2, or 3. Got {self.tier!r}")

    # ------------------------------------------------------------------
    def describe(self) -> str:
        if not self._info:
            return self._self_assess()
        t = self._info
        lines = [
            f"\n{'='*60}",
            f"  Tier {self.tier} — {t['name']}",
            f"{'='*60}",
            f"\n  \"{t['tagline']}\"",
            f"\nPrerequisites:\n  {t['prerequisites']}",
            "\nMilestones (complete all before advancing):",
        ]
        for i, m in enumerate(t["milestones"], 1):
            lines.append(f"  {i}. {m}")
        lines.append("\nKey files to work through:")
        for f_ in t["key_files"]:
            lines.append(f"  • {f_}")
        lines.append("\nNext steps when ready:")
        for s in t["next_steps"]:
            lines.append(f"  → {s}")
        lines.append(f"\nFull guide: {t['entry_point']}")
        lines.append("")
        return "\n".join(lines)

    def _self_assess(self) -> str:
        questions = [
            ("Have you run a PINN before?",
             "If no → start at Tier 1."),
            ("Have you compared multiple architectures (FNO, DeepONet, PINN)?",
             "If no → start at Tier 2."),
            ("Do you have a trained, validated model ready to deploy?",
             "If yes → start at Tier 3."),
        ]
        lines = ["\n  Self-assessment — which tier are you?\n"]
        for q, a in questions:
            lines.append(f"  Q: {q}")
            lines.append(f"     {a}\n")
        lines.append("  Call learning_path(tier=N) with your answer.")
        return "\n".join(lines)

    def __repr__(self) -> str:
        if self.tier is None:
            return "LearningPath(tier=None) — call .describe() for self-assessment"
        return f"LearningPath(tier={self.tier}, name='{_TIERS[self.tier]['name']}')"


def where_am_i() -> None:
    """Print a quick self-assessment to identify your tier."""
    print(LearningPath()._self_assess())
    print("\nPhysicsNeMo is the destination once you complete Tier 3.")
    print("See: pinneaple_learning/physicsnemo_roadmap/README.md\n")


def learning_path(tier: int) -> LearningPath:
    """Return a LearningPath for the given tier and print the guide.

    Parameters
    ----------
    tier : int
        1 = Explorer, 2 = Experimenter, 3 = Builder
    """
    lp = LearningPath(tier=tier)
    print(lp.describe())
    return lp
