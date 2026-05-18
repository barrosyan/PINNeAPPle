from __future__ import annotations
"""
TrainingAdvisor — automatic identification of training improvement points.

Analyses per-epoch loss histories and final metrics, then emits a ranked list
of Suggestion objects that the AutoRetrainer can apply to build an improved
ExperimentConfig.

Diagnostic signals
------------------
plateau        : loss stagnated in the last 25% of training
divergence     : final loss > initial loss (training went backwards)
slow_start     : loss barely moved in the first 25% of epochs
oscillation    : high relative std in the recent loss tail
high_residual  : final PDE or BC loss is still above a threshold
bc_imbalance   : BC loss dominates total loss (needs higher BC weight)
physics_imbalance : physics loss dominates (needs higher physics weight)
vanishing_grad : loss stopped decreasing and final value is very small
                 but metrics are still bad (suggests gradient saturation)
too_few_epochs : convergence was still happening at the end of training
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Suggestion:
    """A single actionable improvement recommendation."""
    code: str                         # machine-readable identifier
    priority: int                     # 1 = highest, 5 = lowest
    description: str                  # human-readable explanation
    config_patch: Dict[str, Any]      # fields to override in ExperimentConfig/ModelRunConfig
    rationale: str = ""               # why this specific value was chosen


@dataclass
class DiagnosticReport:
    """Complete diagnostic output for one model run."""
    model_name: str
    signals: List[str]                # triggered diagnostic codes
    suggestions: List[Suggestion]     # ranked best-first
    metrics_summary: Dict[str, float]
    diagnosis_text: str               # plain-text narrative


# ──────────────────────────────────────────────────────────────────────────────
# Helper statistics
# ──────────────────────────────────────────────────────────────────────────────

def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _safe_std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _safe_mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def _relative_reduction(start: float, end: float) -> float:
    """Fraction of loss reduced: (start-end)/start.  Negative = diverged."""
    if abs(start) < 1e-30:
        return 0.0
    return (start - end) / abs(start)


def _tail(lst: List[float], frac: float = 0.25) -> List[float]:
    n = max(1, int(len(lst) * frac))
    return lst[-n:]


def _head(lst: List[float], frac: float = 0.25) -> List[float]:
    n = max(1, int(len(lst) * frac))
    return lst[:n]


# ──────────────────────────────────────────────────────────────────────────────
# Advisor
# ──────────────────────────────────────────────────────────────────────────────

class TrainingAdvisor:
    """
    Inspects training histories and final metrics to produce ranked
    Suggestion objects for improving the next training run.

    Parameters
    ----------
    plateau_rel_std_thresh   : relative std of tail loss below which we call a plateau
    divergence_rel_thresh    : minimum relative reduction required (below → divergence)
    slow_start_thresh        : minimum reduction in first 25% before flagging slow start
    high_residual_thresh     : absolute PDE/BC residual above which we flag high residual
    oscillation_rel_std      : relative std of tail loss above which we call oscillation
    bc_dominance_ratio       : BC/total ratio above which BC is considered dominant
    physics_dominance_ratio  : physics/total ratio above which physics is dominant
    convergence_still_going  : tail slope / head_mean below which we say "still converging"
    max_retrain_epochs_mult  : cap on epoch multiplier suggested
    """

    def __init__(
        self,
        plateau_rel_std_thresh: float = 0.005,
        divergence_rel_thresh: float = -0.05,
        slow_start_thresh: float = 0.05,
        high_residual_thresh: float = 1e-2,
        oscillation_rel_std: float = 0.10,
        bc_dominance_ratio: float = 0.80,
        physics_dominance_ratio: float = 0.95,
        convergence_still_going: float = -0.02,
        max_retrain_epochs_mult: float = 4.0,
    ):
        self.plateau_rel_std_thresh = plateau_rel_std_thresh
        self.divergence_rel_thresh = divergence_rel_thresh
        self.slow_start_thresh = slow_start_thresh
        self.high_residual_thresh = high_residual_thresh
        self.oscillation_rel_std = oscillation_rel_std
        self.bc_dominance_ratio = bc_dominance_ratio
        self.physics_dominance_ratio = physics_dominance_ratio
        self.convergence_still_going = convergence_still_going
        self.max_retrain_epochs_mult = max_retrain_epochs_mult

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse(
        self,
        model_name: str,
        loss_history: List[float],
        physics_loss_history: List[float],
        bc_loss_history: List[float],
        metrics: Dict[str, float],
        current_lr: float,
        current_epochs: int,
        current_batch_size: int,
    ) -> DiagnosticReport:
        """
        Analyse a completed training run and return a DiagnosticReport with
        ranked Suggestions.
        """
        signals: List[str] = []
        suggestions: List[Suggestion] = []

        if len(loss_history) < 4:
            return DiagnosticReport(
                model_name=model_name,
                signals=["insufficient_history"],
                suggestions=[],
                metrics_summary=metrics,
                diagnosis_text="Training history too short to analyse.",
            )

        total_loss = [v for v in loss_history if math.isfinite(v)]
        phys_loss  = [v for v in physics_loss_history  if math.isfinite(v)]
        bc_loss    = [v for v in bc_loss_history   if math.isfinite(v)]

        head_total = _safe_mean(_head(total_loss))
        tail_total = _safe_mean(_tail(total_loss))
        tail_std   = _safe_std(_tail(total_loss))
        rel_std    = tail_std / (abs(tail_total) + 1e-30)

        head_phys = _safe_mean(_head(phys_loss))  if phys_loss  else 0.0
        tail_phys = _safe_mean(_tail(phys_loss))  if phys_loss  else 0.0
        tail_bc   = _safe_mean(_tail(bc_loss))    if bc_loss    else 0.0

        # ── Detect signals ────────────────────────────────────────────────

        # 1. Divergence / instability
        rel_red = _relative_reduction(head_total, tail_total)
        if rel_red < self.divergence_rel_thresh:
            signals.append("divergence")
            new_lr = current_lr * 0.2
            suggestions.append(Suggestion(
                code="lower_lr_divergence",
                priority=1,
                description="Loss increased during training — learning rate is too high.",
                config_patch={"lr": round(new_lr, 8)},
                rationale=f"Reduced lr by 5× ({current_lr:.2e} → {new_lr:.2e}) to stabilise gradients.",
            ))
            suggestions.append(Suggestion(
                code="add_grad_clip",
                priority=1,
                description="Add gradient clipping (norm=1.0) to prevent gradient explosion.",
                config_patch={"grad_clip": 1.0},
                rationale="Gradient clipping bounds update steps and prevents divergence.",
            ))

        # 2. NaN / inf in loss
        n_bad = sum(1 for v in loss_history if not math.isfinite(v))
        if n_bad > 0:
            signals.append("nan_inf")
            suggestions.append(Suggestion(
                code="lower_lr_nan",
                priority=1,
                description=f"{n_bad} NaN/Inf loss values detected — reduce lr and clip gradients.",
                config_patch={"lr": round(current_lr * 0.1, 8), "grad_clip": 0.5},
                rationale="NaN usually indicates an exploding gradient; aggressive lr reduction + clipping.",
            ))

        # 3. Plateau
        if (rel_std < self.plateau_rel_std_thresh
                and rel_red > self.slow_start_thresh   # some convergence happened
                and "divergence" not in signals):
            signals.append("plateau")
            mult = min(2.0, self.max_retrain_epochs_mult)
            suggestions.append(Suggestion(
                code="more_epochs_plateau",
                priority=2,
                description="Loss plateaued — model may need more iterations or a lower learning rate.",
                config_patch={"epochs": int(current_epochs * mult)},
                rationale=f"Doubled epochs to {current_epochs * mult:.0f} to allow continued descent.",
            ))
            suggestions.append(Suggestion(
                code="lower_lr_plateau",
                priority=2,
                description="Reduce lr to escape the plateau with smaller gradient steps.",
                config_patch={"lr": round(current_lr * 0.3, 8)},
                rationale="Lower lr helps the optimiser navigate flat loss landscapes.",
            ))

        # 4. Still converging at end of training
        n_q = max(1, len(total_loss) // 4)
        q3_mean = _safe_mean(total_loss[-2 * n_q: -n_q])
        q4_mean = tail_total
        still_converging = (q3_mean - q4_mean) / (abs(q3_mean) + 1e-30)
        if still_converging > -self.convergence_still_going and "plateau" not in signals:
            signals.append("still_converging")
            mult = min(self.max_retrain_epochs_mult, max(1.5, 1.0 / max(1e-6, 1.0 + self.convergence_still_going)))
            suggestions.append(Suggestion(
                code="more_epochs_converging",
                priority=2,
                description="Training was still improving at the last epoch — increase training budget.",
                config_patch={"epochs": int(current_epochs * mult)},
                rationale=f"Still-converging slope detected; epochs × {mult:.1f}.",
            ))

        # 5. Slow start
        head_red = _relative_reduction(total_loss[0], _safe_mean(_head(total_loss, 0.10)))
        if abs(head_red) < self.slow_start_thresh and "divergence" not in signals:
            signals.append("slow_start")
            suggestions.append(Suggestion(
                code="higher_lr_slow_start",
                priority=3,
                description="Loss barely moved in the first 10% of epochs — lr may be too low.",
                config_patch={"lr": round(current_lr * 3.0, 8)},
                rationale=f"Tripled lr ({current_lr:.2e} → {current_lr * 3:.2e}) to accelerate early descent.",
            ))

        # 6. Oscillation
        if rel_std > self.oscillation_rel_std and "divergence" not in signals:
            signals.append("oscillation")
            suggestions.append(Suggestion(
                code="lower_lr_oscillation",
                priority=2,
                description="Loss is oscillating in the tail — training is unstable.",
                config_patch={"lr": round(current_lr * 0.5, 8), "grad_clip": 1.0},
                rationale="Halved lr to reduce the step amplitude causing oscillations.",
            ))

        # 7. BC imbalance
        if tail_total > 1e-8 and bc_loss:
            bc_ratio = tail_bc / (tail_total + 1e-30)
            if bc_ratio > self.bc_dominance_ratio:
                signals.append("bc_dominance")
                suggestions.append(Suggestion(
                    code="increase_bc_weight",
                    priority=3,
                    description=(
                        f"BC loss accounts for {bc_ratio:.0%} of total — "
                        "increase BC weight to push the model harder on boundaries."
                    ),
                    config_patch={"weight_override": {"bc": 5.0, "physics": 1.0}},
                    rationale="High BC fraction means the physics residual is already small; rebalance.",
                ))

        # 8. Physics imbalance
        if tail_total > 1e-8 and phys_loss:
            phys_ratio = tail_phys / (tail_total + 1e-30)
            if phys_ratio > self.physics_dominance_ratio:
                signals.append("physics_dominance")
                suggestions.append(Suggestion(
                    code="increase_physics_weight",
                    priority=3,
                    description=(
                        f"Physics loss accounts for {phys_ratio:.0%} of total — "
                        "the boundary conditions are being ignored."
                    ),
                    config_patch={"weight_override": {"bc": 10.0, "physics": 1.0}},
                    rationale="Normalising the BC weight forces the solver to respect BCs.",
                ))

        # 9. High final PDE residual
        pde_res = metrics.get("pde_residual", float("nan"))
        if math.isfinite(pde_res) and pde_res > self.high_residual_thresh:
            signals.append("high_pde_residual")
            more_pts = min(current_batch_size * 2, 16384)
            suggestions.append(Suggestion(
                code="more_collocation_points",
                priority=3,
                description=f"PDE residual {pde_res:.2e} is high — more interior collocation points may help.",
                config_patch={"batch_size": more_pts},
                rationale=f"Doubled collocation points from {current_batch_size} to {more_pts}.",
            ))

        # 10. High final BC residual
        bc_res = metrics.get("bc_residual", float("nan"))
        if math.isfinite(bc_res) and bc_res > self.high_residual_thresh:
            signals.append("high_bc_residual")
            suggestions.append(Suggestion(
                code="increase_bc_weight_residual",
                priority=2,
                description=f"BC residual {bc_res:.2e} still high — increase BC loss weight.",
                config_patch={"weight_override": {"bc": 20.0, "physics": 1.0}},
                rationale="High BC residual means boundaries are not enforced tightly enough.",
            ))

        # 11. High L2 relative error
        l2 = metrics.get("l2_relative", float("nan"))
        if math.isfinite(l2) and l2 > 0.05:
            signals.append("high_l2_error")
            suggestions.append(Suggestion(
                code="more_epochs_l2",
                priority=2,
                description=f"L2 relative error {l2:.2%} is above 5% — model needs more training.",
                config_patch={"epochs": int(current_epochs * 2)},
                rationale="Doubling epochs to allow further accuracy improvement.",
            ))

        # Sort suggestions by priority then by code for determinism
        suggestions.sort(key=lambda s: (s.priority, s.code))

        # Build narrative
        lines = [f"Model: {model_name}"]
        if signals:
            lines.append(f"Signals detected: {', '.join(signals)}")
        else:
            lines.append("No issues detected — training looks healthy.")
        if suggestions:
            lines.append("Top recommendations:")
            for s in suggestions[:3]:
                lines.append(f"  [{s.priority}] {s.description}")
                lines.append(f"       Patch: {s.config_patch}")

        return DiagnosticReport(
            model_name=model_name,
            signals=signals,
            suggestions=suggestions,
            metrics_summary={k: v for k, v in metrics.items() if math.isfinite(v) if isinstance(v, float)},
            diagnosis_text="\n".join(lines),
        )

    # ── Config patching ───────────────────────────────────────────────────────

    def apply_suggestions(
        self,
        suggestions: List[Suggestion],
        current_config: Dict[str, Any],
        max_priority: int = 3,
    ) -> Dict[str, Any]:
        """
        Merge the top suggestions (up to max_priority) into current_config.
        Returns a new config dict with the patches applied.
        """
        new_cfg = dict(current_config)
        applied = []
        for s in suggestions:
            if s.priority > max_priority:
                continue
            for k, v in s.config_patch.items():
                new_cfg[k] = v
                applied.append(s.code)
        new_cfg["_advisor_patches"] = applied
        return new_cfg
