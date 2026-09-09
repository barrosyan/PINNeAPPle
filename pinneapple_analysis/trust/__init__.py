"""pinneapple_analysis.trust — pre-consumption trust scoring for a trained
model's own predictions. See ``trust_gate.py`` for how this differs from
``pinneapple_llm.guardrail.PhysicsGuardrail`` (verifies LLM-proposed
specs, not deployed-model predictions) and
``pinneapple_systems.digital_twin.monitoring.anomaly.MahalanobisDetector``
(scores live sensor innovation vectors, not a model's prediction)."""
from __future__ import annotations

from .trust_gate import TrustGate, TrustScore

__all__ = ["TrustGate", "TrustScore"]
