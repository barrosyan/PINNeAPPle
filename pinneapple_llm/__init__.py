"""LLM-assisted physics-AI pipeline drafting, gated by a physics-grounded
verification layer.

See ``draft.py``, ``geometry_draft.py``, ``twin_draft.py``,
``research.py`` and ``guardrail.py`` module docstrings for the design
rationale: the LLM only ever *selects and parametrises* one of
PINNeAPPle's own already-implemented presets/generators/config schemas
(never invents new physics/CAD/twin machinery), and ``PhysicsGuardrail``
is the mechanical, re-computed check every result should pass before
being reported as trustworthy -- whether or not an LLM was involved in
producing it.

``local_llm.py`` and ``conversation_store.py`` add a fully local path
(Ollama + a SQLite conversation log, no hosted API, no data leaving your
machine); ``finetune.py`` fine-tunes a local model on that logged data.
"""
from .draft import draft_problem, DraftResult
from .geometry_draft import draft_geometry, GeometryDraftResult
from .twin_draft import draft_digital_twin, build_3d_live_twin, TwinDraftResult
from .research import search_literature, ResearchReport
from .guardrail import PhysicsGuardrail, GuardrailReport, CheckResult
from .conversation_store import ConversationStore, ConversationRecord
from ._dispatch import call_llm

try:
    from . import local_llm
except Exception:
    local_llm = None  # type: ignore

try:
    from .finetune import FinetuneConfig, prepare_dataset, finetune_lora
except Exception:
    FinetuneConfig = prepare_dataset = finetune_lora = None  # type: ignore

__all__ = [
    "draft_problem", "DraftResult",
    "draft_geometry", "GeometryDraftResult",
    "draft_digital_twin", "build_3d_live_twin", "TwinDraftResult",
    "search_literature", "ResearchReport",
    "PhysicsGuardrail", "GuardrailReport", "CheckResult",
    "ConversationStore", "ConversationRecord",
    "call_llm", "local_llm",
    "FinetuneConfig", "prepare_dataset", "finetune_lora",
]
