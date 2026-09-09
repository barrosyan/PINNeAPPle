from .agent import DesignAgent
from .schema import ProblemSpec, Gap, Plan, PlanStep, DesignReport, PinneappleSpec
from .state import DesignState
from .protocol import LLMProvider, LLMMessage, LLMResponse, GeminiProvider
from .codegen import build_pinneapple_spec

__all__ = [
    # Agent
    "DesignAgent",
    # Schema
    "ProblemSpec",
    "Gap",
    "Plan",
    "PlanStep",
    "DesignReport",
    # Pinneapple API output
    "PinneappleSpec",
    "build_pinneapple_spec",
    # State
    "DesignState",
    # LLM protocol
    "LLMProvider",
    "LLMMessage",
    "LLMResponse",
    "GeminiProvider",
]

# UnifiedPhysicsAgent composes DesignAgent with pinneapple_worldmodel's
# PhysicsOrchestrator (see unified_agent.py's module docstring for the full
# design rationale). pinneapple_worldmodel is treated the same way
# knowledge/mapping.py already treats it -- as an optional dependency of
# pinneapple_problemdesign -- so importing this package must not hard-fail
# for anyone missing it or its (heavier) transitive deps.
try:
    from .unified_agent import UnifiedPhysicsAgent, UnifiedAgentResult
    __all__ += ["UnifiedPhysicsAgent", "UnifiedAgentResult"]
except ImportError:
    pass
