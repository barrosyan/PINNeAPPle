"""pinneaple_learning.tier3_builder — Resources for Builder-level users.

You are at Tier 3 if you:
  - Have a validated Physics AI model
  - Want to scale it to multiple GPUs or production hardware
  - Are preparing to deploy or migrate to NVIDIA PhysicsNeMo

Start here
----------
>>> import pinneaple_learning.tier3_builder as t3
>>> t3.production_checklist()
>>> t3.physicsnemo_readiness()
"""

from .guide import production_checklist, physicsnemo_readiness

__all__ = ["production_checklist", "physicsnemo_readiness"]
