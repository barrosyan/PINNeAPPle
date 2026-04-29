"""pinneaple_learning.tier2_experimenter — Resources for Experimenter-level users.

You are at Tier 2 if you:
  - Have run at least one PINN before
  - Understand the loss function and collocation points
  - Want to compare approaches, solve inverse problems, quantify uncertainty

Start here
----------
>>> import pinneaple_learning.tier2_experimenter as t2
>>> t2.architecture_guide()
>>> t2.pipeline_anatomy()
"""

from .guide import architecture_guide, pipeline_anatomy

__all__ = ["architecture_guide", "pipeline_anatomy"]
