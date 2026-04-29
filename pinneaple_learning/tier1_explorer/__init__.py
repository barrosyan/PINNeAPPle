"""pinneaple_learning.tier1_explorer — Resources for Explorer-level users.

You are at Tier 1 if you:
  - Understand some physics (ODEs / PDEs / conservation laws)
  - Are new to Physics AI / PINNs
  - Want to see what neural networks can do with physics

Start here
----------
>>> import pinneaple_learning.tier1_explorer as t1
>>> t1.quickstart()
"""

from .guide import quickstart, what_is_pinn, what_is_loss

__all__ = ["quickstart", "what_is_pinn", "what_is_loss"]
