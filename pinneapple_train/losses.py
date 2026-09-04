"""``pinneapple_train.losses`` compatibility submodule -- see ``trainer.py``'s
docstring for how this gap was found (the pre-existing test suite
couldn't even collect) and which callers across the repo needed it.
"""
from pinneapple_neural.trainer.losses import SupervisedLoss, PhysicsLossHook, CombinedLoss, build_loss

__all__ = ["SupervisedLoss", "PhysicsLossHook", "CombinedLoss", "build_loss"]
