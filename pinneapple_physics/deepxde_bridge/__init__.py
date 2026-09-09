"""pinneapple_physics.deepxde_bridge -- DeepXDE <-> PINNeAPPle integration glue.

Bridges a real PINNeAPPle
:class:`~pinneapple_physics.pde_environment.spec.ProblemSpec` (as produced by
e.g. ``pinneapple_physics.pde_environment.presets.get_preset``) to real
DeepXDE (Lu Lu, Xuhui Meng, Zhiping Mao, George Em Karniadakis, "DeepXDE: A
Deep Learning Library for Solving Differential Equations," SIAM Review,
63(1):208-228, 2021; https://github.com/lululxvi/deepxde) training, so a
PINNeAPPle problem can be solved with DeepXDE as an alternative solving
backend to PINNeAPPle's own native compiler
(``pinneapple_physics.pinn_solver.compiler``) -- without reimplementing any
of DeepXDE's geometry sampling, network, optimizer loop, or loss bookkeeping.

Sub-modules
-----------
adapter
    :func:`~pinneapple_physics.deepxde_bridge.adapter.solve_with_deepxde`
    translates ``spec.coords``/``spec.domain_bounds`` into a real
    ``dde.geometry`` geometry, ``spec.pde`` into a real DeepXDE residual
    callable, ``spec.conditions`` into real ``dde.icbc.DirichletBC`` /
    ``NeumannBC`` / ``IC`` objects, builds a real ``dde.nn.FNN`` and
    ``dde.Model``, and actually calls DeepXDE's own
    ``model.compile()``/``model.train()`` -- returning the trained model
    wrapped in :class:`~pinneapple_physics.deepxde_bridge.adapter.DeepXDESolveResult`.

    Only two ``pde.kind`` values are translated for real: ``"burgers"``
    (e.g. the ``burgers_1d`` preset) and ``"poisson"`` (e.g. the
    ``poisson_2d`` preset), matching the residual sign conventions of
    PINNeAPPle's own native compiler exactly. Any other ``pde.kind``, an
    untranslatable condition, or an unsupported geometry shape raises a
    clear ``NotImplementedError`` naming what is unsupported, rather than
    silently mistranslating the problem.

Optional dependency
--------------------
``deepxde`` is genuinely optional: importing this package, or anything in
it, never requires deepxde to be installed. Only
:func:`~pinneapple_physics.deepxde_bridge.adapter.solve_with_deepxde` (and
:func:`~pinneapple_physics.deepxde_bridge.adapter.require_deepxde`) perform
a real ``import deepxde`` and raise a clear, actionable ``ImportError``
(with an install hint) if the package is missing.
"""
from __future__ import annotations

from pinneapple_physics.deepxde_bridge.adapter import (
    DeepXDESolveResult,
    SUPPORTED_PDE_KINDS,
    is_deepxde_available,
    require_deepxde,
    solve_with_deepxde,
)

__all__ = [
    "DeepXDESolveResult",
    "SUPPORTED_PDE_KINDS",
    "is_deepxde_available",
    "require_deepxde",
    "solve_with_deepxde",
]
