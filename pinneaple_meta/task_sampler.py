"""PDE task sampler for meta-learning over parametric PDE families.

A *task* is one PDE instance with a specific set of physical parameter values
(e.g. ``nu=0.01``, ``Re=500``).  The sampler draws parameter values uniformly
from user-supplied ranges, builds the corresponding physics residual function
and optional data batches, and packages everything into a task dictionary that
MAML / Reptile trainers can consume directly.

Example
-------
>>> from pinneaple_meta.task_sampler import PDETaskSampler
>>> import torch
>>>
>>> def physics_factory(params):
...     nu = params["nu"]
...     def physics_fn(model, x):
...         x = x.requires_grad_(True)
...         u = model(x)
...         dudx = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
...         return (dudx + nu * u).pow(2).mean()
...     return physics_fn
>>>
>>> sampler = PDETaskSampler(
...     param_ranges={"nu": (0.001, 0.1)},
...     physics_fn_factory=physics_factory,
... )
>>> task = sampler.sample_task()
>>> batch = sampler.sample_batch(n_tasks=4)
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


class PDETaskSampler:
    """Samples tasks (PDE instances with varying parameters) for meta-training.

    A *task* is one PDE instance with specific parameter values drawn
    uniformly from the user-supplied ``param_ranges``.

    Parameters
    ----------
    param_ranges:
        Mapping of parameter name to ``(low, high)`` sampling interval.
        E.g. ``{"nu": (0.001, 0.1), "Re": (100.0, 1000.0)}``.
    physics_fn_factory:
        A callable ``factory(params: dict) -> physics_loss_fn`` that, given
        a concrete parameter dict, returns a physics loss function with
        signature ``physics_fn(model, x_collocation) -> scalar_tensor``.
    data_factory:
        Optional callable ``factory(params: dict) -> dict`` that returns a
        dict of tensors used as support/query data.  The returned dict must
        at minimum contain the key ``"x_col"`` (collocation points) as a
        float tensor of shape ``(N, d)``.  Additional keys such as
        ``"x_bc"``, ``"u_bc"`` are passed through unchanged.
        When *None*, collocation points are sampled uniformly in ``[0, 1]^d``
        with ``d`` determined by ``input_dim``.
    n_support:
        Number of points in the support set (used for inner-loop adaptation).
    n_query:
        Number of points in the query set (used for outer-loop meta-loss).
    input_dim:
        Spatial/temporal input dimension used when ``data_factory`` is
        *None* and collocation points are generated internally.
    seed:
        Base random seed.  Each call to :meth:`sample_task` increments an
        internal counter so tasks are reproducible but distinct.
    """

    def __init__(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        physics_fn_factory: Callable,
        data_factory: Optional[Callable] = None,
        n_support: int = 64,
        n_query: int = 64,
        input_dim: int = 1,
        seed: int = 0,
    ) -> None:
        if not param_ranges:
            raise ValueError("param_ranges must contain at least one entry.")
        for k, (lo, hi) in param_ranges.items():
            if lo >= hi:
                raise ValueError(
                    f"param_ranges['{k}']: low ({lo}) must be < high ({hi})."
                )
        self.param_ranges = param_ranges
        self.physics_fn_factory = physics_fn_factory
        self.data_factory = data_factory
        self.n_support = n_support
        self.n_query = n_query
        self.input_dim = input_dim
        self._rng = np.random.default_rng(seed)
        self._call_count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_params(self) -> Dict[str, float]:
        """Draw one set of PDE parameters uniformly from ``param_ranges``."""
        return {
            name: float(self._rng.uniform(lo, hi))
            for name, (lo, hi) in self.param_ranges.items()
        }

    def _make_batch(self, params: Dict[str, float], n_points: int) -> dict:
        """Build a data batch for *params* with *n_points* collocation points.

        If ``data_factory`` was provided it is called; otherwise collocation
        points are sampled uniformly in ``[0, 1]^input_dim``.
        """
        if self.data_factory is not None:
            batch = self.data_factory(params)
            # Allow factory to return a fixed-size batch; we slice if needed.
            if "x_col" not in batch:
                raise KeyError(
                    "data_factory must return a dict containing key 'x_col'."
                )
            # Ensure tensors are float
            return {k: v.float() if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
        else:
            x_col = torch.from_numpy(
                self._rng.uniform(0.0, 1.0, size=(n_points, self.input_dim))
            ).float()
            return {"x_col": x_col}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_task(self) -> dict:
        """Sample a single task.

        Returns
        -------
        dict with keys:

        * ``"params"`` — dict of sampled PDE parameter values.
        * ``"support"`` — batch dict (at least ``{"x_col": Tensor}``).
        * ``"query"`` — batch dict (at least ``{"x_col": Tensor}``).
        * ``"physics_fn"`` — callable ``(model, x) -> scalar_tensor``.
        """
        self._call_count += 1
        params = self._sample_params()
        support = self._make_batch(params, self.n_support)
        query = self._make_batch(params, self.n_query)
        physics_fn = self.physics_fn_factory(params)
        return {
            "params": params,
            "support": support,
            "query": query,
            "physics_fn": physics_fn,
        }

    def sample_batch(self, n_tasks: int) -> List[dict]:
        """Sample a batch of *n_tasks* independent tasks.

        Parameters
        ----------
        n_tasks:
            Number of tasks to sample.

        Returns
        -------
        List[dict]
            A list of task dicts as returned by :meth:`sample_task`.
        """
        if n_tasks < 1:
            raise ValueError("n_tasks must be >= 1.")
        return [self.sample_task() for _ in range(n_tasks)]
