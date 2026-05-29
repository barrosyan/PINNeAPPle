from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np


def pareto_dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if solution *a* dominates *b* under minimization.

    Domination requires that every objective of *a* is <= the corresponding
    objective of *b*, with at least one strictly less.
    """
    return bool(np.all(a <= b) and np.any(a < b))


def compute_pareto_front(objectives: np.ndarray) -> np.ndarray:
    """Return a boolean mask of Pareto-optimal solutions.

    Parameters
    ----------
    objectives:
        Array of shape (N, M) where N is the number of solutions and M is
        the number of objectives (all minimized).

    Returns
    -------
    np.ndarray
        Boolean mask of shape (N,); True where the solution is Pareto-optimal.
    """
    n = objectives.shape[0]
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        # A single vectorised comparison beats i in all objectives AND strictly
        # better in at least one — so we can drop i from the Pareto set.
        dominated_by = (
            np.all(objectives <= objectives[i], axis=1)
            & np.any(objectives < objectives[i], axis=1)
        )
        dominated_by[i] = False  # a solution cannot dominate itself
        if np.any(dominated_by):
            is_pareto[i] = False

    return is_pareto


@dataclass
class ParetoFront:
    """Container for a multi-objective Pareto front.

    Attributes
    ----------
    objectives:
        All evaluated objective vectors, shape (N, M).
    params:
        Corresponding design parameter vectors, shape (N, p).  May be None
        when params were not recorded.
    mask:
        Boolean mask of Pareto-optimal entries, shape (N,).
    """

    objectives: np.ndarray
    params: Optional[np.ndarray]
    mask: np.ndarray

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def filter(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Return ``(objectives_pareto, params_pareto)`` for optimal solutions."""
        obj_pareto = self.objectives[self.mask]
        par_pareto = self.params[self.mask] if self.params is not None else None
        return obj_pareto, par_pareto

    # ------------------------------------------------------------------
    # Hypervolume
    # ------------------------------------------------------------------

    def hypervolume(self, ref_point: np.ndarray) -> float:
        """Compute the hypervolume indicator for a 2-D Pareto front.

        Parameters
        ----------
        ref_point:
            Reference (worst) point, shape (2,).  Every Pareto-optimal
            solution must dominate it for the contribution to be positive.

        Returns
        -------
        float
            Hypervolume indicator value.

        Raises
        ------
        ValueError
            If the number of objectives is not exactly 2.
        """
        obj_pareto, _ = self.filter()
        if obj_pareto.shape[1] != 2:
            raise ValueError("hypervolume() currently supports 2-D objective spaces only.")

        ref = np.asarray(ref_point, dtype=float)

        # Sort by first objective ascending so we can sweep right-to-left.
        idx = np.argsort(obj_pareto[:, 0])
        pts = obj_pareto[idx]

        hv = 0.0
        prev_y = ref[1]
        for i in range(len(pts) - 1, -1, -1):
            x, y = pts[i]
            width = ref[0] - x
            height = prev_y - y
            if width > 0 and height > 0:
                hv += width * height
                prev_y = y

        return float(hv)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        ax: Any = None,
        labels: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> Any:
        """Plot the Pareto front for the first two objectives.

        Parameters
        ----------
        ax:
            Existing matplotlib ``Axes`` object.  A new figure is created
            when *ax* is ``None``.
        labels:
            Axis labels for the two objectives; defaults to ``["f1", "f2"]``.
        save_path:
            If given, the figure is saved to this path.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt  # deferred: optional dependency

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))

        labs = labels if labels is not None else ["f1", "f2"]

        dominated_mask = ~self.mask
        if np.any(dominated_mask):
            ax.scatter(
                self.objectives[dominated_mask, 0],
                self.objectives[dominated_mask, 1],
                color="lightgray",
                label="Dominated",
                alpha=0.6,
                s=30,
            )

        obj_p, _ = self.filter()
        # Sort by first objective for a clean step-function staircase visual.
        idx = np.argsort(obj_p[:, 0])
        obj_p = obj_p[idx]

        ax.scatter(obj_p[:, 0], obj_p[:, 1], color="steelblue", label="Pareto front", zorder=5)
        ax.step(
            obj_p[:, 0],
            obj_p[:, 1],
            where="post",
            color="steelblue",
            linewidth=1.2,
            alpha=0.7,
        )

        ax.set_xlabel(labs[0])
        ax.set_ylabel(labs[1])
        ax.set_title("Pareto Front")
        ax.legend()

        if save_path is not None:
            ax.figure.savefig(save_path, bbox_inches="tight")

        return ax
