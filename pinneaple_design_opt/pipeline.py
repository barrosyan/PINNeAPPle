from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .constraints import ConstraintSet
from .objective import ObjectiveBase
from .optimizer import (
    BayesianDesignOptimizer,
    DesignOptimizerConfig,
    EvolutionaryDesignOptimizer,
    GradientDesignOptimizer,
)
from .pareto import ParetoFront, compute_pareto_front
from .refinement import PINNRefinement, RefinementResult
from .surrogate import PhysicsSurrogate


@dataclass
class DesignOptConfig:
    """Top-level configuration for :class:`DesignOptLoop`.

    Attributes
    ----------
    n_iterations:
        Total number of optimization iterations.
    optimizer_cfg:
        Configuration for the inner design optimizer.
    refine_top_k:
        Number of top solutions to pass through PINN refinement at the end;
        0 disables refinement.
    save_history:
        Whether to record per-iteration theta and objective history.
    verbose:
        Print progress to stdout.
    convergence_tol:
        Stop early when the absolute objective improvement is below this
        threshold for 5 consecutive iterations.
    seed:
        Master random seed; propagated to all stochastic components.
    checkpoint_dir:
        If set, save intermediate results to this directory (not yet
        implemented; reserved for future use).
    """

    n_iterations: int = 50
    optimizer_cfg: DesignOptimizerConfig = field(default_factory=DesignOptimizerConfig)
    refine_top_k: int = 0
    save_history: bool = True
    verbose: bool = True
    convergence_tol: float = 1e-6
    seed: int = 0
    checkpoint_dir: Optional[str] = None


@dataclass
class DesignOptResult:
    """Stores the full result of a :class:`DesignOptLoop` run.

    Attributes
    ----------
    best_theta:
        Design vector achieving the lowest (single-objective) or Pareto-
        front-representative (multi-objective) result.
    best_objective:
        Best scalar objective seen during the run (first objective for
        multi-objective problems).
    history_theta:
        Per-iteration best design vector.
    history_objectives:
        Per-iteration best objective value.
    history_all_objectives:
        All evaluated objective tuples per iteration.
    pareto_front:
        Populated for multi-objective runs; ``None`` for single-objective.
    refinement_results:
        List of :class:`~pinneaple_design_opt.refinement.RefinementResult`
        objects when PINN refinement was requested.
    convergence_iter:
        Iteration at which convergence was detected (``None`` if the run
        completed all iterations).
    elapsed_s:
        Wall-clock time in seconds.
    """

    best_theta: np.ndarray
    best_objective: float
    history_theta: List[np.ndarray]
    history_objectives: List[float]
    history_all_objectives: List[List[float]]
    pareto_front: Optional[ParetoFront] = None
    refinement_results: Optional[List[RefinementResult]] = None
    convergence_iter: Optional[int] = None
    elapsed_s: float = 0.0

    # ------------------------------------------------------------------

    def plot_convergence(self, save_path: Optional[str] = None) -> Any:
        """Plot the best-so-far objective over iterations.

        Parameters
        ----------
        save_path:
            If provided, save the figure to this file path.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt  # deferred: optional dependency

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(self.history_objectives, linewidth=1.8, color="steelblue", label="Best objective")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective value")
        ax.set_title("Design Optimization Convergence")

        if self.convergence_iter is not None:
            ax.axvline(
                self.convergence_iter,
                color="tomato",
                linestyle="--",
                label=f"Converged @ iter {self.convergence_iter}",
            )

        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")

        return ax

    def summary(self) -> Dict[str, Any]:
        """Return a concise summary dictionary of the optimization run."""
        out: Dict[str, Any] = {
            "best_objective": float(self.best_objective),
            "best_theta": self.best_theta.tolist(),
            "n_iterations": len(self.history_objectives),
            "elapsed_s": round(self.elapsed_s, 3),
            "converged": self.convergence_iter is not None,
            "convergence_iter": self.convergence_iter,
        }
        if self.pareto_front is not None:
            n_pareto = int(self.pareto_front.mask.sum())
            out["n_pareto_solutions"] = n_pareto
        if self.refinement_results is not None:
            out["n_refined"] = len(self.refinement_results)
            out["refinement_improvement_ratios"] = [
                round(r.improvement_ratio, 4) for r in self.refinement_results
            ]
        return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


class DesignOptLoop:
    """Closed-loop physics-guided design optimizer.

    Orchestrates the full pipeline:
    1. Parametrize design space via *param_space*.
    2. Evaluate physics with *surrogate*.
    3. Optimize with the chosen back-end (gradient / Bayesian / evolutionary).
    4. Optionally refine with PINNs.
    5. Build multi-objective Pareto fronts when *objective* is a list.

    Parameters
    ----------
    param_space:
        :class:`~pinneaple_geom.optimize.loop.ParamSpace` that defines
        parameter names, bounds, and initial point.
    surrogate:
        Trained :class:`~pinneaple_design_opt.surrogate.PhysicsSurrogate`.
    objective:
        Either a single :class:`~pinneaple_design_opt.objective.ObjectiveBase`
        or a list of them for multi-objective optimization.
    constraints:
        Optional :class:`~pinneaple_design_opt.constraints.ConstraintSet`.
    refinement:
        Optional :class:`~pinneaple_design_opt.refinement.PINNRefinement`.
    cfg:
        Loop configuration; defaults to :class:`DesignOptConfig`.
    """

    def __init__(
        self,
        param_space: Any,
        surrogate: PhysicsSurrogate,
        objective: Union[ObjectiveBase, List[ObjectiveBase]],
        *,
        constraints: Optional[ConstraintSet] = None,
        refinement: Optional[PINNRefinement] = None,
        cfg: Optional[DesignOptConfig] = None,
    ) -> None:
        self.param_space = param_space
        self.surrogate = surrogate
        self.objectives: List[ObjectiveBase] = (
            objective if isinstance(objective, list) else [objective]
        )
        self._multi_objective = isinstance(objective, list) and len(objective) > 1
        self.constraints = constraints
        self.refinement = refinement
        self.cfg = cfg or DesignOptConfig()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _param_space_to_array(self) -> np.ndarray:
        """Return initial theta as an ordered 1-D numpy array."""
        keys = list(self.param_space.bounds.keys())
        return np.array([self.param_space.x0[k] for k in keys], dtype=np.float64)

    def _param_space_bounds(self) -> np.ndarray:
        """Return bounds as an (p, 2) array [[lo, hi], ...]."""
        return np.array(
            list(self.param_space.bounds.values()), dtype=np.float64
        )

    def _evaluate(self, theta_np: np.ndarray) -> tuple[float, list[float]]:
        """Predict u and evaluate all objectives + constraint penalties.

        Returns
        -------
        tuple
            ``(primary_scalar, [obj_1, obj_2, ...])`` where ``primary_scalar``
            already includes constraint penalties.
        """
        import torch

        theta_t = torch.from_numpy(theta_np.astype(np.float32))
        u_t = self.surrogate.predict(theta_t)

        obj_vals: List[float] = []
        for obj in self.objectives:
            v = obj(theta_t, u_t)
            obj_vals.append(float(v.item()))

        primary = obj_vals[0]

        if self.constraints is not None:
            pen = self.constraints.total_penalty(theta_t, u_t)
            primary = primary + float(pen.item())

        return primary, obj_vals

    def _build_optimizer(self, theta0: np.ndarray) -> Any:
        cfg = self.cfg.optimizer_cfg
        method = cfg.method.lower()
        if method == "gradient":
            return GradientDesignOptimizer(theta0, cfg)
        elif method == "bayesian":
            return BayesianDesignOptimizer(cfg, seed=self.cfg.seed)
        elif method == "evolutionary":
            return EvolutionaryDesignOptimizer(cfg, seed=self.cfg.seed)
        else:
            raise ValueError(
                f"Unknown optimizer method '{cfg.method}'. "
                "Choose 'gradient', 'bayesian', or 'evolutionary'."
            )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> DesignOptResult:
        """Execute the full design optimization loop.

        Returns
        -------
        DesignOptResult
        """
        import torch

        cfg = self.cfg
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        t_start = time.perf_counter()

        theta = self._param_space_to_array()
        bounds = self._param_space_bounds()
        opt_cfg = cfg.optimizer_cfg

        optimizer = self._build_optimizer(theta)

        history_theta: List[np.ndarray] = []
        history_objectives: List[float] = []
        history_all_objectives: List[List[float]] = []

        # For multi-objective: accumulate all evaluated (theta, obj_vec) pairs.
        all_thetas: List[np.ndarray] = []
        all_obj_vecs: List[List[float]] = []

        best_theta = theta.copy()
        best_obj = float("inf")
        convergence_iter: Optional[int] = None
        stagnation_count = 0

        for it in range(cfg.n_iterations):
            method = opt_cfg.method.lower()

            # ---- propose candidate(s) ----
            if method == "gradient":
                new_theta, primary = optimizer.step(
                    self.surrogate, self.objectives[0], self.constraints, theta
                )
                primary_val, obj_vals = self._evaluate(new_theta)
                theta = new_theta
                candidates_x = [theta.copy()]
                candidates_y = [primary_val]

            elif method == "bayesian":
                candidate = optimizer.propose(bounds)
                primary_val, obj_vals = self._evaluate(candidate)
                optimizer.update(candidate, primary_val)
                theta = candidate.copy()
                candidates_x = [candidate]
                candidates_y = [primary_val]

            elif method == "evolutionary":
                n_pop = opt_cfg.population_size
                xs = optimizer.ask(bounds, n_pop)
                ys: List[float] = []
                all_obj_vecs_iter: List[List[float]] = []
                for x in xs:
                    pv, ov = self._evaluate(x)
                    ys.append(pv)
                    all_obj_vecs_iter.append(ov)
                optimizer.tell(xs, ys)
                best_idx = int(np.argmin(ys))
                theta = xs[best_idx].copy()
                primary_val = ys[best_idx]
                obj_vals = all_obj_vecs_iter[best_idx]
                # Record all evaluated points in this iteration.
                for x_i, ov_i in zip(xs, all_obj_vecs_iter):
                    all_thetas.append(x_i.copy())
                    all_obj_vecs.append(ov_i)
                candidates_x = xs
                candidates_y = ys
            else:
                raise ValueError(f"Unknown method: '{method}'")

            # Record for multi-objective tracking (non-evolutionary methods).
            if method != "evolutionary":
                all_thetas.append(theta.copy())
                all_obj_vecs.append(obj_vals)

            # ---- update best ----
            if primary_val < best_obj:
                improvement = abs(best_obj - primary_val)
                best_obj = primary_val
                best_theta = theta.copy()
                if improvement < cfg.convergence_tol:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
            else:
                stagnation_count += 1

            if cfg.save_history:
                history_theta.append(theta.copy())
                history_objectives.append(best_obj)
                history_all_objectives.append(list(candidates_y))

            if cfg.verbose:
                print(
                    f"[DesignOpt] iter {it+1:4d}/{cfg.n_iterations}"
                    f"  best={best_obj:.6e}"
                    f"  theta={np.round(best_theta, 4).tolist()}"
                )

            # ---- convergence check ----
            if stagnation_count >= 5:
                convergence_iter = it
                if cfg.verbose:
                    print(
                        f"[DesignOpt] Converged at iteration {it+1} "
                        f"(stagnation for 5 steps, tol={cfg.convergence_tol})."
                    )
                break

        # ---- Pareto front (multi-objective) ----
        pareto_front: Optional[ParetoFront] = None
        if self._multi_objective and all_obj_vecs:
            obj_matrix = np.array(all_obj_vecs, dtype=np.float64)
            theta_matrix = np.array(all_thetas, dtype=np.float64)
            mask = compute_pareto_front(obj_matrix)
            pareto_front = ParetoFront(
                objectives=obj_matrix,
                params=theta_matrix,
                mask=mask,
            )
            # Override best_theta with the first Pareto-optimal solution
            # (lowest primary objective among Pareto set).
            pareto_obj, pareto_par = pareto_front.filter()
            if pareto_par is not None and len(pareto_par) > 0:
                best_pareto_idx = int(np.argmin(pareto_obj[:, 0]))
                best_theta = pareto_par[best_pareto_idx]
                best_obj = float(pareto_obj[best_pareto_idx, 0])

        # ---- PINN refinement ----
        refinement_results: Optional[List[RefinementResult]] = None
        if cfg.refine_top_k > 0 and self.refinement is not None:
            # Gather top-k thetas sorted by objective.
            if history_theta:
                paired = sorted(
                    zip(history_objectives, history_theta), key=lambda t: t[0]
                )
                top_k_thetas = [th for _, th in paired[: cfg.refine_top_k]]
            else:
                top_k_thetas = [best_theta]
            refinement_results = self.refinement.refine_top_k(
                top_k_thetas, self.surrogate, k=cfg.refine_top_k
            )

        elapsed = time.perf_counter() - t_start

        return DesignOptResult(
            best_theta=best_theta,
            best_objective=best_obj,
            history_theta=history_theta,
            history_objectives=history_objectives,
            history_all_objectives=history_all_objectives,
            pareto_front=pareto_front,
            refinement_results=refinement_results,
            convergence_iter=convergence_iter,
            elapsed_s=elapsed,
        )
