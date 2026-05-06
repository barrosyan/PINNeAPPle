"""General-purpose physics problem solver — tool-based orchestrator.

:class:`PhysicsOrchestrator` is the *secondary pipeline* that treats every
Pinneaple capability as a callable tool and chains them intelligently to
solve an arbitrary physics problem described in a :class:`ProblemStatement`.

Unlike the world-model pipeline (which is opinionated about architecture and
training flow), the orchestrator is **goal-directed**: you describe *what* you
want to solve (e.g. "predict heat distribution in a 2D fin") and the
orchestrator selects, configures, and runs the right sequence of tools
from the :class:`~.physics_tools.PhysicsToolRegistry`.

Supported problem kinds
-----------------------
* ``"forward"`` — solve for the evolution of a physical system.
* ``"inverse"`` — identify parameters from observations.
* ``"design"`` — optimise a design for a physics objective.
* ``"forecast"`` — time-series forecasting of a physics quantity.
* ``"uncertainty"`` — quantify prediction uncertainty.
* ``"discovery"`` — discover governing equations from data.
* ``"digital_twin"`` — build a real-time digital twin.
* ``"world_model"`` — train a generalist physics AI from scratch.

Quick start::

    from pinneaple_worldmodel.orchestrator import PhysicsOrchestrator, ProblemStatement

    # Solve a forward problem
    result = PhysicsOrchestrator().solve(ProblemStatement(
        kind="forward",
        description="2D heat conduction in a square domain",
        pde_hint="heat_2d",
        domain_hint="unit_square",
        params={"alpha": 0.01},
        output=["trajectory", "validation_report"],
    ))

    # Full world-model pipeline
    result = PhysicsOrchestrator().solve(ProblemStatement(
        kind="world_model",
        scenarios=["heat_2d", "burgers_1d", "ns2d_cavity"],
        n_samples=500,
        device="cuda",
        output=["mega_model", "zoo", "benchmark"],
    ))
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .physics_tools import PhysicsToolRegistry

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ProblemStatement
# ---------------------------------------------------------------------------

@dataclass
class ProblemStatement:
    """A high-level description of a physics problem to solve.

    Parameters
    ----------
    kind : str — problem category (see module docstring for full list).
    description : str — free-form description (used for PDE auto-discovery).
    pde_hint : str or None — hint for PDE kind (e.g. ``"heat_2d"``).
    domain_hint : str or None — hint for domain type (e.g. ``"unit_square"``).
    scenarios : list of scenario names — used for world_model / forward kinds.
    params : dict — physics parameters or search ranges.
    observations : any — observed data (used for inverse / digital_twin).
    model : any — pre-trained model to use as starting point (optional).
    n_samples : int — number of training trajectories.
    n_steps : int — rollout or simulation steps.
    device : str
    output : list of str — what to produce.
        Options: ``"trajectory"``, ``"model"``, ``"mega_model"``, ``"zoo"``,
        ``"validation_report"``, ``"uncertainty"``, ``"benchmark"``,
        ``"params_estimate"``, ``"equations"``, ``"design"``, ``"twin"``,
        ``"forecast"``.
    save_dir : str or None
    verbose : bool
    extra : dict — additional keyword arguments forwarded to tools.
    """
    kind: str = "forward"
    description: str = ""
    pde_hint: Optional[str] = None
    domain_hint: Optional[str] = "unit_square"
    scenarios: List[str] = field(
        default_factory=lambda: ["heat_2d", "burgers_1d", "advection_2d"]
    )
    params: Dict[str, Any] = field(default_factory=dict)
    observations: Any = None
    model: Any = None
    n_samples: int = 200
    n_steps: int = 50
    device: str = "cpu"
    output: List[str] = field(default_factory=lambda: ["model"])
    save_dir: Optional[str] = None
    verbose: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# OrchestratorResult
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorResult:
    """Output of :class:`PhysicsOrchestrator.solve`.

    Attributes
    ----------
    kind : str
    artifacts : dict — all produced artifacts (model, zoo, trajectory, …).
    plan : list of str — sequence of tool names actually executed.
    elapsed_s : float
    logs : list of str
    """
    kind: str
    artifacts: Dict[str, Any] = field(default_factory=dict)
    plan: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    logs: List[str] = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return self.artifacts[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.artifacts.get(key, default)

    def summary(self) -> str:
        lines = [
            f"OrchestratorResult [{self.kind}]  ({self.elapsed_s:.1f}s)",
            f"  Plan: {' → '.join(self.plan) if self.plan else '(empty)'}",
            f"  Artifacts: {list(self.artifacts.keys())}",
        ]
        for log_line in self.logs[-5:]:
            lines.append(f"  {log_line}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PhysicsOrchestrator
# ---------------------------------------------------------------------------

class PhysicsOrchestrator:
    """Solve arbitrary physics problems by chaining Pinneaple tools.

    Parameters
    ----------
    registry : PhysicsToolRegistry or None.
        If None, a new registry is built and all tools are registered.
    verbose : bool

    Example
    -------
    >>> orch = PhysicsOrchestrator()
    >>> result = orch.solve(ProblemStatement(
    ...     kind="world_model",
    ...     scenarios=["heat_2d", "burgers_1d", "ns2d_cavity"],
    ...     n_samples=500,
    ...     device="cuda",
    ...     output=["mega_model", "zoo", "benchmark"],
    ... ))
    >>> mega_model = result["mega_model"]
    """

    def __init__(
        self,
        registry: Optional[PhysicsToolRegistry] = None,
        verbose: bool = True,
    ) -> None:
        if registry is None:
            registry = PhysicsToolRegistry()
            registry.register_all()
        self.registry = registry
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def solve(self, problem: ProblemStatement) -> OrchestratorResult:
        """Select and execute the tool chain for *problem*.

        Parameters
        ----------
        problem : ProblemStatement

        Returns
        -------
        OrchestratorResult
        """
        t0 = time.time()
        if self.verbose:
            print(f"\n[PhysicsOrchestrator] Solving: kind={problem.kind!r}  "
                  f"device={problem.device}")
            if problem.description:
                print(f"  Description: {problem.description}")

        # Route to the appropriate plan
        dispatch = {
            "forward":      self._plan_forward,
            "inverse":      self._plan_inverse,
            "design":       self._plan_design,
            "forecast":     self._plan_forecast,
            "uncertainty":  self._plan_uncertainty,
            "discovery":    self._plan_discovery,
            "digital_twin": self._plan_digital_twin,
            "world_model":  self._plan_world_model,
        }
        planner = dispatch.get(problem.kind)
        if planner is None:
            raise ValueError(
                f"Unknown problem kind: {problem.kind!r}. "
                f"Available: {list(dispatch)}"
            )

        result = OrchestratorResult(kind=problem.kind)
        try:
            planner(problem, result)
        except Exception as exc:
            result.logs.append(f"ERROR: {exc}")
            log.error("Orchestrator failed for kind=%s: %s", problem.kind, exc)
            if self.verbose:
                print(f"[PhysicsOrchestrator] ERROR: {exc}")
            raise

        result.elapsed_s = time.time() - t0
        if self.verbose:
            print(result.summary())
        return result

    # ------------------------------------------------------------------
    # Plan: world_model (full physics AI pipeline)
    # ------------------------------------------------------------------

    def _plan_world_model(
        self, problem: ProblemStatement, result: OrchestratorResult
    ) -> None:
        """Full pipeline: scenarios → datasets → specialists → meta → mega."""
        from .dataset_factory import PhysicsDatasetFactory, FactoryConfig
        from .specialist_trainer import SpecialistTrainer, SpecialistConfig
        from .meta_learning import MetaLearner, MetaConfig
        from .mega_model import PhysicsFoundationModel, FoundationConfig
        from .model_zoo import ModelZoo
        from .benchmark import PhysicsBenchmark

        p = problem
        wants = set(p.output)

        # Step 1: Multi-source dataset generation
        self._log(result, "Step 1/5: Generating multi-source datasets …")
        sources = p.extra.get("sources", ["solver"])
        factory_cfg = FactoryConfig(
            sources=sources,
            scenarios=p.scenarios,
            n_samples_per_scenario=p.n_samples,
            device=p.device,
            save_dir=str(p.save_dir) + "/datasets" if p.save_dir else None,
            verbose=p.verbose,
        )
        catalog = PhysicsDatasetFactory(factory_cfg).build()
        result.artifacts["catalog"] = catalog
        result.plan.append("build_multi_source_datasets")

        # Step 2: Specialist training
        self._log(result, "Step 2/5: Training specialist models …")
        spec_cfg = SpecialistConfig(
            epochs=p.extra.get("specialist_epochs", 50),
            lr=p.extra.get("lr", 1e-3),
            batch_size=p.extra.get("batch_size", 32),
            device=p.device,
            save_dir=str(p.save_dir) + "/specialists" if p.save_dir else None,
            verbose=p.verbose,
        )
        zoo = SpecialistTrainer(spec_cfg).train_all(catalog)
        result.artifacts["zoo"] = zoo
        result.plan.append("train_specialists")

        # Step 3: Meta-learning
        if "mega_model" in wants or p.extra.get("meta_train", True):
            self._log(result, "Step 3/5: Meta-learning (Reptile) …")
            meta_cfg = MetaConfig(
                algorithm=p.extra.get("meta_algorithm", "reptile"),
                n_meta_epochs=p.extra.get("meta_epochs", 200),
                device=p.device,
                verbose=p.verbose,
            )
            meta_learner = MetaLearner(meta_cfg)
            meta_model = meta_learner.meta_train(catalog)
            result.artifacts["meta_model"] = meta_model
            result.plan.append("meta_train_reptile")
        else:
            meta_model = None

        # Step 4: Physics Foundation Model (mega model)
        if "mega_model" in wants:
            self._log(result, "Step 4/5: Building & training PhysicsFoundationModel …")
            # Start from a weight-averaged soup of all specialists
            if len(zoo) > 0:
                try:
                    soup_model = zoo.soup()
                    self._log(result, f"  Initialising from weight-averaged soup "
                              f"of {len(zoo)} specialists.")
                except Exception as exc:
                    log.debug("Soup failed: %s", exc)
                    soup_model = None
            else:
                soup_model = None

            found_cfg = FoundationConfig(
                n_modes=p.extra.get("n_modes", 16),
                width=p.extra.get("width", 128),
                depth=p.extra.get("depth", 6),
                device=p.device,
            )
            ref_ds = next(iter(catalog.entries)).dataset
            mega_model = PhysicsFoundationModel(
                found_cfg,
                n_fields=ref_ds.n_fields,
                grid_shape=ref_ds.grid_shape,
            )
            if soup_model is not None:
                try:
                    mega_model.load_state_dict(soup_model.state_dict(), strict=False)
                    self._log(result, "  Loaded soup weights → mega model.")
                except Exception:
                    pass

            # Fine-tune on merged catalog
            from .trainer import WorldModelTrainer, WorldModelTrainConfig
            merged_ds = catalog.merged()
            train_cfg = WorldModelTrainConfig(
                epochs=p.extra.get("mega_epochs", 100),
                lr=p.extra.get("mega_lr", 5e-4),
                batch_size=p.extra.get("batch_size", 32),
                device=p.device,
                patience=20,
            )
            trainer = WorldModelTrainer(mega_model, train_cfg)
            history = trainer.fit(merged_ds)
            result.artifacts["mega_model"] = mega_model
            result.artifacts["mega_history"] = history
            result.plan.append("train_mega_model")

            if p.save_dir:
                mega_model.save(str(p.save_dir) + "/mega_model.pt")
        else:
            self._log(result, "Step 4/5: Skipped (mega_model not in output).")

        # Step 5: Benchmark
        if "benchmark" in wants:
            self._log(result, "Step 5/5: Running benchmark …")
            bench = PhysicsBenchmark(device=p.device, verbose=p.verbose)
            eval_model = result.artifacts.get("mega_model") or (
                zoo.get(p.scenarios[0]) if p.scenarios and p.scenarios[0] in zoo else None
            )
            if eval_model is not None:
                bench_results = bench.run(
                    eval_model,
                    tasks=[t for t in p.scenarios if t in
                           ["heat_2d_smooth", "burgers_1d_shock", "advection_2d_gaussian",
                            "ns2d_lid_driven"]]
                    or None,
                )
                result.artifacts["benchmark"] = bench_results
                result.plan.append("benchmark")
        else:
            self._log(result, "Step 5/5: Skipped (benchmark not in output).")

    # ------------------------------------------------------------------
    # Plan: forward (solve for evolution)
    # ------------------------------------------------------------------

    def _plan_forward(
        self, problem: ProblemStatement, result: OrchestratorResult
    ) -> None:
        """Forward problem: simulate → train → rollout."""
        p = problem

        # 1. Build dataset
        tool = self.registry.get("build_world_model_dataset")
        scenarios = p.scenarios or (
            [p.pde_hint] if p.pde_hint else ["heat_2d"]
        )
        dataset = tool.call(
            scenarios=scenarios,
            n_samples=p.n_samples,
            device=p.device,
        )
        result.artifacts["dataset"] = dataset
        result.plan.append("build_world_model_dataset")

        # 2. Train world model
        if "model" in p.output or "trajectory" in p.output:
            train_tool = self.registry.get("train_world_model")
            artifacts = train_tool.call(
                dataset=dataset,
                model=p.model,
                epochs=p.extra.get("epochs", 50),
                device=p.device,
            )
            model = artifacts["model"]
            result.artifacts["model"] = model
            result.artifacts["history"] = artifacts["history"]
            result.plan.append("train_world_model")

        # 3. Rollout
        if "trajectory" in p.output and "model" in result.artifacts:
            import torch
            model = result.artifacts["model"]
            model.eval()
            state_0 = torch.randn(1, dataset.n_fields, *dataset.grid_shape)
            traj = result.artifacts["model"].rollout(state_0, n_steps=p.n_steps)
            result.artifacts["trajectory"] = traj
            result.plan.append("rollout_trajectory")

        # 4. Validation
        if "validation_report" in p.output:
            try:
                val_tool = self.registry.get("validate_physics")
                report = val_tool.call(
                    model=result.artifacts.get("model"),
                    dataset=dataset,
                    scenario=scenarios[0] if scenarios else "",
                )
                result.artifacts["validation_report"] = report
                result.plan.append("validate_physics")
            except Exception as exc:
                self._log(result, f"Validation skipped: {exc}")

        # 5. Uncertainty
        if "uncertainty" in p.output and "model" in result.artifacts:
            try:
                import torch
                uq_tool = self.registry.get("decompose_uncertainty")
                x = torch.randn(4, dataset.n_fields, *dataset.grid_shape)
                uq_result = uq_tool.call(model=result.artifacts["model"], x=x)
                result.artifacts["uncertainty"] = uq_result
                result.plan.append("decompose_uncertainty")
            except Exception as exc:
                self._log(result, f"UQ skipped: {exc}")

    # ------------------------------------------------------------------
    # Plan: inverse
    # ------------------------------------------------------------------

    def _plan_inverse(
        self, problem: ProblemStatement, result: OrchestratorResult
    ) -> None:
        """Inverse problem: observe → EKI or SINDy → parameter estimate."""
        p = problem
        if p.observations is None:
            raise ValueError("Inverse problem requires `observations`.")

        if "equations" in p.output:
            tool = self.registry.get("sindy_equation_discovery")
            equations = tool.call(trajectory=p.observations)
            result.artifacts["equations"] = equations
            result.plan.append("sindy_equation_discovery")
        else:
            model = p.model
            if model is None:
                # Train a quick forward model first
                self._plan_forward(problem, result)
                model = result.artifacts.get("model")

            tool = self.registry.get("eki_parameter_inversion")
            params_est = tool.call(model=model, observations=p.observations)
            result.artifacts["params_estimate"] = params_est
            result.plan.append("eki_parameter_inversion")

        if "uncertainty" in p.output:
            try:
                uq_tool = self.registry.get("decompose_uncertainty")
                uq_result = uq_tool.call(
                    model=result.artifacts.get("model"), x=p.observations
                )
                result.artifacts["uncertainty"] = uq_result
                result.plan.append("decompose_uncertainty")
            except Exception as exc:
                self._log(result, f"UQ skipped: {exc}")

    # ------------------------------------------------------------------
    # Plan: discovery
    # ------------------------------------------------------------------

    def _plan_discovery(
        self, problem: ProblemStatement, result: OrchestratorResult
    ) -> None:
        """Equation discovery: data → SINDy → governing equations."""
        p = problem
        if p.observations is None:
            # Generate data first
            self._plan_forward(problem, result)
            obs = result.artifacts.get("trajectory")
        else:
            obs = p.observations

        tool = self.registry.get("sindy_equation_discovery")
        equations = tool.call(trajectory=obs)
        result.artifacts["equations"] = equations
        result.plan.append("sindy_equation_discovery")

        if p.description:
            try:
                id_tool = self.registry.get("identify_pde")
                pde_info = id_tool.call(description=p.description)
                result.artifacts["pde_identification"] = pde_info
                result.plan.append("identify_pde")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Plan: design
    # ------------------------------------------------------------------

    def _plan_design(
        self, problem: ProblemStatement, result: OrchestratorResult
    ) -> None:
        """Design optimisation: surrogate model + Bayesian optimiser."""
        p = problem

        # Build surrogate
        self._plan_forward(problem, result)
        surrogate = result.artifacts.get("model")

        tool = self.registry.get("bayesian_design_opt")
        import torch
        bounds = p.extra.get("bounds", torch.zeros(2, 2))
        best = tool.call(objective=surrogate, bounds=bounds)
        result.artifacts["design"] = best
        result.plan.append("bayesian_design_opt")

    # ------------------------------------------------------------------
    # Plan: forecast
    # ------------------------------------------------------------------

    def _plan_forecast(
        self, problem: ProblemStatement, result: OrchestratorResult
    ) -> None:
        """Time-series forecasting via pinneaple_timeseries."""
        p = problem
        data = p.observations
        if data is None:
            # Generate a trajectory and use it as input time series
            traj_tool = self.registry.get("simulate_trajectory")
            traj = traj_tool.call(
                scenario=p.pde_hint or "heat_2d",
                n_steps=p.n_steps,
                params=p.params,
                device=p.device,
            )
            data = traj.states
            result.artifacts["generated_trajectory"] = traj
            result.plan.append("simulate_trajectory")

        tool = self.registry.get("timeseries_forecast")
        forecast = tool.call(data=data, horizon=p.extra.get("horizon", 10))
        result.artifacts["forecast"] = forecast
        result.plan.append("timeseries_forecast")

        if "uncertainty" in p.output:
            try:
                psd_tool = self.registry.get("power_spectrum")
                psd = psd_tool.call(data=data)
                result.artifacts["power_spectrum"] = psd
                result.plan.append("power_spectrum")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Plan: uncertainty
    # ------------------------------------------------------------------

    def _plan_uncertainty(
        self, problem: ProblemStatement, result: OrchestratorResult
    ) -> None:
        """Full UQ: train model → decompose aleatoric + epistemic."""
        p = problem
        # Build forward model if needed
        if p.model is None:
            self._plan_forward(problem, result)

        import torch
        model = problem.model or result.artifacts.get("model")
        dataset = result.artifacts.get("dataset")

        if model is None or dataset is None:
            raise RuntimeError("Model and dataset required for uncertainty analysis.")

        x = torch.randn(8, dataset.n_fields, *dataset.grid_shape)

        # Aleatoric
        try:
            al_tool = self.registry.get("aleatoric_uncertainty")
            result.artifacts["aleatoric"] = al_tool.call(model=model, x=x)
            result.plan.append("aleatoric_uncertainty")
        except Exception as exc:
            self._log(result, f"Aleatoric UQ skipped: {exc}")

        # Full decomposition
        try:
            dc_tool = self.registry.get("decompose_uncertainty")
            result.artifacts["uncertainty_decomposition"] = dc_tool.call(model=model, x=x)
            result.plan.append("decompose_uncertainty")
        except Exception as exc:
            self._log(result, f"Decomposition UQ skipped: {exc}")

    # ------------------------------------------------------------------
    # Plan: digital_twin
    # ------------------------------------------------------------------

    def _plan_digital_twin(
        self, problem: ProblemStatement, result: OrchestratorResult
    ) -> None:
        """Digital twin: train model → attach data stream → build twin."""
        p = problem

        # Build forward model
        self._plan_forward(problem, result)
        model = result.artifacts.get("model")
        if model is None:
            raise RuntimeError("Forward model required for digital twin.")

        # Build digital twin (stream from extra or None)
        stream = p.extra.get("stream")
        if stream is not None:
            tool = self.registry.get("build_digital_twin")
            twin = tool.call(model=model, stream=stream)
            result.artifacts["twin"] = twin
            result.plan.append("build_digital_twin")

        # Data assimilation with Kalman filter
        if p.observations is not None:
            try:
                ekf_tool = self.registry.get("kalman_data_assimilation")
                state_est = ekf_tool.call(model=model, obs=p.observations)
                result.artifacts["state_estimate"] = state_est
                result.plan.append("kalman_data_assimilation")
            except Exception as exc:
                self._log(result, f"Kalman assimilation skipped: {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, result: OrchestratorResult, msg: str) -> None:
        result.logs.append(msg)
        if self.verbose:
            print(f"  {msg}")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def list_tools(self, category: Optional[str] = None) -> None:
        """Print all available tools (optionally filtered by category)."""
        if category:
            tools = self.registry.list_by_category(category)
        else:
            tools = self.registry.available_tools()
        print(f"\n[PhysicsOrchestrator] {len(tools)} tool(s):")
        for t in tools:
            status = "✓" if t.is_available() else "✗"
            print(f"  {status} [{t.category:15s}] {t.name:35s} — {t.description}")

    def tool_chain_for(self, kind: str) -> List[str]:
        """Return the list of tool names that would be called for *kind*."""
        dummy = ProblemStatement(kind=kind, verbose=False)
        r = OrchestratorResult(kind=kind)
        try:
            dispatch = {
                "forward":      self._plan_forward,
                "inverse":      self._plan_inverse,
                "design":       self._plan_design,
                "forecast":     self._plan_forecast,
                "uncertainty":  self._plan_uncertainty,
                "discovery":    self._plan_discovery,
                "digital_twin": self._plan_digital_twin,
                "world_model":  self._plan_world_model,
            }
            if kind in dispatch:
                dispatch[kind](dummy, r)
        except Exception:
            pass
        return r.plan
