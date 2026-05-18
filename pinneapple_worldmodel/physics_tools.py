"""Pinneapple module capability inventory for tool-based orchestration.

This module treats every Pinneapple sub-package as a *tool* with a declared
capability signature.  The :class:`PhysicsToolRegistry` catalogues what each
tool can do, what inputs it requires, and what outputs it produces.  The
:class:`~.orchestrator.PhysicsOrchestrator` uses this registry to discover
and chain tools automatically when solving an arbitrary physics problem.

Tool taxonomy
-------------
::

    Category         | Module(s)
    -----------------+----------------------------------------------------
    simulation       | pinneapple_solvers, pinneapple_dynamics
    pde_solving      | pinneapple_pinn, pinneapple_environment
    symbolic         | pinneapple_symbolic
    data_generation  | pinneapple_data, pinneapple_worldmodel.dataset_factory
    training         | pinneapple_train, pinneapple_worldmodel.trainer
    validation       | pinneapple_validate
    uncertainty      | pinneapple_uq
    inverse          | pinneapple_inverse
    transfer         | pinneapple_transfer
    meta_learning    | pinneapple_meta
    timeseries       | pinneapple_timeseries
    co_simulation    | pinneapple_cosim
    geometry         | pinneapple_geom, pinneapple_worldmodel.geometry
    inference        | pinneapple_inference
    design_opt       | pinneapple_design_opt
    digital_twin     | pinneapple_digital_twin
    visualization    | pinneapple_viz
    world_model      | pinneapple_worldmodel (this package)

Each tool exposes a standardised ``call(**kwargs) → result`` interface.
Tools are *lazy*: the underlying module is imported only when the tool
is first called, so missing optional dependencies never break the registry.

Usage::

    from pinneapple_worldmodel.physics_tools import PhysicsToolRegistry

    reg = PhysicsToolRegistry()
    tool = reg.get("simulate_trajectory")
    traj = tool.call(scenario="heat_2d", n_steps=50, params={"alpha": 0.01})

    available = reg.list_by_category("simulation")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PhysicsTool — single callable capability
# ---------------------------------------------------------------------------

@dataclass
class PhysicsTool:
    """One registered physics capability.

    Parameters
    ----------
    name : str — unique tool identifier.
    category : str — tool category (see taxonomy above).
    description : str — human-readable description.
    input_schema : dict — expected input keys with type hints as strings.
    output_schema : dict — produced output keys with type hints as strings.
    module_path : str — dotted path to the underlying Pinneapple module.
    fn : callable — the actual implementation (lazy-imported).
    tags : list of str — extra search tags.
    """
    name: str
    category: str
    description: str
    input_schema: Dict[str, str]
    output_schema: Dict[str, str]
    module_path: str
    fn: Optional[Callable] = field(default=None, repr=False)
    tags: List[str] = field(default_factory=list)

    def call(self, **kwargs: Any) -> Any:
        """Invoke the tool with keyword arguments.

        Raises
        ------
        RuntimeError if the underlying module is unavailable.
        """
        if self.fn is None:
            raise RuntimeError(
                f"Tool '{self.name}' has no implementation registered. "
                f"Module: {self.module_path}"
            )
        try:
            return self.fn(**kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Tool '{self.name}' failed: {exc}"
            ) from exc

    def is_available(self) -> bool:
        return self.fn is not None


# ---------------------------------------------------------------------------
# PhysicsToolRegistry
# ---------------------------------------------------------------------------

class PhysicsToolRegistry:
    """Catalogue of all Pinneapple capabilities as callable tools.

    Call :meth:`register_all` to auto-discover tools for all installed
    Pinneapple modules, or use :meth:`register` to add custom tools.

    Example
    -------
    >>> reg = PhysicsToolRegistry()
    >>> reg.register_all()
    >>> print(reg.summary())
    >>> tool = reg.get("simulate_trajectory")
    >>> traj = tool.call(scenario="heat_2d", n_steps=50)
    """

    def __init__(self) -> None:
        self._tools: Dict[str, PhysicsTool] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: PhysicsTool, *, overwrite: bool = False) -> None:
        if tool.name in self._tools and not overwrite:
            raise ValueError(f"Tool '{tool.name}' already registered.")
        self._tools[tool.name] = tool

    def register_all(self) -> None:
        """Auto-discover and register tools for all installed Pinneapple modules."""
        self._register_simulation_tools()
        self._register_pde_tools()
        self._register_data_tools()
        self._register_training_tools()
        self._register_validation_tools()
        self._register_uq_tools()
        self._register_inverse_tools()
        self._register_transfer_tools()
        self._register_meta_tools()
        self._register_timeseries_tools()
        self._register_cosim_tools()
        self._register_geometry_tools()
        self._register_inference_tools()
        self._register_design_opt_tools()
        self._register_digital_twin_tools()
        self._register_worldmodel_tools()
        log.info("PhysicsToolRegistry: %d tools registered.", len(self._tools))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, name: str) -> PhysicsTool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found. Available: {list(self._tools)}")
        return self._tools[name]

    def list_by_category(self, category: str) -> List[PhysicsTool]:
        return [t for t in self._tools.values() if t.category == category]

    def list_by_tag(self, tag: str) -> List[PhysicsTool]:
        return [t for t in self._tools.values() if tag in t.tags]

    def search(self, query: str) -> List[PhysicsTool]:
        """Fuzzy search across name, description, and tags."""
        q = query.lower()
        results = []
        for t in self._tools.values():
            if (q in t.name.lower() or q in t.description.lower()
                    or any(q in tag for tag in t.tags)):
                results.append(t)
        return results

    def available_tools(self) -> List[PhysicsTool]:
        return [t for t in self._tools.values() if t.is_available()]

    def categories(self) -> Set[str]:
        return {t.category for t in self._tools.values()}

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"PhysicsToolRegistry: {len(self._tools)} tools "
            f"({sum(1 for t in self._tools.values() if t.is_available())} available)\n"
        ]
        for cat in sorted(self.categories()):
            tools = self.list_by_category(cat)
            avail = sum(1 for t in tools if t.is_available())
            lines.append(f"  [{cat}] {avail}/{len(tools)} available")
            for t in tools:
                status = "✓" if t.is_available() else "✗"
                lines.append(f"    {status} {t.name:35s} — {t.description[:60]}")
        return "\n".join(lines)

    # ====================================================================
    # Tool registrations — one method per module
    # ====================================================================

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _register_simulation_tools(self) -> None:
        """pinneapple_solvers + pinneapple_dynamics"""

        def _simulate_trajectory(
            scenario: str = "heat_2d",
            n_steps: int = 50,
            params: Optional[Dict] = None,
            seed: int = 0,
            device: str = "cpu",
        ) -> Any:
            from .scenario import BUILTIN_SCENARIOS
            from .simulator import PhysicsSimulator
            sc = BUILTIN_SCENARIOS[scenario]
            sim = PhysicsSimulator(sc, device=device, verbose=False)
            return sim.generate_trajectory(params=params, seed=seed)

        self.register(PhysicsTool(
            name="simulate_trajectory",
            category="simulation",
            description="Generate a single physics trajectory via classical solver",
            input_schema={"scenario": "str", "n_steps": "int", "params": "dict",
                          "seed": "int", "device": "str"},
            output_schema={"states": "Tensor(T+1,C,*grid)", "params": "dict"},
            module_path="pinneapple_worldmodel.simulator",
            fn=_simulate_trajectory,
            tags=["fdm", "trajectory", "physics"],
        ))

        def _simulate_batch(
            scenario: str = "heat_2d",
            n_samples: int = 100,
            device: str = "cpu",
        ) -> Any:
            from .scenario import BUILTIN_SCENARIOS
            from .simulator import PhysicsSimulator
            sc = BUILTIN_SCENARIOS[scenario]
            sim = PhysicsSimulator(sc, device=device, verbose=False)
            return sim.generate_batch(n_samples=n_samples)

        self.register(PhysicsTool(
            name="simulate_batch",
            category="simulation",
            description="Generate a batch of physics trajectories",
            input_schema={"scenario": "str", "n_samples": "int", "device": "str"},
            output_schema={"trajectories": "List[TrajectoryData]"},
            module_path="pinneapple_worldmodel.simulator",
            fn=_simulate_batch,
            tags=["batch", "trajectory", "physics"],
        ))

        # pinneapple_solvers: FDM, FEM, LBM, SPH
        def _run_fdm_solver(pde_kind: str, grid_shape: tuple, **kwargs: Any) -> Any:
            try:
                from pinneapple_simulation.numerical_solvers import SolverRegistry  # type: ignore
                solver = SolverRegistry.get(pde_kind)
                return solver.solve(grid_shape=grid_shape, **kwargs)
            except Exception as exc:
                raise RuntimeError(f"FDM solver failed: {exc}") from exc

        self.register(PhysicsTool(
            name="run_fdm_solver",
            category="simulation",
            description="Run pinneapple_solvers FDM solver for a given PDE",
            input_schema={"pde_kind": "str", "grid_shape": "tuple"},
            output_schema={"solution": "Tensor"},
            module_path="pinneapple_solvers",
            fn=self._safe_wrap(_run_fdm_solver, "pinneapple_solvers"),
            tags=["fdm", "solver", "classical"],
        ))

        def _run_sph(n_particles: int = 1000, **kwargs: Any) -> Any:
            try:
                from pinneapple_simulation.particle_dynamics import SPHParticles, ParticleSystem  # type: ignore
                system = ParticleSystem(SPHParticles(n_particles), **kwargs)
                return system.run()
            except Exception as exc:
                raise RuntimeError(f"SPH solver failed: {exc}") from exc

        self.register(PhysicsTool(
            name="run_sph_simulation",
            category="simulation",
            description="Particle-based fluid simulation via pinneapple_dynamics SPH",
            input_schema={"n_particles": "int"},
            output_schema={"particle_states": "List[Tensor]"},
            module_path="pinneapple_dynamics",
            fn=self._safe_wrap(_run_sph, "pinneapple_dynamics"),
            tags=["sph", "particles", "fluid"],
        ))

    # ------------------------------------------------------------------
    # PDE solving
    # ------------------------------------------------------------------

    def _register_pde_tools(self) -> None:
        def _compile_pinn(pde_kind: str, domain: Any, **kwargs: Any) -> Any:
            from pinneapple_physics.pinn_solver import compile_problem  # type: ignore
            return compile_problem(pde_kind=pde_kind, domain=domain, **kwargs)

        self.register(PhysicsTool(
            name="compile_pinn",
            category="pde_solving",
            description="Compile a PINN problem via pinneapple_pinn",
            input_schema={"pde_kind": "str", "domain": "PhysicsDomain"},
            output_schema={"pinn_problem": "CompiledProblem"},
            module_path="pinneapple_pinn",
            fn=self._safe_wrap(_compile_pinn, "pinneapple_pinn"),
            tags=["pinn", "pde", "compile"],
        ))

        def _identify_pde(description: str, **kwargs: Any) -> Any:
            from pinneapple_physics.pde_environment.capabilities import identify_pde  # type: ignore
            return identify_pde(description, **kwargs)

        self.register(PhysicsTool(
            name="identify_pde",
            category="pde_solving",
            description="Auto-identify PDE structure from a natural-language description",
            input_schema={"description": "str"},
            output_schema={"pde_kind": "str", "suggested_params": "dict"},
            module_path="pinneapple_environment.capabilities",
            fn=self._safe_wrap(_identify_pde, "pinneapple_environment"),
            tags=["pde", "auto", "discovery"],
        ))

        def _suggest_problem_spec(description: str, **kwargs: Any) -> Any:
            from pinneapple_physics.pde_environment.capabilities import suggest_problem_spec  # type: ignore
            return suggest_problem_spec(description, **kwargs)

        self.register(PhysicsTool(
            name="suggest_problem_spec",
            category="pde_solving",
            description="Auto-suggest a ProblemSpec from a description",
            input_schema={"description": "str"},
            output_schema={"problem_spec": "ProblemSpec"},
            module_path="pinneapple_environment.capabilities",
            fn=self._safe_wrap(_suggest_problem_spec, "pinneapple_environment"),
            tags=["spec", "auto", "problem"],
        ))

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def _register_data_tools(self) -> None:
        def _build_dataset(scenarios: list, n_samples: int = 500, **kwargs: Any) -> Any:
            from .dataset import DatasetBuilder, DatasetConfig
            cfg = DatasetConfig(scenarios=scenarios, n_samples_per_scenario=n_samples,
                                **kwargs)
            return DatasetBuilder(cfg).build()

        self.register(PhysicsTool(
            name="build_world_model_dataset",
            category="data_generation",
            description="Build a WorldModelDataset from scenario list",
            input_schema={"scenarios": "List[str]", "n_samples": "int"},
            output_schema={"dataset": "WorldModelDataset"},
            module_path="pinneapple_worldmodel.dataset",
            fn=_build_dataset,
            tags=["dataset", "world_model"],
        ))

        def _sample_collocation(domain: Any, n_points: int = 1000, **kwargs: Any) -> Any:
            from pinneapple_data import CollocationSampler, CollocationConfig  # type: ignore
            cfg = CollocationConfig(n_points=n_points, **kwargs)
            sampler = CollocationSampler(cfg)
            return sampler.sample(domain)

        self.register(PhysicsTool(
            name="sample_collocation",
            category="data_generation",
            description="Sample collocation points on a domain via pinneapple_data",
            input_schema={"domain": "PhysicsDomain", "n_points": "int"},
            output_schema={"points": "Tensor(N, d)"},
            module_path="pinneapple_data",
            fn=self._safe_wrap(_sample_collocation, "pinneapple_data"),
            tags=["collocation", "sampling"],
        ))

        def _active_learn(model: Any, domain: Any, **kwargs: Any) -> Any:
            from pinneapple_data import ResidualBasedAL, AdaptiveCollocationTrainer  # type: ignore
            al = ResidualBasedAL(**kwargs)
            return al.suggest_points(model, domain)

        self.register(PhysicsTool(
            name="active_learning_sampling",
            category="data_generation",
            description="Residual-based active learning for collocation sampling",
            input_schema={"model": "nn.Module", "domain": "PhysicsDomain"},
            output_schema={"new_points": "Tensor"},
            module_path="pinneapple_data",
            fn=self._safe_wrap(_active_learn, "pinneapple_data"),
            tags=["active_learning", "sampling", "adaptive"],
        ))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _register_training_tools(self) -> None:
        def _train_world_model(dataset: Any, model: Any = None, **kwargs: Any) -> Any:
            from .trainer import WorldModelTrainer, WorldModelTrainConfig
            from .model import PhysicsWorldModel, WorldModelConfig
            if model is None:
                model = PhysicsWorldModel(
                    WorldModelConfig(), n_fields=dataset.n_fields,
                    grid_shape=dataset.grid_shape
                )
            cfg = WorldModelTrainConfig(**{k: v for k, v in kwargs.items()
                                          if k in WorldModelTrainConfig.__dataclass_fields__})
            trainer = WorldModelTrainer(model, cfg)
            history = trainer.fit(dataset)
            return {"model": model, "history": history}

        self.register(PhysicsTool(
            name="train_world_model",
            category="training",
            description="Train a PhysicsWorldModel on a WorldModelDataset",
            input_schema={"dataset": "WorldModelDataset", "model": "PhysicsWorldModel"},
            output_schema={"model": "PhysicsWorldModel", "history": "List[dict]"},
            module_path="pinneapple_worldmodel.trainer",
            fn=_train_world_model,
            tags=["training", "world_model", "fno"],
        ))

        def _train_pinn(problem: Any, model: Any = None, **kwargs: Any) -> Any:
            try:
                from pinneapple_neural.trainer import Trainer, TrainConfig  # type: ignore
                cfg = TrainConfig(**kwargs)
                trainer = Trainer(model or problem.default_model(), cfg)
                return trainer.fit(problem)
            except Exception as exc:
                raise RuntimeError(f"PINN training failed: {exc}") from exc

        self.register(PhysicsTool(
            name="train_pinn",
            category="training",
            description="Train a PINN model via pinneapple_train",
            input_schema={"problem": "CompiledProblem", "model": "nn.Module"},
            output_schema={"model": "nn.Module", "history": "List[dict]"},
            module_path="pinneapple_train",
            fn=self._safe_wrap(_train_pinn, "pinneapple_train"),
            tags=["pinn", "training"],
        ))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _register_validation_tools(self) -> None:
        def _validate_physics(model: Any, dataset: Any, scenario: str = "", **kwargs: Any) -> Any:
            try:
                from pinneapple_analysis.validation import PhysicsValidator  # type: ignore
                pv = PhysicsValidator()
                return pv.check(model, dataset, scenario=scenario, **kwargs)
            except Exception as exc:
                raise RuntimeError(f"Physics validation failed: {exc}") from exc

        self.register(PhysicsTool(
            name="validate_physics",
            category="validation",
            description="Check conservation laws and physics consistency of predictions",
            input_schema={"model": "nn.Module", "dataset": "Dataset", "scenario": "str"},
            output_schema={"passed": "bool", "metrics": "dict"},
            module_path="pinneapple_validate",
            fn=self._safe_wrap(_validate_physics, "pinneapple_validate"),
            tags=["validation", "conservation", "physics"],
        ))

    # ------------------------------------------------------------------
    # Uncertainty
    # ------------------------------------------------------------------

    def _register_uq_tools(self) -> None:
        def _mc_dropout_uq(model: Any, x: Any, n_samples: int = 50, **kwargs: Any) -> Any:
            from pinneapple_analysis.uncertainty import uq_predict  # type: ignore
            return uq_predict(model, x, method="mc_dropout", n_samples=n_samples)

        self.register(PhysicsTool(
            name="mc_dropout_uncertainty",
            category="uncertainty",
            description="Monte-Carlo Dropout uncertainty estimation",
            input_schema={"model": "nn.Module", "x": "Tensor", "n_samples": "int"},
            output_schema={"mean": "Tensor", "std": "Tensor", "epistemic_std": "Tensor"},
            module_path="pinneapple_uq",
            fn=self._safe_wrap(_mc_dropout_uq, "pinneapple_uq"),
            tags=["uq", "mc_dropout", "epistemic"],
        ))

        def _aleatoric_uq(model: Any, x: Any, **kwargs: Any) -> Any:
            from pinneapple_analysis.uncertainty import uq_predict  # type: ignore
            return uq_predict(model, x, method="aleatoric")

        self.register(PhysicsTool(
            name="aleatoric_uncertainty",
            category="uncertainty",
            description="Aleatoric (data) uncertainty via heteroscedastic head",
            input_schema={"model": "nn.Module", "x": "Tensor"},
            output_schema={"mean": "Tensor", "aleatoric_std": "Tensor"},
            module_path="pinneapple_uq",
            fn=self._safe_wrap(_aleatoric_uq, "pinneapple_uq"),
            tags=["uq", "aleatoric", "heteroscedastic"],
        ))

        def _decompose_uq(model: Any, x: Any, n_samples: int = 50, **kwargs: Any) -> Any:
            from pinneapple_analysis.uncertainty import decompose_uncertainty  # type: ignore
            return decompose_uncertainty(model, x, n_samples=n_samples)

        self.register(PhysicsTool(
            name="decompose_uncertainty",
            category="uncertainty",
            description="Decompose total uncertainty into aleatoric + epistemic",
            input_schema={"model": "nn.Module", "x": "Tensor", "n_samples": "int"},
            output_schema={"aleatoric_std": "Tensor", "epistemic_std": "Tensor",
                           "total_std": "Tensor"},
            module_path="pinneapple_uq",
            fn=self._safe_wrap(_decompose_uq, "pinneapple_uq"),
            tags=["uq", "decompose", "aleatoric", "epistemic"],
        ))

    # ------------------------------------------------------------------
    # Inverse problems
    # ------------------------------------------------------------------

    def _register_inverse_tools(self) -> None:
        def _eki_inversion(model: Any, observations: Any, prior: Any = None, **kwargs: Any) -> Any:
            from pinneapple_analysis.inverse_problems import EnsembleKalmanInversion  # type: ignore
            solver = EnsembleKalmanInversion(model, prior=prior, **kwargs)
            return solver.solve(observations)

        self.register(PhysicsTool(
            name="eki_parameter_inversion",
            category="inverse",
            description="Ensemble Kalman Inversion for parameter identification",
            input_schema={"model": "nn.Module", "observations": "Tensor",
                          "prior": "distribution"},
            output_schema={"params": "Tensor", "uncertainty": "Tensor"},
            module_path="pinneapple_inverse",
            fn=self._safe_wrap(_eki_inversion, "pinneapple_inverse"),
            tags=["inverse", "eki", "parameter"],
        ))

        def _sindy(trajectory: Any, **kwargs: Any) -> Any:
            from pinneapple_analysis.inverse_problems import SINDyIdentifier  # type: ignore
            sindy = SINDyIdentifier(**kwargs)
            return sindy.fit(trajectory)

        self.register(PhysicsTool(
            name="sindy_equation_discovery",
            category="inverse",
            description="SINDy sparse equation discovery from trajectory data",
            input_schema={"trajectory": "Tensor"},
            output_schema={"equation": "str", "coefficients": "Tensor"},
            module_path="pinneapple_inverse",
            fn=self._safe_wrap(_sindy, "pinneapple_inverse"),
            tags=["inverse", "sindy", "discovery", "sparse"],
        ))

        def _sensitivity(model: Any, params: Any, **kwargs: Any) -> Any:
            from pinneapple_analysis.inverse_problems import LocalSensitivity  # type: ignore
            return LocalSensitivity(model, **kwargs).compute(params)

        self.register(PhysicsTool(
            name="local_sensitivity",
            category="inverse",
            description="Local parameter sensitivity analysis (Jacobian-based)",
            input_schema={"model": "nn.Module", "params": "Tensor"},
            output_schema={"jacobian": "Tensor", "sensitivity_scores": "dict"},
            module_path="pinneapple_inverse",
            fn=self._safe_wrap(_sensitivity, "pinneapple_inverse"),
            tags=["inverse", "sensitivity", "jacobian"],
        ))

    # ------------------------------------------------------------------
    # Transfer learning
    # ------------------------------------------------------------------

    def _register_transfer_tools(self) -> None:
        def _transfer_train(source_model: Any, target_dataset: Any, **kwargs: Any) -> Any:
            from pinneapple_adaptation.transfer_learning import TransferTrainer  # type: ignore
            trainer = TransferTrainer(source_model, **kwargs)
            return trainer.fit(target_dataset)

        self.register(PhysicsTool(
            name="transfer_train",
            category="transfer",
            description="Fine-tune a pre-trained model on a new physics domain",
            input_schema={"source_model": "nn.Module", "target_dataset": "Dataset"},
            output_schema={"model": "nn.Module", "history": "List[dict]"},
            module_path="pinneapple_transfer",
            fn=self._safe_wrap(_transfer_train, "pinneapple_transfer"),
            tags=["transfer", "fine-tuning", "adaptation"],
        ))

        def _parametric_transfer(model: Any, family: Any, **kwargs: Any) -> Any:
            from pinneapple_adaptation.transfer_learning import ParametricFamilyTransfer  # type: ignore
            return ParametricFamilyTransfer(model, family, **kwargs).run()

        self.register(PhysicsTool(
            name="parametric_family_transfer",
            category="transfer",
            description="Transfer across a parametric family of PDEs",
            input_schema={"model": "nn.Module", "family": "PDE family descriptor"},
            output_schema={"adapted_model": "nn.Module"},
            module_path="pinneapple_transfer",
            fn=self._safe_wrap(_parametric_transfer, "pinneapple_transfer"),
            tags=["transfer", "parametric", "pde_family"],
        ))

    # ------------------------------------------------------------------
    # Meta-learning
    # ------------------------------------------------------------------

    def _register_meta_tools(self) -> None:
        def _reptile(catalog: Any, **kwargs: Any) -> Any:
            from .meta_learning import MetaLearner, MetaConfig
            learner = MetaLearner(MetaConfig(algorithm="reptile", **{
                k: v for k, v in kwargs.items()
                if k in MetaConfig.__dataclass_fields__
            }))
            return learner.meta_train(catalog)

        self.register(PhysicsTool(
            name="reptile_meta_train",
            category="meta_learning",
            description="Reptile meta-learning for fast physics adaptation",
            input_schema={"catalog": "DatasetCatalog", "n_meta_epochs": "int"},
            output_schema={"meta_model": "PhysicsWorldModel"},
            module_path="pinneapple_worldmodel.meta_learning",
            fn=_reptile,
            tags=["reptile", "meta", "few-shot"],
        ))

        def _maml(catalog: Any, **kwargs: Any) -> Any:
            from .meta_learning import MetaLearner, MetaConfig
            learner = MetaLearner(MetaConfig(algorithm="maml", **{
                k: v for k, v in kwargs.items()
                if k in MetaConfig.__dataclass_fields__
            }))
            return learner.meta_train(catalog)

        self.register(PhysicsTool(
            name="maml_meta_train",
            category="meta_learning",
            description="MAML meta-learning for few-shot physics adaptation",
            input_schema={"catalog": "DatasetCatalog", "n_meta_epochs": "int"},
            output_schema={"meta_model": "PhysicsWorldModel"},
            module_path="pinneapple_worldmodel.meta_learning",
            fn=_maml,
            tags=["maml", "meta", "few-shot"],
        ))

        def _fast_adapt(meta_model: Any, support_dataset: Any, n_steps: int = 10, **kwargs: Any) -> Any:
            from .meta_learning import MetaLearner, MetaConfig
            learner = MetaLearner(MetaConfig())
            return learner.adapt(meta_model, support_dataset, n_steps=n_steps)

        self.register(PhysicsTool(
            name="fast_adapt",
            category="meta_learning",
            description="Fast-adapt a meta-trained model to a new physics task",
            input_schema={"meta_model": "PhysicsWorldModel",
                          "support_dataset": "WorldModelDataset", "n_steps": "int"},
            output_schema={"adapted_model": "PhysicsWorldModel"},
            module_path="pinneapple_worldmodel.meta_learning",
            fn=_fast_adapt,
            tags=["adapt", "meta", "few-shot"],
        ))

    # ------------------------------------------------------------------
    # Time series
    # ------------------------------------------------------------------

    def _register_timeseries_tools(self) -> None:
        def _ts_forecast(data: Any, horizon: int = 10, **kwargs: Any) -> Any:
            from pinneapple_systems.time_series import LSTMForecaster, TimeSeriesSpec  # type: ignore
            spec = TimeSeriesSpec(**kwargs)
            model = LSTMForecaster(spec)
            return model.forecast(data, horizon=horizon)

        self.register(PhysicsTool(
            name="timeseries_forecast",
            category="timeseries",
            description="Time-series forecasting via pinneapple_timeseries (LSTM/NBeats/TFT)",
            input_schema={"data": "Tensor", "horizon": "int"},
            output_schema={"forecast": "Tensor", "uncertainty": "Tensor"},
            module_path="pinneapple_timeseries",
            fn=self._safe_wrap(_ts_forecast, "pinneapple_timeseries"),
            tags=["timeseries", "forecast", "lstm"],
        ))

        def _power_spectrum(data: Any, **kwargs: Any) -> Any:
            from pinneapple_systems.time_series import power_spectrum  # type: ignore
            return power_spectrum(data, **kwargs)

        self.register(PhysicsTool(
            name="power_spectrum",
            category="timeseries",
            description="Compute power spectral density of a time series",
            input_schema={"data": "Tensor"},
            output_schema={"frequencies": "Tensor", "psd": "Tensor"},
            module_path="pinneapple_timeseries",
            fn=self._safe_wrap(_power_spectrum, "pinneapple_timeseries"),
            tags=["timeseries", "psd", "spectral"],
        ))

    # ------------------------------------------------------------------
    # Co-simulation
    # ------------------------------------------------------------------

    def _register_cosim_tools(self) -> None:
        def _build_cosim(nodes: list, edges: list, **kwargs: Any) -> Any:
            from pinneapple_systems.cosimulation import CoSimGraph, CoSimEngine  # type: ignore
            graph = CoSimGraph(nodes, edges)
            engine = CoSimEngine(graph, **kwargs)
            return engine

        self.register(PhysicsTool(
            name="build_cosim",
            category="co_simulation",
            description="Build a multi-physics co-simulation graph",
            input_schema={"nodes": "List[CoSimNode]", "edges": "List[tuple]"},
            output_schema={"engine": "CoSimEngine"},
            module_path="pinneapple_cosim",
            fn=self._safe_wrap(_build_cosim, "pinneapple_cosim"),
            tags=["cosim", "multi-physics", "graph"],
        ))

        def _run_cosim(engine: Any, t_span: tuple = (0.0, 1.0), **kwargs: Any) -> Any:
            from pinneapple_systems.cosimulation import TrajectoryRecorder  # type: ignore
            rec = TrajectoryRecorder()
            engine.run(t_span=t_span, callbacks=[rec], **kwargs)
            return rec.get_trajectory()

        self.register(PhysicsTool(
            name="run_cosim",
            category="co_simulation",
            description="Execute a co-simulation and record the trajectory",
            input_schema={"engine": "CoSimEngine", "t_span": "tuple"},
            output_schema={"trajectory": "dict"},
            module_path="pinneapple_cosim",
            fn=self._safe_wrap(_run_cosim, "pinneapple_cosim"),
            tags=["cosim", "simulation", "multi-physics"],
        ))

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _register_geometry_tools(self) -> None:
        def _make_domain(domain_type: str = "unit_square", **kwargs: Any) -> Any:
            from .geometry import BUILTIN_DOMAINS
            if domain_type in BUILTIN_DOMAINS:
                return BUILTIN_DOMAINS[domain_type]
            from .geometry import make_unit_square
            return make_unit_square()

        self.register(PhysicsTool(
            name="make_physics_domain",
            category="geometry",
            description="Create a named physics domain with boundary regions",
            input_schema={"domain_type": "str"},
            output_schema={"domain": "PhysicsDomain"},
            module_path="pinneapple_worldmodel.geometry",
            fn=_make_domain,
            tags=["geometry", "domain", "boundary"],
        ))

        def _make_sdf(shape: str = "circle", **kwargs: Any) -> Any:
            try:
                from pinneapple_design.geometry import sdf2d_circle, SDF  # type: ignore
                if shape == "circle":
                    return sdf2d_circle(**kwargs)
                return SDF(**kwargs)
            except Exception as exc:
                raise RuntimeError(f"SDF creation failed: {exc}") from exc

        self.register(PhysicsTool(
            name="make_sdf",
            category="geometry",
            description="Create a Signed Distance Function for a shape",
            input_schema={"shape": "str"},
            output_schema={"sdf": "SDF"},
            module_path="pinneapple_geom",
            fn=self._safe_wrap(_make_sdf, "pinneapple_geom"),
            tags=["geometry", "sdf", "implicit"],
        ))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _register_inference_tools(self) -> None:
        def _infer_2d(model: Any, x: Any, y: Any, **kwargs: Any) -> Any:
            from pinneapple_neural.predictor import infer_on_grid_2d  # type: ignore
            return infer_on_grid_2d(model, x, y, **kwargs)

        self.register(PhysicsTool(
            name="infer_on_grid_2d",
            category="inference",
            description="Run model inference on a 2D grid",
            input_schema={"model": "nn.Module", "x": "Tensor", "y": "Tensor"},
            output_schema={"field": "Tensor(H, W)"},
            module_path="pinneapple_inference",
            fn=self._safe_wrap(_infer_2d, "pinneapple_inference"),
            tags=["inference", "grid", "2d"],
        ))

        def _plot_field(field: Any, **kwargs: Any) -> Any:
            from pinneapple_neural.predictor import plot_field_2d  # type: ignore
            return plot_field_2d(field, **kwargs)

        self.register(PhysicsTool(
            name="plot_physics_field",
            category="inference",
            description="Visualise a 2D physics field",
            input_schema={"field": "Tensor(H, W)"},
            output_schema={"figure": "matplotlib.Figure"},
            module_path="pinneapple_inference",
            fn=self._safe_wrap(_plot_field, "pinneapple_inference"),
            tags=["viz", "field", "2d"],
        ))

    # ------------------------------------------------------------------
    # Design optimisation
    # ------------------------------------------------------------------

    def _register_design_opt_tools(self) -> None:
        def _bayesian_opt(objective: Any, bounds: Any, **kwargs: Any) -> Any:
            from pinneapple_design.design_optimizer import BayesianDesignOptimizer  # type: ignore
            opt = BayesianDesignOptimizer(objective, bounds=bounds, **kwargs)
            return opt.run()

        self.register(PhysicsTool(
            name="bayesian_design_opt",
            category="design_opt",
            description="Bayesian optimisation of a physics design objective",
            input_schema={"objective": "callable", "bounds": "Tensor(D, 2)"},
            output_schema={"best_design": "Tensor", "best_value": "float"},
            module_path="pinneapple_design_opt",
            fn=self._safe_wrap(_bayesian_opt, "pinneapple_design_opt"),
            tags=["design", "bayesian", "optimisation"],
        ))

    # ------------------------------------------------------------------
    # Digital twin
    # ------------------------------------------------------------------

    def _register_digital_twin_tools(self) -> None:
        def _build_twin(model: Any, stream: Any, **kwargs: Any) -> Any:
            from pinneapple_systems.digital_twin import DigitalTwin  # type: ignore
            return DigitalTwin(model, data_source=stream, **kwargs)

        self.register(PhysicsTool(
            name="build_digital_twin",
            category="digital_twin",
            description="Build a real-time digital twin from a trained model + data stream",
            input_schema={"model": "nn.Module", "stream": "DataStream"},
            output_schema={"twin": "DigitalTwin"},
            module_path="pinneapple_digital_twin",
            fn=self._safe_wrap(_build_twin, "pinneapple_digital_twin"),
            tags=["digital_twin", "real-time", "assimilation"],
        ))

        def _ekf(model: Any, obs: Any, **kwargs: Any) -> Any:
            from pinneapple_systems.digital_twin import ExtendedKalmanFilter  # type: ignore
            ekf = ExtendedKalmanFilter(model, **kwargs)
            return ekf.update(obs)

        self.register(PhysicsTool(
            name="kalman_data_assimilation",
            category="digital_twin",
            description="Extended Kalman Filter data assimilation",
            input_schema={"model": "nn.Module", "obs": "Tensor"},
            output_schema={"state_estimate": "Tensor", "covariance": "Tensor"},
            module_path="pinneapple_digital_twin",
            fn=self._safe_wrap(_ekf, "pinneapple_digital_twin"),
            tags=["kalman", "assimilation", "digital_twin"],
        ))

    # ------------------------------------------------------------------
    # World model tools
    # ------------------------------------------------------------------

    def _register_worldmodel_tools(self) -> None:
        def _build_foundation(config: Any = None, n_fields: int = 1,
                              grid_shape: tuple = (64, 64), **kwargs: Any) -> Any:
            from .mega_model import PhysicsFoundationModel, FoundationConfig
            cfg = config or FoundationConfig(**{
                k: v for k, v in kwargs.items()
                if k in FoundationConfig.__dataclass_fields__
            })
            return PhysicsFoundationModel(cfg, n_fields=n_fields, grid_shape=grid_shape)

        self.register(PhysicsTool(
            name="build_foundation_model",
            category="world_model",
            description="Build a PhysicsFoundationModel (mega generalist model)",
            input_schema={"config": "FoundationConfig", "n_fields": "int",
                          "grid_shape": "tuple"},
            output_schema={"model": "PhysicsFoundationModel"},
            module_path="pinneapple_worldmodel.mega_model",
            fn=_build_foundation,
            tags=["foundation", "fno", "mega_model"],
        ))

        def _rollout(model: Any, state_0: Any, descriptor: Any = None,
                     n_steps: int = 20, **kwargs: Any) -> Any:
            return model.rollout(state_0, descriptor, n_steps=n_steps)

        self.register(PhysicsTool(
            name="rollout_trajectory",
            category="world_model",
            description="Auto-regressive rollout of a world model",
            input_schema={"model": "PhysicsWorldModel", "state_0": "Tensor",
                          "descriptor": "Any", "n_steps": "int"},
            output_schema={"trajectory": "Tensor(B, T, C, *grid)"},
            module_path="pinneapple_worldmodel.model",
            fn=_rollout,
            tags=["rollout", "world_model", "autoregressive"],
        ))

        def _benchmark(model: Any, tasks: list = None, **kwargs: Any) -> Any:
            from .benchmark import PhysicsBenchmark
            bench = PhysicsBenchmark(**{k: v for k, v in kwargs.items()
                                        if k in ("device", "verbose")})
            return bench.run(model, tasks=tasks)

        self.register(PhysicsTool(
            name="benchmark_model",
            category="world_model",
            description="Run the physics AI benchmark suite on a model",
            input_schema={"model": "nn.Module", "tasks": "List[str]"},
            output_schema={"results": "Dict[str, BenchmarkResult]"},
            module_path="pinneapple_worldmodel.benchmark",
            fn=_benchmark,
            tags=["benchmark", "evaluation", "physics"],
        ))

    # ------------------------------------------------------------------
    # Utility: safe-wrap
    # ------------------------------------------------------------------

    def _safe_wrap(self, fn: Callable, module_name: str) -> Optional[Callable]:
        """Return fn if the module is importable, else None."""
        try:
            __import__(module_name.split(".")[0])
            return fn
        except ImportError:
            log.debug("Module '%s' not installed; tool will be unavailable.", module_name)
            return None
