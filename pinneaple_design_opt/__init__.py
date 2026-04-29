"""Physics-guided design optimization pipeline.

Provides a closed-loop optimizer that:
1. Parametrizes design space (via ParamSpace)
2. Evaluates physics with a surrogate model (FNO / MLP / any nn.Module)
3. Optimizes design via gradient / Bayesian / evolutionary methods
4. Optionally refines with PINNs
5. Handles multi-objective optimization with Pareto fronts

Quick start::

    from pinneaple_design_opt import DesignOptLoop, DesignOptConfig, PhysicsSurrogate
    from pinneaple_design_opt import DragObjective, BoxConstraint, ConstraintSet
    from pinneaple_geom.optimize.loop import ParamSpace

    space = ParamSpace(
        bounds={"chord": (0.5, 2.0), "camber": (0.0, 0.3), "thickness": (0.05, 0.3)},
        x0={"chord": 1.0, "camber": 0.1, "thickness": 0.1},
    )

    surrogate = PhysicsSurrogate.build_mlp(in_dim=3, out_dim=32)
    objective = DragObjective()
    constraints = ConstraintSet()
    constraints.add(BoxConstraint(
        theta_min=torch.tensor([0.5, 0.0, 0.05]),
        theta_max=torch.tensor([2.0, 0.3, 0.3]),
    ))

    loop = DesignOptLoop(space, surrogate, objective, constraints=constraints)
    result = loop.run()
    print(result.summary())
    result.plot_convergence()
"""
from .adjoint import (
    ShapeParametrization,
    ContinuousAdjointSolver,
    DragAdjointObjective,
    naca_parametric,
)
from .pareto import ParetoFront, pareto_dominates, compute_pareto_front
from .objective import (
    ObjectiveBase,
    DragObjective,
    ThermalEfficiencyObjective,
    StructuralObjective,
    WeightMinimizationObjective,
    CompositeObjective,
)
from .constraints import (
    ConstraintBase,
    BoxConstraint,
    MassConservationConstraint,
    GeometricConstraint,
    ManufacturabilityConstraint,
    ConstraintSet,
)
from .surrogate import SurrogateConfig, PhysicsSurrogate
from .optimizer import (
    DesignOptimizerConfig,
    GradientDesignOptimizer,
    BayesianDesignOptimizer,
    EvolutionaryDesignOptimizer,
)
from .refinement import RefinementConfig, PINNRefinement, RefinementResult
from .pipeline import DesignOptConfig, DesignOptResult, DesignOptLoop

__all__ = [
    # Adjoint shape optimisation (Feature 16)
    "ShapeParametrization",
    "ContinuousAdjointSolver",
    "DragAdjointObjective",
    "naca_parametric",
    # Pareto / multi-objective
    "ParetoFront", "pareto_dominates", "compute_pareto_front",
    "ObjectiveBase", "DragObjective", "ThermalEfficiencyObjective",
    "StructuralObjective", "WeightMinimizationObjective", "CompositeObjective",
    "ConstraintBase", "BoxConstraint", "MassConservationConstraint",
    "GeometricConstraint", "ManufacturabilityConstraint", "ConstraintSet",
    "SurrogateConfig", "PhysicsSurrogate",
    "DesignOptimizerConfig", "GradientDesignOptimizer",
    "BayesianDesignOptimizer", "EvolutionaryDesignOptimizer",
    "RefinementConfig", "PINNRefinement", "RefinementResult",
    "DesignOptConfig", "DesignOptResult", "DesignOptLoop",
]
