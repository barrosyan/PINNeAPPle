from .conditions import (
    ConditionSpec,
    DirichletBC,
    NeumannBC,
    RobinBC,
    InitialCondition,
    DataConstraint,
)
from .spec import PDETermSpec, ProblemSpec
from .scales import ScaleSpec

from .presets.academics import burgers_1d_default, laplace_2d_default, poisson_2d_default
from .presets.cfd import ns_incompressible_2d_default, ns_incompressible_3d_default
from .presets.industry import (
    steady_heat_conduction_3d_default,
    transient_heat_3d_default,
    linear_elasticity_3d_default,
    darcy_pressure_only_3d_default,
    helmholtz_acoustics_3d_default,
    wave_ultrasound_3d_default,
    reaction_diffusion_2d_default,
)

__all__ = [
    "PDETermSpec",
    "ProblemSpec",
    "ScaleSpec",
    "ConditionSpec",
    "DirichletBC",
    "NeumannBC",
    "RobinBC",
    "InitialCondition",
    "DataConstraint",
    "burgers_1d_default",
    "laplace_2d_default",
    "poisson_2d_default",
    "ns_incompressible_2d_default",
    "ns_incompressible_3d_default",
    "steady_heat_conduction_3d_default",
    "transient_heat_3d_default",
    "linear_elasticity_3d_default",
    "darcy_pressure_only_3d_default",
    "helmholtz_acoustics_3d_default",
    "wave_ultrasound_3d_default",
    "reaction_diffusion_2d_default",
]