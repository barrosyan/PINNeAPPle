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
from .builder import ProblemBuilder

from .presets.academics import burgers_1d_default, laplace_2d_default, poisson_2d_default
from .presets.cfd import (
    ns_incompressible_2d_default,
    ns_incompressible_3d_default,
    lid_driven_cavity_3d,
    channel_flow_3d,
    pipe_flow_3d,
)
from .presets.registry import get_preset, list_presets, register_preset
from .presets.industry import (
    steady_heat_conduction_3d_default,
    transient_heat_3d_default,
    linear_elasticity_3d_default,
    darcy_pressure_only_3d_default,
    helmholtz_acoustics_3d_default,
    wave_ultrasound_3d_default,
    reaction_diffusion_2d_default,
)
try:
    from .presets.structural import (
        plane_stress_2d_default,
        plane_strain_2d_default,
        von_mises_2d_default,
        linear_elasticity_3d,
        drill_pipe_torsion_default,
        thermoelasticity_2d_default,
    )
except ImportError:
    pass

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
    "lid_driven_cavity_3d",
    "channel_flow_3d",
    "pipe_flow_3d",
    "steady_heat_conduction_3d_default",
    "transient_heat_3d_default",
    "linear_elasticity_3d_default",
    "darcy_pressure_only_3d_default",
    "helmholtz_acoustics_3d_default",
    "wave_ultrasound_3d_default",
    "reaction_diffusion_2d_default",
    "ProblemBuilder",
    "get_preset",
    "list_presets",
    "register_preset",
    "plane_stress_2d_default",
    "plane_strain_2d_default",
    "von_mises_2d_default",
    "linear_elasticity_3d",
    "drill_pipe_torsion_default",
    "thermoelasticity_2d_default",
]