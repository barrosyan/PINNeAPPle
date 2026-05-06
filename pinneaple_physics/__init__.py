"""pinneaple_physics — Physics problem definition and PINN solving.

Sub-modules
-----------
pde_environment  (was pinneaple_environment)
    PDE problem specification: ProblemSpec, boundary/initial conditions,
    presets (NS, heat, wave, Burgers, elasticity …), RANS turbulence models,
    and PDE-family knowledge base.

pinn_solver      (was pinneaple_pinn)
    PINN compiler: translates a ProblemSpec into callable loss functions.
    Includes DoMINO domain-decomposition PINN.

symbolic_pde     (was pinneaple_symbolic)
    SymPy-to-autograd compiler: define PDE residuals as SymPy expressions,
    get a PyTorch-differentiable residual function. HardBC / SoftBC support.

Integration helpers
------------------
``define_problem(pde_type, ...)``   — quick ProblemSpec builder
``compile_physics(spec)``           — wraps pinn_solver.compile_problem
``solve_pde(spec, model, ...)``     — one-shot: compile → train
``identify(description)``           — wraps pde_environment.identify_pde

Usage
-----
>>> from pinneaple_physics import ProblemSpec, DirichletBC, compile_physics, identify
>>> spec = ProblemSpec(...)
>>> losses = compile_physics(spec)
>>> info = identify("Navier-Stokes incompressible 2D")
"""
from __future__ import annotations

# ── sub-modules (new descriptive names) ───────────────────────────────────────
from . import pde_environment
from . import pinn_solver
from . import symbolic_pde

# backward-compat aliases (old names still work)
environment = pde_environment
pinn        = pinn_solver
symbolic    = symbolic_pde

# ── pde_environment re-exports ────────────────────────────────────────────────
from .pde_environment import (
    ConditionSpec,
    DirichletBC,
    NeumannBC,
    RobinBC,
    InitialCondition,
    DataConstraint,
    PDETermSpec,
    ProblemSpec,
    ScaleSpec,
    ProblemBuilder,
    # Presets — academic
    burgers_1d_default,
    laplace_2d_default,
    poisson_2d_default,
    # Presets — CFD
    ns_incompressible_2d_default,
    ns_incompressible_3d_default,
    lid_driven_cavity_3d,
    channel_flow_3d,
    pipe_flow_3d,
    # Presets — industry
    steady_heat_conduction_3d_default,
    transient_heat_3d_default,
    linear_elasticity_3d_default,
    darcy_pressure_only_3d_default,
    helmholtz_acoustics_3d_default,
    wave_ultrasound_3d_default,
    reaction_diffusion_2d_default,
    # Preset registry
    get_preset,
    list_presets,
    register_preset,
    # RANS turbulence
    KOmegaSSTResiduals,
    SpalartAllmarasResiduals,
    get_rans_preset,
    SST_CONSTS,
    # PDE knowledge base
    PDEFamily,
    list_pde_families,
    get_pde_family,
    identify_pde,
    suggest_problem_spec,
)

try:
    from .pde_environment import (
        plane_stress_2d_default,
        plane_strain_2d_default,
        von_mises_2d_default,
        linear_elasticity_3d,
        drill_pipe_torsion_default,
        thermoelasticity_2d_default,
    )
except ImportError:
    pass

# ── pinn_solver re-exports ────────────────────────────────────────────────────
from .pinn_solver import (
    LossWeights,
    compile_problem,
    Subdomain,
    SubdomainPINN,
    DoMINO,
)

# ── symbolic_pde re-exports ───────────────────────────────────────────────────
from .symbolic_pde import (
    SymbolicPDE,
    pde_from_sympy,
    auto_residual,
    HardBC,
    PeriodicBC,
    DirichletBC as SymbolicDirichletBC,
    NeumannBC as SymbolicNeumannBC,
)


# ── Integration helpers ────────────────────────────────────────────────────────

def compile_physics(spec: "ProblemSpec", **kwargs):
    """Compile a ProblemSpec into weighted PINN loss functions."""
    return compile_problem(spec, **kwargs)


def identify(description: str):
    """Identify PDE family from a natural-language description."""
    return identify_pde(description)


def define_problem(preset: str | None = None, **spec_kwargs) -> "ProblemSpec":
    """Quick ProblemSpec builder.

    Parameters
    ----------
    preset : str, optional
        Named preset string (e.g. ``"ns_incompressible_2d"``).
    """
    if preset is not None:
        base = get_preset(preset)
        for k, v in spec_kwargs.items():
            setattr(base, k, v)
        return base
    return ProblemSpec(**spec_kwargs)


def solve_pde(spec: "ProblemSpec", model, *, epochs: int = 5000, device: str = "cpu", **train_kwargs):
    """One-shot: compile physics losses and train model on a ProblemSpec."""
    from pinneaple_neural.trainer import Trainer, TrainConfig
    losses = compile_problem(spec)
    cfg = TrainConfig(n_epochs=epochs, device=device, **train_kwargs)
    trainer = Trainer(model, losses, cfg)
    return trainer.train()


__all__ = [
    # Sub-modules (new names)
    "pde_environment", "pinn_solver", "symbolic_pde",
    # Sub-modules (old aliases — backward compat)
    "environment", "pinn", "symbolic",
    # Integration helpers
    "compile_physics", "identify", "define_problem", "solve_pde",
    # pde_environment
    "ConditionSpec", "DirichletBC", "NeumannBC", "RobinBC",
    "InitialCondition", "DataConstraint",
    "PDETermSpec", "ProblemSpec", "ScaleSpec", "ProblemBuilder",
    "burgers_1d_default", "laplace_2d_default", "poisson_2d_default",
    "ns_incompressible_2d_default", "ns_incompressible_3d_default",
    "lid_driven_cavity_3d", "channel_flow_3d", "pipe_flow_3d",
    "steady_heat_conduction_3d_default", "transient_heat_3d_default",
    "linear_elasticity_3d_default", "darcy_pressure_only_3d_default",
    "helmholtz_acoustics_3d_default", "wave_ultrasound_3d_default",
    "reaction_diffusion_2d_default",
    "get_preset", "list_presets", "register_preset",
    "KOmegaSSTResiduals", "SpalartAllmarasResiduals", "get_rans_preset", "SST_CONSTS",
    "PDEFamily", "list_pde_families", "get_pde_family", "identify_pde", "suggest_problem_spec",
    # pinn_solver
    "LossWeights", "compile_problem", "Subdomain", "SubdomainPINN", "DoMINO",
    # symbolic_pde
    "SymbolicPDE", "pde_from_sympy", "auto_residual",
    "HardBC", "PeriodicBC", "SymbolicDirichletBC", "SymbolicNeumannBC",
]
