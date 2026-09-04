"""pinneapple_physics — Physics problem definition and PINN solving.

Sub-modules
-----------
pde_environment  (was pinneapple_environment)
    PDE problem specification: ProblemSpec, boundary/initial conditions,
    presets (NS, heat, wave, Burgers, elasticity …), RANS turbulence models,
    and PDE-family knowledge base.

pinn_solver      (was pinneapple_pinn)
    PINN compiler: translates a ProblemSpec into callable loss functions.
    Includes DoMINO domain-decomposition PINN.

symbolic_pde     (was pinneapple_symbolic)
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
>>> from pinneapple_physics import ProblemSpec, DirichletBC, compile_physics, identify
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
    # RANS/LES turbulence
    KOmegaSSTResiduals,
    SpalartAllmarasResiduals,
    WALEResiduals,
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
    AdaptiveWeights,
    compile_problem,
    Subdomain,
    SubdomainPINN,
    DoMINO,
    LatentConditionedModel,
    sample_latent,
    ensemble_forward,
    mean_covariance_loss,
)

# ── symbolic_pde re-exports ───────────────────────────────────────────────────
from .symbolic_pde import (
    SymbolicPDE,
    pde_from_sympy,
    auto_residual,
    HardBC,
    PeriodicBC,
    MultiPeriodicBC,
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


def solve_pde(
    spec: "ProblemSpec",
    model,
    *,
    epochs: int = 5000,
    device: str = "cpu",
    lr: float = 1e-3,
    n_collocation: int = 2048,
    seed: int = 0,
    weights=None,
    x_bc=None, y_bc=None, n_bc=None,
    x_ic=None, y_ic=None,
    x_data=None, y_data=None,
    ctx=None,
    **cond_masks,
):
    """One-shot: compile physics losses and train *model* on a ProblemSpec.

    Was previously broken in two ways at once: ``TrainConfig(n_epochs=...)``
    (the field is ``epochs``, not ``n_epochs``), and
    ``Trainer(model, losses, cfg)`` followed by ``trainer.train()`` (the
    constructor's second positional argument is ``loss_fn``, not a config
    -- the config belongs in ``.fit(train_loader, val_loader, cfg)``, and
    there is no ``.train()`` method at all). Both would have raised
    immediately on any actual call.

    Routing through ``Trainer.fit`` at all turned out not to make sense
    here either: it needs a ``DataLoader``, but a ``ProblemSpec`` is a
    *problem definition*, not a dataset -- there is nothing to load, only
    points to sample fresh from ``spec.domain_bounds`` -- and its
    validation pass runs inside ``torch.no_grad()``, which breaks any
    ``create_graph=True`` second-derivative residual ``compile_problem``'s
    own output commonly needs (any PDE with second-order terms, e.g.
    ``laplace``/``poisson``/``navier_stokes_incompressible``). So this is a
    small, self-contained Adam loop instead, calling ``compile_problem``'s
    ``loss_fn`` directly every step -- no ``Trainer`` involved.

    Collocation points are sampled fresh every step from
    ``spec.domain_bounds``. Boundary/initial/data condition points ARE
    sampled automatically too, for every condition whose
    ``selector_type`` is ``"all"`` or ``"callable"`` (both are
    self-describing -- ``cond.mask(X, ctx)`` alone decides which sampled
    points belong to it, e.g. "the left wall" via
    ``selector=lambda X, ctx: X[:, 0] <= bounds["x"][0] + eps``): each such
    condition draws its own domain-wide sample every step and keeps
    whatever ``cond.mask()`` selects. A ``selector_type="tag"`` condition
    needs an externally supplied ``ctx["tag_masks"]`` this function cannot
    derive from the spec alone (there is no PDE-agnostic way to know how
    to build the tags), so those conditions are skipped automatically --
    pass ``x_bc``/``y_bc``/``ctx`` (with ``ctx["tag_masks"]`` populated)
    and matching ``mask_<condition_name>=...`` boolean tensors via
    ``**cond_masks`` yourself for a spec that uses tag selectors, the same
    way ``examples/pde_environment/03_ns2d_channel_tags.py`` builds its
    batch by hand. Any of ``x_bc``/``x_ic``/``x_data`` passed in explicitly
    disables auto-sampling for that condition kind entirely (all-or-
    nothing per kind, to avoid silently mixing an auto-sampled batch with
    a hand-built one that used a different point layout).

    Returns ``{"model": model, "history": {"loss": [...]}}``.
    """
    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    device_t = torch.device(device)
    model = model.to(device_t)
    coords = list(spec.coords)
    bounds = spec.domain_bounds
    n_fields = len(spec.fields)
    ctx = dict(ctx or {})
    kind_to_suffix = {"dirichlet": "bc", "neumann": "bc", "robin": "bc", "initial": "ic"}

    loss_fn = compile_problem(spec, weights=weights)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def _sample_domain_np(n: int) -> "np.ndarray":
        return np.stack([np.random.uniform(*bounds[c], size=n) for c in coords], axis=1).astype(np.float32)

    def _sample_domain(n: int) -> "torch.Tensor":
        return torch.as_tensor(_sample_domain_np(n), device=device_t)

    def _empty_xy():
        return torch.zeros((0, len(coords)), device=device_t), torch.zeros((0, n_fields), device=device_t)

    explicit = {"bc": x_bc is not None, "ic": x_ic is not None, "data": x_data is not None}
    explicit_xy = {
        "bc": (x_bc.to(device_t), y_bc.to(device_t)) if x_bc is not None else None,
        "ic": (x_ic.to(device_t), y_ic.to(device_t)) if x_ic is not None else None,
        "data": (x_data.to(device_t), y_data.to(device_t)) if x_data is not None else None,
    }
    n_bc_t = n_bc.to(device_t) if n_bc is not None else None
    explicit_masks = {k: v.to(device_t) for k, v in cond_masks.items() if k.startswith("mask_")}

    auto_conditions = [
        c for c in spec.conditions
        if c.selector_type in ("all", "callable") and not explicit[kind_to_suffix.get(c.kind, "data")]
    ]

    history = {"loss": []}
    for _epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        x_col = _sample_domain(n_collocation).requires_grad_(True)

        buckets = {"bc": ([], []), "ic": ([], []), "data": ([], [])}
        # per-suffix list of (condition_name, n_points_this_condition), in
        # the same order points are appended to `buckets`, so the absolute
        # offsets to build each condition's boolean mask can be recovered
        # after concatenation without a second pass over spec.conditions.
        spans = {"bc": [], "ic": [], "data": []}
        for cond in auto_conditions:
            suffix = kind_to_suffix.get(cond.kind, "data")
            x_np = _sample_domain_np(n_collocation)
            m = np.asarray(cond.mask(x_np, ctx), dtype=bool)
            x_sel = x_np[m]
            if x_sel.shape[0] == 0:
                continue
            y_np = (
                np.asarray(cond.value_fn(x_sel, ctx), dtype=np.float32)
                if cond.value_fn is not None
                else np.zeros((x_sel.shape[0], n_fields), dtype=np.float32)
            )
            if y_np.ndim == 1:
                y_np = y_np[:, None]
            spans[suffix].append((cond.name, x_sel.shape[0]))
            buckets[suffix][0].append(x_sel)
            buckets[suffix][1].append(y_np)

        auto_xy, mask_tensors = {}, {}
        for suffix in ("bc", "ic", "data"):
            xs, ys = buckets[suffix]
            if xs:
                x_cat = torch.as_tensor(np.concatenate(xs, axis=0), device=device_t)
                y_cat = torch.as_tensor(np.concatenate(ys, axis=0), device=device_t)
            else:
                x_cat, y_cat = _empty_xy()
            auto_xy[suffix] = (x_cat, y_cat)

            n_total = x_cat.shape[0]
            offset = 0
            for name, n_pts in spans[suffix]:
                m = torch.zeros(n_total, dtype=torch.bool, device=device_t)
                m[offset : offset + n_pts] = True
                mask_tensors[f"mask_{name}"] = m
                offset += n_pts
        mask_tensors.update(explicit_masks)

        x_bc_t, y_bc_t = explicit_xy["bc"] if explicit["bc"] else auto_xy["bc"]
        x_ic_t, y_ic_t = explicit_xy["ic"] if explicit["ic"] else auto_xy["ic"]
        x_data_t, y_data_t = explicit_xy["data"] if explicit["data"] else auto_xy["data"]

        batch = {
            "x_col": x_col, "ctx": ctx,
            "x_bc": x_bc_t, "y_bc": y_bc_t, "n_bc": n_bc_t,
            "x_ic": x_ic_t, "y_ic": y_ic_t,
            "x_data": x_data_t, "y_data": y_data_t,
            **mask_tensors,
        }
        y_hat = model(x_col)
        out = loss_fn(model, y_hat, batch)
        loss = out["total"] if isinstance(out, dict) else out
        loss.backward()
        opt.step()
        history["loss"].append(float(loss.item()))

    return {"model": model, "history": history}


class PhysicsPipeline:
    """Result of :func:`pipeline` -- a trained model bound to the
    ``ProblemSpec`` it was trained on, with a ``.predict()`` convenience
    method. ``.model``/``.spec``/``.history`` are the same objects
    ``get_preset``/``solve_pde`` already produce -- this is a thin
    wrapper, not a new abstraction to learn."""

    def __init__(self, model, spec: "ProblemSpec", history: dict):
        self.model = model
        self.spec = spec
        self.history = history

    def predict(self, x):
        """``x``: array-like (N, len(spec.coords)). Returns a numpy array
        (N, len(spec.fields))."""
        import numpy as np
        import torch

        self.model.eval()
        with torch.no_grad():
            x_t = torch.as_tensor(np.asarray(x, dtype="float32"))
            y = self.model(x_t)
            if hasattr(y, "y"):
                y = y.y
        return y.numpy()


def pipeline(
    task: str,
    *,
    architecture: str = "modified_mlp",
    hidden_dim: int = 64,
    n_layers: int = 4,
    epochs: int = 2000,
    device: str = "cpu",
    lr: float = 1e-3,
    n_collocation: int = 2048,
    seed: int = 0,
    weights=None,
    **preset_kwargs,
) -> PhysicsPipeline:
    """One call, named-task physics AI pipeline: preset -> model -> trained
    solution -> ``.predict()``-ready object.

    ``task`` is any name registered via ``get_preset``/``@register_preset``
    (``list_presets()`` for the full list -- fluid, thermal, structural,
    aerospace, automotive, ... presets already ship with the package;
    register your own with ``@register_preset("my_task")`` and it becomes
    a valid ``task`` here immediately, no separate wiring needed).
    ``**preset_kwargs`` are forwarded to ``get_preset`` (e.g. ``Re=500``
    for a fluid preset, ``nu=0.01`` for ``burgers_1d``). Everything else
    is forwarded to :func:`solve_pde`.

    Examples
    --------
    >>> import pinneapple_physics as pp
    >>> pipe = pp.pipeline("burgers_1d", nu=0.01, epochs=3000)
    >>> pipe.predict([[0.0, 0.5]])   # u(x=0, t=0.5)
    array([[...]], dtype=float32)

    This is exactly ``get_preset(task, **preset_kwargs)`` +
    ``ModelRegistry.build(architecture, ...)`` + ``solve_pde(...)`` in one
    call -- nothing here that wasn't already directly usable, just the
    common path collapsed to one line the way a Hugging-Face-style
    ``pipeline()`` call does for a named task.
    """
    import pinneapple_neural.architectures  # noqa: F401  registers the model zoo
    from pinneapple_neural.architectures.registry import ModelRegistry

    spec = get_preset(task, **preset_kwargs)
    model = ModelRegistry.build(
        architecture,
        in_dim=len(spec.coords), out_dim=len(spec.fields),
        hidden_dim=hidden_dim, n_layers=n_layers,
    )
    result = solve_pde(
        spec, model, epochs=epochs, device=device, lr=lr,
        n_collocation=n_collocation, seed=seed, weights=weights,
    )
    return PhysicsPipeline(result["model"], spec, result["history"])


__all__ = [
    # Sub-modules (new names)
    "pde_environment", "pinn_solver", "symbolic_pde",
    # Sub-modules (old aliases — backward compat)
    "environment", "pinn", "symbolic",
    # Integration helpers
    "compile_physics", "identify", "define_problem", "solve_pde",
    "pipeline", "PhysicsPipeline",
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
    "KOmegaSSTResiduals", "SpalartAllmarasResiduals", "WALEResiduals", "get_rans_preset", "SST_CONSTS",
    "PDEFamily", "list_pde_families", "get_pde_family", "identify_pde", "suggest_problem_spec",
    # pinn_solver
    "LossWeights", "AdaptiveWeights", "compile_problem", "Subdomain", "SubdomainPINN", "DoMINO",
    "LatentConditionedModel", "sample_latent", "ensemble_forward", "mean_covariance_loss",
    # symbolic_pde
    "SymbolicPDE", "pde_from_sympy", "auto_residual",
    "HardBC", "PeriodicBC", "MultiPeriodicBC", "SymbolicDirichletBC", "SymbolicNeumannBC",
]
