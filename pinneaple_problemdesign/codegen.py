"""Code generation: translate elicited ProblemSpec → concrete Pinneaple API objects.

Entry point
-----------
``build_pinneaple_spec(design_spec)`` reads the elicited
:class:`~pinneaple_problemdesign.schema.ProblemSpec` (business-level) and returns a
:class:`~pinneaple_problemdesign.schema.PinneapleSpec` containing:

* The identified PDE kind (via ``pinneaple_environment.capabilities.identify_pde``)
* Constructor kwargs for ``pinneaple_environment.ProblemSpec``
* Model name + kwargs for ``pp.build_model()``
* ``TrainConfig`` kwargs tuned to the task type
* ``CollocationSampler`` kwargs
* A ready-to-run Python code snippet
"""
from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Optional, Tuple

from .schema import ProblemSpec as DesignSpec, PinneapleSpec


# ---------------------------------------------------------------------------
# Task-type dispatch tables
# ---------------------------------------------------------------------------

# (model_name, model_kwargs)
_TASK_MODEL: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "pde_solution":      ("VanillaPINN", {"hidden": [64, 64, 64, 64]}),
    "inverse_problem":   ("VanillaPINN", {"hidden": [64, 64, 64]}),
    "forecasting":       ("FNO",         {"modes": 16, "width": 64, "depth": 4}),
    "neural_operator":   ("FNO",         {"modes": 16, "width": 64, "depth": 4}),
    "anomaly_detection": ("VanillaPINN", {"hidden": [32, 32]}),
    "control":           ("VanillaPINN", {"hidden": [64, 64]}),
    "optimization":      ("VanillaPINN", {"hidden": [64, 64, 64]}),
    "other":             ("VanillaPINN", {"hidden": [64, 64, 64]}),
}

_TASK_TRAIN: Dict[str, Dict[str, Any]] = {
    "pde_solution":      {"epochs": 5000, "lr": 1e-3,  "amp": False},
    "inverse_problem":   {"epochs": 8000, "lr": 5e-4,  "amp": False},
    "forecasting":       {"epochs": 200,  "lr": 1e-3,  "amp": True},
    "neural_operator":   {"epochs": 300,  "lr": 1e-3,  "amp": True},
    "anomaly_detection": {"epochs": 100,  "lr": 1e-3,  "amp": True},
    "control":           {"epochs": 3000, "lr": 1e-3,  "amp": False},
    "optimization":      {"epochs": 2000, "lr": 1e-3,  "amp": False},
    "other":             {"epochs": 1000, "lr": 1e-3,  "amp": False},
}

_TASK_COLLOCATION: Dict[str, Dict[str, Any]] = {
    "pde_solution":      {"n_col": 8000, "n_bc": 500,  "n_ic": 500,  "strategy": "lhs"},
    "inverse_problem":   {"n_col": 6000, "n_bc": 1000, "n_ic": 500,  "strategy": "lhs"},
    "forecasting":       {"n_col": 0,    "n_bc": 0,    "n_ic": 0,    "strategy": "uniform"},
    "neural_operator":   {"n_col": 0,    "n_bc": 0,    "n_ic": 0,    "strategy": "uniform"},
    "anomaly_detection": {"n_col": 0,    "n_bc": 0,    "n_ic": 0,    "strategy": "uniform"},
    "control":           {"n_col": 4000, "n_bc": 500,  "n_ic": 500,  "strategy": "lhs"},
    "optimization":      {"n_col": 4000, "n_bc": 500,  "n_ic": 0,    "strategy": "lhs"},
    "other":             {"n_col": 4000, "n_bc": 500,  "n_ic": 0,    "strategy": "uniform"},
}

# Tasks that use pinneaple_pinn (compile_problem path) vs supervised (FNO path)
_PINN_TASKS = {"pde_solution", "inverse_problem", "control", "optimization"}


# ---------------------------------------------------------------------------
# PDE identification helper
# ---------------------------------------------------------------------------

def _identify_pde(design_spec: DesignSpec) -> Dict[str, Any]:
    """Run suggest_problem_spec against physics + domain context.

    Falls back gracefully if pinneaple_environment is not installed.
    """
    try:
        from pinneaple_environment.capabilities import suggest_problem_spec
    except ImportError:
        return {
            "kind": "custom", "fields": ["u"], "coords": ["x", "t"],
            "default_params": {}, "tags": [], "confidence": 0.0,
        }

    # Build a description from everything the user told us about the physics
    fragments: List[str] = []
    if design_spec.domain_context:
        fragments.append(design_spec.domain_context)
    fragments.extend(design_spec.physics.governing_equations)
    fragments.extend(design_spec.physics.constraints)
    description = " ".join(fragments)

    if not description.strip():
        # Fall back to task_type name as hint
        description = design_spec.task_type

    return suggest_problem_spec(description)


# ---------------------------------------------------------------------------
# in_dim / out_dim inference
# ---------------------------------------------------------------------------

def _infer_dims(
    pde_hint: Dict[str, Any],
    design_spec: DesignSpec,
) -> Tuple[int, int]:
    """Infer (in_dim, out_dim) for model construction."""
    coords = pde_hint.get("coords") or []
    fields = pde_hint.get("fields") or []

    # Override with explicitly elicited info when available
    if design_spec.inputs:
        in_dim = len(design_spec.inputs)
    else:
        in_dim = max(len(coords), 1)

    if design_spec.outputs:
        out_dim = len(design_spec.outputs)
    elif design_spec.data.target_variables:
        out_dim = len(design_spec.data.target_variables)
    else:
        out_dim = max(len(fields), 1)

    return in_dim, out_dim


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def _pinn_code(ps: PinneapleSpec, design_spec: DesignSpec) -> str:
    coords_tuple = repr(tuple(ps.coords))
    fields_tuple = repr(tuple(ps.fields))
    params_repr = repr(ps.pde_params)
    problem_name = (design_spec.title or "my_problem").lower().replace(" ", "_")
    dim = len(ps.coords)

    mk = dict(ps.model_kwargs)
    mk.update({"in_dim": ps.model_kwargs.get("in_dim", dim),
                "out_dim": ps.model_kwargs.get("out_dim", len(ps.fields))})

    model_kwargs_str = ", ".join(
        f"{k}={repr(v)}" for k, v in mk.items()
    )

    tc = ps.train_config_kwargs
    train_kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in tc.items())

    col = ps.collocation_kwargs
    col_kwargs_str = ", ".join(
        f"{k}={repr(v)}" for k, v in col.items() if v
    )

    inv_block = ""
    if design_spec.task_type == "inverse_problem" and design_spec.physics.parameters_unknown:
        inv_names = repr(design_spec.physics.parameters_unknown)
        inv_block = textwrap.dedent(f"""\
            # Inverse-problem: trainable parameters to recover
            # Pass them to VanillaPINN via inverse_params_names= or use PINNFactory
            # inverse_params = {inv_names}
        """)

    return textwrap.dedent(f"""\
        # ── Generated by Pinneaple DesignAgent ───────────────────────────────
        # Task type : {design_spec.task_type}
        # PDE kind  : {ps.pde_kind}  (confidence {ps.pde_confidence:.1f})
        # Adjust PDE kind, coords, fields, params and domain_bounds as needed.
        # ─────────────────────────────────────────────────────────────────────

        import pinneaple as pp
        from pinneaple_environment import ProblemSpec, PDETermSpec
        from pinneaple_pinn import compile_problem
        from pinneaple_data import CollocationSampler, CollocationConfig
        from pinneaple_train import Trainer, TrainConfig

        # 1 ── Problem specification
        spec = ProblemSpec(
            name={repr(problem_name)},
            dim={dim},
            coords={coords_tuple},
            fields={fields_tuple},
            pde=PDETermSpec(
                kind={repr(ps.pde_kind)},
                fields={fields_tuple},
                coords={coords_tuple},
                params={params_repr},
            ),
        )
        {inv_block}
        # 2 ── Compile physics loss from spec
        loss_fn = compile_problem(spec)

        # 3 ── Build model
        model = pp.build_model({repr(ps.model_name)}, {model_kwargs_str})

        # 4 ── Collocation + boundary / initial-condition points
        col_cfg = CollocationConfig({col_kwargs_str})
        sampler = CollocationSampler.from_problem_spec(spec, col_cfg)
        batch   = sampler.sample()

        # 5 ── Train
        train_cfg = TrainConfig({train_kwargs_str})
        trainer   = Trainer(model, loss_fn=loss_fn)
        result    = trainer.fit(batch, cfg=train_cfg)
        print("Training complete. Losses:", result.metrics)

        # 6 ── Inference (adjust grid to your domain)
        import torch, numpy as np
        grid = [torch.linspace(0, 1, 100) for _ in spec.coords]
        mesh = torch.stack(torch.meshgrid(*grid, indexing="ij"), dim=-1).reshape(-1, {dim})
        with torch.no_grad():
            u_pred = model(mesh)
        print("Prediction shape:", u_pred.shape)
    """)


def _fno_code(ps: PinneapleSpec, design_spec: DesignSpec) -> str:
    mk = dict(ps.model_kwargs)
    mk.setdefault("in_dim", max(len(ps.coords), 1))
    mk.setdefault("out_dim", max(len(ps.fields), 1))
    model_kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in mk.items())

    tc = ps.train_config_kwargs
    train_kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in tc.items())

    is_forecasting = design_spec.task_type == "forecasting"
    window_hint = ""
    if is_forecasting:
        window_hint = textwrap.dedent("""\
            # from pinneaple_timeseries import WindowedDataset
            # dataset = WindowedDataset(your_data, input_window=64, horizon=24)
            # train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        """)

    return textwrap.dedent(f"""\
        # ── Generated by Pinneaple DesignAgent ───────────────────────────────
        # Task type : {design_spec.task_type}
        # Model     : {ps.model_name}
        # ─────────────────────────────────────────────────────────────────────

        import pinneaple as pp
        from pinneaple_train import Trainer, TrainConfig
        from pinneaple_train.losses import SupervisedLoss
        from torch.utils.data import DataLoader

        # 1 ── Model ({ps.model_name} for {design_spec.task_type})
        model = pp.build_model({repr(ps.model_name)}, {model_kwargs_str})

        # 2 ── Dataset  (replace with your actual DataLoader)
        {window_hint}
        # train_loader = DataLoader(your_dataset, batch_size=32, shuffle=True)
        train_loader = ...  # TODO: wire up your DataLoader here

        # 3 ── Loss
        loss_fn = SupervisedLoss()

        # 4 ── Train
        train_cfg = TrainConfig({train_kwargs_str})
        trainer   = Trainer(model, loss_fn=loss_fn)
        result    = trainer.fit(train_loader, cfg=train_cfg)
        print("Training complete. Losses:", result.metrics)

        # 5 ── Inference
        import torch
        x_test = ...  # TODO: your test input tensor
        with torch.no_grad():
            y_pred = model(x_test)
        print("Prediction shape:", y_pred.shape)
    """)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_pinneaple_spec(design_spec: DesignSpec) -> PinneapleSpec:
    """Translate an elicited :class:`~pinneaple_problemdesign.schema.ProblemSpec`
    into a :class:`~pinneaple_problemdesign.schema.PinneapleSpec`.

    Uses ``pinneaple_environment.capabilities.identify_pde`` for PDE detection
    and falls back gracefully when the environment package is unavailable.

    Parameters
    ----------
    design_spec:
        The business-level spec produced by :class:`~pinneaple_problemdesign.agent.DesignAgent`.

    Returns
    -------
    PinneapleSpec
        Concrete API objects + runnable code snippet.
    """
    task = design_spec.task_type or "other"

    # --- PDE identification ---
    pde_hint = _identify_pde(design_spec)
    pde_kind = pde_hint.get("kind", "custom")
    pde_confidence = float(pde_hint.get("confidence", 0.0))
    coords: List[str] = list(pde_hint.get("coords") or ["x"])
    fields: List[str] = list(pde_hint.get("fields") or ["u"])
    pde_params: Dict[str, Any] = dict(pde_hint.get("default_params") or {})

    # Override with explicitly elicited values when present
    if design_spec.inputs:
        coords = list(design_spec.inputs)
    if design_spec.outputs:
        fields = list(design_spec.outputs)
    elif design_spec.data.target_variables:
        fields = list(design_spec.data.target_variables)
    if design_spec.physics.parameters_known:
        # surface any known params as stubs in pde_params
        for p in design_spec.physics.parameters_known:
            pde_params.setdefault(p, 1.0)

    # --- Model selection ---
    model_name, model_kwargs_base = _TASK_MODEL.get(task, _TASK_MODEL["other"])
    in_dim, out_dim = _infer_dims(pde_hint, design_spec)
    model_kwargs = dict(model_kwargs_base)
    model_kwargs["in_dim"] = in_dim
    model_kwargs["out_dim"] = out_dim

    # --- Training config ---
    train_config_kwargs = dict(_TASK_TRAIN.get(task, _TASK_TRAIN["other"]))

    # --- Collocation ---
    collocation_kwargs = dict(_TASK_COLLOCATION.get(task, _TASK_COLLOCATION["other"]))

    # --- Environment kwargs ---
    dim = len(coords)
    problem_name = (design_spec.title or "my_problem").lower().replace(" ", "_")
    environment_kwargs: Dict[str, Any] = {
        "name": problem_name,
        "dim": dim,
        "coords": tuple(coords),
        "fields": tuple(fields),
        "pde": {
            "kind": pde_kind,
            "fields": tuple(fields),
            "coords": tuple(coords),
            "params": pde_params,
        },
    }

    # --- Assemble partial spec (no code yet) ---
    ps = PinneapleSpec(
        pde_kind=pde_kind,
        pde_confidence=pde_confidence,
        coords=coords,
        fields=fields,
        pde_params=pde_params,
        environment_kwargs=environment_kwargs,
        model_name=model_name,
        model_kwargs=model_kwargs,
        train_config_kwargs=train_config_kwargs,
        collocation_kwargs=collocation_kwargs,
    )

    # --- Code generation ---
    if task in _PINN_TASKS:
        ps.pipeline_code = _pinn_code(ps, design_spec)
    else:
        ps.pipeline_code = _fno_code(ps, design_spec)

    return ps
