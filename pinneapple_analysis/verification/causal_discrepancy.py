"""Two roadmap points from the product README's honest 21-point scope table:

Point 15 -- Causal / intervention engine
-----------------------------------------
"What happens if I increase pressure 20%?" -- ``run_intervention`` re-runs
the EXISTING ``core.pipeline.analyze`` pipeline once as a baseline, then
once per requested parameter value with that one parameter overridden in
the drafted preset's kwargs, and reports REAL percentage changes (final
training loss, the guardrail's re-computed PDE residual, and the trained
model's own prediction at a fixed query point for every field the preset
produces) -- never a fabricated number, and never a percentage change
presented as reliable when the guardrail marked that run (or the
baseline itself) NOT TRUSTWORTHY.

Deliberately does NOT touch ``saas/physics_verification_engine/core/pipeline.py``:
``draft_problem`` is only ever invoked once (inside the single baseline
``analyze()`` call); every perturbed run below re-implements just the
execution + verification steps directly against
``pinneapple_physics.get_preset``/``solve_pde`` and
``pinneapple_llm.PhysicsGuardrail`` -- the same three calls
``pipeline.py``'s own ``analyze()`` makes internally, just with one
kwarg overridden -- so nothing shared with other in-progress roadmap work
in that saas app is modified.

Note on ``run_intervention``'s one saas dependency: its single baseline
call goes through ``saas.physics_verification_engine.core.pipeline.analyze``,
the LLM-driven "draft a ProblemSpec from a natural-language description"
orchestration -- that orchestration is specific to the
``physics_verification_engine`` saas app (not generic, reusable physics
logic like the rest of this module), so it deliberately stays there
rather than moving into this core package. The import is done lazily
inside ``run_intervention`` itself (like this function's other
``pinneapple_llm``/``pinneapple_physics`` imports) precisely so importing
this module -- or the rest of ``pinneapple_analysis.verification`` --
never requires the ``saas`` app to be importable; only actually calling
``run_intervention`` does.

Cost note / future optimization (not built here): every intervention
value in this module currently retrains a PINN from scratch. A real
deployment would want to cache/reuse the baseline-trained model as a
warm start (or freeze/fine-tune) when the architecture and epoch budget
are unchanged between the baseline and the perturbed run, rather than
paying the full training cost per intervention value. Kept out of scope
for this pass; test epoch counts are kept small (100-200) specifically
to keep this honest (no shortcuts) but still fast to run for real.

Point 10 -- Model discrepancy / physics+ML hybrid model
---------------------------------------------------------
``fit_discrepancy_model`` implements the standard SciML "model-form
discrepancy" technique: Kennedy & O'Hagan, "Bayesian Calibration of
Computer Models", J. R. Stat. Soc. B, 63(3), 2001 -- the reference model
is ``y(x) = eta(x) + delta(x)``, where ``eta`` is the (imperfect) physics
model and ``delta`` is a small, separately-fit statistical correction for
whatever the physics model's structural form is missing. Here ``delta``
is a small MLP fit by ordinary MSE regression against the REAL residual
``reference_y - physics_model(reference_x)`` -- this is data regression,
not PDE-residual training, so it deliberately does not go through
``solve_pde``. Training reuses PINNeAPPle's own generic supervised
``pinneapple_neural.trainer.Trainer``/``TrainConfig`` (not a hand-rolled
loop) and its real ``pinneapple_data.splits.split_indices`` train/val/test
splitter, so the reported diagnostics (variance explained, RMSE) are
computed on a genuine held-out test split the correction model never saw
during training.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _as_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.as_tensor(x, dtype=torch.float32)


def _call_model(model: Any, x: torch.Tensor) -> torch.Tensor:
    """Call either a PINNeAPPle ``nn.Module`` (whose ``forward`` returns a
    ``ModelOutput`` with a ``.y`` tensor) or a plain ``callable(x) ->
    Tensor`` uniformly -- ``fit_discrepancy_model``'s ``physics_model``
    argument may legitimately be either (a hand-written analytic function
    in a test, or a real trained PINN elsewhere)."""
    out = model(x)
    y = getattr(out, "y", None)
    return y if y is not None else out


def _eval_mode(model: Any) -> None:
    if hasattr(model, "eval"):
        model.eval()


def _pct_change(old: Optional[float], new: Optional[float]) -> Optional[float]:
    """Real percentage change, or None when it is not honestly
    computable (either value missing, or old==0 making a percentage
    change undefined) -- never silently divides by zero into +/-inf."""
    if old is None or new is None:
        return None
    if old == 0:
        return None
    return (new - old) / abs(old) * 100.0


# ---------------------------------------------------------------------------
# Point 15: Causal / intervention engine
# ---------------------------------------------------------------------------

@dataclass
class InterventionRunResult:
    param_value: float
    final_loss: Optional[float] = None
    loss_pct_change: Optional[float] = None
    residual_value: Optional[float] = None
    residual_pct_change: Optional[float] = None
    guardrail_trustworthy: Optional[bool] = None
    guardrail_checks: List[Dict[str, Any]] = field(default_factory=list)
    field_predictions: Dict[str, float] = field(default_factory=dict)
    field_pct_change: Dict[str, Optional[float]] = field(default_factory=dict)
    reliable: bool = False
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class InterventionReport:
    base_description: str
    param_name: str
    baseline_rejected: bool = False
    rejection_reason: str = ""
    query_point: Dict[str, float] = field(default_factory=dict)
    baseline_param_value: Optional[float] = None
    baseline_final_loss: Optional[float] = None
    baseline_residual_value: Optional[float] = None
    baseline_field_predictions: Dict[str, float] = field(default_factory=dict)
    baseline_trustworthy: Optional[bool] = None
    architecture_used: str = ""
    runs: List[InterventionRunResult] = field(default_factory=list)
    summary: List[str] = field(default_factory=list)


def _build_query_point(spec) -> torch.Tensor:
    """A single fixed point (midpoint of each coordinate's domain bound),
    reused across the baseline and every intervention run so predicted
    field values are directly comparable."""
    values = []
    for c in spec.coords:
        lo, hi = spec.domain_bounds.get(c, (0.0, 1.0))
        values.append(0.5 * (float(lo) + float(hi)))
    return torch.tensor([values], dtype=torch.float32)


def _evaluate_fields_at_point(model, point: torch.Tensor, field_names: Sequence[str]) -> Dict[str, float]:
    _eval_mode(model)
    with torch.no_grad():
        y = _call_model(model, point)
    return {name: float(y[0, i].item()) for i, name in enumerate(field_names)}


def run_intervention(
    base_description: str,
    param_name: str,
    param_values: Sequence[float],
    *,
    physical_parameters: Optional[Dict[str, float]] = None,
    provider: str = "ollama",
    model_name: str = "llama3.2:3b",
    **analyze_kwargs: Any,
) -> InterventionReport:
    """"What happens if I increase pressure 20%?" -- run the real
    pipeline once as a baseline, then once per value in ``param_values``
    with ``param_name`` overridden in the drafted preset's kwargs
    (e.g. ``param_name="nu"`` for a Burgers preset's viscosity), and
    report REAL percentage changes vs. the baseline in: final training
    loss, the guardrail's re-computed PDE residual (when computable for
    this preset's ``pde_kind``), and the trained model's own prediction
    at one fixed query point for every field the preset produces.

    ``analyze_kwargs`` (``epochs``, ``n_collocation``, ``residual_threshold``,
    ``reference_benchmark``, ``candidate_architectures``, ``flow_geometry``,
    ...) are forwarded to the one baseline ``analyze()`` call exactly as
    given; ``epochs``/``n_collocation``/``residual_threshold`` are then
    reused (same values, not re-drafted) for every perturbed re-run so the
    comparison is apples-to-apples. If more than one
    ``candidate_architectures`` was given, only the baseline's WINNING
    architecture is retrained per intervention value (retraining the
    whole shortlist per value would multiply the already-nontrivial
    training cost with little extra signal for a "what if I change one
    parameter" question).
    """
    import pinneapple_llm as pl
    import pinneapple_physics as pp
    from pinneapple_neural.architectures.registry import ModelRegistry
    import pinneapple_neural.architectures  # noqa: F401  registers the zoo
    # Lazy, saas-specific import -- see this module's docstring's
    # "Note on run_intervention's one saas dependency" for why.
    from saas.physics_verification_engine.core.pipeline import analyze

    baseline = analyze(
        base_description,
        physical_parameters=physical_parameters,
        provider=provider,
        model_name=model_name,
        **analyze_kwargs,
    )

    if baseline.rejected or baseline.best_candidate is None:
        return InterventionReport(
            base_description=base_description,
            param_name=param_name,
            baseline_rejected=True,
            rejection_reason=baseline.rejection_reason or "baseline analyze() produced no usable candidate",
        )

    preset_name = baseline.provenance.drafted_preset
    base_kwargs = dict(baseline.provenance.drafted_preset_kwargs or {})
    arch = baseline.best_candidate.architecture

    # Same defaults as core.pipeline.analyze()'s own signature -- reused
    # (not re-derived) so every perturbed run trains/verifies under
    # exactly the same budget as the baseline.
    epochs = analyze_kwargs.get("epochs", 1500)
    n_collocation = analyze_kwargs.get("n_collocation", 1024)
    residual_threshold = analyze_kwargs.get("residual_threshold", 1e-2)
    reference_benchmark = analyze_kwargs.get("reference_benchmark")

    base_spec = pp.get_preset(preset_name, **base_kwargs)
    query_point = _build_query_point(base_spec)
    query_point_named = {c: float(query_point[0, i].item()) for i, c in enumerate(base_spec.coords)}

    baseline_field_predictions = _evaluate_fields_at_point(baseline.best_candidate.model, query_point, base_spec.fields)
    baseline_residual_value = next(
        (c["value"] for c in baseline.best_candidate.guardrail_checks if c["name"] == "pde_residual"), None
    )

    report = InterventionReport(
        base_description=base_description,
        param_name=param_name,
        query_point=query_point_named,
        baseline_param_value=base_kwargs.get(param_name),
        baseline_final_loss=baseline.best_candidate.final_loss,
        baseline_residual_value=baseline_residual_value,
        baseline_field_predictions=baseline_field_predictions,
        baseline_trustworthy=baseline.best_candidate.guardrail_trustworthy,
        architecture_used=arch,
    )

    for value in param_values:
        override_kwargs = dict(base_kwargs)
        override_kwargs[param_name] = value

        try:
            spec = pp.get_preset(preset_name, **override_kwargs)
            torch_model = ModelRegistry.build(
                arch, in_dim=len(spec.coords), out_dim=len(spec.fields), hidden_dim=64, n_layers=4,
            )
            result = pp.solve_pde(spec, torch_model, epochs=epochs, n_collocation=n_collocation)
        except Exception as e:  # noqa: BLE001 -- a real, reportable failure, not a silent skip
            report.runs.append(InterventionRunResult(param_value=value, error=f"{type(e).__name__}: {e}"))
            report.summary.append(f"{param_name}={value}: FAILED to train ({type(e).__name__}: {e})")
            continue

        guardrail = pl.PhysicsGuardrail(spec, residual_threshold=residual_threshold)
        check_kwargs = {} if reference_benchmark is None else {"reference_benchmark": reference_benchmark}
        grep = guardrail.check(torch_model, **check_kwargs)
        residual_value = next((c.value for c in grep.checks if c.name == "pde_residual"), None)
        final_loss = result["history"]["loss"][-1]

        # Same fixed query point -- same coord ordering, since only one
        # preset kwarg (not the preset itself) changed.
        field_predictions = _evaluate_fields_at_point(torch_model, query_point, spec.fields)
        field_pct_change = {
            name: _pct_change(baseline_field_predictions.get(name), field_predictions.get(name))
            for name in field_predictions
        }

        reliable = bool(grep.trustworthy) and bool(baseline.best_candidate.guardrail_trustworthy)
        warnings: List[str] = []
        if not grep.trustworthy:
            warnings.append(
                "PhysicsGuardrail marked THIS intervention run NOT TRUSTWORTHY -- the percentage "
                "changes below are real numbers computed from a real run, but must NOT be treated "
                "as a reliable physical conclusion."
            )
        if not baseline.best_candidate.guardrail_trustworthy:
            warnings.append(
                "The BASELINE run itself was NOT TRUSTWORTHY -- every percentage change in this "
                "report is relative to an untrustworthy reference and should not be trusted either."
            )

        run_result = InterventionRunResult(
            param_value=value,
            final_loss=final_loss,
            loss_pct_change=_pct_change(baseline.best_candidate.final_loss, final_loss),
            residual_value=residual_value,
            residual_pct_change=_pct_change(baseline_residual_value, residual_value),
            guardrail_trustworthy=grep.trustworthy,
            guardrail_checks=[
                {"name": c.name, "passed": c.passed, "detail": c.detail, "value": c.value, "threshold": c.threshold}
                for c in grep.checks
            ],
            field_predictions=field_predictions,
            field_pct_change=field_pct_change,
            reliable=reliable,
            warnings=warnings,
        )
        report.runs.append(run_result)

        parts = ", ".join(
            f"{name} {pct:+.1f}%" if pct is not None else f"{name} n/a"
            for name, pct in field_pct_change.items()
        )
        trust_tag = "TRUSTWORTHY" if reliable else "NOT TRUSTWORTHY -- do not treat as reliable"
        report.summary.append(f"{param_name}={value}: {parts} [{trust_tag}]")

    return report


# ---------------------------------------------------------------------------
# Point 10: Model discrepancy / physics+ML hybrid model
# ---------------------------------------------------------------------------

@dataclass
class DiscrepancyModel:
    """``y(x) ~= physics_model(x) + delta(x)`` -- Kennedy & O'Hagan (2001)
    model-form discrepancy. ``delta`` (``correction_model``) is a small MLP
    fit by MSE regression on the real residual ``reference_y -
    physics_model(reference_x)``."""

    physics_model: Any
    correction_model: Any
    in_dim: int
    out_dim: int
    n_train: int
    n_test: int
    variance_explained_test: float  # R^2 of delta(x) vs. the TRUE residual, held-out points
    test_rmse_physics_only: float
    test_rmse_hybrid: float
    reference: str = (
        "Kennedy, M. C. & O'Hagan, A. (2001). 'Bayesian Calibration of "
        "Computer Models'. J. R. Stat. Soc. B, 63(3), 425-464."
    )

    def discrepancy_only(self, x: Any) -> torch.Tensor:
        xt = _as_tensor(x)
        if xt.ndim == 1:
            xt = xt.unsqueeze(-1)
        _eval_mode(self.correction_model)
        with torch.no_grad():
            return _call_model(self.correction_model, xt)

    def predict(self, x: Any) -> torch.Tensor:
        """The hybrid prediction: physics_model(x) + discrepancy_only(x)."""
        xt = _as_tensor(x)
        if xt.ndim == 1:
            xt = xt.unsqueeze(-1)
        _eval_mode(self.physics_model)
        with torch.no_grad():
            phys = _call_model(self.physics_model, xt)
        return phys + self.discrepancy_only(xt)


def fit_discrepancy_model(
    spec: Any,
    physics_model: Any,
    reference_x: Any,
    reference_y: Any,
    *,
    hidden_dim: int = 32,
    n_layers: int = 2,
    epochs: int = 400,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 42,
) -> DiscrepancyModel:
    """Fit a small MLP discrepancy-correction model on the real residual
    between ``reference_y`` and ``physics_model(reference_x)`` (Kennedy &
    O'Hagan 2001's model-form discrepancy technique -- see this module's
    docstring and ``DiscrepancyModel.reference``).

    ``spec`` is accepted (mirroring the rest of this codebase's
    ``ProblemSpec``-centric API, and so a future caller already holding a
    ``ProblemSpec`` doesn't need to separately track ``in_dim``/``out_dim``
    itself) but is not otherwise required -- the correction network's
    input/output dimensions are inferred directly from
    ``reference_x``/``reference_y``'s own shapes, which is what actually
    determines what the network needs to consume/produce.

    Training is via ``pinneapple_neural.trainer.Trainer``/``TrainConfig``
    (PINNeAPPle's own generic supervised trainer) with a plain MSE
    ``loss_fn`` -- deliberately NOT ``solve_pde``/a PDE-residual loss,
    since this fits an ML correction to DATA, not a physics equation.
    The train/val/test split is a real, seeded random split via
    ``pinneapple_data.splits.split_indices`` (re-exported as
    ``pinneapple_neural.trainer.splits``); diagnostics
    (``variance_explained_test``, the two RMSEs) are computed ONLY on the
    held-out test split, which never appears in any training batch.
    """
    del spec  # see docstring: accepted for API symmetry, dims come from the data itself

    from pinneapple_neural.architectures.registry import ModelRegistry
    import pinneapple_neural.architectures  # noqa: F401  registers the zoo
    from pinneapple_neural.trainer import Trainer, TrainConfig
    from pinneapple_neural.trainer.splits import SplitSpec, split_indices
    from pinneapple_neural.trainer.metrics import R2, RMSE

    x = _as_tensor(reference_x)
    y = _as_tensor(reference_y)
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    if y.ndim == 1:
        y = y.unsqueeze(-1)

    n = x.shape[0]
    if n < 10:
        raise ValueError(
            f"fit_discrepancy_model needs at least 10 reference points for an honest "
            f"train/val/test split, got {n}"
        )
    if y.shape[0] != n:
        raise ValueError(f"reference_x has {n} points but reference_y has {y.shape[0]}")

    in_dim = x.shape[1]
    out_dim = y.shape[1]

    split_spec = SplitSpec(method="random", train=0.7, val=0.15, test=0.15, seed=seed)
    idx = split_indices(n, split_spec)
    train_idx, val_idx, test_idx = idx["train"], idx["val"], idx["test"]
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("the train/val/test split produced an empty train or test set -- supply more reference points")

    _eval_mode(physics_model)
    with torch.no_grad():
        phys_pred_all = _call_model(physics_model, x)
    residual_all = y - phys_pred_all  # the real model-form discrepancy target

    x_train_t = x[train_idx]
    r_train_t = residual_all[train_idx]
    # val split falls back to (a copy of) the train split only in the
    # degenerate case where SplitSpec's val ratio rounds to zero points
    # (e.g. a very small n) -- Trainer.fit requires a non-empty val_loader.
    if len(val_idx) > 0:
        x_val_t, r_val_t = x[val_idx], residual_all[val_idx]
    else:
        x_val_t, r_val_t = x_train_t, r_train_t

    correction_model = ModelRegistry.build(
        "modified_mlp", in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, n_layers=n_layers,
    )

    train_loader = DataLoader(TensorDataset(x_train_t, r_train_t), batch_size=min(batch_size, len(x_train_t)), shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val_t, r_val_t), batch_size=min(batch_size, len(x_val_t)), shuffle=False)

    def _mse_loss_fn(model, y_hat, batch):  # noqa: ARG001 -- Trainer's own loss_fn contract
        return torch.mean((y_hat - batch["y"]) ** 2)

    log_dir = tempfile.mkdtemp(prefix="discrepancy_trainer_")
    trainer = Trainer(correction_model, _mse_loss_fn, metrics=[R2(), RMSE()])
    cfg = TrainConfig(
        epochs=epochs, lr=lr, log_dir=log_dir, run_name="discrepancy_fit",
        save_best=False,  # this is a small research fit, not a deployable checkpoint
        physics_aware_validation=False,  # plain data regression -- no PDE-residual gradient needed at val time
    )
    trainer.fit(train_loader, val_loader, cfg)

    # -- honest, held-out diagnostics: the test split was in none of the above --
    x_test_t = x[test_idx]
    y_test_t = y[test_idx]
    _eval_mode(correction_model)
    with torch.no_grad():
        phys_pred_test = _call_model(physics_model, x_test_t)
        true_residual_test = y_test_t - phys_pred_test
        discrepancy_pred_test = _call_model(correction_model, x_test_t)
        hybrid_pred_test = phys_pred_test + discrepancy_pred_test

    r2_metric, rmse_metric = R2(), RMSE()
    variance_explained_test = r2_metric(discrepancy_pred_test, true_residual_test)
    test_rmse_physics_only = rmse_metric(phys_pred_test, y_test_t)
    test_rmse_hybrid = rmse_metric(hybrid_pred_test, y_test_t)

    return DiscrepancyModel(
        physics_model=physics_model,
        correction_model=correction_model,
        in_dim=in_dim,
        out_dim=out_dim,
        n_train=int(len(train_idx)),
        n_test=int(len(test_idx)),
        variance_explained_test=variance_explained_test,
        test_rmse_physics_only=test_rmse_physics_only,
        test_rmse_hybrid=test_rmse_hybrid,
    )
