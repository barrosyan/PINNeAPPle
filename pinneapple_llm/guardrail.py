"""``PhysicsGuardrail``: the actual anti-hallucination mechanism.

State this precisely, because overclaiming it is exactly the failure mode
it exists to prevent: **this does not guarantee a result is correct, and
nothing can.** What it does is compute a small set of independent,
numeric, re-checkable signals -- a re-evaluated PDE residual on fresh
points, basic parameter sanity, and (when a reference is supplied) an
error metric against real data -- and refuse to label a result
"trustworthy" unless every signal it can compute passes. A raw LLM asked
to solve a physics problem, by contrast, produces an answer with no
residual behind it at all; there is nothing to check. The differentiation
this module gives PINNeAPPle over "just ask a language model" is not that
its answers are smarter, it's that every claim is required to pass through
this gate, or be reported as *not* having passed it -- explicitly, with
the failing check named, not silently.

Independent of ``pinneapple_llm``'s drafting module (``draft.py``): this
is useful on *any* ``solve_pde``/``pipeline`` result, whether or not an
LLM was involved in producing the ``ProblemSpec`` at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class GuardrailReport:
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        """True only if every check that actually ran, passed. A check
        that could not run (e.g. no reference data supplied) is neither a
        pass nor a fail -- it is absent, and does not count either way;
        see ``checked_names``/``skipped`` to see exactly what was and
        was not evaluated, rather than inferring it from a single bool."""
        return all(c.passed for c in self.checks)

    def as_error(self) -> RuntimeError:
        failing = [c for c in self.checks if not c.passed]
        lines = [f"PhysicsGuardrail: {len(failing)} check(s) failed:"]
        for c in failing:
            lines.append(f"  - {c.name}: {c.detail}")
        return RuntimeError("\n".join(lines))

    def summary(self) -> str:
        lines = ["PhysicsGuardrail report:"]
        for c in self.checks:
            mark = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{mark}] {c.name}: {c.detail}")
        lines.append(f"  Overall: {'TRUSTWORTHY' if self.trustworthy else 'NOT TRUSTWORTHY'}")
        return "\n".join(lines)


class PhysicsGuardrail:
    """Runs a fixed sequence of independent checks against a trained
    model + the ``ProblemSpec`` it was trained on.

    Parameters
    ----------
    spec : the ``ProblemSpec`` (from ``pinneapple_physics.pde_environment``)
        the model was trained against.
    residual_threshold : max acceptable mean-squared PDE residual
        (re-evaluated on FRESH collocation points, never the ones used
        during training -- catches memorisation as well as an under-
        converged fit).
    n_check_points : how many fresh collocation points to re-evaluate the
        residual on.
    """

    def __init__(self, spec: Any, *, residual_threshold: float = 1e-2, n_check_points: int = 4096):
        self.spec = spec
        self.residual_threshold = residual_threshold
        self.n_check_points = n_check_points

    # ------------------------------------------------------------------
    def _check_parameter_sanity(self) -> CheckResult:
        """Cheap, first-line check: catches an obviously unphysical
        parameter before spending any compute on the expensive checks --
        e.g. a negative viscosity/diffusivity/Reynolds number, which is
        exactly the kind of value an LLM asked to "make up something
        plausible" can produce without any grounding."""
        params = dict(getattr(self.spec.pde, "params", {}) or {})
        problems = []
        # Parameters that are physically required to be positive for any
        # of this module's supported pde_kinds (diffusivity/viscosity/
        # Reynolds-number-like quantities). Deliberately conservative: an
        # unrecognised parameter name is not flagged (this checks known
        # physical quantities, not the shape of an arbitrary dict).
        positive_only = {"nu", "Re", "alpha", "D", "k", "inv_Re", "diffusivity"}
        for name in positive_only & params.keys():
            val = params[name]
            if isinstance(val, (int, float)) and val <= 0:
                problems.append(f"{name}={val} must be > 0")
        passed = not problems
        detail = "all recognised physical parameters are positive" if passed else "; ".join(problems)
        return CheckResult(name="parameter_sanity", passed=passed, detail=detail)

    def _check_residual(self, model) -> CheckResult:
        import numpy as np
        import torch
        from pinneapple_physics.pinn_solver.compiler.compile import compile_problem

        loss_fn = compile_problem(self.spec)
        coords = list(self.spec.coords)
        bounds = self.spec.domain_bounds
        cols = [
            torch.as_tensor(np.random.uniform(*bounds[c], size=self.n_check_points), dtype=torch.float32).reshape(-1, 1)
            for c in coords
        ]
        x = torch.cat(cols, dim=1).requires_grad_(True)
        n_fields = len(self.spec.fields)
        batch = {
            "x_col": x, "ctx": {},
            "x_bc": torch.zeros((0, len(coords))), "y_bc": torch.zeros((0, n_fields)),
            "x_ic": torch.zeros((0, len(coords))), "y_ic": torch.zeros((0, n_fields)),
            "x_data": torch.zeros((0, len(coords))), "y_data": torch.zeros((0, n_fields)),
        }
        y_hat = model(x)
        if hasattr(y_hat, "y"):
            y_hat = y_hat.y
        out = loss_fn(model, y_hat, batch)
        residual = float(out["pde"].item()) if isinstance(out, dict) and "pde" in out else float(out["total"].item())
        passed = residual <= self.residual_threshold
        return CheckResult(
            name="pde_residual", passed=passed,
            detail=f"mean-squared PDE residual on {self.n_check_points} fresh points = {residual:.4g} "
                   f"({'<=' if passed else '>'} threshold {self.residual_threshold:.4g})",
            value=residual, threshold=self.residual_threshold,
        )

    def _check_reference(self, model, reference_x, reference_y, rmse_threshold: float) -> CheckResult:
        import numpy as np
        import torch

        model.eval()
        with torch.no_grad():
            x_t = torch.as_tensor(np.asarray(reference_x, dtype="float32"))
            y_hat = model(x_t)
            if hasattr(y_hat, "y"):
                y_hat = y_hat.y
            y_hat = y_hat.numpy()
        y_true = np.asarray(reference_y, dtype="float32")
        rmse = float(np.sqrt(np.mean((y_hat - y_true) ** 2)))
        passed = rmse <= rmse_threshold
        return CheckResult(
            name="reference_data_match", passed=passed,
            detail=f"RMSE against supplied reference data = {rmse:.4g} "
                   f"({'<=' if passed else '>'} threshold {rmse_threshold:.4g})",
            value=rmse, threshold=rmse_threshold,
        )

    # ------------------------------------------------------------------
    def check(
        self,
        model,
        *,
        reference_x=None,
        reference_y=None,
        reference_rmse_threshold: Optional[float] = None,
    ) -> GuardrailReport:
        """Run every applicable check and return a :class:`GuardrailReport`.

        Parameters
        ----------
        model : the trained model (an ``nn.Module`` mapping
            ``(N, len(coords)) -> (N, len(fields))``).
        reference_x, reference_y : optional real reference data
            (e.g. a DNS/experimental dataset) to check the model against;
            skipped entirely if not given -- see ``GuardrailReport
            .trustworthy``'s docstring for what a skipped check means.
        reference_rmse_threshold : required if ``reference_x``/
            ``reference_y`` are given (no default -- what counts as an
            acceptable RMSE is problem-specific and must not be silently
            assumed).
        """
        checks = [self._check_parameter_sanity(), self._check_residual(model)]
        if reference_x is not None or reference_y is not None:
            if reference_x is None or reference_y is None:
                raise ValueError("reference_x and reference_y must both be given, or neither")
            if reference_rmse_threshold is None:
                raise ValueError(
                    "reference_rmse_threshold is required when reference_x/reference_y are given "
                    "-- there is no problem-agnostic default for 'acceptable error'"
                )
            checks.append(self._check_reference(model, reference_x, reference_y, reference_rmse_threshold))
        return GuardrailReport(checks=checks)
