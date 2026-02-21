"""Adapter from Physical Schema (UPD) to PINN problem and sampling specs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from pinneaple_pinn.factory import PINNProblemSpec
from pinneaple_pinn.io import ConditionSpec, SamplingSpec, PINNMapping


@dataclass
class SchemaAdapterOptions:
    """
    Options controlling how a Physical Schema is converted into a PINN problem.

    MVP assumptions:
      - PDE residuals are provided as a list of strings in schema["equations"]["residuals"]
        (or compatible aliases).
      - Conditions are provided as a list with entries specifying type/name/equation.
      - For grid UPD, we can auto-map condition types:
          * "initial" -> sample on first time index
          * "boundary" -> sample on lat/lon edges (axis can be specified)
          * "slice" -> sample on coord slice (lev, etc.)
    """
    # Default loss weights
    default_loss_weights: Dict[str, float] = None  # set in __post_init__
    # Default condition weight if missing
    default_condition_weight: float = 1.0
    # Whether to create sampling specs from schema conditions
    build_sampling_from_conditions: bool = True

    def __post_init__(self):
        if self.default_loss_weights is None:
            self.default_loss_weights = {"pde": 1.0, "conditions": 1.0, "data": 1.0}


# ----------------------------
# Helpers: tolerant schema access
# ----------------------------

def _get(d: Dict[str, Any], keys: List[str], default=None):
    """Get nested value at path keys; return default if any key missing."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _as_list(x) -> List[Any]:
    """Convert value to list (None->[], list->identity, else wrap)."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _pick_first_present(schema: Dict[str, Any], candidates: List[List[str]], default=None):
    """Return first value found at any candidate path in schema; else default."""
    for path in candidates:
        val = _get(schema, path, default=None)
        if val is not None:
            return val
    return default


# ----------------------------
# Main adapter
# ----------------------------

class SchemaAdapter:
    """
    Converts a Physical Schema (UPD/Pinneaple) into:
      - PINNProblemSpec (for PINNFactory)
      - Optional: SamplingSpec/ConditionSpec (for UPDDataset)

    Intended usage:
      adapter = SchemaAdapter(mapping, options)
      problem, sampling = adapter.to_pinn_problem(schema, base_sampling=...)
    """

    def __init__(
        self,
        mapping: PINNMapping,
        options: Optional[SchemaAdapterOptions] = None,
    ) -> None:
        self.mapping = mapping
        self.options = options or SchemaAdapterOptions()

    def to_pinn_problem(
        self,
        schema: Dict[str, Any],
        *,
        base_sampling: Optional[SamplingSpec] = None,
        verbose: bool = False,
    ) -> Tuple[PINNProblemSpec, Optional[SamplingSpec]]:
        """
        Returns:
          (problem_spec, sampling_spec_or_none)

        base_sampling:
          if provided, we will copy n_collocation/n_data/replace/seed and only
          append conditions derived from schema (unless already given).
        """
        # 1) PDE residuals
        residuals = self._extract_pde_residuals(schema)
        if not residuals:
            # MVP fallback: allow empty PDE residuals; user can still do data-only / condition-only
            residuals = []

        # 2) Conditions (symbolic equations)
        conditions = self._extract_conditions(schema)

        # 3) Inverse parameters
        inverse_params = self._extract_inverse_params(schema)

        # 4) Loss weights
        loss_weights = self._extract_loss_weights(schema)

        problem = PINNProblemSpec(
            pde_residuals=residuals,
            conditions=conditions,
            independent_vars=self.mapping.independent_vars(),
            dependent_vars=self.mapping.dependent_vars(),
            inverse_params=inverse_params,
            loss_weights=loss_weights,
            verbose=verbose,
        )

        sampling = None
        if self.options.build_sampling_from_conditions:
            sampling = self._build_sampling(schema, base_sampling=base_sampling, conditions=conditions)

        return problem, sampling

    # ----------------------------
    # Extractors
    # ----------------------------

    def _extract_pde_residuals(self, schema: Dict[str, Any]) -> List[str]:
        """
        Tries common locations:
          schema["equations"]["residuals"]
          schema["governing_equations"]["residuals"]
          schema["pde"]["residuals"]
          schema["pde_residuals"]
        """
        val = _pick_first_present(
            schema,
            candidates=[
                ["equations", "residuals"],
                ["governing_equations", "residuals"],
                ["pde", "residuals"],
                ["pde_residuals"],
                ["residuals"],
            ],
            default=[],
        )
        residuals = [str(x) for x in _as_list(val) if str(x).strip()]
        return residuals

    def _extract_inverse_params(self, schema: Dict[str, Any]) -> List[str]:
        """
        Common locations:
          schema["inverse"]["params"]  -> list[str]
          schema["inverse_params"]     -> list[str]
        """
        val = _pick_first_present(
            schema,
            candidates=[
                ["inverse", "params"],
                ["inverse_params"],
            ],
            default=[],
        )
        return [str(x) for x in _as_list(val) if str(x).strip()]

    def _extract_loss_weights(self, schema: Dict[str, Any]) -> Dict[str, float]:
        """
        Common locations:
          schema["loss"]["weights"]
          schema["loss_weights"]
        """
        val = _pick_first_present(
            schema,
            candidates=[
                ["loss", "weights"],
                ["loss_weights"],
            ],
            default=None,
        )
        out = dict(self.options.default_loss_weights)
        if isinstance(val, dict):
            for k, v in val.items():
                try:
                    out[str(k)] = float(v)
                except Exception:
                    pass
        return out

    def _extract_conditions(self, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Conditions format (MVP):
          schema["conditions"] = [
            {"name":"ic_T", "type":"initial", "equation":"T - T0", "weight":1.0, "n":1024, "options":{...}},
            {"name":"bc_wall", "type":"boundary", "equation":"U", "axis":"lat", ...},
          ]

        We output a list of dicts compatible with PINNProblemSpec.conditions:
          {"name":..., "equation":..., "weight":...}
        """
        val = _pick_first_present(
            schema,
            candidates=[
                ["conditions"],
                ["ics_bcs"],   # sometimes people put mixed
                ["bc_ic"],
            ],
            default=[],
        )

        conds_in = _as_list(val)
        conds_out: List[Dict[str, Any]] = []

        for i, c in enumerate(conds_in):
            if not isinstance(c, dict):
                continue
            eq = c.get("equation") or c.get("expr") or c.get("residual")
            if not eq:
                continue
            name = c.get("name") or f"cond_{i}"
            w = c.get("weight", self.options.default_condition_weight)

            conds_out.append(
                {
                    "name": str(name),
                    "equation": str(eq),
                    "weight": float(w) if w is not None else self.options.default_condition_weight,
                    # keep extra fields for sampling builder (non-breaking)
                    "type": c.get("type"),
                    "n": c.get("n"),
                    "options": c.get("options") or {},
                    "axis": c.get("axis"),
                    "coord": c.get("coord"),
                    "value": c.get("value"),
                }
            )

        return conds_out

    # ----------------------------
    # Sampling builder
    # ----------------------------

    def _build_sampling(
        self,
        schema: Dict[str, Any],
        *,
        base_sampling: Optional[SamplingSpec],
        conditions: List[Dict[str, Any]],
    ) -> SamplingSpec:
        """
        Creates a SamplingSpec for UPDDataset from schema conditions.

        We interpret condition type fields:
          - initial: ConditionSpec(type="initial")
          - boundary: ConditionSpec(type="boundary", options={"axis": ...})
          - slice: ConditionSpec(type="slice", options={"coord": ..., "value": ...})
          - interior: ConditionSpec(type="interior")
        """
        if base_sampling is None:
            # try schema defaults
            n_collocation = int(_pick_first_present(schema, [["sampling", "n_collocation"], ["n_collocation"]], 4096))
            n_data = int(_pick_first_present(schema, [["sampling", "n_data"], ["n_data"]], 0))
            replace = bool(_pick_first_present(schema, [["sampling", "replace"], ["replace"]], True))
            seed = int(_pick_first_present(schema, [["sampling", "seed"], ["seed"]], 0))
            sampling = SamplingSpec(
                n_collocation=n_collocation,
                n_data=n_data,
                replace=replace,
                seed=seed,
                conditions=[],
            )
        else:
            sampling = SamplingSpec(
                n_collocation=base_sampling.n_collocation,
                n_data=base_sampling.n_data,
                replace=base_sampling.replace,
                seed=base_sampling.seed,
                conditions=list(base_sampling.conditions),
            )

        # If base_sampling already has conditions, keep them and append schema-derived ones
        schema_specs: List[ConditionSpec] = []
        for c in conditions:
            ctype = (c.get("type") or "interior").lower().strip()
            n = c.get("n")
            n = int(n) if n is not None else 1024

            options = dict(c.get("options") or {})
            # convenience shorthands
            if c.get("axis") and "axis" not in options:
                options["axis"] = c["axis"]
            if c.get("coord") and "coord" not in options:
                options["coord"] = c["coord"]
            if c.get("value") is not None and "value" not in options:
                options["value"] = c["value"]

            schema_specs.append(
                ConditionSpec(
                    name=str(c.get("name", "cond")),
                    type=ctype,
                    equation=str(c.get("equation", "0")),
                    n=n,
                    options=options,
                )
            )

        sampling.conditions.extend(schema_specs)
        return sampling
