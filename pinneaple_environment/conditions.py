from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union, Literal

import numpy as np

from .typing import FieldNames

SelectorType = Literal["all", "tag", "callable"]


@dataclass(frozen=True)
class ConditionSpec:
    """
    Generic constraint spec.

    kind:
      - "dirichlet" -> u(x)=g(x)
      - "neumann"   -> n·∇u(x)=g(x)
      - "robin"     -> a u + b n·∇u = g
      - "initial"   -> u(x,t0)=g(x)
      - "data"      -> supervised constraint at points

    selector:
      - "all": applies to all points of corresponding set
      - "tag": applies to points with ctx["tag_masks"][tag]==True
      - "callable": selector(X, ctx)->bool mask

    value_fn:
      - callable returning values for selected points
    """
    name: str
    kind: str
    fields: FieldNames
    selector_type: SelectorType = "all"
    selector: Optional[Union[Dict[str, Any], Callable[[np.ndarray, Dict[str, Any]], np.ndarray]]] = None
    value_fn: Optional[Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = None
    weight: float = 1.0

    def mask(self, X: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        if self.selector_type == "all":
            return np.ones((X.shape[0],), dtype=bool)

        if self.selector_type == "tag":
            if not isinstance(self.selector, dict) or "tag" not in self.selector:
                raise ValueError(f"Condition '{self.name}' selector_type='tag' requires selector={{'tag':...}}")
            tag = self.selector["tag"]
            tag_masks = ctx.get("tag_masks", {})
            if tag not in tag_masks:
                return np.zeros((X.shape[0],), dtype=bool)
            m = np.asarray(tag_masks[tag], dtype=bool)
            if m.shape[0] != X.shape[0]:
                # If tag masks provided correspond to boundary set only, user must ensure X matches.
                raise ValueError(f"Tag mask '{tag}' shape mismatch: {m.shape} vs X={X.shape}")
            return m

        if self.selector_type == "callable":
            if not callable(self.selector):
                raise ValueError(f"Condition '{self.name}' selector_type='callable' requires callable selector")
            m = self.selector(X, ctx)
            return np.asarray(m, dtype=bool)

        raise ValueError(f"Unknown selector_type: {self.selector_type}")

    def values(self, X: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        if self.value_fn is None:
            # default zeros
            return np.zeros((X.shape[0], len(self.fields)), dtype=np.float32)
        v = self.value_fn(X, ctx)
        v = np.asarray(v, dtype=np.float32)
        if v.ndim == 1:
            v = v[:, None]
        return v


# ---------------------------------------------------------------------------
# Internal helper: build a value_fn from a dict of {field: scalar} values.
# ---------------------------------------------------------------------------

def _value_fn_from_dict(values: Dict[str, float]) -> Callable[[np.ndarray, Dict[str, Any]], np.ndarray]:
    """Return a value_fn that outputs a constant array from a {field: value} dict."""
    vals = list(values.values())
    n_fields = len(vals)
    def _fn(X: np.ndarray, ctx: Dict[str, Any], _vals=vals, _nf=n_fields) -> np.ndarray:
        out = np.zeros((X.shape[0], _nf), dtype=np.float32)
        for i, v in enumerate(_vals):
            out[:, i] = float(v)
        return out
    return _fn


# Convenience typed constructors.
#
# These support two call signatures:
#   1. Simple dict form (paper-style):
#      DirichletBC({"u": 0.0})
#      DirichletBC({"u": 0.0, "v": 0.0})
#
#   2. Full positional/keyword form (internal builder):
#      DirichletBC(name, fields, selector_type, selector, value_fn, weight)


def DirichletBC(
    name: Union[str, Dict[str, float]],
    fields: Optional[FieldNames] = None,
    selector_type: SelectorType = "all",
    selector: Optional[Union[Dict[str, Any], Callable[[np.ndarray, Dict[str, Any]], np.ndarray]]] = None,
    value_fn: Optional[Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = None,
    weight: float = 1.0,
) -> ConditionSpec:
    """Construct a Dirichlet boundary condition.

    Simple dict form (paper-style)::

        DirichletBC({"u": 0.0})          # zero Dirichlet for field u
        DirichletBC({"u": 0.0, "v": 0.0})

    Full form::

        DirichletBC("wall", ("u",), "all", None, value_fn, 10.0)
    """
    if isinstance(name, dict):
        values = name
        _fields: FieldNames = tuple(values.keys())
        return ConditionSpec(
            name="dirichlet_bc",
            kind="dirichlet",
            fields=_fields,
            selector_type="all",
            selector=None,
            value_fn=_value_fn_from_dict(values),
            weight=weight,
        )
    return ConditionSpec(
        name=name,
        kind="dirichlet",
        fields=fields or (),
        selector_type=selector_type,
        selector=selector,
        value_fn=value_fn,
        weight=weight,
    )


def NeumannBC(
    name: Union[str, Dict[str, float]],
    fields: Optional[FieldNames] = None,
    selector_type: SelectorType = "all",
    selector: Optional[Union[Dict[str, Any], Callable[[np.ndarray, Dict[str, Any]], np.ndarray]]] = None,
    value_fn: Optional[Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = None,
    weight: float = 1.0,
) -> ConditionSpec:
    """Construct a Neumann boundary condition.

    Simple dict form::

        NeumannBC({"u": 1.0})
    """
    if isinstance(name, dict):
        values = name
        _fields: FieldNames = tuple(values.keys())
        return ConditionSpec(
            name="neumann_bc",
            kind="neumann",
            fields=_fields,
            selector_type="all",
            selector=None,
            value_fn=_value_fn_from_dict(values),
            weight=weight,
        )
    return ConditionSpec(
        name=name,
        kind="neumann",
        fields=fields or (),
        selector_type=selector_type,
        selector=selector,
        value_fn=value_fn,
        weight=weight,
    )


def RobinBC(
    name: Union[str, Dict[str, float]],
    fields: Optional[FieldNames] = None,
    selector_type: SelectorType = "all",
    selector: Optional[Union[Dict[str, Any], Callable[[np.ndarray, Dict[str, Any]], np.ndarray]]] = None,
    value_fn: Optional[Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = None,
    weight: float = 1.0,
) -> ConditionSpec:
    """Construct a Robin boundary condition."""
    if isinstance(name, dict):
        values = name
        _fields: FieldNames = tuple(values.keys())
        return ConditionSpec(
            name="robin_bc",
            kind="robin",
            fields=_fields,
            selector_type="all",
            selector=None,
            value_fn=_value_fn_from_dict(values),
            weight=weight,
        )
    return ConditionSpec(
        name=name,
        kind="robin",
        fields=fields or (),
        selector_type=selector_type,
        selector=selector,
        value_fn=value_fn,
        weight=weight,
    )


def InitialCondition(
    name: Union[str, Dict[str, float]],
    fields: Optional[FieldNames] = None,
    selector_type: SelectorType = "all",
    selector: Optional[Union[Dict[str, Any], Callable[[np.ndarray, Dict[str, Any]], np.ndarray]]] = None,
    value_fn: Optional[Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = None,
    weight: float = 1.0,
) -> ConditionSpec:
    """Construct an initial condition.

    Simple dict form::

        InitialCondition({"u": 1.0})
    """
    if isinstance(name, dict):
        values = name
        _fields: FieldNames = tuple(values.keys())
        return ConditionSpec(
            name="initial_condition",
            kind="initial",
            fields=_fields,
            selector_type="all",
            selector=None,
            value_fn=_value_fn_from_dict(values),
            weight=weight,
        )
    return ConditionSpec(
        name=name,
        kind="initial",
        fields=fields or (),
        selector_type=selector_type,
        selector=selector,
        value_fn=value_fn,
        weight=weight,
    )


def DataConstraint(
    name: Union[str, Dict[str, float]],
    fields: Optional[FieldNames] = None,
    selector_type: SelectorType = "all",
    selector: Optional[Union[Dict[str, Any], Callable[[np.ndarray, Dict[str, Any]], np.ndarray]]] = None,
    value_fn: Optional[Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = None,
    weight: float = 1.0,
) -> ConditionSpec:
    """Construct a data constraint (supervised loss at specific points)."""
    if isinstance(name, dict):
        values = name
        _fields: FieldNames = tuple(values.keys())
        return ConditionSpec(
            name="data_constraint",
            kind="data",
            fields=_fields,
            selector_type="all",
            selector=None,
            value_fn=_value_fn_from_dict(values),
            weight=weight,
        )
    return ConditionSpec(
        name=name,
        kind="data",
        fields=fields or (),
        selector_type=selector_type,
        selector=selector,
        value_fn=value_fn,
        weight=weight,
    )