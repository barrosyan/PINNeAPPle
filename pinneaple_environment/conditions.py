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


# Convenience typed constructors
def DirichletBC(
    name: str,
    fields: FieldNames,
    selector_type: SelectorType = "all",
    selector: Optional[Union[Dict[str, Any], Callable[[np.ndarray, Dict[str, Any]], np.ndarray]]] = None,
    value_fn: Optional[Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = None,
    weight: float = 1.0,
) -> ConditionSpec:
    return ConditionSpec(
        name=name,
        kind="dirichlet",
        fields=fields,
        selector_type=selector_type,
        selector=selector,
        value_fn=value_fn,
        weight=weight,
    )


def NeumannBC(
    name: str,
    fields: FieldNames,
    selector_type: SelectorType = "all",
    selector: Optional[Union[Dict[str, Any], Callable[[np.ndarray, Dict[str, Any]], np.ndarray]]] = None,
    value_fn: Optional[Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = None,
    weight: float = 1.0,
) -> ConditionSpec:
    return ConditionSpec(
        name=name,
        kind="neumann",
        fields=fields,
        selector_type=selector_type,
        selector=selector,
        value_fn=value_fn,
        weight=weight,
    )


def RobinBC(
    name: str,
    fields: FieldNames,
    selector_type: SelectorType = "all",
    selector: Optional[Union[Dict[str, Any], Callable[[np.ndarray, Dict[str, Any]], np.ndarray]]] = None,
    value_fn: Optional[Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = None,
    weight: float = 1.0,
) -> ConditionSpec:
    return ConditionSpec(
        name=name,
        kind="robin",
        fields=fields,
        selector_type=selector_type,
        selector=selector,
        value_fn=value_fn,
        weight=weight,
    )


def InitialCondition(
    name: str,
    fields: FieldNames,
    selector_type: SelectorType = "all",
    selector: Optional[Union[Dict[str, Any], Callable[[np.ndarray, Dict[str, Any]], np.ndarray]]] = None,
    value_fn: Optional[Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = None,
    weight: float = 1.0,
) -> ConditionSpec:
    return ConditionSpec(
        name=name,
        kind="initial",
        fields=fields,
        selector_type=selector_type,
        selector=selector,
        value_fn=value_fn,
        weight=weight,
    )


def DataConstraint(
    name: str,
    fields: FieldNames,
    selector_type: SelectorType = "all",
    selector: Optional[Union[Dict[str, Any], Callable[[np.ndarray, Dict[str, Any]], np.ndarray]]] = None,
    value_fn: Optional[Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = None,
    weight: float = 1.0,
) -> ConditionSpec:
    return ConditionSpec(
        name=name,
        kind="data",
        fields=fields,
        selector_type=selector_type,
        selector=selector,
        value_fn=value_fn,
        weight=weight,
    )