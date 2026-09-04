from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class LossWeights:
    w_pde: float = 1.0
    w_bc: float = 10.0
    w_ic: float = 10.0
    w_data: float = 1.0


@dataclass
class AdaptiveWeights:
    """Self-normalising weights for a multi-term PINN loss, keyed by
    arbitrary term names (unlike :class:`LossWeights`'s four fixed
    ``w_pde``/``w_bc``/``w_ic``/``w_data`` fields, which ``compile_problem``
    reads directly by attribute and is unaffected by this addition -- this
    class is a standalone utility for training loops that manage their own
    named loss-term dict, e.g. from ``compile_problem``'s own per-term
    output keys (``"pde"``, ``"bc_<name>"``, ``"ic_<name>"``,
    ``"data_<name>"``) or from a hand-rolled residual/loss set entirely).

    Tracks a lagged exponential moving average of each term's *raw* loss
    value and rescales that term's configured base weight *relative to the
    currently hardest (largest-EMA) term*: every term's multiplier is
    ``max_j(ema_j) / (ema_i + eps)``, clamped to ``[1, max_ratio]``.
    "Lagged" matters -- the multiplier used for step *n* is computed from
    the EMA *before* step *n*'s losses are folded in via :meth:`update`, so
    the weights are plain Python floats with no gradient path of their own.

    Why relative-to-hardest and not a plain per-term ``base / (ema + eps)``
    inverse magnitude (a natural first attempt): a PDE with a trivial exact
    solution is a real trap for the naive version. Incompressible
    Navier-Stokes with no-slip walls, for instance, has ``U ≡ 0, p ≡ const``
    as an exact solution of the momentum/continuity residual and the wall
    BC simultaneously -- if the network drifts anywhere near that
    degenerate field, a physics/BC term's EMA collapses toward zero, its
    *per-term* inverse weight blows up to the configured cap, and any
    data-fit term is left floating down near the cap's reciprocal. The
    optimiser is then explicitly rewarded for finishing the collapse to the
    trivial solution and ignoring the data entirely (observed in practice:
    a per-term-inverse-weighted run on exactly this problem converged to a
    uniformly-zero velocity field instead of the real, highly
    non-uniform mean profile). Rescaling relative to the *hardest* term
    instead of each term's own history removes the failure mode: a term's
    multiplier can only ever be pulled *up* to match the hardest term,
    never pulled *down* below its base weight, so no term can be abandoned
    just because it became easy (possibly trivially, possibly
    degenerately) to satisfy.

    Examples
    --------
    >>> aw = AdaptiveWeights(base_weights={"pde": 1.0, "bc": 10.0, "data": 5.0})
    >>> for step in range(n_steps):
    ...     w = aw.current()
    ...     loss = sum(w[k] * raw_losses[k] for k in raw_losses)
    ...     loss.backward(); opt.step()
    ...     aw.update({k: float(v.item()) for k, v in raw_losses.items()})
    """

    base_weights: Dict[str, float]
    momentum: float = 0.9
    eps: float = 1e-6
    max_ratio: float = 20.0
    enabled: bool = True
    _ema: Dict[str, Optional[float]] = field(default_factory=dict)

    def __post_init__(self):
        self._ema = {k: None for k in self.base_weights}

    def current(self) -> Dict[str, float]:
        """Effective weights for *this* step, from the EMA state left by
        the previous step's :meth:`update`."""
        if not self.enabled:
            return dict(self.base_weights)
        emas = {k: v for k, v in self._ema.items() if v is not None}
        if not emas:
            return dict(self.base_weights)
        hardest = max(emas.values())
        out = {}
        for name, base in self.base_weights.items():
            ema = self._ema.get(name)
            if ema is None:
                out[name] = base
                continue
            multiplier = hardest / (ema + self.eps)
            multiplier = min(max(multiplier, 1.0), self.max_ratio)
            out[name] = base * multiplier
        return out

    def update(self, raw_values: Dict[str, float]) -> None:
        """Fold this step's *detached* raw loss values into the EMA, for
        the *next* step's :meth:`current`."""
        if not self.enabled:
            return
        for name, val in raw_values.items():
            if name not in self.base_weights:
                continue
            prev = self._ema.get(name)
            self._ema[name] = val if prev is None else self.momentum * prev + (1 - self.momentum) * val

    def state_dict(self) -> Dict[str, float]:
        return dict(self._ema)
