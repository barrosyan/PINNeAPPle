"""Multi-round L-BFGS fine-tuning: resample-per-round quasi-Newton polish.

A single L-BFGS run on one fixed batch keeps improving right up to its
iteration budget -- promising, but risky: quasi-Newton steps on a single
fixed sample can overfit to that sample's particular noise instead of the
true objective (observed in practice on a PINN training run: a single
600-iteration/large-batch L-BFGS phase thrashed memory for days with zero
usable progress, traced to ``history_size=50``'s curvature-pair buffer
never resetting across one very long phase at a large batch). Splitting
the same total iteration budget into several rounds, each on a freshly
resampled (and optionally larger) batch, keeps the fast local convergence
L-BFGS is good at while forcing every few dozen steps to re-check
themselves against new points -- closer in spirit to stochastic
optimization, without giving up the second-order convergence rate within
each round -- and bounds ``history_size``'s memory growth to one round's
worth of curvature pairs instead of the whole budget's.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import torch


@dataclass
class LBFGSFinetuneResult:
    """Per-round bookkeeping from :func:`multi_round_lbfgs`."""
    round_losses: List[float] = field(default_factory=list)
    round_iters: List[int] = field(default_factory=list)

    @property
    def final_loss(self) -> float:
        return self.round_losses[-1] if self.round_losses else float("nan")


def multi_round_lbfgs(
    params: Sequence[torch.nn.Parameter],
    closure_builder: Callable[[], Callable[[], torch.Tensor]],
    n_rounds: int,
    n_steps_per_round: int,
    lr: float = 1.0,
    history_size: int = 50,
    line_search_fn: Optional[str] = "strong_wolfe",
    post_round_eval: Optional[Callable[[], torch.Tensor]] = None,
    on_round_end: Optional[Callable[[int, float, int], None]] = None,
) -> LBFGSFinetuneResult:
    """Run ``n_rounds`` independent :class:`torch.optim.LBFGS` phases, each
    ``max_iter=n_steps_per_round``, each against a freshly built closure
    (so a new mini-batch, if the caller's ``closure_builder`` resamples
    one) -- rather than one L-BFGS phase for the whole budget.

    ``n_rounds=1`` reproduces plain single-round L-BFGS exactly (a no-op
    wrapper in that case).

    Parameters
    ----------
    params : parameters to optimize (passed straight to ``torch.optim.LBFGS``).
    closure_builder : called once per round, returns a fresh ``closure()``
        callable (the standard PyTorch closure contract: zero grad, compute
        loss, ``loss.backward()``, return loss) -- this is where a caller
        resamples a new batch for the round, if desired. The returned
        closure is reused for every one of that round's ``n_steps_per_round``
        L-BFGS iterations (and any internal line-search evaluations),
        i.e. the batch is fixed *within* a round, fresh *across* rounds.
    n_steps_per_round : ``max_iter`` for each round's ``LBFGS``.
    post_round_eval : optional callable returning a scalar loss tensor,
        invoked after each round's ``lbfgs.step()`` completes, used for
        this round's reported/returned loss instead of ``lbfgs.step()``'s
        own return value. Use this if you need the reported loss computed
        under conditions ``lbfgs.step()``'s internal closure calls can't
        provide -- e.g. a PDE residual needing ``torch.autograd.grad(...,
        create_graph=True)``, which raises under the implicit no-grad
        context some closure call sites run in. If omitted, the loss
        ``lbfgs.step()`` itself returns is used.
    on_round_end : optional callback ``(round_idx, loss, n_iters) -> None``
        for progress logging.

    Returns
    -------
    :class:`LBFGSFinetuneResult` with one entry per round in
    ``round_losses``/``round_iters``.
    """
    result = LBFGSFinetuneResult()
    for round_idx in range(int(n_rounds)):
        closure = closure_builder()
        lbfgs = torch.optim.LBFGS(
            params, lr=lr, max_iter=int(n_steps_per_round),
            history_size=int(history_size), line_search_fn=line_search_fn,
        )
        step_count = {"n": 0}

        def _wrapped_closure(closure=closure, step_count=step_count):
            step_count["n"] += 1
            return closure()

        step_loss = lbfgs.step(_wrapped_closure)

        if post_round_eval is not None:
            loss_value = float(post_round_eval().item())
        elif step_loss is not None:
            loss_value = float(step_loss.item() if torch.is_tensor(step_loss) else step_loss)
        else:
            loss_value = float("nan")

        result.round_losses.append(loss_value)
        result.round_iters.append(step_count["n"])
        if on_round_end is not None:
            on_round_end(round_idx, loss_value, step_count["n"])

    return result
