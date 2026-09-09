"""Tests for pinneapple_neural.trainer.lbfgs_finetune.multi_round_lbfgs."""
from __future__ import annotations

import torch
import torch.nn as nn

from pinneapple_neural.trainer.lbfgs_finetune import LBFGSFinetuneResult, multi_round_lbfgs


def test_single_round_matches_plain_lbfgs():
    """n_rounds=1 must be a pure passthrough to a single torch.optim.LBFGS run."""
    torch.manual_seed(0)
    target = torch.tensor([3.0, -2.0])

    def make_closure_builder(param):
        def closure_builder():
            def closure():
                param.grad = None
                loss = ((param - target) ** 2).sum()
                loss.backward()
                return loss
            return closure
        return closure_builder

    p_multi = nn.Parameter(torch.zeros(2))
    res = multi_round_lbfgs([p_multi], make_closure_builder(p_multi), n_rounds=1, n_steps_per_round=20)

    p_plain = nn.Parameter(torch.zeros(2))
    lbfgs = torch.optim.LBFGS([p_plain], lr=1.0, max_iter=20, history_size=50, line_search_fn="strong_wolfe")

    def plain_closure():
        p_plain.grad = None
        loss = ((p_plain - target) ** 2).sum()
        loss.backward()
        return loss

    lbfgs.step(plain_closure)

    # Both must converge to the actual target (torch.optim.LBFGS.step()'s
    # returned loss is the FIRST closure evaluation, i.e. the loss BEFORE
    # that step -- not a measure of final convergence -- so the meaningful
    # check is the parameter value itself, not the reported loss).
    assert torch.allclose(p_multi, target, atol=1e-4)
    assert torch.allclose(p_multi, p_plain, atol=1e-6)
    assert isinstance(res, LBFGSFinetuneResult)
    assert len(res.round_losses) == 1
    assert res.round_losses[0] == 13.0  # loss at the start of the (only) round: (0-3)^2+(0-(-2))^2


def test_multi_round_converges_a_toy_quadratic():
    torch.manual_seed(1)
    target = torch.tensor([1.5, -0.5, 2.0])
    param = nn.Parameter(torch.randn(3))

    def closure_builder():
        def closure():
            param.grad = None
            loss = ((param - target) ** 2).sum()
            loss.backward()
            return loss
        return closure

    res = multi_round_lbfgs([param], closure_builder, n_rounds=4, n_steps_per_round=10)
    assert len(res.round_losses) == 4
    assert len(res.round_iters) == 4
    assert res.final_loss < 1e-6
    assert torch.allclose(param, target, atol=1e-3)


def test_resampling_across_rounds_uses_fresh_batches():
    """Confirm closure_builder is called once PER ROUND (so a caller that
    resamples data inside it genuinely gets a fresh batch each round)."""
    torch.manual_seed(2)
    param = nn.Parameter(torch.zeros(1))
    build_calls = {"n": 0}

    def closure_builder():
        build_calls["n"] += 1
        target = torch.tensor([float(build_calls["n"])])  # a different target "batch" each round

        def closure():
            param.grad = None
            loss = ((param - target) ** 2).sum()
            loss.backward()
            return loss
        return closure

    multi_round_lbfgs([param], closure_builder, n_rounds=3, n_steps_per_round=5)
    assert build_calls["n"] == 3


def test_post_round_eval_overrides_reported_loss():
    torch.manual_seed(3)
    param = nn.Parameter(torch.zeros(2))
    target = torch.tensor([1.0, 1.0])

    def closure_builder():
        def closure():
            param.grad = None
            loss = ((param - target) ** 2).sum()
            loss.backward()
            return loss
        return closure

    def post_eval():
        with torch.no_grad():
            return torch.tensor(12345.0)  # a deliberately distinctive sentinel value

    res = multi_round_lbfgs([param], closure_builder, n_rounds=2, n_steps_per_round=5,
                             post_round_eval=post_eval)
    assert all(loss == 12345.0 for loss in res.round_losses)


def test_on_round_end_callback_invoked_correctly():
    torch.manual_seed(4)
    param = nn.Parameter(torch.zeros(1))
    target = torch.tensor([2.0])
    seen = []

    def closure_builder():
        def closure():
            param.grad = None
            loss = ((param - target) ** 2).sum()
            loss.backward()
            return loss
        return closure

    def on_round_end(round_idx, loss, n_iters):
        seen.append((round_idx, n_iters))

    multi_round_lbfgs([param], closure_builder, n_rounds=3, n_steps_per_round=4, on_round_end=on_round_end)
    assert [r for r, _ in seen] == [0, 1, 2]
    assert all(n > 0 for _, n in seen)


def test_works_with_create_graph_residual_style_closure():
    """The real motivating use case: a closure whose loss requires
    torch.autograd.grad(..., create_graph=True) internally (a PDE
    residual), which must NOT be run under any implicit no-grad context."""
    torch.manual_seed(5)
    net = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))

    def closure_builder():
        x = torch.rand(16, 1, requires_grad=True)

        def closure():
            for p in net.parameters():
                p.grad = None
            y = net(x)
            dy_dx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
            residual = dy_dx - 2.0 * x  # target: y' = 2x  (y = x^2 + C)
            loss = (residual ** 2).mean()
            loss.backward()
            return loss
        return closure

    res = multi_round_lbfgs(list(net.parameters()), closure_builder, n_rounds=2, n_steps_per_round=15)
    assert len(res.round_losses) == 2
    assert all(torch.isfinite(torch.tensor(l)) for l in res.round_losses)
