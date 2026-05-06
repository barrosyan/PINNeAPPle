"""Variance decomposition: total predictive uncertainty = aleatoric + epistemic.

Theory
------
Given *S* stochastic forward passes of a model that returns ``(μ_i, log σ²_i)``:

  Predictive mean   μ̄  = E[μ_i]
  Aleatoric var        = E[σ²_i]       (average predicted data noise)
  Epistemic var        = Var[μ_i]      (spread of the predicted means)
  Total var            = aleatoric + epistemic

For purely deterministic models (ensemble / MC-dropout *without* a variance head),
only epistemic variance can be estimated; set ``has_aleatoric=False``.

Quick start::

    from pinneaple_uq import AleatoricHead, MCDropoutWrapper, MCDropoutConfig
    from pinneaple_uq import decompose_uncertainty

    # 1) Build a model that outputs (mean, log_var)
    aleatoric_model = AleatoricHead(my_base_model, out_dim=1)

    # 2) Wrap with MC Dropout to activate stochasticity
    mcd = MCDropoutWrapper(aleatoric_model, MCDropoutConfig(n_samples=50))

    # 3) Decompose
    result = decompose_uncertainty(mcd, x_test, has_aleatoric=True, device="cuda")
    print("aleatoric:", result.aleatoric_std.mean().item())
    print("epistemic:", result.epistemic_std.mean().item())
"""
from __future__ import annotations

from typing import Any, Optional, Union

import torch
from torch import Tensor

from .core import UQResult


def decompose_uncertainty(
    model: Any,
    x: Tensor,
    *,
    n_samples: int = 100,
    has_aleatoric: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> UQResult:
    """Decompose predictive uncertainty into aleatoric and epistemic parts.

    The *model* is called ``n_samples`` times in **training mode** (so dropout
    layers remain active). Each call should return either:

    * ``Tensor`` of shape ``(N, D)`` — point prediction (epistemic only).
    * ``(Tensor, Tensor)`` — ``(mean, log_var)`` (aleatoric + epistemic).

    Parameters
    ----------
    model : Any
        A callable (e.g., ``MCDropoutWrapper`` wrapping ``AleatoricHead``).
    x : Tensor — input, shape ``(N, ...)``.
    n_samples : int — number of stochastic forward passes.
    has_aleatoric : bool
        ``True`` when *model* returns ``(mean, log_var)`` tuples.
        ``False`` when it returns plain tensors (epistemic only).
    device : optional — moves model and data to this device.

    Returns
    -------
    UQResult
        ``mean`` : predictive mean ``μ̄``.
        ``std`` : total predictive std ``sqrt(aleatoric_var + epistemic_var)``.
        ``aleatoric_std`` : ``sqrt(E[σ²_i])``.
        ``epistemic_std`` : ``sqrt(Var[μ_i])``.
        ``samples`` : stacked mean samples ``(S, N, D)``.
    """
    if device is not None:
        device = torch.device(device) if isinstance(device, str) else device
        if hasattr(model, "to"):
            model.to(device)
        x = x.to(device)

    means_list: list[Tensor] = []
    vars_list: list[Tensor] = []

    model.train()  # keep dropout/stochastic layers active
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(x)
            if has_aleatoric and isinstance(out, (tuple, list)) and len(out) == 2:
                mu_i, lv_i = out
                means_list.append(mu_i)
                vars_list.append(torch.exp(lv_i))  # σ²_i
            else:
                if isinstance(out, (tuple, list)):
                    out = out[0]
                means_list.append(out)

    means = torch.stack(means_list, dim=0)          # (S, N, D)
    mu = means.mean(dim=0)                          # (N, D)
    epistemic_var = means.var(dim=0, unbiased=False)  # Var[μ_i]

    if has_aleatoric and vars_list:
        aleatoric_var = torch.stack(vars_list, dim=0).mean(dim=0)  # E[σ²_i]
    else:
        aleatoric_var = torch.zeros_like(epistemic_var)

    total_var = aleatoric_var + epistemic_var
    eps = torch.finfo(total_var.dtype).eps

    return UQResult(
        mean=mu,
        std=torch.sqrt(total_var + eps),
        aleatoric_std=torch.sqrt(aleatoric_var + eps),
        epistemic_std=torch.sqrt(epistemic_var + eps),
        samples=means,
        metadata={
            "method": "decomposition",
            "n_samples": n_samples,
            "has_aleatoric": has_aleatoric,
        },
    )
