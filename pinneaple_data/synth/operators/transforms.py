"""Tensor normalization transforms for synth operators."""
import torch


def normalize_01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize a tensor to the [0, 1] range.

    The function rescales the input tensor linearly using its
    global minimum and maximum values. A small epsilon is used
    to avoid division by zero when the tensor has constant values.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to normalize.
    eps : float, optional
        Minimum denominator value to prevent division by zero.
        Default is 1e-8.

    Returns
    -------
    torch.Tensor
        Tensor scaled to the [0, 1] interval.
    """
    mn = x.amin()
    mx = x.amax()
    return (x - mn) / (mx - mn).clamp_min(eps)