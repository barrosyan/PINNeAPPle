"""Gaussian noise addition for synthetic data augmentation."""
import torch


def add_gaussian_noise(x: torch.Tensor, std: float = 0.01, seed: int = 42) -> torch.Tensor:
    """
    Add Gaussian noise to a tensor.

    Noise is sampled from a normal distribution with mean 0 and
    standard deviation `std`, using a deterministic CPU generator
    for reproducibility.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to which noise will be added.
    std : float, optional
        Standard deviation of the Gaussian noise. Default is 0.01.
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape as `x` with additive Gaussian noise.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return x + std * torch.randn_like(x, generator=g)