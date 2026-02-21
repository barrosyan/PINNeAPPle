"""Random block mask generation for 2D grids."""
import torch


def random_block_mask(H: int, W: int, frac: float = 0.2, seed: int = 42) -> torch.Tensor:
    """
    Generate a random rectangular block mask within a 2D grid.

    The mask contains a single contiguous rectangular region of True values,
    occupying approximately `frac` of the total height and width. The block
    location is randomly selected using a deterministic CPU generator.

    Parameters
    ----------
    H : int
        Height of the grid.
    W : int
        Width of the grid.
    frac : float, optional
        Fraction of the grid size used to determine the block height and width.
        Default is 0.2.
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape (H, W), where the selected block region is True
        and the remaining entries are False.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    m = torch.zeros((H, W), dtype=torch.bool)
    h = max(1, int(H * frac))
    w = max(1, int(W * frac))
    i = int(torch.randint(0, max(1, H - h), (1,), generator=g).item())
    j = int(torch.randint(0, max(1, W - w), (1,), generator=g).item())
    m[i:i + h, j:j + w] = True
    return m