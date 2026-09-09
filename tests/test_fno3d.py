"""FNO3d correctness tests.

The key correctness property exploited here: a Fourier spectral convolution
(even truncated to a subset of low-frequency modes) is a pure circular
convolution, which is mathematically EXACT shift-equivariant under a
circular (``torch.roll``) shift along any spatial axis -- a shift only
multiplies each retained frequency's coefficient by a phase factor, which
commutes with the layer's per-frequency linear weight and with the
low-pass truncation mask. This holds regardless of how many modes are
kept, so it's an implementation-independent invariant: if
``_SpectralConv3d``'s 4-corner low-frequency indexing (the generalisation
from FNO2d's 2 corners to FNO3d's 4, needed because FNO3d has two
full-spectrum axes instead of one) has a bug -- wrong axis, mismatched
positive/negative corner pairing, an unpopulated corner -- shift
equivariance breaks for shifts along the affected axis. This is a much
stronger, more targeted check than a generic "doesn't crash" smoke test.
"""
from __future__ import annotations

import torch

from pinneapple_neural.architectures.neural_operators.fno import (
    _SpectralConv3d,
    FNO3d,
)
from pinneapple_neural.architectures.neural_operators.registry import NeuralOperatorCatalog


def test_spectral_conv3d_shape():
    conv = _SpectralConv3d(in_c=3, out_c=5, modes1=4, modes2=4, modes3=4)
    x = torch.randn(2, 3, 8, 10, 12)
    y = conv(x)
    assert y.shape == (2, 5, 8, 10, 12)
    assert torch.isfinite(y).all()


def test_spectral_conv3d_is_circular_shift_equivariant():
    torch.manual_seed(0)
    conv = _SpectralConv3d(in_c=2, out_c=2, modes1=3, modes2=3, modes3=3)
    x = torch.randn(1, 2, 8, 8, 8)
    y = conv(x)

    for dim, shift in [(2, 3), (3, 5), (4, 2)]:
        x_shifted = torch.roll(x, shifts=shift, dims=dim)
        y_from_shifted_input = conv(x_shifted)
        y_shifted_expected = torch.roll(y, shifts=shift, dims=dim)
        max_err = (y_from_shifted_input - y_shifted_expected).abs().max().item()
        assert max_err < 1e-4, f"shift-equivariance broken along dim={dim}: max_err={max_err}"


def test_spectral_conv3d_all_four_corners_receive_gradient():
    """A bug that leaves one of the 4 low-frequency corners unpopulated
    (e.g. a copy-paste error reusing weights1's slice for weights3/4) would
    silently give that weight tensor zero gradient -- catch it directly.

    Uses a random-weighted sum (Parseval-sensitive to every retained
    frequency, not just DC) rather than a plain ``.sum()``: a plain sum of
    a real periodic field equals exactly its own DC/zero-frequency
    coefficient (``sum_n y[n] = Y[0]``), so weights populating any
    non-zero-frequency corner (weights2/3/4 here -- only weights1's corner
    contains the joint index (0,0,0)) would show a spuriously-zero
    gradient under a plain-sum loss even with fully correct code; that's a
    property of the loss, not a bug. Confirmed by hand before fixing this
    test: a plain-sum loss did make weights2/3/4 look like they had zero
    gradient here, which is the expected, correct behaviour for that
    (badly-chosen) loss -- not a real bug."""
    torch.manual_seed(0)
    conv = _SpectralConv3d(in_c=2, out_c=2, modes1=3, modes2=3, modes3=3)
    x = torch.randn(2, 2, 8, 8, 8, requires_grad=True)
    y = conv(x)
    weight_mask = torch.randn_like(y)
    (y * weight_mask).sum().backward()
    for name in ("weights1", "weights2", "weights3", "weights4"):
        g = getattr(conv, name).grad
        assert g is not None, f"{name} received no gradient"
        assert g.abs().sum().item() > 0, f"{name}'s gradient is all zero"


def test_fno3d_forward_shape_and_grid():
    torch.manual_seed(0)
    net = FNO3d(in_channels=4, out_channels=4, width=8, modes1=3, modes2=3, modes3=3, layers=2, use_grid=True)
    x = torch.randn(2, 4, 6, 8, 10)
    out = net(x)
    assert out.y.shape == (2, 4, 6, 8, 10)
    assert torch.isfinite(out.y).all()


def test_fno3d_without_grid():
    torch.manual_seed(0)
    net = FNO3d(in_channels=1, out_channels=1, width=8, modes1=2, modes2=2, modes3=2, layers=1, use_grid=False)
    x = torch.randn(1, 1, 6, 6, 6)
    out = net(x)
    assert out.y.shape == (1, 1, 6, 6, 6)


def test_fno3d_return_loss():
    torch.manual_seed(0)
    net = FNO3d(in_channels=1, out_channels=1, width=8, modes1=2, modes2=2, modes3=2, layers=1)
    x = torch.randn(1, 1, 6, 6, 6)
    y_true = torch.randn(1, 1, 6, 6, 6)
    out = net(x, y_true=y_true, return_loss=True)
    assert "mse" in out.losses and "total" in out.losses
    assert torch.isfinite(out.losses["total"])


def test_fno3d_gradients_reach_input_projection():
    torch.manual_seed(0)
    net = FNO3d(in_channels=2, out_channels=2, width=8, modes1=2, modes2=2, modes3=2, layers=2)
    x = torch.randn(1, 2, 6, 6, 6)
    out = net(x)
    out.y.sum().backward()
    grad_norm = sum(p.grad.abs().sum().item() for p in net.parameters() if p.grad is not None)
    assert grad_norm > 0


def test_fno3d_registered_in_neural_operator_catalog():
    catalog = NeuralOperatorCatalog()
    assert "fno3d" in catalog.list()
    model = catalog.build("fno3d", in_channels=1, out_channels=1, width=8, modes1=2, modes2=2, modes3=2, layers=1)
    x = torch.randn(1, 1, 6, 6, 6)
    out = model(x)
    assert out.y.shape == (1, 1, 6, 6, 6)
