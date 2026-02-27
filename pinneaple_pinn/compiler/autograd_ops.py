from __future__ import annotations

import torch


def ensure_tensor(y):
    if hasattr(y, "y"):
        return y.y
    return y


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]


def jacobian(Y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    assert Y.ndim == 2
    outs = []
    for j in range(Y.shape[1]):
        g = torch.autograd.grad(
            outputs=Y[:, j:j + 1],
            inputs=x,
            grad_outputs=torch.ones_like(Y[:, j:j + 1]),
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        outs.append(g[:, None, :])
    return torch.cat(outs, dim=1)


def divergence(V: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    assert V.shape[1] == x.shape[1]
    J = jacobian(V, x)
    div = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
    for i in range(x.shape[1]):
        div = div + J[:, i:i + 1, i:i + 1].reshape(-1, 1)
    return div


def laplacian(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    g = grad(y, x)
    out = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
    for i in range(x.shape[1]):
        gi = g[:, i:i + 1]
        gii = torch.autograd.grad(
            outputs=gi,
            inputs=x,
            grad_outputs=torch.ones_like(gi),
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0][:, i:i + 1]
        out = out + gii
    return out


def time_derivative(y: torch.Tensor, x: torch.Tensor, t_index: int) -> torch.Tensor:
    g = grad(y, x)
    return g[:, t_index:t_index + 1]


def norm_dot_grad(y: torch.Tensor, x: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    g = grad(y, x)
    return torch.sum(g * normals, dim=1, keepdim=True)


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a - b) ** 2)