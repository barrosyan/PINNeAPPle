from __future__ import annotations
import torch
import torch.nn as nn

from .base import NeuralOperatorBase, OperatorOutput


class GalerkinNeuralOperator(NeuralOperatorBase):
    """
    Galerkin Neural Operator (mesh-based, 1D/2D/3D):
      - Learn a basis Phi(x) from coords (x in R^d, d can be 1,2,3)
      - Project u -> coefficients using quadrature/lumped-mass weights
      - Apply operator in coefficient space
      - Reconstruct y = Phi * c_out

    Inputs
    ------
    u:      (B,N,in_dim) or (N,in_dim)
    coords: (B,N,d)      or (N,d)   where d in {1,2,3} (also works for any d>=1)
    w:      (B,N) or (N,) optional node weights (areas/volumes). If None -> uniform.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        basis_dim: int = 64,
        basis_hidden: int = 128,
        op_hidden: int = 128,
        op_depth: int = 2,
        orthonormalize_basis: bool = True,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.basis_dim = basis_dim
        self.orthonormalize_basis = orthonormalize_basis
        self.eps = eps

        # coords -> Phi(x) (Lazy so coord_dim can be 1D/2D/3D)
        self.basis_net = nn.Sequential(
            nn.LazyLinear(basis_hidden),
            nn.GELU(),
            nn.Linear(basis_hidden, basis_hidden),
            nn.GELU(),
            nn.Linear(basis_hidden, basis_dim),
        )

        # Optional pointwise encoding of input channels
        self.u_encoder = nn.Linear(in_dim, in_dim)

        # Coefficient operator: (basis_dim*in_dim) -> (basis_dim*out_dim)
        in_c = basis_dim * in_dim
        out_c = basis_dim * out_dim

        layers = [nn.Linear(in_c, op_hidden), nn.GELU()]
        for _ in range(max(op_depth - 1, 0)):
            layers += [nn.Linear(op_hidden, op_hidden), nn.GELU()]
        layers += [nn.Linear(op_hidden, out_c)]
        self.coeff_operator = nn.Sequential(*layers)

    def _ensure_batch(self, u, coords, w):
        batched = (u.dim() == 3)
        if not batched:
            u = u.unsqueeze(0)
            coords = coords.unsqueeze(0)
            if w is not None:
                w = w.unsqueeze(0)
        return batched, u, coords, w

    def _make_weights(self, B, N, device, dtype, w):
        if w is None:
            # uniform weights => mean-like quadrature
            return torch.full((B, N), 1.0 / float(N), device=device, dtype=dtype)
        return w

    def _build_basis(self, coords, w):
        # coords: (B,N,d)
        Phi = self.basis_net(coords)  # (B,N,r)

        if not self.orthonormalize_basis:
            return Phi

        # Weighted QR orthonormalization (approx. orthonormal wrt W)
        # Phi_w = sqrt(w) * Phi; QR gives Q with Q^T Q = I in weighted sense approximately.
        sqrtw = torch.sqrt(w.clamp_min(self.eps)).unsqueeze(-1)  # (B,N,1)
        Phi_w = Phi * sqrtw  # (B,N,r)

        # QR per batch element
        Qs = []
        for b in range(Phi_w.shape[0]):
            Q, _ = torch.linalg.qr(Phi_w[b], mode="reduced")  # (N,r)
            Qs.append(Q)
        return torch.stack(Qs, dim=0)  # (B,N,r)

    def forward(self, u, *, coords, w=None, y_true=None, return_loss=False):
        batched, u, coords, w = self._ensure_batch(u, coords, w)
        B, N, _ = u.shape

        w = self._make_weights(B, N, u.device, u.dtype, w)  # (B,N)

        # 1) Basis Phi(x)
        Phi = self._build_basis(coords, w)  # (B,N,r)

        # 2) Encode u (optional but ok)
        u_enc = self.u_encoder(u)  # (B,N,in_dim)

        # 3) Project: c = Phi^T (W u)
        u_w = u_enc * w.unsqueeze(-1)  # (B,N,in_dim)
        c = torch.einsum("bnr,bni->bri", Phi, u_w)  # (B,r,in_dim)

        # 4) Operator on coefficients
        c_flat = c.reshape(B, self.basis_dim * self.in_dim)
        c_out_flat = self.coeff_operator(c_flat)
        c_out = c_out_flat.reshape(B, self.basis_dim, self.out_dim)  # (B,r,out_dim)

        # 5) Reconstruct: y = Phi c_out
        y = torch.einsum("bnr,bro->bno", Phi, c_out)  # (B,N,out_dim)

        losses = {"total": torch.tensor(0.0, device=y.device)}
        if return_loss and y_true is not None:
            if y_true.dim() == 2:
                y_true = y_true.unsqueeze(0)
            losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]

        extras = {
            "Phi": Phi,
            "coeffs_in": c,
            "coeffs_out": c_out,
            "w": w,
            "coord_dim": coords.shape[-1],
        }

        if not batched:
            y = y.squeeze(0)

        return OperatorOutput(y=y, losses=losses, extras=extras)