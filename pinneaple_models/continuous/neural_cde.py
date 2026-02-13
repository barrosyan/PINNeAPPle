from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import ContinuousModelBase, ContOutput


class NeuralCDE(ContinuousModelBase):
    """
    Neural CDE (literature-style MVP, no torchcde):

    Canonical form:
        dh_t = F(h_t, t) dX_t
    where X_t is the control path. To include drift, we *augment the control*
    with time as an extra channel:
        X̃_t = [t, x_t]  ->  dX̃ = [dt, dx]

    With piecewise-linear interpolation of X̃ and Euler discretization:
        h_{i+1} = h_i + (F(h_i, t_i) @ dX̃_i)

    Inputs:
      x: (B, T, input_dim) control observations
      t: (T,) strictly increasing
    Output:
      y_hat: (B, T, out_dim) at same timestamps
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        f_hidden: int = 128,   # kept for API stability; used as width for F net
        g_hidden: int = 128,   # kept for API stability; used as width for F net
        num_layers: int = 2,
        activation: str = "tanh",
        solver: str = "euler",               # optional hook for future
        check_strict_t: bool = True,          # validation as requested
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)

        self.solver = str(solver).lower()
        self.check_strict_t = bool(check_strict_t)

        act = (activation or "tanh").lower()
        act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}.get(act, nn.Tanh)

        # initial hidden from first observation (not time-augmented on purpose; standard/simple)
        self.h0_net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            act_fn(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # We augment control with time: X̃ has D+1 channels => F outputs (H, D+1)
        self.aug_dim = self.input_dim + 1

        # Build F: (h, t) -> (hidden_dim * (aug_dim))
        # Keep f_hidden/g_hidden params but just pick a reasonable width (max of them)
        width = int(max(f_hidden, g_hidden))

        layers = [nn.Linear(self.hidden_dim + 1, width), act_fn()]
        for _ in range(int(num_layers) - 1):
            layers += [nn.Linear(width, width), act_fn()]
        layers += [nn.Linear(width, self.hidden_dim * self.aug_dim)]
        self.F = nn.Sequential(*layers)

        self.readout = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def _Ft(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns F(h,t) reshaped as (B, H, D+1).
        """
        # h: (B,H), t: scalar or (B,) or (B,1)
        if t.ndim == 0:
            tt = t.view(1, 1).expand(h.shape[0], 1)
        elif t.ndim == 1:
            tt = t[:, None]
        else:
            tt = t
        inp = torch.cat([h, tt.to(h.device, h.dtype)], dim=-1)
        out = self.F(inp)  # (B, H*(D+1))
        return out.view(h.shape[0], self.hidden_dim, self.aug_dim)

    @staticmethod
    def _assert_strictly_increasing(t: torch.Tensor) -> None:
        if t.ndim != 1:
            raise ValueError(f"t must be 1D (T,), got shape {tuple(t.shape)}")
        if t.numel() < 2:
            return
        # strict monotonicity
        if not torch.all(t[1:] > t[:-1]):
            raise ValueError("t must be strictly increasing (t[i+1] > t[i] for all i).")

    def forward(
        self,
        x: torch.Tensor,            # (B,T,input_dim)
        t: torch.Tensor,            # (T,)
        *,
        y_true: Optional[torch.Tensor] = None,  # (B,T,out_dim)
        return_loss: bool = False,
    ) -> ContOutput:
        B, T, D = x.shape
        if D != self.input_dim:
            raise ValueError(f"Expected x.shape[-1]=={self.input_dim}, got {D}")
        if t.numel() != T:
            raise ValueError(f"Expected t length {T}, got {t.numel()}")

        if self.check_strict_t:
            self._assert_strictly_increasing(t)

        # init
        h = self.h0_net(x[:, 0, :])  # (B,H)
        ys = [self.readout(h)]

        # Euler over piecewise-linear control (time-augmented)
        if self.solver not in ("euler",):
            raise ValueError(f"Unsupported solver='{self.solver}'. Currently only 'euler' is implemented.")

        for i in range(T - 1):
            dt = (t[i + 1] - t[i]).to(dtype=h.dtype, device=h.device)  # scalar
            dx = (x[:, i + 1, :] - x[:, i, :])                          # (B,D)

            # dX̃ = [dt, dx]
            dX_aug = torch.cat(
                [dt.expand(B, 1), dx.to(dtype=h.dtype, device=h.device)],
                dim=-1,
            )  # (B, D+1)

            Fval = self._Ft(h, t[i])  # (B,H,D+1)
            dh = torch.einsum("bhd,bd->bh", Fval, dX_aug)  # (B,H)

            h = h + dh
            ys.append(self.readout(h))

        y_hat = torch.stack(ys, dim=1)  # (B,T,out_dim)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=x.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y_hat, y_true)
            losses["total"] = losses["mse"]

        return ContOutput(y=y_hat, losses=losses, extras={"h_last": h, "solver": self.solver})
