from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from .base import AEBase, AEOutput
from .dense_ae import _mlp


def _as_dt(dt: Union[float, torch.Tensor], batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Normaliza dt para shape (B,).
    Aceita:
      - float
      - tensor escalar
      - tensor (B,)
    """
    if isinstance(dt, (float, int)):
        return torch.full((batch_size,), float(dt), device=device, dtype=dtype)

    dt_t = dt.to(device=device, dtype=dtype)
    if dt_t.ndim == 0:
        return dt_t.expand(batch_size)
    if dt_t.ndim == 1 and dt_t.shape[0] == batch_size:
        return dt_t
    raise ValueError(f"dt deve ser float, escalar, ou shape (B,), mas veio {tuple(dt_t.shape)}")


class AEROMHybrid(AEBase):
    """
    AE-ROM Hybrid (Koopman/latent linear dynamics) com:
      - AE denso (encoder/decoder)
      - Dinâmica contínua linear no latente: dz/dt = K z + c
      - Passo discreto via Euler: z_next_pred = z + dt*(K z + c)
      - Loss de previsão em x (decode do z_next_pred)
      - Rollout multi-step (latente e x)
      - Regularização de estabilidade (contratividade via parte simétrica de K)
      - Atualização de K,c por Least Squares (ridge)

    forward suporta:
      - forward(x)
      - forward(x, x_next=..., dt=...) adiciona losses 1-step
      - forward(x, x_seq=..., dt_seq=...) adiciona rollout multi-step

    x_seq: Tensor shape (B, K, input_dim) ou (B, K, ...) que será achatado para (B, K, input_dim)
    dt_seq: float / tensor escalar / tensor (B,) / tensor (B, K)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden: List[int] = (512, 256),
        activation: str = "gelu",
        # pesos de loss
        rom_weight: float = 1.0,          # loss no latente (1-step)
        pred_weight: float = 1.0,         # loss no espaço original (1-step)
        rollout_latent_weight: float = 0.5,  # loss latente multi-step
        rollout_x_weight: float = 0.5,       # loss x multi-step
        stability_weight: float = 1e-3,      # penalidade de estabilidade
        # estabilidade
        stability_margin: float = 0.0,    # força max_eig(sym(K)) <= -margin (margin >= 0)
        # LS (ridge)
        ls_ridge: float = 1e-6,
        ls_update_in_forward: bool = False,  # se True, atualiza K,c por LS dentro do forward quando tiver pares
        detach_ls_targets: bool = True,      # evita LS "puxar" encoder pelo grad (normalmente desejável)
    ):
        super().__init__()
        act = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU()}.get(activation.lower(), nn.GELU())

        self.rom_weight = float(rom_weight)
        self.pred_weight = float(pred_weight)
        self.rollout_latent_weight = float(rollout_latent_weight)
        self.rollout_x_weight = float(rollout_x_weight)
        self.stability_weight = float(stability_weight)
        self.stability_margin = float(stability_margin)

        self.ls_ridge = float(ls_ridge)
        self.ls_update_in_forward = bool(ls_update_in_forward)
        self.detach_ls_targets = bool(detach_ls_targets)

        self.encoder = _mlp([input_dim, *list(hidden), latent_dim], act, last_act=False)
        self.decoder = _mlp([latent_dim, *list(reversed(hidden)), input_dim], act, last_act=False)

        # ROM contínuo: dz/dt = K z + c
        self.K = nn.Parameter(torch.zeros(latent_dim, latent_dim))
        self.c = nn.Parameter(torch.zeros(latent_dim))

        # init opcional: pequena estabilidade (K ~ 0 => quase identidade no Euler)
        nn.init.zeros_(self.K)
        nn.init.zeros_(self.c)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    # -------------------------
    # ROM helpers
    # -------------------------
    def latent_step(self, z: torch.Tensor, dt: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Um passo Euler no latente:
          z_next = z + dt*(K z + c)
        """
        B, d = z.shape
        dt_b = _as_dt(dt, B, z.device, z.dtype)  # (B,)
        dz = z @ self.K.t() + self.c[None, :]    # (B,d)
        return z + dt_b[:, None] * dz

    @torch.no_grad()
    def fit_rom_ls(
        self,
        z: torch.Tensor,
        z_next: torch.Tensor,
        dt: Union[float, torch.Tensor],
        ridge: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Ajusta K,c por least squares (ridge) no modelo:
          (z_next - z)/dt = z @ K^T + c

        z, z_next: (B,d)
        dt: float | escalar | (B,)
        """
        B, d = z.shape
        ridge = self.ls_ridge if ridge is None else float(ridge)

        dt_b = _as_dt(dt, B, z.device, z.dtype)  # (B,)
        # evita divisão por zero
        eps = torch.finfo(z.dtype).eps
        dt_b = torch.clamp(dt_b, min=eps)

        dz_target = (z_next - z) / dt_b[:, None]  # (B,d)

        # Monta regressão: dz = [z, 1] @ Theta, Theta shape (d+1, d)
        ones = torch.ones((B, 1), device=z.device, dtype=z.dtype)
        Z = torch.cat([z, ones], dim=1)  # (B, d+1)

        # Ridge: Theta = (Z^T Z + ridge I)^{-1} Z^T dz
        # (d+1, d+1)
        ZTZ = Z.t() @ Z
        I = torch.eye(d + 1, device=z.device, dtype=z.dtype)
        ZTZ_reg = ZTZ + ridge * I
        ZTdz = Z.t() @ dz_target  # (d+1, d)

        Theta = torch.linalg.solve(ZTZ_reg, ZTdz)  # (d+1, d)

        K_T = Theta[:d, :]      # (d,d)  corresponde a K^T
        c = Theta[d, :]         # (d,)

        self.K.copy_(K_T.t().contiguous())
        self.c.copy_(c.contiguous())

        return {"K": self.K, "c": self.c}

    def stability_penalty(self) -> torch.Tensor:
        """
        Penaliza instabilidade incentivando contração na dinâmica contínua.

        Usamos a parte simétrica:
          S = 0.5*(K + K^T)
        Para um sistema linear contínuo, exigir S negativo (autovalor máximo <= -margin)
        ajuda a evitar crescimento de energia.
        """
        S = 0.5 * (self.K + self.K.t())
        # eigvalsh (simétrica) é estável numericamente
        eigs = torch.linalg.eigvalsh(S)  # (d,)
        max_eig = eigs.max()
        # queremos max_eig <= -margin
        target = -self.stability_margin
        return torch.relu(max_eig - target) ** 2

    # -------------------------
    # Forward
    # -------------------------
    def forward(
        self,
        x: torch.Tensor,
        *,
        x_next: Optional[torch.Tensor] = None,
        dt: Union[float, torch.Tensor] = 1.0,
        # rollout
        x_seq: Optional[torch.Tensor] = None,
        dt_seq: Optional[Union[float, torch.Tensor]] = None,
    ) -> AEOutput:
        """
        x: (B, ...) -> recon
        x_next + dt -> 1-step losses
        x_seq + dt_seq -> rollout multi-step (x_seq shape (B,K,...) e dt_seq float | (B,) | (B,K))
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        losses = self.loss_from_parts(x_hat=x_hat, z=z, x=x)

        extras: Dict[str, torch.Tensor] = {
            "K": self.K,
            "c": self.c,
            "z": z,
            "x_hat": x_hat,
        }

        # Estabilidade (sempre pode aplicar)
        if self.stability_weight > 0:
            stab = self.stability_penalty()
            losses["stability"] = stab
            losses["total"] = losses["total"] + self.stability_weight * stab

        # ---------- 1-step losses ----------
        if x_next is not None:
            z_next = self.encode(x_next)
            z_pred = self.latent_step(z, dt=dt)

            rom_lat = torch.mean((z_next - z_pred) ** 2)
            losses["rom_latent"] = rom_lat
            losses["total"] = losses["total"] + self.rom_weight * rom_lat

            # previsão no espaço original: decode do z_pred deve bater com x_next
            x_pred = self.decode(z_pred)
            pred_x = torch.mean((x_next.view(x_next.shape[0], -1) - x_pred) ** 2)
            losses["pred_x"] = pred_x
            losses["total"] = losses["total"] + self.pred_weight * pred_x

            extras.update({"z_next": z_next, "z_pred": z_pred, "x_pred": x_pred})

            # LS update opcional (fechado-form) para K,c
            if self.ls_update_in_forward:
                z_ls = z.detach() if self.detach_ls_targets else z
                znext_ls = z_next.detach() if self.detach_ls_targets else z_next
                self.fit_rom_ls(z_ls, znext_ls, dt=dt)

        # ---------- rollout multi-step ----------
        # x_seq esperado: (B,K,...) onde passo k corresponde ao estado futuro k+1 (ou seja, t+1..t+K)
        if x_seq is not None:
            B = x.shape[0]
            K_steps = x_seq.shape[1]
            x_seq_flat = x_seq.view(B, K_steps, -1)

            # dt_seq pode ser: None -> usa dt (mesmo dt para todos); float; (B,); (B,K)
            if dt_seq is None:
                dt_seq_t = _as_dt(dt, B, z.device, z.dtype).unsqueeze(1).expand(B, K_steps)  # (B,K)
            else:
                if isinstance(dt_seq, (float, int)):
                    dt_seq_t = torch.full((B, K_steps), float(dt_seq), device=z.device, dtype=z.dtype)
                else:
                    dt_seq_tt = dt_seq.to(device=z.device, dtype=z.dtype)
                    if dt_seq_tt.ndim == 0:
                        dt_seq_t = dt_seq_tt.expand(B, K_steps)
                    elif dt_seq_tt.ndim == 1 and dt_seq_tt.shape[0] == B:
                        dt_seq_t = dt_seq_tt.unsqueeze(1).expand(B, K_steps)
                    elif dt_seq_tt.ndim == 2 and dt_seq_tt.shape == (B, K_steps):
                        dt_seq_t = dt_seq_tt
                    else:
                        raise ValueError(f"dt_seq deve ser float, escalar, (B,) ou (B,K). Veio {tuple(dt_seq_tt.shape)}")

            # targets no latente
            z_targets = self.encode(x_seq_flat.reshape(B * K_steps, -1)).view(B, K_steps, -1)

            z_roll = z
            lat_errs = []
            x_errs = []
            z_preds = []
            x_preds = []

            for k in range(K_steps):
                z_roll = self.latent_step(z_roll, dt=dt_seq_t[:, k])  # (B,d)
                z_preds.append(z_roll)

                # erro latente vs encode(x_{t+k+1})
                lat_errs.append(torch.mean((z_targets[:, k, :] - z_roll) ** 2))

                # erro em x vs x_{t+k+1}
                xk_pred = self.decode(z_roll)  # (B,input_dim)
                x_preds.append(xk_pred)
                x_errs.append(torch.mean((x_seq_flat[:, k, :] - xk_pred) ** 2))

            rollout_lat = torch.stack(lat_errs).mean()
            rollout_x = torch.stack(x_errs).mean()

            losses["rollout_latent"] = rollout_lat
            losses["rollout_x"] = rollout_x
            losses["total"] = losses["total"] + self.rollout_latent_weight * rollout_lat + self.rollout_x_weight * rollout_x

            extras["z_rollout"] = torch.stack(z_preds, dim=1)  # (B,K,d)
            extras["x_rollout"] = torch.stack(x_preds, dim=1)  # (B,K,input_dim)

            # LS update opcional usando todos os pares do rollout (t+k -> t+k+1)
            if self.ls_update_in_forward:
                # monta pares (z_k, z_{k+1}) e dt_k
                with torch.no_grad():
                    # z0 = z; z1 target = z_targets[:,0], etc.
                    z_pairs = []
                    z_next_pairs = []
                    dt_pairs = []

                    z_curr = z
                    for k in range(K_steps):
                        z_next_t = z_targets[:, k, :]
                        z_pairs.append(z_curr)
                        z_next_pairs.append(z_next_t)
                        dt_pairs.append(dt_seq_t[:, k])
                        z_curr = z_next_t  # usa target como "estado atual" para pares seguintes (teacher forcing)

                    Z = torch.cat(z_pairs, dim=0)
                    ZN = torch.cat(z_next_pairs, dim=0)
                    DT = torch.cat(dt_pairs, dim=0)

                    if self.detach_ls_targets:
                        Z = Z.detach()
                        ZN = ZN.detach()
                        DT = DT.detach()

                    self.fit_rom_ls(Z, ZN, dt=DT)

        return AEOutput(x_hat=x_hat, z=z, losses=losses, extras=extras)