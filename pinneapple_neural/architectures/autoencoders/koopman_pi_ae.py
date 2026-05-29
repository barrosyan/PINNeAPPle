from __future__ import annotations
"""Physics-informed Koopman autoencoder for dynamical systems."""

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from .base import AEBase, AEOutput
from .dense_ae import _mlp


class PhysicsInformedKoopmanAutoencoder(AEBase):
    """
    Koopman AE (prático / literature-grade):
      - Dense AE
      - Operador linear treinável no latente (K, opcionalmente com bias b)
      - Suporte explícito a dt via gerador contínuo A (K_eff = expm(A*dt)) se desejado
      - Loss no latente: ||z_next - K_eff z||^2
      - Loss de previsão no espaço original: ||x_next - D(K_eff E(x))||^2
      - Multi-step / rollout loss se x_next vier como sequência (B,H,dim)
      - Regularização de estabilidade (penaliza sigma_max(K_eff) > 1)
      - Opção de atualizar K por least squares (EDMD-like) quando x_next é fornecido

    forward:
      - forward(x) -> recon apenas
      - forward(x, x_next=...) -> adiciona losses dinâmicas
        * x_next pode ser (B, D) ou (B, H, D)

    Args:
      input_dim, latent_dim, hidden
      koopman_weight: peso da loss no latente (1-step)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden: List[int] = (512, 256),
        activation: str = "gelu",
        koopman_weight: float = 1.0,
        *,
        # extras (defaults "não mudam" o comportamento atual)
        pred_weight: float = 1.0,
        rollout_weight: float = 1.0,
        stability_weight: float = 0.0,
        use_affine: bool = True,
        use_generator_A: bool = False,   # se True, usa A e matrix_exp(A*dt) como operador efetivo
        ls_update: bool = False,         # se True, atualiza K por least squares quando x_next existe
        ls_ridge: float = 1e-6,
        power_iters: int = 10,           # p/ estimar sigma_max via power iteration
    ):
        super().__init__()
        act = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU()}.get(
            activation.lower(), nn.GELU()
        )

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)

        self.koopman_weight = float(koopman_weight)
        self.pred_weight = float(pred_weight)
        self.rollout_weight = float(rollout_weight)
        self.stability_weight = float(stability_weight)

        self.use_affine = bool(use_affine)
        self.use_generator_A = bool(use_generator_A)

        self.ls_update = bool(ls_update)
        self.ls_ridge = float(ls_ridge)
        self.power_iters = int(power_iters)

        self.encoder = _mlp([input_dim, *list(hidden), latent_dim], act, last_act=False)
        self.decoder = _mlp([latent_dim, *list(reversed(hidden)), input_dim], act, last_act=False)

        # Discreto: z_next = K z + b
        self.K = nn.Parameter(torch.eye(latent_dim))

        # Contínuo: z_dot = A z  =>  z_next = expm(A*dt) z
        self.A = nn.Parameter(torch.zeros(latent_dim, latent_dim))

        if self.use_affine:
            self.b = nn.Parameter(torch.zeros(latent_dim))
        else:
            self.register_parameter("b", None)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)  # mais robusto que view
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    # -----------------------------
    # Koopman helpers
    # -----------------------------
    def _as_dt(
        self,
        dt: Optional[Union[float, torch.Tensor]],
        *,
        device: torch.device,
        dtype: torch.dtype,
        steps: int,
        batch: int,
    ) -> torch.Tensor:
        """
        Retorna dt como tensor broadcastável para (B, steps, 1, 1) se necessário.
        Aceita:
          - None -> 1.0
          - float
          - tensor escalar
          - (steps,)
          - (B, steps)
        """
        if dt is None:
            dt_t = torch.tensor(1.0, device=device, dtype=dtype)
        elif isinstance(dt, (float, int)):
            dt_t = torch.tensor(float(dt), device=device, dtype=dtype)
        else:
            dt_t = dt.to(device=device, dtype=dtype)

        # Normaliza para forma (B, steps)
        if dt_t.ndim == 0:
            dt_t = dt_t.view(1, 1).expand(batch, steps)
        elif dt_t.ndim == 1:
            if dt_t.shape[0] != steps:
                raise ValueError(f"dt com shape {tuple(dt_t.shape)} não bate steps={steps}.")
            dt_t = dt_t.view(1, steps).expand(batch, steps)
        elif dt_t.ndim == 2:
            if dt_t.shape != (batch, steps):
                raise ValueError(f"dt com shape {tuple(dt_t.shape)} esperado {(batch, steps)}.")
        else:
            raise ValueError(f"dt ndim={dt_t.ndim} não suportado.")

        return dt_t

    def _effective_operator(self, dt_scalar: torch.Tensor) -> torch.Tensor:
        """
        Retorna K_eff (latent_dim x latent_dim) para um dt escalar (tensor 0-d).
        - Se use_generator_A: expm(A*dt)
        - Caso contrário: K (assumindo dt fixo no dataset)
        """
        if self.use_generator_A:
            # torch.matrix_exp é estável e differentiable
            return torch.matrix_exp(self.A * dt_scalar)
        return self.K

    def _apply_K(self, z: torch.Tensor, K_eff: torch.Tensor) -> torch.Tensor:
        z_pred = z @ K_eff.t()
        if self.use_affine and (self.b is not None):
            z_pred = z_pred + self.b
        return z_pred

    @torch.no_grad()
    def _least_squares_update_K(self, z: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        """
        Atualiza self.K (e opcionalmente b) via mínimos quadrados (ridge).
        Resolve: z_next ≈ z @ K^T + b
        Retorna K_ls (para logging).
        """
        B, d = z.shape
        ridge = self.ls_ridge

        if self.use_affine and (self.b is not None):
            ones = torch.ones(B, 1, device=z.device, dtype=z.dtype)
            Z = torch.cat([z, ones], dim=1)           # (B, d+1)
            Y = z_next                                 # (B, d)

            # Solve (Z^T Z + λI) W = Z^T Y  where W is (d+1, d)
            I = torch.eye(d + 1, device=z.device, dtype=z.dtype)
            M = Z.t() @ Z + ridge * I
            RHS = Z.t() @ Y
            W = torch.linalg.solve(M, RHS)            # (d+1, d)

            K_t = W[:d, :]                            # (d, d) -> this is K^T
            b = W[d, :]                               # (d,)

            self.K.copy_(K_t.t())
            self.b.copy_(b)
            return self.K.detach()
        else:
            Z = z                                      # (B, d)
            Y = z_next                                 # (B, d)
            I = torch.eye(d, device=z.device, dtype=z.dtype)
            M = Z.t() @ Z + ridge * I
            RHS = Z.t() @ Y
            K_t = torch.linalg.solve(M, RHS)           # (d, d) = K^T
            self.K.copy_(K_t.t())
            return self.K.detach()

    def _sigma_max_penalty(self, K_eff: torch.Tensor) -> torch.Tensor:
        """
        Penaliza sigma_max(K_eff) > 1 (estimado por power iteration em K^T K).
        """
        if self.stability_weight <= 0.0:
            return K_eff.new_tensor(0.0)

        d = K_eff.shape[0]
        v = torch.randn(d, 1, device=K_eff.device, dtype=K_eff.dtype)
        v = v / (v.norm() + 1e-12)

        # power iteration on (K^T K)
        M = K_eff.t() @ K_eff
        for _ in range(max(self.power_iters, 1)):
            v = M @ v
            v = v / (v.norm() + 1e-12)

        # Rayleigh quotient ~ largest eigenvalue of M = sigma_max^2
        lam = (v.t() @ (M @ v)).squeeze()
        sigma = torch.sqrt(torch.clamp(lam, min=0.0))

        # penaliza apenas se > 1
        return torch.relu(sigma - 1.0) ** 2

    # -----------------------------
    # Forward
    # -----------------------------
    def forward(
        self,
        x: torch.Tensor,
        *,
        x_next: Optional[torch.Tensor] = None,
        dt: Optional[Union[float, torch.Tensor]] = None,
    ) -> AEOutput:
        """
        x_next:
          - None: só recon
          - (B, D): 1-step
          - (B, H, D): rollout multi-step supervision
        dt:
          - None -> 1.0
          - float/tensor escalar -> mesmo dt para todos
          - (H,) ou (B,H) se x_next for sequência
        """
        z = self.encode(x)
        x_hat = self.decode(z)

        losses = self.loss_from_parts(x_hat=x_hat, z=z, x=x)

        extras: Dict[str, torch.Tensor] = {
            "K": self.K,
            "A": self.A,
        }

        if x_next is None:
            return AEOutput(x_hat=x_hat, z=z, losses=losses, extras=extras)

        # Normaliza x_next para (B, H, D)
        if x_next.ndim == 2:
            x_next_seq = x_next.unsqueeze(1)  # (B, 1, D)
        elif x_next.ndim == 3:
            x_next_seq = x_next
        else:
            raise ValueError(f"x_next ndim={x_next.ndim} não suportado; use (B,D) ou (B,H,D).")

        B, H, D = x_next_seq.shape
        if D != self.input_dim:
            # se seu input estiver flatten, isso pode acontecer; ajuste aqui se necessário
            x_next_seq = x_next_seq.reshape(B, H, -1)
            if x_next_seq.shape[-1] != self.input_dim:
                raise ValueError(f"x_next last dim {x_next_seq.shape[-1]} != input_dim {self.input_dim}")

        dt_BH = self._as_dt(dt, device=x.device, dtype=x.dtype, steps=H, batch=B)

        # (opcional) least squares update para o 1-step (dt implícito)
        # Obs: LS update só faz sentido direto quando use_generator_A=False (K discreto).
        # Se use_generator_A=True, você normalmente faz LS no A (mais complexo).
        if self.ls_update and (not self.use_generator_A):
            z1 = z
            z_next_1 = self.encode(x_next_seq[:, 0, :])
            extras["K_ls"] = self._least_squares_update_K(z1.detach(), z_next_1.detach())

        # 1-step losses (primeiro passo)
        x_next_1 = x_next_seq[:, 0, :]
        z_next_1 = self.encode(x_next_1)

        dt0 = dt_BH[:, 0].mean()  # usa dt médio do batch para montar K_eff (mantém simples e estável)
        K_eff_0 = self._effective_operator(dt0)

        z_pred_1 = self._apply_K(z, K_eff_0)

        koop_lat = torch.mean((z_next_1 - z_pred_1) ** 2)
        losses["koopman"] = koop_lat
        losses["total"] = losses["total"] + self.koopman_weight * koop_lat

        # Previsão no espaço original (1-step)
        x_pred_1 = self.decode(z_pred_1)
        pred_x = torch.mean((x_next_1 - x_pred_1) ** 2)
        losses["pred_x"] = pred_x
        losses["total"] = losses["total"] + self.pred_weight * pred_x

        # Estabilidade (no operador efetivo do 1-step)
        stab = self._sigma_max_penalty(K_eff_0)
        losses["stability"] = stab
        losses["total"] = losses["total"] + self.stability_weight * stab

        extras.update(
            {
                "z_next": z_next_1,
                "z_pred": z_pred_1,
                "x_pred": x_pred_1,
                "K_eff": K_eff_0,
            }
        )

        # Rollout multi-step (se H > 1)
        if H > 1:
            z_roll = z
            rollout_loss = z.new_tensor(0.0)

            # opcional: acumular previsões
            x_preds = []

            for k in range(H):
                dtk = dt_BH[:, k].mean()
                K_eff_k = self._effective_operator(dtk)
                z_roll = self._apply_K(z_roll, K_eff_k)
                xk_pred = self.decode(z_roll)
                x_preds.append(xk_pred)

                xk_true = x_next_seq[:, k, :]
                rollout_loss = rollout_loss + torch.mean((xk_true - xk_pred) ** 2)

            rollout_loss = rollout_loss / float(H)
            losses["rollout_x"] = rollout_loss
            losses["total"] = losses["total"] + self.rollout_weight * rollout_loss

            extras["x_rollout_pred"] = torch.stack(x_preds, dim=1)  # (B,H,D)

        return AEOutput(x_hat=x_hat, z=z, losses=losses, extras=extras)
