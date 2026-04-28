"""Two-phase PINN trainer: supervised (Phase 1) → physics (Phase 2).

Phase 1 fits the model to reference solver/measurement data (MSE).
Phase 2 enforces PDE residuals and boundary conditions.

Typical workflow
----------------
>>> from pinneaple_train.two_phase import TwoPhaseTrainer, TwoPhaseConfig
>>> cfg = TwoPhaseConfig(phase1_epochs=300, phase2_epochs=800)
>>> trainer = TwoPhaseTrainer(model, supervised_loss, physics_loss, cfg)
>>> trainer.fit(solver_loader, collocation_loader)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim


LossOut = Union[torch.Tensor, Dict[str, torch.Tensor]]


@dataclass
class TwoPhaseConfig:
    # Phase 1 — supervised (fit to reference/solver data)
    phase1_epochs: int = 200
    phase1_lr: float = 1e-3

    # Phase 2 — physics (PDE residual + BC loss)
    phase2_epochs: int = 500
    phase2_lr: float = 5e-4

    # Shared
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = False
    seed: Optional[int] = None
    print_every: int = 50

    # When combined=True, Phase 1 runs a weighted sum of supervised + physics.
    # The physics weight ramps from physics_weight_start → physics_weight_end
    # over phase1_epochs, then stays at physics_weight_end for Phase 2.
    combined: bool = False
    supervised_weight: float = 1.0
    physics_weight_start: float = 0.0
    physics_weight_end: float = 1.0


@dataclass
class TwoPhaseHistory:
    phase1_epochs: List[int] = field(default_factory=list)
    phase1_loss: List[float] = field(default_factory=list)
    phase2_epochs: List[int] = field(default_factory=list)
    phase2_loss: List[float] = field(default_factory=list)
    phase2_pde_loss: List[float] = field(default_factory=list)
    phase2_bc_loss: List[float] = field(default_factory=list)


class TwoPhaseTrainer:
    """
    Auto-scheduled two-phase trainer for PINNs.

    Parameters
    ----------
    model : nn.Module
    supervised_loss_fn : callable(model, batch, extras) -> Tensor | dict
        Loss for Phase 1.  Return value is a scalar Tensor or a dict with
        a mandatory ``"total"`` key.
    physics_loss_fn : callable(model, batch, extras) -> Tensor | dict
        Loss for Phase 2.  Dict keys ``"pde"`` and ``"bc"`` are logged
        separately if present.
    config : TwoPhaseConfig, optional
    """

    def __init__(
        self,
        model: nn.Module,
        supervised_loss_fn: Callable,
        physics_loss_fn: Callable,
        config: Optional[TwoPhaseConfig] = None,
    ):
        self.model = model
        self.supervised_loss_fn = supervised_loss_fn
        self.physics_loss_fn = physics_loss_fn
        self.cfg = config or TwoPhaseConfig()
        self.history = TwoPhaseHistory()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_loss(self, out: LossOut) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(out, torch.Tensor):
            return out, {}
        total = out["total"]
        rest = {k: v for k, v in out.items() if k != "total"}
        return total, rest

    def _make_optimizer(self, lr: float) -> optim.Optimizer:
        return optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.cfg.weight_decay,
        )

    def _clip_grad(self) -> None:
        if self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

    @staticmethod
    def _move_batch(batch: Any, device: torch.device) -> Any:
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        if isinstance(batch, dict):
            return {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
        if isinstance(batch, (list, tuple)):
            return type(batch)(
                b.to(device) if isinstance(b, torch.Tensor) else b for b in batch
            )
        return batch

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        supervised_loader,
        physics_loader,
        *,
        extras: Optional[Dict[str, Any]] = None,
    ) -> "TwoPhaseTrainer":
        """
        Run Phase 1 (supervised) then Phase 2 (physics).

        Parameters
        ----------
        supervised_loader : iterable
            Mini-batches of reference data (e.g. ``torch.utils.data.DataLoader``
            over solver output).  Each batch is passed to ``supervised_loss_fn``.
        physics_loader : iterable
            Collocation batches (e.g. from ``CollocationSampler``). Each batch
            is passed to ``physics_loss_fn``.
        extras : dict, optional
            Additional context forwarded to both loss functions (e.g. PDE
            parameters, geometry info).
        """
        cfg = self.cfg
        extras = extras or {}
        device = torch.device(cfg.device)
        self.model.to(device)

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

        # Phase 1 — supervised
        print("[TwoPhaseTrainer] -- Phase 1: supervised ----------------------")
        self._run_phase(
            phase_id=1,
            loader=supervised_loader,
            loss_fn=self.supervised_loss_fn,
            epochs=cfg.phase1_epochs,
            lr=cfg.phase1_lr,
            device=device,
            extras=extras,
        )

        # Phase 2 — physics
        print("[TwoPhaseTrainer] -- Phase 2: physics --------------------------")
        self._run_phase(
            phase_id=2,
            loader=physics_loader,
            loss_fn=self.physics_loss_fn,
            epochs=cfg.phase2_epochs,
            lr=cfg.phase2_lr,
            device=device,
            extras=extras,
        )

        print("[TwoPhaseTrainer] -- Done --------------------------------------")
        return self

    # ------------------------------------------------------------------

    def _run_phase(
        self,
        phase_id: int,
        loader,
        loss_fn: Callable,
        epochs: int,
        lr: float,
        device: torch.device,
        extras: Dict,
    ) -> None:
        opt = self._make_optimizer(lr)
        use_amp = self.cfg.amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_total = 0.0
            epoch_pde = 0.0
            epoch_bc = 0.0
            n_batches = 0

            for batch in loader:
                batch = self._move_batch(batch, device)
                opt.zero_grad()

                with torch.autocast(device_type=device.type, enabled=use_amp):
                    out = loss_fn(self.model, batch, extras)
                total, rest = self._parse_loss(out)

                scaler.scale(total).backward()
                scaler.unscale_(opt)
                self._clip_grad()
                scaler.step(opt)
                scaler.update()

                epoch_total += float(total.detach())
                if "pde" in rest:
                    epoch_pde += float(rest["pde"].detach())
                if "bc" in rest:
                    epoch_bc += float(rest["bc"].detach())
                n_batches += 1

            if n_batches == 0:
                break

            avg_total = epoch_total / n_batches
            avg_pde   = epoch_pde   / n_batches
            avg_bc    = epoch_bc    / n_batches

            if phase_id == 1:
                self.history.phase1_epochs.append(epoch)
                self.history.phase1_loss.append(avg_total)
            else:
                self.history.phase2_epochs.append(epoch)
                self.history.phase2_loss.append(avg_total)
                self.history.phase2_pde_loss.append(avg_pde)
                self.history.phase2_bc_loss.append(avg_bc)

            if self.cfg.print_every > 0 and epoch % self.cfg.print_every == 0:
                tag = f"  [P{phase_id} {epoch:>5}/{epochs}]"
                if phase_id == 2:
                    print(
                        f"{tag}  total={avg_total:.4e}"
                        + (f"  pde={avg_pde:.4e}" if avg_pde else "")
                        + (f"  bc={avg_bc:.4e}"  if avg_bc  else "")
                    )
                else:
                    print(f"{tag}  loss={avg_total:.4e}")


class UnnormModel(nn.Module):
    """Wraps a normalised model to output physical-scale predictions.

    Training surrogates on normalised targets (Y ~ N(0,1)) is numerically
    stable, but PDE residuals must be computed in physical units.  Wrap the
    normalised model *before* computing any physics loss:

        unnorm = UnnormModel(model, Y_mean, Y_std)
        pde_residual = pde_loss_fn(unnorm, collocation_pts)

    Parameters
    ----------
    model : nn.Module
        Any model that accepts coordinates and returns normalised predictions.
    Y_mean : array-like or Tensor, shape (out_dim,) or scalar
    Y_std  : array-like or Tensor, shape (out_dim,) or scalar

    Forward
    -------
    ``forward(x)  =  model(x) * Y_std + Y_mean``
    """

    def __init__(self, model: nn.Module, Y_mean, Y_std):
        super().__init__()
        self.model = model
        self.register_buffer(
            "Y_mean", torch.as_tensor(Y_mean, dtype=torch.float32).reshape(1, -1)
        )
        self.register_buffer(
            "Y_std", torch.as_tensor(Y_std, dtype=torch.float32).reshape(1, -1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_norm = self.model(x)
        if hasattr(y_norm, "y"):
            y_norm = y_norm.y
        return y_norm * self.Y_std + self.Y_mean


__all__ = ["TwoPhaseConfig", "TwoPhaseHistory", "TwoPhaseTrainer", "UnnormModel"]
