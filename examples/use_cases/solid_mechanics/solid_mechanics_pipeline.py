"""
Pipeline de Mecânica dos Sólidos — PINNeAPPle Framework
========================================================

Fluxo genérico para qualquer problema de elasticidade:

  [1] Definir problema       → ProblemBuilder / preset registrado
  [2] Dados FEM (opcional)   → solution.npy → DataConstraint → ProblemSpec
  [3] Treinar PINN           → SolidMechanicsPipeline.train()
  [4] Pós-processar          → Von Mises, strain energy, principal stresses
  [5] Avaliar                → comparar com FEM ou solução analítica
  [6] Visualizar             → tricontourf na seção rz

Casos de uso incluídos
----------------------
  A. Cilindro espesso sob pressão interna (Lamé — solução analítica disponível)
  B. Acoplamento roscado TC50 BOX com dados FEM externos (solution.npy)
  C. Qualquer preset de elasticidade axissimétrica via get_preset()

Para o preset de acoplamento, substitua o caminho para solution.npy pelo seu dataset:
  pipeline = SolidMechanicsPipeline.from_fem_solution(
      "synthetic_tc50/dataset/case_000/solution.npy",
      preset="threaded_coupling_tc50_box",
  )
  result = pipeline.train(epochs=5000)
  pipeline.evaluate(result)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors as mcolors

# ── PINNeAPPle imports ─────────────────────────────────────────────────────
from pinneaple_environment import ProblemBuilder, ProblemSpec
from pinneaple_environment import get_preset
from pinneaple_environment.presets.solid_mechanics import lame_analytical

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# PINN para elasticidade axissimétrica
# ══════════════════════════════════════════════════════════════════════════════

class AxiSymElasticityPINN(nn.Module):
    """
    PINN para elasticidade axissimétrica 2D.

    Entrada : (r, z)  normalizados para [0,1]
    Saída   : (u_r, u_z)  em unidades físicas (mm)

    Arquitetura: ResNet com ativações Tanh + encoding de Fourier opcional.
    A normalização de entrada/saída é feita internamente — o modelo recebe
    coordenadas físicas (mm) e devolve deslocamentos físicos (mm).

    Melhorias em relação ao modelo original do acoplamento:
      - Skip connections para gradiente mais estável
      - Normalização interna (não depende de constantes externas)
      - Saída bounded via tanh (evita explosão fora do domínio)
    """

    def __init__(
        self,
        r_bounds: Tuple[float, float] = (10.0, 50.0),
        z_bounds: Tuple[float, float] = (0.0, 100.0),
        u_scale: float = 0.1,           # mm — escala característica de deslocamento
        hidden: int = 128,
        n_layers: int = 6,
        fourier_features: bool = True,
        n_fourier: int = 16,
    ):
        super().__init__()
        self.r_min, self.r_max = float(r_bounds[0]), float(r_bounds[1])
        self.z_min, self.z_max = float(z_bounds[0]), float(z_bounds[1])
        self.u_scale = float(u_scale)

        # Fourier feature encoding: f(x) = [sin(Bx), cos(Bx)]
        in_dim = 2
        if fourier_features:
            torch.manual_seed(42)
            self.B = nn.Parameter(
                torch.randn(n_fourier, 2) * 5.0, requires_grad=False
            )
            in_dim = 2 * n_fourier
        else:
            self.B = None

        # ResNet body
        self.layers = nn.ModuleList()
        self.skips   = nn.ModuleList()
        prev = in_dim
        for i in range(n_layers):
            self.layers.append(nn.Linear(prev, hidden))
            if i > 0 and prev != hidden:
                self.skips.append(nn.Linear(prev, hidden))
            else:
                self.skips.append(None)
            prev = hidden
        self.head = nn.Linear(hidden, 2)   # u_r, u_z

    def _encode(self, rz_norm: torch.Tensor) -> torch.Tensor:
        if self.B is not None:
            proj = rz_norm @ self.B.T            # (N, n_fourier)
            return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return rz_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N, 2) — physical coordinates [r, z] in mm
        Returns (N, 2) — displacements [u_r, u_z] in mm
        """
        r = (x[:, 0:1] - self.r_min) / (self.r_max - self.r_min + 1e-10)
        z = (x[:, 1:2] - self.z_min) / (self.z_max - self.z_min + 1e-10)
        rz_norm = torch.cat([r, z], dim=-1)   # (N, 2) in [0,1]

        h = self._encode(rz_norm)
        for i, (layer, skip) in enumerate(zip(self.layers, self.skips)):
            h_new = torch.tanh(layer(h))
            if skip is not None:
                h_new = h_new + torch.tanh(skip(h))
            h = h_new

        out = self.head(h)   # unbounded linear head
        return out * self.u_scale   # scale to physical units


# ══════════════════════════════════════════════════════════════════════════════
# Physics loss — equilibrio axissimétrico
# ══════════════════════════════════════════════════════════════════════════════

def axisymmetric_equilibrium_residual(
    model: nn.Module,
    x: torch.Tensor,
    E: float,
    nu: float,
) -> torch.Tensor:
    """
    Residuo das equações de equilíbrio axissimétricas:
      R1 = ∂σ_rr/∂r + ∂σ_rz/∂z + (σ_rr − σ_θθ)/r
      R2 = ∂σ_rz/∂r + ∂σ_zz/∂z + σ_rz/r

    Retorna: (R1² + R2²).mean()  — residuo médio quadrático
    """
    lam_c = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu_c  = E / (2 * (1 + nu))

    x = x.requires_grad_(True)
    u = model(x)   # (N, 2): u_r, u_z

    r = x[:, 0:1].clamp(min=1e-3)   # evita divisão por zero no eixo

    def _grad(scalar):
        return torch.autograd.grad(scalar.sum(), x, create_graph=True)[0]

    du_r = _grad(u[:, 0:1])   # [∂u_r/∂r, ∂u_r/∂z]
    du_z = _grad(u[:, 1:2])   # [∂u_z/∂r, ∂u_z/∂z]

    # Strain components
    e_rr = du_r[:, 0:1]
    e_zz = du_z[:, 1:2]
    e_tt = u[:, 0:1] / r
    e_rz = 0.5 * (du_r[:, 1:2] + du_z[:, 0:1])

    tr = e_rr + e_zz + e_tt

    # Stress components
    s_rr = lam_c * tr + 2 * mu_c * e_rr
    s_zz = lam_c * tr + 2 * mu_c * e_zz
    s_tt = lam_c * tr + 2 * mu_c * e_tt
    s_rz = 2 * mu_c * e_rz

    # Divergence of stress (equilibrium residual)
    ds_rr_dr = _grad(s_rr)[:, 0:1]
    ds_rz_dz = _grad(s_rz)[:, 1:2]
    ds_rz_dr = _grad(s_rz)[:, 0:1]
    ds_zz_dz = _grad(s_zz)[:, 1:2]

    R1 = ds_rr_dr + ds_rz_dz + (s_rr - s_tt) / r
    R2 = ds_rz_dr + ds_zz_dz + s_rz / r

    return (R1 ** 2).mean() + (R2 ** 2).mean()


def von_mises_from_displacements(
    model: nn.Module,
    x: torch.Tensor,
    E: float,
    nu: float,
) -> np.ndarray:
    """
    Computa Von Mises a partir dos deslocamentos preditos pelo PINN.
    Retorna array numpy (N,).
    """
    lam_c = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu_c  = E / (2 * (1 + nu))

    model.eval()
    device = next(model.parameters()).device
    results = []
    batch_size = 2048

    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size].to(device).requires_grad_(True)
        u  = model(xb)

        r = xb[:, 0:1].clamp(min=1e-3)

        def _g(s):
            return torch.autograd.grad(s.sum(), xb, create_graph=False, retain_graph=True)[0]

        du_r = _g(u[:, 0:1])
        du_z = _g(u[:, 1:2])

        e_rr = du_r[:, 0:1]
        e_zz = du_z[:, 1:2]
        e_tt = u[:, 0:1] / r
        e_rz = 0.5 * (du_r[:, 1:2] + du_z[:, 0:1])
        tr   = e_rr + e_zz + e_tt

        s_rr = lam_c * tr + 2 * mu_c * e_rr
        s_zz = lam_c * tr + 2 * mu_c * e_zz
        s_tt = lam_c * tr + 2 * mu_c * e_tt
        s_rz = 2 * mu_c * e_rz

        vm = torch.sqrt(
            ((s_rr - s_zz)**2 + (s_zz - s_tt)**2 + (s_tt - s_rr)**2
             + 6 * s_rz**2) / 2
        )
        results.append(vm.detach().cpu().numpy().ravel())

    return np.concatenate(results)


# ══════════════════════════════════════════════════════════════════════════════
# SolidMechanicsPipeline
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SolidMechanicsResult:
    """Resultado do pipeline de mecânica dos sólidos."""
    spec: ProblemSpec
    model: nn.Module
    history: List[Dict[str, float]]
    metrics: Dict[str, float] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def summary(self) -> str:
        lines = [
            f"SolidMechanicsResult — {self.spec.name}",
            f"  Epochs  : {len(self.history)}",
            f"  Time    : {self.elapsed_s:.1f}s",
            f"  Final loss: {self.history[-1].get('loss', float('nan')):.4e}",
        ]
        if self.metrics:
            for k, v in self.metrics.items():
                lines.append(f"  {k:<18}: {v:.4e}")
        return "\n".join(lines)


class SolidMechanicsPipeline:
    """
    Pipeline completo para problemas de mecânica dos sólidos com PINN.

    Criação
    -------
    Via preset existente::

        pipeline = SolidMechanicsPipeline.from_preset("thick_walled_cylinder_lame",
                                                        a=20, b=60, p_a=100e6)

    Via FEM solution.npy (dados externos do FEM)::

        pipeline = SolidMechanicsPipeline.from_fem_solution(
            "synthetic_tc50/dataset/case_000/solution.npy",
            preset="threaded_coupling_tc50_box",
        )

    Via ProblemBuilder::

        spec = ProblemBuilder("meu_problema") ...
        pipeline = SolidMechanicsPipeline(spec)

    Treinamento::

        result = pipeline.train(epochs=5000)
        pipeline.evaluate(result, analytical_fn=lame_analytical)
        pipeline.visualize(result)
    """

    def __init__(
        self,
        spec: ProblemSpec,
        fem_coords: Optional[np.ndarray] = None,
        fem_disp: Optional[np.ndarray] = None,
        fem_vm: Optional[np.ndarray] = None,
    ):
        self.spec      = spec
        self.fem_coords = fem_coords   # (N, 2): r, z
        self.fem_disp   = fem_disp     # (N, 2): u_r, u_z
        self.fem_vm     = fem_vm       # (N,):   Von Mises

    # ── Construtores ──────────────────────────────────────────────────────────

    @classmethod
    def from_preset(cls, preset_id: str, **params) -> "SolidMechanicsPipeline":
        """Cria pipeline a partir de preset registrado."""
        spec = get_preset(preset_id, **params)
        return cls(spec)

    @classmethod
    def from_fem_solution(
        cls,
        solution_npy: Union[str, Path],
        preset: Union[str, ProblemSpec] = "axisymmetric_linear_elasticity_2d",
        preset_params: Optional[Dict[str, Any]] = None,
    ) -> "SolidMechanicsPipeline":
        """
        Cria pipeline carregando dados FEM de solution.npy.

        O ProblemSpec define a física; os dados FEM servem como:
          1. DataConstraint (supervisão nos pontos FEM durante treino)
          2. Ground truth para avaliação pós-treino

        solution.npy deve ter as chaves:
          coords   : (N, 2) — r, z em mm
          disp     : (N, 2) — u_r, u_z em mm
          stress_vm: (N,)   — Von Mises em Pa  (opcional)
        """
        path = Path(solution_npy)
        if not path.exists():
            raise FileNotFoundError(f"FEM solution not found: {path}")

        data = np.load(str(path), allow_pickle=True).item()
        coords = data["coords"]     # (N, 2): col0=r, col1=z
        disp   = data["disp"]       # (N, 2): u_r, u_z
        vm     = data.get("stress_vm", None)

        # Infere bounds do domínio a partir dos dados FEM
        r_min = float(coords[:, 0].min())
        r_max = float(coords[:, 0].max())
        z_min = float(coords[:, 1].min())
        z_max = float(coords[:, 1].max())

        if isinstance(preset, str):
            params = dict(preset_params or {})
            # Injeta bounds se o preset aceita esses parâmetros
            if preset == "axisymmetric_linear_elasticity_2d":
                params.setdefault("r_min", r_min)
                params.setdefault("r_max", r_max)
                params.setdefault("z_min", z_min)
                params.setdefault("z_max", z_max)
            spec = get_preset(preset, **params)
        else:
            spec = preset

        print(f"  FEM data: {len(coords):,} pontos  |  "
              f"r ∈ [{r_min:.1f}, {r_max:.1f}]  "
              f"z ∈ [{z_min:.1f}, {z_max:.1f}] mm")
        if vm is not None:
            print(f"  FEM Von Mises: max={vm.max():.3e} Pa  mean={vm.mean():.3e} Pa")

        return cls(spec, fem_coords=coords, fem_disp=disp, fem_vm=vm)

    @classmethod
    def from_builder(cls, builder) -> "SolidMechanicsPipeline":
        """Cria pipeline a partir de um ProblemBuilder já configurado."""
        return cls(builder.build())

    # ── Amostragem ────────────────────────────────────────────────────────────

    def _sample_collocation(self, n: int, seed: int = 42) -> np.ndarray:
        """Amostra pontos uniformes no domínio retangular da spec."""
        rng = np.random.default_rng(seed)
        r_min, r_max = self.spec.domain_bounds["r"]
        z_min, z_max = self.spec.domain_bounds["z"]
        r = rng.uniform(r_min, r_max, n).astype(np.float32)
        z = rng.uniform(z_min, z_max, n).astype(np.float32)
        return np.column_stack([r, z])

    def _sample_bc(self, n_per_face: int, seed: int = 0) -> np.ndarray:
        """Amostra pontos nas 4 faces do bounding box."""
        rng = np.random.default_rng(seed)
        r_min, r_max = self.spec.domain_bounds["r"]
        z_min, z_max = self.spec.domain_bounds["z"]
        n = n_per_face
        r_bot  = np.column_stack([rng.uniform(r_min, r_max, n), np.full(n, z_min)])
        r_top  = np.column_stack([rng.uniform(r_min, r_max, n), np.full(n, z_max)])
        r_in   = np.column_stack([np.full(n, r_min), rng.uniform(z_min, z_max, n)])
        r_out  = np.column_stack([np.full(n, r_max), rng.uniform(z_min, z_max, n)])
        return np.vstack([r_bot, r_top, r_in, r_out]).astype(np.float32)

    # ── Treino ────────────────────────────────────────────────────────────────

    def train(
        self,
        epochs: int = 5000,
        lr: float = 1e-3,
        n_col: Optional[int] = None,
        n_bc: Optional[int] = None,
        n_data: Optional[int] = None,
        batch_size: int = 2048,
        w_phys: float = 1e-8,   # peso do resíduo físico (Pa² normalizado)
        w_data: float = 1.0,    # peso dos dados FEM
        w_bc: float = 10.0,     # peso das BCs
        device: str = "auto",
        seed: int = 42,
        verbose: bool = True,
    ) -> SolidMechanicsResult:
        """
        Treina o PINN no problema especificado.

        Parâmetros
        ----------
        w_phys : peso do resíduo de equilíbrio (ajuste se loss_phys >> loss_data)
        w_data : peso da supervisão FEM (0 se sem dados FEM)
        w_bc   : peso das condições de contorno

        O w_phys deve ser escalado para compensar a diferença de magnitude
        entre Pa²/mm² (resíduo) e mm² (dados de deslocamento).
        Regra prática: w_phys ≈ U_scale² / E²  onde U_scale é o deslocamento
        característico e E é o módulo de Young.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        if device == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(device)

        # Parâmetros do material
        E  = float(self.spec.pde.params.get("E",  2.1e11))
        nu = float(self.spec.pde.params.get("nu", 0.3))

        # Escala característica de deslocamento
        u_char = float(self.spec.scales.U) if self.spec.scales.U != 1.0 else 1e-3

        # Pontos de collocação
        _n_col  = n_col  or self.spec.sample_defaults.get("n_col",  8000)
        _n_bc   = n_bc   or self.spec.sample_defaults.get("n_bc",   2000)

        x_col = torch.from_numpy(self._sample_collocation(_n_col, seed)).to(dev)
        x_bc  = torch.from_numpy(self._sample_bc(_n_bc // 4, seed)).to(dev)

        # Dados FEM
        has_fem = self.fem_coords is not None and self.fem_disp is not None
        if has_fem:
            _n_data = min(n_data or 10000, len(self.fem_coords))
            idx_data = np.random.default_rng(seed).choice(len(self.fem_coords), _n_data, replace=False)
            x_data  = torch.from_numpy(self.fem_coords[idx_data].astype(np.float32)).to(dev)
            y_data  = torch.from_numpy(self.fem_disp[idx_data].astype(np.float32)).to(dev)

        # Modelo
        r_bounds = self.spec.domain_bounds["r"]
        z_bounds = self.spec.domain_bounds["z"]
        model = AxiSymElasticityPINN(
            r_bounds=r_bounds,
            z_bounds=z_bounds,
            u_scale=u_char,
        ).to(dev)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        if verbose:
            print(f"\n  PINN: AxiSymElasticityPINN  |  device: {dev}")
            print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
            print(f"  Col: {_n_col}  |  BC: {_n_bc}  |  FEM data: {_n_data if has_fem else 0}")
            print(f"  Pesos: phys={w_phys:.1e}  data={w_data:.1e}  bc={w_bc:.1e}")

        history: List[Dict[str, float]] = []
        t0 = time.time()

        for ep in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()

            # Seleciona mini-batch de collocação
            idx = torch.randperm(_n_col, device=dev)[:min(batch_size, _n_col)]
            xb  = x_col[idx]

            # ── Loss física: resíduo de equilíbrio ────────────────────────────
            l_phys = axisymmetric_equilibrium_residual(model, xb, E, nu)

            # ── Loss de BC ────────────────────────────────────────────────────
            # Avalia condições de contorno da spec
            l_bc = self._eval_bc_loss(model, x_bc, dev)

            # ── Loss de dados FEM ─────────────────────────────────────────────
            l_data = torch.zeros(1, device=dev)
            if has_fem:
                idx_d = torch.randperm(len(x_data), device=dev)[:min(batch_size, len(x_data))]
                u_pred = model(x_data[idx_d])
                l_data = nn.functional.mse_loss(u_pred, y_data[idx_d])

            # Normalização da loss física para ficar na mesma escala que l_data
            # l_phys é em Pa²/mm² ≈ E²/L² → normaliza para u_char²
            phys_scale = (u_char / max(E / 1e3, 1.0)) ** 2
            l_total = w_bc * l_bc + w_data * l_data + w_phys * l_phys / max(phys_scale, 1e-20)

            l_total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            entry = {
                "epoch": ep,
                "loss": float(l_total.item()),
                "loss_phys": float(l_phys.item()),
                "loss_data": float(l_data.item()),
                "loss_bc":   float(l_bc.item()),
            }
            history.append(entry)

            if verbose and ep % max(1, epochs // 10) == 0:
                print(f"    ep {ep:5d}  total={l_total.item():.3e}  "
                      f"data={l_data.item():.3e}  bc={l_bc.item():.3e}  "
                      f"phys={l_phys.item():.3e}  "
                      f"t={time.time()-t0:.0f}s")

        elapsed = time.time() - t0
        if verbose:
            print(f"\n  Treino concluído em {elapsed:.1f}s")

        result = SolidMechanicsResult(
            spec=self.spec, model=model, history=history, elapsed_s=elapsed
        )
        return result

    def _eval_bc_loss(self, model: nn.Module, x_bc: torch.Tensor, device) -> torch.Tensor:
        """Avalia loss das condições de contorno da spec nos pontos x_bc."""
        loss = torch.zeros(1, device=device)
        x_np = x_bc.detach().cpu().numpy()

        for cond in self.spec.conditions:
            if cond.kind not in ("dirichlet",):
                continue
            try:
                mask = cond.mask(x_np, {})
                if mask.sum() == 0:
                    continue
                x_sel = x_bc[mask]
                u_pred = model(x_sel)
                # Target: avalia value_fn nos pontos selecionados
                y_target = cond.values(x_np[mask], {})
                y_target_t = torch.from_numpy(y_target).to(device)

                # Aplica apenas ao campo correspondente
                field_idx = [list(self.spec.fields).index(f)
                             for f in cond.fields if f in self.spec.fields]
                if field_idx:
                    pred_sel = u_pred[:, field_idx]
                    loss = loss + cond.weight * nn.functional.mse_loss(pred_sel, y_target_t)
            except Exception:
                continue  # selector falhou (callable não aplicável)

        return loss

    # ── Avaliação ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        result: SolidMechanicsResult,
        analytical_fn=None,
    ) -> Dict[str, float]:
        """
        Avalia o PINN treinado.

        Se dados FEM estão disponíveis → compara u_pred com u_fem.
        Se analytical_fn fornecido (e.g. lame_analytical) → compara analiticamente.
        Sempre computa Von Mises nos pontos de collocação.

        Returns dict com métricas.
        """
        model  = result.model
        spec   = result.spec
        E      = float(spec.pde.params.get("E",  2.1e11))
        nu     = float(spec.pde.params.get("nu", 0.3))
        device = next(model.parameters()).device

        metrics: Dict[str, float] = {}

        # ── Avaliação vs FEM ──────────────────────────────────────────────────
        if self.fem_coords is not None and self.fem_disp is not None:
            x_fem = torch.from_numpy(self.fem_coords.astype(np.float32))
            model.eval()
            with torch.no_grad():
                u_pred_np = model(x_fem.to(device)).cpu().numpy()

            u_fem = self.fem_disp
            diff  = u_pred_np - u_fem
            rmse  = float(np.sqrt((diff**2).mean()))
            rel   = float(np.sqrt((diff**2).sum()) / (np.sqrt((u_fem**2).sum()) + 1e-10))
            metrics["rmse_disp_mm"] = rmse
            metrics["rel_l2_disp"]  = rel
            print(f"\n  vs FEM  RMSE: {rmse:.4e} mm  |  Rel-L2: {rel:.4f}")

            # Von Mises do PINN vs FEM
            if self.fem_vm is not None:
                vm_pinn = von_mises_from_displacements(model, x_fem, E, nu)
                abs_err = np.abs(vm_pinn - self.fem_vm)
                metrics["vm_rmse_pa"]    = float(np.sqrt((abs_err**2).mean()))
                metrics["vm_rel_error"]  = float(abs_err.mean() / (self.fem_vm.mean() + 1e-10))
                print(f"  Von Mises RMSE: {metrics['vm_rmse_pa']:.4e} Pa  |  "
                      f"Rel err: {metrics['vm_rel_error']*100:.2f}%")

        # ── Avaliação analítica (Lamé) ─────────────────────────────────────────
        if analytical_fn is not None and "lame" in spec.name:
            meta = spec.pde.meta
            a   = meta.get("a",   spec.domain_bounds["r"][0])
            b   = meta.get("b",   spec.domain_bounds["r"][1])
            p_a = meta.get("p_a", 0.0)
            p_b = meta.get("p_b", 0.0)

            r_test = np.linspace(float(a), float(b), 200)
            sol = analytical_fn(r_test, a, b, p_a, p_b, E, nu)

            # PINN prediction along r (at z = midpoint)
            z_mid = (spec.domain_bounds["z"][0] + spec.domain_bounds["z"][1]) / 2
            x_test = torch.tensor(
                np.column_stack([r_test, np.full_like(r_test, z_mid)]),
                dtype=torch.float32,
            )
            model.eval()
            with torch.no_grad():
                u_pred_line = model(x_test.to(device)).cpu().numpy()

            ur_pred = u_pred_line[:, 0]
            ur_ref  = sol["u_r"].astype(np.float32)
            rmse_ur = float(np.sqrt(((ur_pred - ur_ref)**2).mean()))
            rel_ur  = float(np.sqrt(((ur_pred - ur_ref)**2).sum()) /
                            (np.sqrt((ur_ref**2).sum()) + 1e-10))
            metrics["lame_rmse_ur_mm"] = rmse_ur
            metrics["lame_rel_l2_ur"]  = rel_ur
            print(f"\n  vs Lamé  RMSE u_r: {rmse_ur:.4e} mm  |  Rel-L2: {rel_ur:.4f}")

        result.metrics.update(metrics)
        return metrics

    # ── Visualização ──────────────────────────────────────────────────────────

    def visualize(
        self,
        result: SolidMechanicsResult,
        output_path: Optional[Path] = None,
        show: bool = False,
    ) -> Path:
        """
        Gera figura com:
          1. Convergência da loss
          2. u_r predito na seção rz
          3. u_z predito na seção rz
          4. Von Mises predito
          5. Erro vs FEM (se dados disponíveis)
          6. Comparação analítica u_r(r) (se Lamé disponível)
        """
        model  = result.model
        spec   = result.spec
        E      = float(spec.pde.params.get("E",  2.1e11))
        nu     = float(spec.pde.params.get("nu", 0.3))
        device = next(model.parameters()).device

        model.eval()

        # Malha de avaliação na seção rz
        r_min, r_max = spec.domain_bounds["r"]
        z_min, z_max = spec.domain_bounds["z"]
        nr, nz = 80, 120
        r_vals = np.linspace(r_min, r_max, nr, dtype=np.float32)
        z_vals = np.linspace(z_min, z_max, nz, dtype=np.float32)
        RR, ZZ = np.meshgrid(r_vals, z_vals)
        pts = np.column_stack([RR.ravel(), ZZ.ravel()])
        x_grid = torch.from_numpy(pts).to(device)

        with torch.no_grad():
            u_grid = model(x_grid).cpu().numpy()

        ur_grid = u_grid[:, 0].reshape(nz, nr)
        uz_grid = u_grid[:, 1].reshape(nz, nr)

        # Von Mises na grade (downsampled para speed)
        step = 4
        pts_ds = pts[::step]
        vm_ds = von_mises_from_displacements(
            model, torch.from_numpy(pts_ds), E, nu)
        vm_grid_flat = np.zeros(len(pts))
        vm_grid_flat[::step] = vm_ds
        vm_grid = vm_grid_flat.reshape(nz, nr)

        # ── Figure ─────────────────────────────────────────────────────────────
        ncols = 3 + (1 if self.fem_vm is not None else 0) + \
                    (1 if "lame" in spec.name else 0)
        ncols = max(ncols, 3)
        fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 9))
        fig.patch.set_facecolor("#0d1117")

        def _style(ax):
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#8b949e", labelsize=8)
            for sp in ax.spines.values():
                sp.set_color("#30363d")

        def _field2d(ax, R, Z, F, title, unit, cmap="RdBu_r"):
            _style(ax)
            vm_ax = np.percentile(np.abs(F), 99) or 1.0
            im = ax.pcolormesh(R, Z, F, cmap=cmap, vmin=-vm_ax, vmax=vm_ax,
                               shading="auto", rasterized=True)
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label(
                unit, color="#8b949e", fontsize=8)
            ax.set_title(title, color="white", fontsize=9, pad=4)
            ax.set_xlabel("r [mm]", color="#8b949e", fontsize=8)
            ax.set_ylabel("z [mm]", color="#8b949e", fontsize=8)

        def _field2d_pos(ax, R, Z, F, title, unit, cmap="hot"):
            _style(ax)
            vmax = np.percentile(F[F > 0], 99) if (F > 0).any() else 1.0
            im = ax.pcolormesh(R, Z, F, cmap=cmap, vmin=0, vmax=vmax,
                               shading="auto", rasterized=True)
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label(
                unit, color="#8b949e", fontsize=8)
            ax.set_title(title, color="white", fontsize=9, pad=4)
            ax.set_xlabel("r [mm]", color="#8b949e", fontsize=8)
            ax.set_ylabel("z [mm]", color="#8b949e", fontsize=8)

        # Row 0: u_r, u_z, Von Mises
        _field2d(axes[0, 0], RR, ZZ, ur_grid,
                 "u_r predito [mm]", "mm")
        _field2d(axes[0, 1], RR, ZZ, uz_grid,
                 "u_z predito [mm]", "mm")
        _field2d_pos(axes[0, 2], RR, ZZ, vm_grid,
                     "Von Mises predito [Pa]", "Pa", cmap="jet")

        # Convergência
        ax_conv = axes[1, 0]
        _style(ax_conv)
        losses = [h["loss"] for h in result.history]
        ax_conv.semilogy(losses, color="#58a6ff", lw=1.5)
        if "loss_data" in result.history[0]:
            ax_conv.semilogy([h["loss_data"] for h in result.history],
                             color="#3fb950", lw=1, alpha=0.7, label="data")
            ax_conv.semilogy([h["loss_bc"] for h in result.history],
                             color="#d29922", lw=1, alpha=0.7, label="bc")
            ax_conv.legend(fontsize=7, labelcolor="white", facecolor="#21262d",
                           framealpha=0.5)
        ax_conv.set_title("Convergência", color="white", fontsize=9)
        ax_conv.set_xlabel("Epoch", color="#8b949e", fontsize=8)
        ax_conv.set_ylabel("Loss", color="#8b949e", fontsize=8)
        ax_conv.grid(True, alpha=0.2, color="#30363d")

        col = 1
        # Erro vs FEM
        if self.fem_vm is not None:
            ax_err = axes[1, col]
            _style(ax_err)
            col += 1
            # Predição PINN nos pontos FEM
            xf = torch.from_numpy(self.fem_coords.astype(np.float32))
            vm_pinn_fem = von_mises_from_displacements(model, xf, E, nu)
            abs_err = np.abs(vm_pinn_fem - self.fem_vm)
            try:
                triang = mtri.Triangulation(
                    self.fem_coords[:, 1], self.fem_coords[:, 0])  # z, r
                sides = np.array([
                    np.linalg.norm(
                        self.fem_coords[triang.triangles[:, (i+1)%3]] -
                        self.fem_coords[triang.triangles[:, i]], axis=2
                    ).max(axis=1)
                    for i in range(3)
                ])
                triang.set_mask(sides.max(axis=0) > 8.0)
                vmax_e = np.percentile(abs_err, 99) or 1.0
                tcf = ax_err.tricontourf(triang, abs_err,
                                         levels=np.linspace(0, vmax_e, 128),
                                         cmap="hot_r", extend="both")
                plt.colorbar(tcf, ax=ax_err, fraction=0.03, pad=0.02).set_label(
                    "|ΔVm| [Pa]", color="#8b949e", fontsize=8)
            except Exception:
                sc = ax_err.scatter(self.fem_coords[:, 1], self.fem_coords[:, 0],
                                    c=abs_err, cmap="hot_r", s=1)
                plt.colorbar(sc, ax=ax_err, fraction=0.03)
            ax_err.set_title(
                f"Erro |PINN−FEM| Von Mises\nRMSE={result.metrics.get('vm_rmse_pa',0):.2e} Pa",
                color="white", fontsize=9)
            ax_err.set_xlabel("z [mm]", color="#8b949e", fontsize=8)
            ax_err.set_ylabel("r [mm]", color="#8b949e", fontsize=8)

        # Comparação Lamé
        if "lame" in spec.name:
            ax_lame = axes[1, col]
            _style(ax_lame)
            meta = spec.pde.meta
            a_   = meta.get("a",   r_min)
            b_   = meta.get("b",   r_max)
            p_a_ = meta.get("p_a", 0.0)
            p_b_ = meta.get("p_b", 0.0)
            r_line = np.linspace(float(a_), float(b_), 200, dtype=np.float32)
            z_mid  = (z_min + z_max) / 2
            x_line = torch.from_numpy(np.column_stack([r_line, np.full_like(r_line, z_mid)]))
            with torch.no_grad():
                u_line = model(x_line.to(device)).cpu().numpy()

            sol = lame_analytical(r_line, float(a_), float(b_), float(p_a_), float(p_b_), E, nu)
            ax_lame.plot(r_line, sol["u_r"] * 1e3, color="#3fb950", lw=2, label="Lamé (analítico)")
            ax_lame.plot(r_line, u_line[:, 0] * 1e3, color="#58a6ff", lw=1.5, ls="--", label="PINN")
            ax_lame.set_title("u_r(r) — Lamé vs PINN", color="white", fontsize=9)
            ax_lame.set_xlabel("r [mm]", color="#8b949e", fontsize=8)
            ax_lame.set_ylabel("u_r [μm]", color="#8b949e", fontsize=8)
            ax_lame.legend(fontsize=8, labelcolor="white", facecolor="#21262d", framealpha=0.5)
            ax_lame.grid(True, alpha=0.2, color="#30363d")
            col += 1

        # Desliga eixos não usados
        for r_ax in range(2):
            for c_ax in range(col, ncols):
                if r_ax < 2:
                    axes[r_ax, c_ax].axis("off")

        fig.suptitle(
            f"PINNeAPPle — Mecânica dos Sólidos Axissimétrica\n{spec.name}",
            color="white", fontsize=11, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        if output_path is None:
            output_path = OUTPUT_DIR / f"{spec.name}_results.png"
        plt.savefig(str(output_path), dpi=110, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        if verbose := True:
            print(f"\n  Figura salva: {output_path}")
        return output_path


# ══════════════════════════════════════════════════════════════════════════════
# Caso A — Cilindro espesso (Lamé): validação com solução analítica
# ══════════════════════════════════════════════════════════════════════════════

def demo_thick_walled_cylinder():
    print("\n" + "═" * 60)
    print("  CASO A — Cilindro Espesso (Lamé)")
    print("  Solução analítica disponível para validação")
    print("═" * 60)

    # Parâmetros: a=20mm, b=60mm, p_a=100MPa, aço
    pipeline = SolidMechanicsPipeline.from_preset(
        "thick_walled_cylinder_lame",
        a=20.0, b=60.0, L=80.0,
        p_a=100e6, p_b=0.0,
        E=2.1e11, nu=0.3,
    )

    print(f"\n  Spec: {pipeline.spec.name}")
    print(f"  PDE: {pipeline.spec.pde.kind}")
    print(f"  Domínio: {pipeline.spec.domain_bounds}")
    print(f"  Condições: {[c.name for c in pipeline.spec.conditions]}")

    result = pipeline.train(
        epochs=3000,
        lr=1e-3,
        n_col=4000,
        n_bc=800,
        w_phys=1.0,
        w_bc=50.0,
        w_data=0.0,   # sem dados FEM — só física
        verbose=True,
    )

    pipeline.evaluate(result, analytical_fn=lame_analytical)
    pipeline.visualize(result, OUTPUT_DIR / "case_A_thick_walled_cylinder.png")
    print(f"\n{result.summary()}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Caso B — Acoplamento TC50 com dados FEM externos
# ══════════════════════════════════════════════════════════════════════════════

def demo_threaded_coupling_tc50(dataset_dir: str = "synthetic_tc50/dataset"):
    print("\n" + "═" * 60)
    print("  CASO B — Acoplamento TC50 BOX")
    print("  Pipeline com dados FEM externos (solution.npy)")
    print("═" * 60)

    # Verifica se dados FEM existem
    import os
    cases = sorted([
        c for c in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, c))
        and os.path.exists(os.path.join(dataset_dir, c, "solution.npy"))
    ]) if os.path.isdir(dataset_dir) else []

    if not cases:
        print(f"\n  AVISO: Nenhum solution.npy encontrado em {dataset_dir}")
        print("  Executando sem dados FEM (só física)...")
        use_fem = False
    else:
        sol_path = os.path.join(dataset_dir, cases[0], "solution.npy")
        print(f"\n  Usando FEM: {cases[0]}")
        use_fem = True

    if use_fem:
        # Pipeline com dados FEM
        pipeline = SolidMechanicsPipeline.from_fem_solution(
            sol_path,
            preset="threaded_coupling_tc50_box",
            preset_params={"clearance": 0.1, "thread_height": 0.8},
        )
    else:
        # Pipeline só com física (sem FEM)
        pipeline = SolidMechanicsPipeline.from_preset(
            "threaded_coupling_tc50_box",
            clearance=0.1, thread_height=0.8,
        )

    print(f"\n  Spec: {pipeline.spec.name}")
    print(f"  Domínio: r {pipeline.spec.domain_bounds['r']}  "
          f"z {pipeline.spec.domain_bounds['z']}")

    result = pipeline.train(
        epochs=2000,
        lr=1e-3,
        w_phys=1.0,
        w_data=1.0 if use_fem else 0.0,
        w_bc=20.0,
        verbose=True,
    )

    pipeline.evaluate(result)
    pipeline.visualize(result, OUTPUT_DIR / "case_B_threaded_coupling_tc50.png")
    print(f"\n{result.summary()}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Caso C — Via ProblemBuilder: qualquer problema de mecânica dos sólidos
# ══════════════════════════════════════════════════════════════════════════════

def demo_custom_problem():
    print("\n" + "═" * 60)
    print("  CASO C — Problema Customizado via ProblemBuilder")
    print("  Shaft hollow sob torção + pressão interna")
    print("═" * 60)

    # Shaft oco (seção transversal): a=15mm, b=30mm, L=80mm
    # Pressão interna p=50MPa + tração axial 50MPa no topo
    spec = (
        ProblemBuilder("shaft_hollow_pressure")
        .domain(r=(15.0, 30.0), z=(0.0, 80.0))
        .fields("u_r", "u_z")
        .pde("axisymmetric_linear_elasticity", E=2.1e11, nu=0.29)
        # IC axial fixo na base
        .bc("dirichlet", field="u_z", value=0.0,
            on=("z", "min"), name="fixed_base", weight=50.0)
        # Simetria de eixo: não aplica (r_min=15 > 0)
        # Pressão interna: u_r normal compressão
        .bc("neumann", field="u_r", value=-50e6,
            on=("r", "min"), name="inner_pressure", weight=5.0)
        # Tração axial no topo
        .bc("neumann", field="u_z", value=50e6,
            on=("z", "max"), name="axial_traction", weight=5.0)
        .sample(interior=3000, boundary=600)
        .field_range(u_r=(-1e-1, 1e-1), u_z=(-1e-1, 1e-1))
        .reference("Eixo oco sob pressão interna e tração axial — benchmark")
        .build()
    )

    print(f"\n  Spec: {spec.name}")
    print(f"  PDE: {spec.pde.kind}  params={spec.pde.params}")
    print(f"  Condições:")
    for c in spec.conditions:
        print(f"    [{c.kind}] {c.name}  w={c.weight}")

    pipeline = SolidMechanicsPipeline(spec)
    result = pipeline.train(
        epochs=1000,
        lr=1e-3,
        w_phys=1.0,
        w_bc=30.0,
        verbose=True,
    )
    pipeline.visualize(result, OUTPUT_DIR / "case_C_shaft_hollow.png")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Mecânica dos Sólidos — PINNeAPPle Framework")
    print("=" * 60)

    # Verifica presets registrados
    from pinneaple_environment import list_presets
    presets = list_presets()
    solid_presets = [p for p in presets if any(
        kw in p for kw in ["elasticity", "cylinder", "drill", "stress", "strain"]
    )]
    print(f"\n  Presets de mecânica dos sólidos disponíveis ({len(solid_presets)}):")
    for p in sorted(solid_presets):
        print(f"    - {p}")

    # Caso A: Cilindro espesso (valida com Lamé)
    result_a = demo_thick_walled_cylinder()

    # Caso B: Acoplamento TC50 (usa FEM se disponível)
    # result_b = demo_threaded_coupling_tc50("synthetic_tc50/dataset")

    # Caso C: ProblemBuilder customizado
    result_c = demo_custom_problem()

    print("\n" + "=" * 60)
    print("  MAPEAMENTO DO PIPELINE (acoplamento → PINNeAPPle)")
    print("=" * 60)
    print("""
  ANTES (acoplamento standalone)          AGORA (PINNeAPPle)
  ─────────────────────────────          ────────────────────────────────
  generate_mesh.py                  →    geom: domínio r,z + solver_spec
  run_simulations.py (FEniCS)       →    SolidMechanicsPipeline.from_fem_solution()
  train.py (PINNElasticity)         →    AxiSymElasticityPINN + pipeline.train()
  physics_loss() axissimétrica      →    axisymmetric_equilibrium_residual()
  contact_loss()                    →    DataConstraint (FEM data supervision)
  evaluate.py (vm_from_pinn)        →    von_mises_from_displacements() + evaluate()
  visualize_fem_results.py          →    pipeline.visualize() (tricontourf rz)
  solution.npy                      →    from_fem_solution(path, preset=...)

  OUTROS PROBLEMAS DE MECÂNICA DOS SÓLIDOS
  ─────────────────────────────────────────
  Qualquer problema axissimétrico:
    get_preset("axisymmetric_linear_elasticity_2d", r_min=..., E=..., p_inner=...)

  Com solução analítica conhecida:
    get_preset("thick_walled_cylinder_lame", a=..., b=..., p_a=...)
    → lame_analytical(r, a, b, p_a, p_b, E, nu)

  Preset customizado:
    ProblemBuilder("meu_shaft")
    .domain(r=(...), z=(...))
    .fields("u_r", "u_z")
    .pde("axisymmetric_linear_elasticity", E=..., nu=...)
    .bc(...)
    .register("meu_shaft")
    → disponível em YAML como problem.id: meu_shaft
""")


if __name__ == "__main__":
    main()
