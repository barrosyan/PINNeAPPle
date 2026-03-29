"""
Pipeline completo — Drill Pipe NC50 em Rotação com PINNeAPPle
=============================================================

Instalação
----------
    pip install PINNeAPPle

    # Ou com extras de geometria e servição:
    pip install "PINNeAPPle[pinn,geom,serve]"

Problema físico
---------------
Conexão roscada NC50 (BOX + PIN) sob carregamento combinado típico de
operação de perfuração rotativa:

    ① Pressão interna (fluido de perfuração)   p_i  = 20 MPa
    ② Tração axial (hook load / WOB)           F    = 500 kN
    ③ Torque de make-up + rotação              T    = 40 kN·m  @ 90 RPM

Formulação PDE
--------------
Coordenadas axissimétricas (r, z). Em elasticidade linear os problemas
meridional e torsional são desacoplados:

  Meridional (u_r, u_z):
    ∂σ_rr/∂r + ∂σ_rz/∂z + (σ_rr − σ_θθ)/r = 0
    ∂σ_rz/∂r + ∂σ_zz/∂z +  σ_rz/r          = 0

  Torsional (u_θ):
    ∂²u_θ/∂r² + (1/r)∂u_θ/∂r − u_θ/r² + ∂²u_θ/∂z² = 0

  Von Mises com 6 componentes de tensão:
    σ_vm = √(½ [(σ_rr−σ_zz)²+(σ_zz−σ_θθ)²+(σ_θθ−σ_rr)²
               + 6(τ_rz²+τ_rθ²+τ_θz²)])

Fluxo do pipeline
-----------------
  [1] Preset registrado  → ProblemSpec  (physics + BCs codificados)
  [2] Dados sintéticos   → gera solution.npy simulando FEM
  [3] Treino PINN        → RotatingDrillPipePINN  (3 saídas: u_r, u_z, u_θ)
  [4] Avaliação          → L2 relativo vs. FEM sintético
  [5] Visualização       → figura 2×3 (campos + Von Mises + histórico)
  [6] Export             → TorchScript .pt  (pronto para C++ / edge deploy)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
import torch.nn as nn

# ── PINNeAPPle ─────────────────────────────────────────────────────────────
from pinneaple_environment import get_preset
from pinneaple_environment.presets.solid_mechanics import lame_analytical

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# PINN — 3 saídas: u_r, u_z, u_θ
# ══════════════════════════════════════════════════════════════════════════════

class RotatingDrillPipePINN(nn.Module):
    """
    PINN de 3 campos para drill pipe NC50 em rotação.

    Entrada : (r, z)      — coordenadas físicas em mm
    Saída   : (u_r, u_z, u_θ) — deslocamentos em mm

    Arquitetura
    -----------
    Dois sub-redes independentes:
      • net_meridional  → (u_r, u_z)  2 saídas
      • net_torsional   → (u_θ,)       1 saída

    A separação reflete o desacoplamento físico e melhora o condicionamento:
    as escalas dos campos meridionais (~μm) e torsionais (~mm) diferem muito.
    Ambas usam Fourier features + ResNet + Tanh.
    """

    def __init__(
        self,
        r_bounds: Tuple[float, float] = (50.0, 84.15),
        z_bounds: Tuple[float, float] = (0.0, 140.0),
        u_rz_scale: float = 1e-3,    # mm — escala meridional
        u_th_scale: float = 1e-2,    # mm — escala torsional (θ·r)
        hidden: int = 128,
        n_layers: int = 6,
        n_fourier: int = 16,
    ):
        super().__init__()
        self.r_min, self.r_max = float(r_bounds[0]), float(r_bounds[1])
        self.z_min, self.z_max = float(z_bounds[0]), float(z_bounds[1])
        self.u_rz_scale = float(u_rz_scale)
        self.u_th_scale = float(u_th_scale)

        torch.manual_seed(42)
        # Fourier features compartilhadas (mesma projeção aleatória)
        self.B = nn.Parameter(torch.randn(n_fourier, 2) * 4.0, requires_grad=False)
        in_dim = 2 * n_fourier

        self.net_meridional = self._make_resnet(in_dim, hidden, n_layers, out_dim=2)
        self.net_torsional   = self._make_resnet(in_dim, hidden, n_layers, out_dim=1)

    @staticmethod
    def _make_resnet(in_dim: int, hidden: int, n_layers: int, out_dim: int) -> nn.ModuleList:
        layers = nn.ModuleList()
        prev = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(prev, hidden))
            prev = hidden
        layers.append(nn.Linear(hidden, out_dim))
        return layers

    def _encode(self, rz_norm: torch.Tensor) -> torch.Tensor:
        proj = rz_norm @ self.B.T
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    @staticmethod
    def _forward_resnet(h: torch.Tensor, net: nn.ModuleList) -> torch.Tensor:
        layers = list(net)
        head   = layers[-1]
        body   = layers[:-1]
        for layer in body:
            h = torch.tanh(layer(h))
        return head(h)   # linear head — no activation

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        r = (x[:, 0:1] - self.r_min) / (self.r_max - self.r_min + 1e-10)
        z = (x[:, 1:2] - self.z_min) / (self.z_max - self.z_min + 1e-10)
        return torch.cat([r, z], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N, 2)  — physical [r, z] in mm
        Returns (N, 3) — [u_r, u_z, u_θ] in mm
        """
        h = self._encode(self._normalize(x))
        u_rz = self._forward_resnet(h, self.net_meridional) * self.u_rz_scale
        u_th = self._forward_resnet(h, self.net_torsional)  * self.u_th_scale
        return torch.cat([u_rz, u_th], dim=-1)   # (N, 3)


# ══════════════════════════════════════════════════════════════════════════════
# Resíduos físicos
# ══════════════════════════════════════════════════════════════════════════════

def meridional_residual(
    model: nn.Module,
    x: torch.Tensor,
    E: float,
    nu: float,
) -> torch.Tensor:
    """
    Resíduo do equilíbrio axissimétrico em (u_r, u_z).
    Retorna (R1² + R2²).mean().

    Formulação via derivadas de segunda ordem DIRETAS de u (não via ∂σ/∂x).
    Evita o "backward through graph a second time": cada _g() é chamado
    exatamente UMA VEZ sobre um tensor diferente — sem nós duplicados no grafo.

    R1 = ∂σ_rr/∂r + ∂σ_rz/∂z + (σ_rr − σ_θθ)/r = 0
    R2 = ∂σ_rz/∂r + ∂σ_zz/∂z + σ_rz/r          = 0

    Expandindo σ em termos de u:
      ∂σ_rr/∂r = (λ+2μ)u_r_rr + λu_z_zr + λu_r_r/r − λu_r/r²
      ∂σ_rz/∂z = μ(u_r_zz + u_z_rz)
      (σ_rr−σ_θθ)/r = 2μ(u_r_r − u_r/r)/r
      ∂σ_rz/∂r = μ(u_r_rz + u_z_rr)
      ∂σ_zz/∂z = (λ+2μ)u_z_zz + λu_r_rz + λu_r_z/r
      σ_rz/r   = μ(u_r_z + u_z_r)/r
    """
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu  = E / (2 * (1 + nu))

    # Folha independente — desacopla o grafo de xb externo
    x = x.detach().requires_grad_(True)
    u = model(x)
    r = x[:, 0:1].clamp(min=1e-3)
    u_r = u[:, 0:1]
    u_z = u[:, 1:2]

    def _g(s, retain=True):
        return torch.autograd.grad(
            s.sum(), x, create_graph=True, retain_graph=retain
        )[0]

    # ── Derivadas de primeira ordem (cada _g chamado UMA VEZ) ────────────
    du_r = _g(u_r)        # [∂u_r/∂r, ∂u_r/∂z]
    du_z = _g(u_z)        # [∂u_z/∂r, ∂u_z/∂z]

    u_r_r = du_r[:, 0:1]   # ∂u_r/∂r
    u_r_z = du_r[:, 1:2]   # ∂u_r/∂z
    u_z_r = du_z[:, 0:1]   # ∂u_z/∂r
    u_z_z = du_z[:, 1:2]   # ∂u_z/∂z

    # ── Derivadas de segunda ordem (cada _g chamado UMA VEZ) ─────────────
    d2u_r_r = _g(u_r_r)          # [∂²u_r/∂r², ∂²u_r/∂r∂z]
    d2u_r_z = _g(u_r_z)          # [∂²u_r/∂z∂r, ∂²u_r/∂z²]
    d2u_z_r = _g(u_z_r)          # [∂²u_z/∂r², ∂²u_z/∂r∂z]
    d2u_z_z = _g(u_z_z)                  # [∂²u_z/∂z∂r, ∂²u_z/∂z²]

    u_r_rr = d2u_r_r[:, 0:1]   # ∂²u_r/∂r²
    u_r_rz = d2u_r_r[:, 1:2]   # ∂²u_r/∂r∂z
    u_r_zz = d2u_r_z[:, 1:2]   # ∂²u_r/∂z²
    u_z_rr = d2u_z_r[:, 0:1]   # ∂²u_z/∂r²
    u_z_rz = d2u_z_r[:, 1:2]   # ∂²u_z/∂r∂z  (= ∂²u_z/∂z∂r por simetria)
    u_z_zr = d2u_z_z[:, 0:1]   # ∂²u_z/∂z∂r
    u_z_zz = d2u_z_z[:, 1:2]   # ∂²u_z/∂z²

    # ── Resíduos de equilíbrio ────────────────────────────────────────────
    R1 = ((lam + 2 * mu) * u_r_rr + lam * u_z_zr
          + lam * u_r_r / r - lam * u_r / r ** 2
          + mu * (u_r_zz + u_z_rz)
          + 2 * mu * (u_r_r - u_r / r) / r)

    R2 = (mu * (u_r_rz + u_z_rr)
          + (lam + 2 * mu) * u_z_zz + lam * u_r_rz
          + lam * u_r_z / r
          + mu * (u_r_z + u_z_r) / r)

    return (R1 ** 2).mean() + (R2 ** 2).mean()


def torsional_residual(
    model: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Resíduo da equação de torção:
        ∂²u_θ/∂r² + (1/r)∂u_θ/∂r − u_θ/r² + ∂²u_θ/∂z² = 0
    """
    # Folha independente — mesmo motivo que meridional_residual
    x = x.detach().requires_grad_(True)
    u = model(x)
    u_th = u[:, 2:3]
    r = x[:, 0:1].clamp(min=1e-3)

    def _g(s, retain=True):
        return torch.autograd.grad(
            s.sum(), x, create_graph=True, retain_graph=retain
        )[0]

    # Primeira ordem — UMA VEZ
    d1  = _g(u_th)
    d_r = d1[:, 0:1]   # ∂u_θ/∂r
    d_z = d1[:, 1:2]   # ∂u_θ/∂z

    # Segunda ordem — cada _g UMA VEZ, retain=True em AMBAS.
    # d_r e d_z compartilham o grafo de d1; se a segunda _g usar retain=False,
    # libera d1's graph antes de o outer backward processar a primeira.
    d2_r = _g(d_r)        # [∂²u_θ/∂r², ∂²u_θ/∂r∂z]
    d2_z = _g(d_z)        # [∂²u_θ/∂z∂r, ∂²u_θ/∂z²]

    u_th_rr = d2_r[:, 0:1]   # ∂²u_θ/∂r²
    u_th_zz = d2_z[:, 1:2]   # ∂²u_θ/∂z²

    R = u_th_rr + d_r / r - u_th / r ** 2 + u_th_zz
    return (R ** 2).mean()


def full_von_mises(
    model: nn.Module,
    x: torch.Tensor,
    E: float,
    nu: float,
) -> np.ndarray:
    """
    Von Mises com todos os 6 componentes de tensão:
        σ_vm = √(½ [(σ_rr−σ_zz)²+(σ_zz−σ_θθ)²+(σ_θθ−σ_rr)²
                   + 6(τ_rz²+τ_rθ²+τ_θz²)])
    """
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu  = E / (2 * (1 + nu))
    model.eval()
    device = next(model.parameters()).device
    results = []

    for i in range(0, len(x), 2048):
        xb = x[i:i + 2048].to(device).detach().requires_grad_(True)
        u  = model(xb)
        r  = xb[:, 0:1].clamp(min=1e-3)

        def _g(s, xb=xb):
            return torch.autograd.grad(
                s.sum(), xb, create_graph=False, retain_graph=True
            )[0]

        du_r = _g(u[:, 0:1])
        du_z = _g(u[:, 1:2])
        du_t = _g(u[:, 2:3])

        e_rr = du_r[:, 0:1]
        e_zz = du_z[:, 1:2]
        e_tt = u[:, 0:1] / r
        e_rz = 0.5 * (du_r[:, 1:2] + du_z[:, 0:1])
        tr   = e_rr + e_zz + e_tt

        s_rr = lam * tr + 2 * mu * e_rr
        s_zz = lam * tr + 2 * mu * e_zz
        s_tt = lam * tr + 2 * mu * e_tt
        t_rz = 2 * mu * e_rz

        # Torsional shear stresses
        t_rt = mu * (du_t[:, 0:1] - u[:, 2:3] / r)   # τ_rθ
        t_tz = mu * du_t[:, 1:2]                        # τ_θz

        vm = torch.sqrt(
            0.5 * (
                (s_rr - s_zz) ** 2 + (s_zz - s_tt) ** 2 + (s_tt - s_rr) ** 2
                + 6 * (t_rz ** 2 + t_rt ** 2 + t_tz ** 2)
            )
        )
        results.append(vm.detach().cpu().numpy().ravel())

    return np.concatenate(results)


# ══════════════════════════════════════════════════════════════════════════════
# Dados sintéticos (substitui FEM real — para demo standalone)
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_fem_data(
    spec,
    n_points: int = 3000,
    seed: int = 0,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Gera dados "FEM" sintéticos usando soluções analíticas aproximadas.

    Meridional: solução de Lamé (cilindro espesso)
    Torsional : u_θ ≈ θ_max · (z/z_max) · r  (torção linear)

    Substitua por dados reais do FEniCS/ABAQUS via:
        np.save("solution.npy", {"coords": X, "disp": U, "stress_vm": S})
    """
    rng = np.random.default_rng(seed)

    r_min, r_max = spec.domain_bounds["r"]
    z_min, z_max = spec.domain_bounds["z"]
    r = rng.uniform(r_min, r_max, n_points).astype(np.float64)
    z = rng.uniform(z_min, z_max, n_points).astype(np.float64)

    p = spec.pde
    E    = p.params["E"]
    nu   = p.params["nu"]
    G    = p.params.get("G", E / (2 * (1 + nu)))
    p_i  = p.params.get("p_inner", 0.0)
    F_ax = p.params.get("F_axial", 0.0)
    T    = p.params.get("T_torque", 0.0)
    theta_max = p.meta.get("theta_max_rad", 0.0)

    A_cross = np.pi * ((r_max * 1e-3) ** 2 - (r_min * 1e-3) ** 2)
    sigma_axial = F_ax / max(A_cross, 1e-10)   # Pa

    # Meridional — Lamé + axial contribution
    lame = lame_analytical(r * 1e-3, r_min * 1e-3, r_max * 1e-3,
                            p_i, 0.0, E, nu)
    u_r = lame["u_r"] * 1e3  # m → mm

    # Axial displacement: linear from 0 at z=z_min (assumed simple)
    lam_c = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu_c  = E / (2 * (1 + nu))
    e_zz_axial = (sigma_axial - lam_c * 2 * (p_i / (2 * max(r_max - r_min, 1) * 1e-3))) / (lam_c + 2 * mu_c)
    u_z = e_zz_axial * (z - z_min) * 1e-3 * 1e3   # mm

    # Torsional — linear twist profile: u_θ(r, z) = θ_max · (z/z_max) · r
    u_th = theta_max * ((z - z_min) / (z_max - z_min)) * r   # mm

    coords = np.column_stack([r, z]).astype(np.float32)
    disp   = np.column_stack([u_r, u_z, u_th]).astype(np.float32)

    # Approximate Von Mises from Lamé stresses + torsional shear
    s_rr = lame["sigma_rr"]
    s_tt = lame["sigma_tt"]
    s_zz = np.full_like(r, sigma_axial)
    tau_tz = G * theta_max / (z_max * 1e-3) * (r * 1e-3)   # τ_θz from linear twist

    vm = np.sqrt(0.5 * (
        (s_rr - s_zz) ** 2 + (s_zz - s_tt) ** 2 + (s_tt - s_rr) ** 2
        + 6 * tau_tz ** 2
    )).astype(np.float32)

    if output_path is None:
        output_path = OUTPUT_DIR / "synthetic_nc50_solution.npy"

    np.save(str(output_path), {"coords": coords, "disp": disp, "stress_vm": vm})
    print(f"  [FEM sintético] {n_points} pontos → {output_path.name}")
    print(f"    u_r  : [{u_r.min() * 1e3:.3f}, {u_r.max() * 1e3:.3f}] μm")
    print(f"    u_z  : [{u_z.min() * 1e3:.3f}, {u_z.max() * 1e3:.3f}] μm")
    print(f"    u_θ  : [{u_th.min() * 1e3:.3f}, {u_th.max() * 1e3:.3f}] μm")
    print(f"    σ_vm : max = {vm.max() / 1e6:.1f} MPa")
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# Resultado
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DrillPipeResult:
    spec: object
    model: nn.Module
    history: List[Dict[str, float]]
    metrics: Dict[str, float] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"Drill Pipe NC50 Rotating — {self.spec.name}",
            "=" * 60,
            f"  Epochs    : {len(self.history)}",
            f"  Tempo     : {self.elapsed_s:.1f} s",
            f"  Loss final: {self.history[-1]['loss']:.4e}",
            "",
            "  Métricas de avaliação:",
        ]
        for k, v in self.metrics.items():
            lines.append(f"    {k:<22}: {v:.4e}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline principal
# ══════════════════════════════════════════════════════════════════════════════

class RotatingDrillPipePipeline:
    """
    Pipeline PINN completo para drill pipe NC50 em rotação.

    Uso rápido
    ----------
    # A partir do preset (sem dados FEM):
    pipeline = RotatingDrillPipePipeline.from_preset(
        "drill_pipe_nc50_rotating", body="BOX",
        p_inner=20e6, F_axial=500e3, T_torque=40e3, rpm=90,
    )
    result = pipeline.train(epochs=3000)

    # Com dados FEM (solution.npy com 3 colunas de deslocamento):
    pipeline = RotatingDrillPipePipeline.from_fem_solution(
        "solution.npy",
        preset="drill_pipe_nc50_rotating",
        preset_params={"body": "BOX", "p_inner": 20e6, ...},
    )
    result = pipeline.train(epochs=3000)
    pipeline.evaluate(result)
    pipeline.visualize(result)
    pipeline.export(result)
    """

    def __init__(self, spec, fem_coords=None, fem_disp=None, fem_vm=None):
        self.spec        = spec
        self.fem_coords  = fem_coords   # (N, 2): r, z
        self.fem_disp    = fem_disp     # (N, 3): u_r, u_z, u_θ
        self.fem_vm      = fem_vm       # (N,)

    # ── Construtores ──────────────────────────────────────────────────────────

    @classmethod
    def from_preset(cls, preset_id: str, **params) -> "RotatingDrillPipePipeline":
        """Cria pipeline a partir de preset registrado, sem dados FEM."""
        spec = get_preset(preset_id, **params)
        return cls(spec)

    @classmethod
    def from_fem_solution(
        cls,
        solution_npy: str | Path,
        preset: str = "drill_pipe_nc50_rotating",
        preset_params: Optional[Dict] = None,
    ) -> "RotatingDrillPipePipeline":
        """
        Carrega dados FEM e cria pipeline.

        solution.npy deve conter:
            coords   : (N, 2) — [r, z] em mm
            disp     : (N, 3) — [u_r, u_z, u_θ] em mm
            stress_vm: (N,)   — Von Mises em Pa (opcional)
        """
        path = Path(solution_npy)
        if not path.exists():
            raise FileNotFoundError(f"FEM solution não encontrada: {path}")

        data   = np.load(str(path), allow_pickle=True).item()
        coords = data["coords"]
        disp   = data["disp"]
        vm     = data.get("stress_vm", None)

        r_min = float(coords[:, 0].min())
        r_max = float(coords[:, 0].max())
        z_min = float(coords[:, 1].min())
        z_max = float(coords[:, 1].max())

        params = dict(preset_params or {})
        spec = get_preset(preset, **params)

        print(f"\n  [FEM] {len(coords):,} pontos  |  "
              f"r ∈ [{r_min:.1f}, {r_max:.1f}]  z ∈ [{z_min:.1f}, {z_max:.1f}] mm")
        if vm is not None:
            print(f"  [FEM] σ_vm: max={vm.max() / 1e6:.1f} MPa  "
                  f"mean={vm.mean() / 1e6:.1f} MPa")

        return cls(spec, fem_coords=coords, fem_disp=disp, fem_vm=vm)

    # ── Amostragem ────────────────────────────────────────────────────────────

    def _sample(self, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        r0, r1 = self.spec.domain_bounds["r"]
        z0, z1 = self.spec.domain_bounds["z"]
        return np.column_stack([
            rng.uniform(r0, r1, n),
            rng.uniform(z0, z1, n),
        ]).astype(np.float32)

    def _sample_bc(self, n_per_face: int, seed: int = 1) -> np.ndarray:
        rng = np.random.default_rng(seed)
        r0, r1 = self.spec.domain_bounds["r"]
        z0, z1 = self.spec.domain_bounds["z"]
        n = n_per_face
        return np.vstack([
            np.column_stack([rng.uniform(r0, r1, n), np.full(n, z0)]),
            np.column_stack([rng.uniform(r0, r1, n), np.full(n, z1)]),
            np.column_stack([np.full(n, r0), rng.uniform(z0, z1, n)]),
            np.column_stack([np.full(n, r1), rng.uniform(z0, z1, n)]),
        ]).astype(np.float32)

    # ── Treino ────────────────────────────────────────────────────────────────

    def train(
        self,
        epochs: int = 5000,
        lr: float = 1e-3,
        n_col: int = 10000,
        n_bc: int = 2500,
        batch_col: int = 2048,
        w_meri: float = 1.0,    # peso resíduo meridional
        w_tors: float = 1.0,    # peso resíduo torsional
        w_bc: float = 20.0,     # peso BCs Dirichlet
        w_data: float = 5.0,    # peso dados FEM (se disponíveis)
        device: str = "auto",
        seed: int = 42,
        verbose: bool = True,
    ) -> DrillPipeResult:
        """
        Treina o PINN nos dois sistemas de PDE (meridional + torsional).

        Pesos sugeridos
        ---------------
        Sem dados FEM     : w_meri=1, w_tors=1, w_bc=20, w_data=0
        Com dados FEM     : w_meri=0.1, w_tors=0.1, w_bc=20, w_data=10
        (Dados FEM dominam; física estabiliza extrapolação fora dos dados)
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        dev = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "cpu" if device == "auto" else device
        )

        E  = float(self.spec.pde.params["E"])
        nu = float(self.spec.pde.params["nu"])

        r_bounds = self.spec.domain_bounds["r"]
        z_bounds = self.spec.domain_bounds["z"]
        u_char   = float(self.spec.scales.U)
        theta_max = float(self.spec.pde.meta.get("theta_max_rad", 1e-3))
        u_th_char = theta_max * r_bounds[1]

        model = RotatingDrillPipePINN(
            r_bounds=r_bounds,
            z_bounds=z_bounds,
            u_rz_scale=max(u_char * 0.5, 1e-6),
            u_th_scale=max(u_th_char * 0.5, 1e-8),
        ).to(dev)

        optim = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=lr * 0.01)

        x_col = torch.from_numpy(self._sample(n_col, seed)).to(dev)
        x_bc  = torch.from_numpy(self._sample_bc(n_bc // 4, seed)).to(dev)

        has_fem = self.fem_coords is not None
        if has_fem:
            _n_d = min(15000, len(self.fem_coords))
            idx  = np.random.default_rng(seed).choice(len(self.fem_coords), _n_d, replace=False)
            x_fd = torch.from_numpy(self.fem_coords[idx].astype(np.float32)).to(dev)
            y_fd = torch.from_numpy(self.fem_disp[idx].astype(np.float32)).to(dev)
            # Verifica se FEM tem 3 campos (u_r, u_z, u_θ) ou apenas 2
            n_fem_fields = y_fd.shape[1]

        n_params = sum(p.numel() for p in model.parameters())
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  RotatingDrillPipePINN — {self.spec.name}")
            print(f"  Parâmetros : {n_params:,}   |   device: {dev}")
            print(f"  Col: {n_col}  BC: {n_bc}  FEM: {_n_d if has_fem else 0}")
            print(f"  Pesos  meri={w_meri}  tors={w_tors}  bc={w_bc}  data={w_data}")
            print(f"  θ_max = {theta_max * 180 / np.pi:.3f}°  "
                  f"u_rz_char = {u_char * 1e3:.3f} μm  "
                  f"u_θ_char  = {u_th_char * 1e3:.3f} μm")
            print(f"{'=' * 60}")

        history = []
        t0 = time.time()

        for ep in range(1, epochs + 1):
            model.train()
            optim.zero_grad()

            # Mini-batch de collocação
            idx_c = torch.randperm(n_col, device=dev)[:batch_col]
            xb    = x_col[idx_c]

            # Physics losses
            l_meri = meridional_residual(model, xb, E, nu)
            l_tors = torsional_residual(model, xb)

            # BC loss — avalia condições Dirichlet registradas na spec
            l_bc = self._bc_loss(model, x_bc, dev)

            # Dados FEM
            l_data = torch.zeros(1, device=dev)
            if has_fem:
                idx_d  = torch.randperm(len(x_fd), device=dev)[:batch_col]
                u_pred = model(x_fd[idx_d])
                target = y_fd[idx_d]
                if n_fem_fields == 2:
                    # FEM só tem u_r, u_z — compara apenas campos meridionais
                    l_data = nn.functional.mse_loss(u_pred[:, :2], target)
                else:
                    l_data = nn.functional.mse_loss(u_pred, target)

            # Escala: resíduos físicos em Pa²/(mm²) → normaliza
            phys_scale_meri = (u_char / max(E * 1e-3, 1.0)) ** 2
            phys_scale_tors = max(theta_max ** 2, 1e-20)

            l_total = (
                w_meri * l_meri / max(phys_scale_meri, 1e-20)
                + w_tors * l_tors / max(phys_scale_tors, 1e-20)
                + w_bc   * l_bc
                + w_data * l_data
            )

            l_total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            entry = {
                "epoch": ep,
                "loss":      float(l_total),
                "loss_meri": float(l_meri),
                "loss_tors": float(l_tors),
                "loss_bc":   float(l_bc),
                "loss_data": float(l_data),
            }
            history.append(entry)

            if verbose and ep % max(1, epochs // 10) == 0:
                print(f"  ep {ep:5d}  total={l_total:.3e}  "
                      f"meri={l_meri:.3e}  tors={l_tors:.3e}  "
                      f"bc={l_bc:.3e}  data={l_data:.3e}  "
                      f"t={time.time() - t0:.0f}s")

        elapsed = time.time() - t0
        if verbose:
            print(f"\n  Treino concluído em {elapsed:.1f}s")

        return DrillPipeResult(
            spec=self.spec, model=model, history=history, elapsed_s=elapsed
        )

    def _bc_loss(
        self, model: nn.Module, x_bc: torch.Tensor, dev: torch.device
    ) -> torch.Tensor:
        """Avalia condições Dirichlet da spec nos pontos de contorno."""
        loss   = torch.zeros(1, device=dev)
        x_np   = x_bc.detach().cpu().numpy()
        fields = list(self.spec.fields)

        for cond in self.spec.conditions:
            if cond.kind != "dirichlet":
                continue
            try:
                mask = cond.mask(x_np, {})
                if mask.sum() == 0:
                    continue
                x_sel  = x_bc[mask]
                u_pred = model(x_sel)

                # Avalia valor-alvo
                y_np  = cond.values(x_np[mask], {})
                y_t   = torch.from_numpy(y_np.astype(np.float32)).to(dev)

                fidxs = [fields.index(f) for f in cond.fields if f in fields]
                if fidxs:
                    loss = loss + cond.weight * nn.functional.mse_loss(
                        u_pred[:, fidxs], y_t
                    )
            except Exception:
                continue

        return loss

    # ── Avaliação ─────────────────────────────────────────────────────────────

    def evaluate(self, result: DrillPipeResult) -> Dict[str, float]:
        """
        Calcula métricas vs. dados FEM (se disponíveis).

        Retorna L2 relativo para cada campo e para Von Mises.
        """
        model = result.model
        model.eval()
        dev   = next(model.parameters()).device

        metrics: Dict[str, float] = {}

        if self.fem_coords is not None and self.fem_disp is not None:
            x_t = torch.from_numpy(self.fem_coords.astype(np.float32)).to(dev)
            with torch.no_grad():
                u_pred = model(x_t).cpu().numpy()

            u_true = self.fem_disp
            n_fields = min(u_pred.shape[1], u_true.shape[1])
            names   = ["u_r", "u_z", "u_θ"][:n_fields]
            for i, name in enumerate(names):
                err = np.linalg.norm(u_pred[:, i] - u_true[:, i])
                ref = np.linalg.norm(u_true[:, i]) + 1e-30
                metrics[f"L2_rel_{name}"] = float(err / ref)

            if self.fem_vm is not None:
                E  = float(self.spec.pde.params["E"])
                nu = float(self.spec.pde.params["nu"])
                vm_pred = full_von_mises(model, x_t, E, nu)
                vm_true = self.fem_vm
                metrics["L2_rel_vm"] = float(
                    np.linalg.norm(vm_pred - vm_true) / (np.linalg.norm(vm_true) + 1e-30)
                )
                metrics["vm_max_pred_MPa"] = float(vm_pred.max() / 1e6)
                metrics["vm_max_true_MPa"] = float(vm_true.max() / 1e6)

        result.metrics = metrics

        print("\n  Métricas de avaliação:")
        for k, v in metrics.items():
            print(f"    {k:<26}: {v:.4e}")

        return metrics

    # ── Visualização ──────────────────────────────────────────────────────────

    def visualize(
        self,
        result: DrillPipeResult,
        output_path: Optional[Path] = None,
        n_grid: int = 80,
    ) -> Path:
        """
        Figura 2×3: u_r, u_z, u_θ, σ_vm (PINN vs FEM) + histórico de loss.

        Layout
        ------
        [u_r PINN] [u_z PINN] [u_θ PINN]
        [σ_vm PINN] [σ_vm FEM (se disp.)] [Histórico de loss]
        """
        model = result.model
        model.eval()
        dev   = next(model.parameters()).device

        r0, r1 = self.spec.domain_bounds["r"]
        z0, z1 = self.spec.domain_bounds["z"]

        r_grid = np.linspace(r0, r1, n_grid).astype(np.float32)
        z_grid = np.linspace(z0, z1, n_grid).astype(np.float32)
        RR, ZZ = np.meshgrid(r_grid, z_grid)
        x_grid = np.column_stack([RR.ravel(), ZZ.ravel()])

        x_t = torch.from_numpy(x_grid).to(dev)
        with torch.no_grad():
            u_grid = model(x_t).cpu().numpy()

        E  = float(self.spec.pde.params["E"])
        nu = float(self.spec.pde.params["nu"])
        vm_grid = full_von_mises(model, x_t, E, nu)

        u_r_grid  = u_grid[:, 0].reshape(n_grid, n_grid)
        u_z_grid  = u_grid[:, 1].reshape(n_grid, n_grid)
        u_th_grid = u_grid[:, 2].reshape(n_grid, n_grid)
        vm_map    = vm_grid.reshape(n_grid, n_grid) / 1e6  # MPa

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(
            f"Drill Pipe NC50 — {self.spec.name}\n"
            f"p_i={self.spec.pde.params.get('p_inner', 0) / 1e6:.0f} MPa  "
            f"F={self.spec.pde.params.get('F_axial', 0) / 1e3:.0f} kN  "
            f"T={self.spec.pde.params.get('T_torque', 0) / 1e3:.0f} kN·m  "
            f"RPM={self.spec.pde.params.get('rpm', 0):.0f}",
            fontsize=13, fontweight="bold"
        )

        kw = dict(cmap="RdBu_r", aspect="auto", origin="lower",
                  extent=[r0, r1, z0, z1])

        def _panel(ax, data, label, unit="μm", scale=1e3):
            im = ax.imshow(data * scale, **kw)
            ax.set_xlabel("r [mm]")
            ax.set_ylabel("z [mm]")
            ax.set_title(label)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(unit)

        _panel(axes[0, 0], u_r_grid,  "u_r — radial",  "μm",   scale=1e3)
        _panel(axes[0, 1], u_z_grid,  "u_z — axial",   "μm",   scale=1e3)
        _panel(axes[0, 2], u_th_grid, "u_θ — torsional","μm",  scale=1e3)
        _panel(axes[1, 0], vm_map,    "σ_vm (PINN)", "MPa", scale=1.0)

        # Von Mises FEM (se disponível)
        ax_fem = axes[1, 1]
        if self.fem_coords is not None and self.fem_vm is not None:
            r_f = self.fem_coords[:, 0]
            z_f = self.fem_coords[:, 1]
            tri = mtri.Triangulation(r_f, z_f)
            cf  = ax_fem.tricontourf(tri, self.fem_vm / 1e6, levels=20,
                                     cmap="RdBu_r")
            ax_fem.set_title("σ_vm (FEM sintético)")
            ax_fem.set_xlabel("r [mm]")
            ax_fem.set_ylabel("z [mm]")
            fig.colorbar(cf, ax=ax_fem, fraction=0.046, pad=0.04).set_label("MPa")
        else:
            ax_fem.axis("off")
            ax_fem.text(0.5, 0.5, "Sem dados FEM", ha="center", va="center",
                        transform=ax_fem.transAxes, fontsize=12, color="gray")

        # Histórico de loss
        ax_loss = axes[1, 2]
        epochs  = [h["epoch"] for h in result.history]
        ax_loss.semilogy(epochs, [h["loss"]      for h in result.history], label="total", lw=2)
        ax_loss.semilogy(epochs, [h["loss_meri"] for h in result.history], label="meridional", ls="--")
        ax_loss.semilogy(epochs, [h["loss_tors"] for h in result.history], label="torsional",  ls="--")
        ax_loss.semilogy(epochs, [h["loss_bc"]   for h in result.history], label="BC",         ls=":")
        if any(h["loss_data"] > 0 for h in result.history):
            ax_loss.semilogy(epochs, [h["loss_data"] for h in result.history], label="FEM data", ls="-.")
        ax_loss.set_xlabel("Época")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Histórico de treinamento")
        ax_loss.legend(fontsize=8)
        ax_loss.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path is None:
            output_path = OUTPUT_DIR / f"drill_pipe_rotating_{self.spec.name}.png"

        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Figura salva → {output_path}")
        return output_path

    # ── Export ────────────────────────────────────────────────────────────────

    def export(
        self,
        result: DrillPipeResult,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """
        Exporta o modelo em 2 formatos:
          • TorchScript (.pt)   — para C++ / edge
          • ONNX (.onnx)        — para inference servers (se onnx instalado)

        Retorna dict com caminhos dos arquivos gerados.
        """
        if output_dir is None:
            output_dir = OUTPUT_DIR

        model = result.model
        model.eval()

        paths: Dict[str, Path] = {}

        # TorchScript
        dummy = torch.zeros(1, 2)
        traced = torch.jit.trace(model, dummy)
        ts_path = output_dir / f"{self.spec.name}_model.pt"
        traced.save(str(ts_path))
        paths["torchscript"] = ts_path
        print(f"  TorchScript → {ts_path.name}")

        # Fidelity check: erro relativo com TorchScript
        with torch.no_grad():
            out_orig   = model(dummy)
            out_traced = traced(dummy)
        fid_err = float((out_orig - out_traced).abs().max())
        print(f"  Fidelidade TorchScript: erro_max = {fid_err:.2e}")

        # ONNX (opcional)
        try:
            import onnx  # noqa: F401
            onnx_path = output_dir / f"{self.spec.name}_model.onnx"
            torch.onnx.export(
                model, dummy, str(onnx_path),
                input_names=["rz_coords"],
                output_names=["displacements_ur_uz_uth"],
                dynamic_axes={"rz_coords": {0: "batch"}, "displacements_ur_uz_uth": {0: "batch"}},
                opset_version=17,
            )
            paths["onnx"] = onnx_path
            print(f"  ONNX        → {onnx_path.name}")
        except ImportError:
            print("  ONNX: não instalado (pip install onnx onnxruntime)")

        return paths


# ══════════════════════════════════════════════════════════════════════════════
# Demo completo
# ══════════════════════════════════════════════════════════════════════════════

def run_demo(epochs: int = 2000, device: str = "auto"):
    """
    Executa o pipeline completo em modo demo.

    Para produção, substitua generate_synthetic_fem_data() pelo seu
    pipeline FEniCS/ABAQUS real e passe o solution.npy para
    RotatingDrillPipePipeline.from_fem_solution().
    """
    print("\n" + "=" * 60)
    print("  DRILL PIPE NC50 — PIPELINE COMPLETO (PINNeAPPle)")
    print("=" * 60)

    # ── [1] Preset ────────────────────────────────────────────────────────────
    print("\n[1] Carregando preset 'drill_pipe_nc50_rotating'...")
    spec = get_preset(
        "drill_pipe_nc50_rotating",
        body="BOX",
        p_inner=20e6,       # 20 MPa — fluido de perfuração
        F_axial=500e3,      # 500 kN — hook load
        T_torque=40e3,      # 40 kN·m — torque de make-up + rotação
        rpm=90.0,           # 90 RPM
    )
    print(f"  Preset: {spec.name}")
    print(f"  Campos: {spec.fields}")
    print(f"  Domínio: r ∈ {spec.domain_bounds['r']} mm  z ∈ {spec.domain_bounds['z']} mm")
    print(f"  BCs registradas: {[c.name for c in spec.conditions]}")
    print(f"  θ_max = {spec.pde.meta['theta_max_rad'] * 180 / np.pi:.3f}°")

    # ── [2] Dados sintéticos (substitua por FEM real) ─────────────────────────
    print("\n[2] Gerando dados FEM sintéticos...")
    fem_path = generate_synthetic_fem_data(spec, n_points=3000)

    # ── [3] Pipeline + treino ─────────────────────────────────────────────────
    print("\n[3] Criando pipeline com dados FEM...")
    pipeline = RotatingDrillPipePipeline.from_fem_solution(
        fem_path,
        preset="drill_pipe_nc50_rotating",
        preset_params={
            "body": "BOX",
            "p_inner": 20e6,
            "F_axial": 500e3,
            "T_torque": 40e3,
            "rpm": 90.0,
        },
    )

    print(f"\n[4] Treinando PINN ({epochs} épocas)...")
    result = pipeline.train(
        epochs=epochs,
        lr=1e-3,
        n_col=8000,
        n_bc=2000,
        w_meri=0.1,    # física meridional — menor peso quando há dados FEM
        w_tors=0.1,    # física torsional
        w_bc=20.0,     # BCs são críticas
        w_data=10.0,   # dados FEM dominam
        device=device,
        verbose=True,
    )

    # ── [4] Avaliação ─────────────────────────────────────────────────────────
    print("\n[5] Avaliando...")
    pipeline.evaluate(result)
    print("\n" + result.summary())

    # ── [5] Visualização ──────────────────────────────────────────────────────
    print("\n[6] Visualizando...")
    pipeline.visualize(result)

    # ── [6] Export ────────────────────────────────────────────────────────────
    print("\n[7] Exportando modelo...")
    exported = pipeline.export(result)
    for fmt, path in exported.items():
        print(f"  {fmt}: {path}")

    print("\n  Pipeline concluído! Resultados em:", OUTPUT_DIR)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Drill Pipe NC50 Rotating — PINNeAPPle")
    parser.add_argument("--epochs",  type=int,   default=2000,  help="Épocas de treino")
    parser.add_argument("--device",  type=str,   default="auto", help="cpu/cuda/auto")
    parser.add_argument("--fem",     type=str,   default=None,   help="Caminho p/ solution.npy real")
    parser.add_argument("--body",    type=str,   default="BOX",  help="BOX ou PIN")
    parser.add_argument("--p-inner", type=float, default=20e6)
    parser.add_argument("--f-axial", type=float, default=500e3)
    parser.add_argument("--torque",  type=float, default=40e3)
    parser.add_argument("--rpm",     type=float, default=90.0)
    args = parser.parse_args()

    if args.fem:
        # Modo produção: usa FEM real
        print(f"\nModo produção — FEM: {args.fem}")
        pipeline = RotatingDrillPipePipeline.from_fem_solution(
            args.fem,
            preset="drill_pipe_nc50_rotating",
            preset_params={
                "body":     args.body,
                "p_inner":  args.p_inner,
                "F_axial":  args.f_axial,
                "T_torque": args.torque,
                "rpm":      args.rpm,
            },
        )
        result = pipeline.train(epochs=args.epochs, device=args.device)
        pipeline.evaluate(result)
        pipeline.visualize(result)
        pipeline.export(result)
    else:
        # Modo demo
        run_demo(epochs=args.epochs, device=args.device)
