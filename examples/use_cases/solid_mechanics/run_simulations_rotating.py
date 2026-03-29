"""
run_simulations_rotating.py — FEM axissimétrico com torção (PINNeAPPle)
========================================================================

Estende o `run_simulations.py` original para resolver o problema completo
do drill pipe em rotação: pressão interna + tração axial + torque.

Formulação FEM (dolfinx)
------------------------
Os problemas meridional e torsional são desacoplados em elasticidade linear,
resolvidos como dois sistemas independentes no mesmo mesh.

  ① Meridional (u_r, u_z) — axissimétrico padrão:
     Forma fraca com peso r:
       ∫ σ:ε(v) · r dr dz = ∫ f·v · r ds

  ② Torsional (u_θ) — equação de Laplace modificada:
     ∂²u_θ/∂r² + (1/r)∂u_θ/∂r − u_θ/r² + ∂²u_θ/∂z² = 0
     Forma fraca com peso r:
       ∫ [∂u_θ/∂r ∂v/∂r + ∂u_θ/∂z ∂v/∂z + u_θ v/r²] r dr dz
         = ∫ τ_θz_top · v · r ds_top
       onde τ_θz_top = T / (2π ∫_{r_bore}^{r_outer} r² dr)

  Von Mises 6-componentes:
     σ_vm = √(½ [(σ_rr−σ_zz)²+(σ_zz−σ_θθ)²+(σ_θθ−σ_rr)²+6(τ_rz²+τ_rθ²+τ_θz²)])

Saída por caso: solution.npy com chaves:
  coords    : (N, 2) — [r, z] em mm
  disp      : (N, 3) — [u_r, u_z, u_θ] em mm
  stress_vm : (N,)   — Von Mises em Pa

Instalação
----------
  Requer FEniCS/dolfinx no Linux (WSL ou container Docker):
    pip install "PINNeAPPle[fenics]"   # instala mpi4py, petsc4py
    # dolfinx deve ser instalado separadamente via conda ou container:
    # conda install -c conda-forge fenics-dolfinx

  O script roda no WSL com:
    cd ~/drill_pipe
    python run_simulations_rotating.py

Uso
---
  # Todos os casos do dataset:
  python run_simulations_rotating.py

  # Dataset alternativo:
  python run_simulations_rotating.py --dataset-dir meu_dataset

  # Só um caso específico:
  python run_simulations_rotating.py --case dataset/case_003

  # Força resimulação mesmo se solution.npy já existir:
  python run_simulations_rotating.py --force

  # Parâmetros de carregamento:
  python run_simulations_rotating.py \\
      --p-inner 20e6 --f-axial 500e3 --torque 40e3
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import meshio
import numpy as np
from mpi4py import MPI

import ufl
from dolfinx import fem, io
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import LinearProblem

# ── PINNeAPPle ─────────────────────────────────────────────────────────────
# O preset fornece os parâmetros físicos validados (E, nu, cargas)
from pinneaple_environment import get_preset

# ══════════════════════════════════════════════════════════════════════════════
# Parâmetros físicos padrão (NC50, AISI 4145H)
# ══════════════════════════════════════════════════════════════════════════════

_E      = 2.1e11    # Pa
_NU     = 0.3
_MU     = _E / (2 * (1 + _NU))
_LMBDA  = _E * _NU / ((1 + _NU) * (1 - 2 * _NU))

DATASET_DIR = "dataset"


# ══════════════════════════════════════════════════════════════════════════════
# Utilitários de mesh
# ══════════════════════════════════════════════════════════════════════════════

def msh_to_dolfinx(
    msh_file: str | Path,
    xdmf_file: str | Path,
) -> Tuple[dmesh.Mesh, float, float]:
    """
    Converte .msh (gmsh) → domínio dolfinx via XDMF.
    Retorna (domain, z_min, z_max).
    """
    msh_data   = meshio.read(str(msh_file))
    tri_blocks = [c for c in msh_data.cells if c.type in ("triangle", "triangle6")]
    if not tri_blocks:
        raise RuntimeError(f"Nenhum triângulo 2D em {msh_file}.")

    cell_data = np.vstack([
        c.data[:, :3] if c.type == "triangle6" else c.data
        for c in tri_blocks
    ])
    points_2d = msh_data.points[:, :2]

    meshio.write(str(xdmf_file),
                 meshio.Mesh(points=points_2d, cells={"triangle": cell_data}))
    print(f"    Nós: {len(points_2d):,}  |  Triângulos: {len(cell_data):,}")

    with io.XDMFFile(MPI.COMM_WORLD, str(xdmf_file), "r") as xf:
        domain = xf.read_mesh(name="Grid")

    coords = domain.geometry.x
    return domain, float(coords[:, 1].min()), float(coords[:, 1].max())


# ══════════════════════════════════════════════════════════════════════════════
# Solve ① — Meridional (u_r, u_z): axissimétrico com pressão + tração axial
# ══════════════════════════════════════════════════════════════════════════════

def solve_meridional(
    domain,
    z_min: float,
    z_max: float,
    r_bore: float,          # mm — raio do bore interno
    E: float     = _E,
    nu: float    = _NU,
    p_inner: float = 20e6, # Pa — pressão interna
    F_axial: float = 500e3, # N  — tração axial
    r_outer: float = 84.15, # mm — raio externo
    petsc_prefix: str = "meri_",
) -> Tuple[object, np.ndarray, np.ndarray]:
    """
    Resolve o problema meridional (u_r, u_z) com:
      - Pressão interna: σ_rr = −p_inner  em r = r_bore
      - Tração axial: σ_zz = F/A  no topo
      - u_z = 0 no nariz (z = z_min)

    Retorna (u_sol, vm_values, W_coords).
    """
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu  = E / (2 * (1 + nu))

    V    = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    u_tr = ufl.TrialFunction(V)
    v_te = ufl.TestFunction(V)

    mu_c  = fem.Constant(domain, np.float64(mu))
    lam_c = fem.Constant(domain, np.float64(lam))

    x = ufl.SpatialCoordinate(domain)
    r = x[0]

    def eps_axi(u):
        return ufl.as_vector([
            u[0].dx(0),
            u[1].dx(1),
            u[0] / r,
            u[0].dx(1) + u[1].dx(0),
        ])

    def sigma_axi(u):
        e   = eps_axi(u)
        tr  = e[0] + e[1] + e[2]
        return ufl.as_vector([
            lam_c * tr + 2 * mu_c * e[0],
            lam_c * tr + 2 * mu_c * e[1],
            lam_c * tr + 2 * mu_c * e[2],
            mu_c * e[3],
        ])

    def inner_axi(s, e):
        return s[0]*e[0] + s[1]*e[1] + s[2]*e[2] + s[3]*e[3]

    a = inner_axi(sigma_axi(u_tr), eps_axi(v_te)) * r * ufl.dx

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    # ── Tração axial no topo ──────────────────────────────────────────────
    A_cross = np.pi * ((r_outer * 1e-3) ** 2 - (r_bore * 1e-3) ** 2)  # m²
    sigma_axial = F_axial / max(A_cross, 1e-12)                          # Pa

    top_facets = dmesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[1], z_max, atol=1.5))
    ft_tags = dmesh.meshtags(domain, fdim, top_facets,
                              np.ones(len(top_facets), dtype=np.int32))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft_tags)
    traction = fem.Constant(domain, np.array([0.0, sigma_axial]))
    L = ufl.dot(traction, v_te) * r * ds(1)

    # ── Pressão interna no bore ───────────────────────────────────────────
    bore_facets = dmesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[0], r_bore, atol=1.5))
    if len(bore_facets) > 0:
        bore_tags = dmesh.meshtags(domain, fdim, bore_facets,
                                    np.full(len(bore_facets), 2, dtype=np.int32))
        ds2 = ufl.Measure("ds", domain=domain, subdomain_data=bore_tags)
        # Pressão interna: força radial para fora = −p_inner · n̂  (n̂ = −r̂ no bore)
        pressure_load = fem.Constant(domain, np.array([p_inner, 0.0]))
        L = L + ufl.dot(pressure_load, v_te) * r * ds2(2)

    # ── BC: u_z = 0 no nariz ─────────────────────────────────────────────
    bot_facets = dmesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[1], z_min, atol=1.5))
    V_uz, _ = V.sub(1).collapse()
    dofs_bot = fem.locate_dofs_topological((V.sub(1), V_uz), fdim, bot_facets)
    bc_z0 = fem.dirichletbc(np.float64(0.0), dofs_bot[0], V.sub(1))

    problem = LinearProblem(a, L, bcs=[bc_z0],
                             petsc_options={"ksp_type": "cg", "pc_type": "gamg",
                                            "ksp_rtol": 1e-10},
                             petsc_options_prefix=petsc_prefix)
    u_sol = problem.solve()

    # ── Von Mises (4 componentes meridionais) ─────────────────────────────
    W = fem.functionspace(domain, ("Lagrange", 1))

    def von_mises_meri(u):
        s = sigma_axi(u)
        srr, szz, stt, srz = s[0], s[1], s[2], s[3]
        return ufl.sqrt(
            ((srr - szz) ** 2 + (szz - stt) ** 2 + (stt - srr) ** 2 + 6 * srz ** 2) / 2
        )

    vm_func = fem.Function(W)
    vm_func.interpolate(fem.Expression(
        von_mises_meri(u_sol), W.element.interpolation_points()))

    W_coords = W.tabulate_dof_coordinates()[:, :2]   # (N, 2): r, z
    return u_sol, vm_func.x.array.copy(), W_coords


# ══════════════════════════════════════════════════════════════════════════════
# Solve ② — Torsional (u_θ): desacoplado
# ══════════════════════════════════════════════════════════════════════════════

def solve_torsional(
    domain,
    z_min: float,
    z_max: float,
    r_bore: float,
    r_outer: float,
    E: float     = _E,
    nu: float    = _NU,
    T_torque: float = 40e3,    # N·m
    petsc_prefix: str = "tors_",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resolve a equação de torção u_θ:

        ∂²u_θ/∂r² + (1/r)∂u_θ/∂r − u_θ/r² + ∂²u_θ/∂z² = 0

    Forma fraca com peso r (volume cilíndrico):
        ∫ [∂u_θ/∂r ∂v/∂r + ∂u_θ/∂z ∂v/∂z + u_θ v/r²] r dr dz
            = ∫ τ_θz_top · v · r ds_top

    Tensão de cisalhamento aplicada no topo:
        τ_θz = T / (2π ∫_{r_bore}^{r_outer} r² dr)

    BCs:
        u_θ = 0  em z = z_min  (nariz fixo — sem rotação)

    Retorna (u_th_array, tau_th_z_array, W_coords) em unidades SI (mm, Pa).
    """
    G = E / (2 * (1 + nu))

    # J polar da seção anular (mm⁴ → m⁴)
    r0_m, r1_m = r_bore * 1e-3, r_outer * 1e-3
    J = np.pi * (r1_m ** 4 - r0_m ** 4) / 2.0   # m⁴

    # Tensão de cisalhamento torsional média no topo
    # τ = T · r / J  → valor médio em r_outer (conservativo)
    tau_top = T_torque * r1_m / max(J, 1e-30)    # Pa

    W = fem.functionspace(domain, ("Lagrange", 1))
    u_th = ufl.TrialFunction(W)
    v    = ufl.TestFunction(W)

    x = ufl.SpatialCoordinate(domain)
    r = x[0]

    # Forma fraca da equação de torção cilíndrica (com peso r):
    # ∫ [du_θ/dr · dv/dr + du_θ/dz · dv/dz + u_θ·v/r²] · r dr dz
    a = (
        u_th.dx(0) * v.dx(0) * r
        + u_th.dx(1) * v.dx(1) * r
        + u_th * v / r
    ) * ufl.dx

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    # ── Carregamento: τ_θz no topo ────────────────────────────────────────
    top_facets = dmesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[1], z_max, atol=1.5))
    ft_tags = dmesh.meshtags(domain, fdim, top_facets,
                              np.ones(len(top_facets), dtype=np.int32))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft_tags)
    tau_c = fem.Constant(domain, np.float64(tau_top))
    L = tau_c * v * r * ds(1)

    # ── BC: u_θ = 0 no nariz (z = z_min) ─────────────────────────────────
    bot_facets = dmesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[1], z_min, atol=1.5))
    dofs_bot = fem.locate_dofs_topological(W, fdim, bot_facets)
    bc_z0 = fem.dirichletbc(np.float64(0.0), dofs_bot, W)

    problem = LinearProblem(a, L, bcs=[bc_z0],
                             petsc_options={"ksp_type": "cg", "pc_type": "amg",
                                            "ksp_rtol": 1e-10},
                             petsc_options_prefix=petsc_prefix)
    u_th_sol = problem.solve()

    u_th_arr = u_th_sol.x.array.copy()    # mm (deslocamento hoop)
    W_coords = W.tabulate_dof_coordinates()[:, :2]

    # τ_θz = G · ∂u_θ/∂z  (aproximado como u_th / z_max · G para exportar)
    # (valor exato calculado na pós-processagem via autograd no PINN)
    tau_th_z = G * u_th_arr / max(float(z_max) * 1e-3, 1e-10)   # Pa (aprox)

    return u_th_arr, tau_th_z, W_coords


# ══════════════════════════════════════════════════════════════════════════════
# Von Mises 6-componentes (combina os dois solves)
# ══════════════════════════════════════════════════════════════════════════════

def compute_full_von_mises(
    domain,
    u_sol,          # solução meridional dolfinx Function
    u_th_arr: np.ndarray,   # (N,) solução torsional
    W_coords: np.ndarray,   # (N, 2)
    E: float = _E,
    nu: float = _NU,
) -> np.ndarray:
    """
    Calcula Von Mises 6-componentes combinando tensões meridionais e torsionais.

    σ_vm = √(½ [(σ_rr−σ_zz)² + (σ_zz−σ_θθ)² + (σ_θθ−σ_rr)²
               + 6(τ_rz² + τ_rθ² + τ_θz²)])

    τ_rθ e τ_θz vêm do campo torsional u_θ (calculados aqui via FEM).
    """
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu  = E / (2 * (1 + nu))
    G   = E / (2 * (1 + nu))

    W   = fem.functionspace(domain, ("Lagrange", 1))
    x   = ufl.SpatialCoordinate(domain)
    r   = x[0]

    def eps_axi(u):
        return ufl.as_vector([
            u[0].dx(0), u[1].dx(1), u[0] / r, u[0].dx(1) + u[1].dx(0)])

    def sigma_axi(u):
        lam_c = fem.Constant(domain, np.float64(lam))
        mu_c  = fem.Constant(domain, np.float64(mu))
        e     = eps_axi(u)
        tr    = e[0] + e[1] + e[2]
        return ufl.as_vector([
            lam_c * tr + 2 * mu_c * e[0],
            lam_c * tr + 2 * mu_c * e[1],
            lam_c * tr + 2 * mu_c * e[2],
            mu_c * e[3],
        ])

    # Interpolação dos componentes meridionais no espaço W
    def _interp(expr):
        f = fem.Function(W)
        f.interpolate(fem.Expression(expr, W.element.interpolation_points()))
        return f.x.array.copy()

    s = sigma_axi(u_sol)
    s_rr = _interp(s[0])
    s_zz = _interp(s[1])
    s_tt = _interp(s[2])
    t_rz = _interp(s[3])

    # Tensões torsionais — aproximação linear (exata no PINN via autograd)
    # τ_θz ≈ G · u_θ / (z_max * 1e-3)  já calculado, mas aqui reusamos u_th_arr
    # τ_rθ ≈ G · u_θ / r  (deformação hoop de cisalhamento torsional)
    r_vals = W_coords[:, 0] * 1e-3   # mm → m
    z_vals = W_coords[:, 1] * 1e-3
    u_th_m = u_th_arr * 1e-3          # mm → m

    z_max_val = float(z_vals.max())
    t_th_z = G * u_th_m / max(z_max_val, 1e-10)              # τ_θz Pa
    t_rt   = G * u_th_m / np.maximum(r_vals, 1e-6)           # τ_rθ Pa

    vm = np.sqrt(
        0.5 * (
            (s_rr - s_zz) ** 2
            + (s_zz - s_tt) ** 2
            + (s_tt - s_rr) ** 2
            + 6 * (t_rz ** 2 + t_rt ** 2 + t_th_z ** 2)
        )
    )
    return vm.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Simulação de um caso
# ══════════════════════════════════════════════════════════════════════════════

def simulate_case(
    case_path: Path,
    body: str = "BOX",
    E: float = _E,
    nu: float = _NU,
    p_inner: float = 20e6,
    F_axial: float = 500e3,
    T_torque: float = 40e3,
    force: bool = False,
) -> bool:
    """
    Simula um caso (BOX ou PIN) e salva solution.npy.

    solution.npy contém:
        coords    : (N, 2) — [r, z] em mm
        disp      : (N, 3) — [u_r, u_z, u_θ] em mm
        stress_vm : (N,)   — Von Mises 6-comp em Pa

    Retorna True se simulado, False se já existia.
    """
    body = body.upper()
    msh_key  = "mesh.msh"     if body == "BOX" else "mesh_pin.msh"
    sol_key  = "solution.npy" if body == "BOX" else "solution_pin.npy"
    xdmf_key = "mesh.xdmf"   if body == "BOX" else "mesh_pin.xdmf"

    msh_file  = case_path / msh_key
    sol_file  = case_path / sol_key
    xdmf_file = case_path / xdmf_key

    if not force and sol_file.exists():
        return False
    if not msh_file.exists():
        raise FileNotFoundError(f"Mesh não encontrado: {msh_file}")

    # Parâmetros geométricos pelo preset
    spec = get_preset("drill_pipe_nc50_rotating", body=body,
                       E=E, nu=nu, p_inner=p_inner, F_axial=F_axial, T_torque=T_torque)
    r_bore  = spec.domain_bounds["r"][0]
    r_outer = spec.domain_bounds["r"][1]

    print(f"  [{body}] Carregando mesh...")
    domain, z_min, z_max = msh_to_dolfinx(msh_file, xdmf_file)
    print(f"  [{body}] r ∈ [{domain.geometry.x[:,0].min():.1f}, "
          f"{domain.geometry.x[:,0].max():.1f}] mm  "
          f"z ∈ [{z_min:.1f}, {z_max:.1f}] mm")

    # ── Solve ① meridional ────────────────────────────────────────────────
    print(f"  [{body}] Solve meridional (u_r, u_z)...")
    u_meri, vm_meri, W_coords = solve_meridional(
        domain, z_min, z_max, r_bore, E, nu, p_inner, F_axial, r_outer,
        petsc_prefix=f"{body.lower()}_meri_"
    )
    u_vals = u_meri.x.array.reshape(-1, 2)[:len(W_coords)]  # (N,2): u_r, u_z (m)

    # ── Solve ② torsional ─────────────────────────────────────────────────
    print(f"  [{body}] Solve torsional (u_θ)...")
    u_th_arr, _, _ = solve_torsional(
        domain, z_min, z_max, r_bore, r_outer, E, nu, T_torque,
        petsc_prefix=f"{body.lower()}_tors_"
    )
    u_th_vals = u_th_arr[:len(W_coords)]   # (N,) m

    # ── Von Mises 6-componentes ───────────────────────────────────────────
    print(f"  [{body}] Computando Von Mises 6-componentes...")
    vm_full = compute_full_von_mises(domain, u_meri, u_th_vals, W_coords, E, nu)

    # ── Salva solution.npy ────────────────────────────────────────────────
    # Converte deslocamentos m → mm
    disp = np.column_stack([
        u_vals[:, 0] * 1e3,    # u_r  mm
        u_vals[:, 1] * 1e3,    # u_z  mm
        u_th_vals * 1e3,        # u_θ  mm
    ]).astype(np.float32)

    np.save(str(sol_file), {
        "coords":    W_coords.astype(np.float32),   # (N,2) mm
        "disp":      disp,                           # (N,3) mm
        "stress_vm": vm_full,                        # (N,)  Pa
    })

    print(f"  [{body}] u_r  : [{disp[:,0].min()*1e3:.2f}, {disp[:,0].max()*1e3:.2f}] μm")
    print(f"  [{body}] u_z  : [{disp[:,1].min()*1e3:.2f}, {disp[:,1].max()*1e3:.2f}] μm")
    print(f"  [{body}] u_θ  : [{disp[:,2].min()*1e3:.2f}, {disp[:,2].max()*1e3:.2f}] μm")
    print(f"  [{body}] σ_vm : max={vm_full.max()/1e6:.1f} MPa  mean={vm_full.mean()/1e6:.1f} MPa")
    print(f"  [{body}] → {sol_file}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Runner do dataset
# ══════════════════════════════════════════════════════════════════════════════

def run(
    dataset_dir: str | Path = DATASET_DIR,
    E: float = _E,
    nu: float = _NU,
    p_inner: float = 20e6,
    F_axial: float = 500e3,
    T_torque: float = 40e3,
    force: bool = False,
) -> None:
    """
    Simula BOX e PIN para todos os casos do dataset.

    Cada caso deve ter mesh.msh e mesh_pin.msh (gerados por generate_geometry.py).
    Salva solution.npy e solution_pin.npy com 3 campos de deslocamento + σ_vm.
    """
    dataset_path = Path(dataset_dir)
    cases = sorted([
        d for d in dataset_path.iterdir()
        if d.is_dir() and not d.name.startswith("test_")
    ])

    if not cases:
        print(f"  Nenhum caso encontrado em '{dataset_path}'.")
        print("  Execute generate_geometry.py primeiro.")
        return

    print(f"\n  Dataset: {dataset_path}  ({len(cases)} casos)")
    print(f"  Cargas: p_inner={p_inner/1e6:.0f} MPa  "
          f"F={F_axial/1e3:.0f} kN  T={T_torque/1e3:.0f} kN·m\n")

    ok = skipped = failed = 0
    for case_path in cases:
        print(f"\n=== {case_path.name} ===")
        try:
            done_box = simulate_case(case_path, "BOX", E, nu,
                                      p_inner, F_axial, T_torque, force=force)
            done_pin = simulate_case(case_path, "PIN", E, nu,
                                      p_inner, F_axial, T_torque, force=force)
            if done_box or done_pin:
                ok += 1
            else:
                print("  [skip] solutions já existem.")
                skipped += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed += 1

    print(f"\n  Concluído: {ok} simulados | {skipped} pulados | {failed} erros")
    if failed:
        raise RuntimeError(f"{failed} caso(s) falharam.")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FEM axissimétrico com torção — NC50 Drill Pipe (PINNeAPPle)"
    )
    parser.add_argument("--dataset-dir", default=DATASET_DIR,
                        help="Pasta do dataset (padrão: dataset/)")
    parser.add_argument("--case",        default=None,
                        help="Simula só um caso (ex: dataset/case_003)")
    parser.add_argument("--body",        default="BOX", choices=["BOX", "PIN", "BOTH"],
                        help="Corpo a simular (padrão: BOX)")
    parser.add_argument("--p-inner",     type=float, default=20e6,  help="Pressão interna Pa")
    parser.add_argument("--f-axial",     type=float, default=500e3, help="Tração axial N")
    parser.add_argument("--torque",      type=float, default=40e3,  help="Torque N·m")
    parser.add_argument("--E",           type=float, default=_E,    help="Young's modulus Pa")
    parser.add_argument("--nu",          type=float, default=_NU,   help="Poisson's ratio")
    parser.add_argument("--force",       action="store_true",
                        help="Força resimulação mesmo se solution.npy existir")
    args = parser.parse_args()

    if args.case:
        # Simula só um caso
        case_path = Path(args.case)
        bodies = ["BOX", "PIN"] if args.body == "BOTH" else [args.body]
        for body in bodies:
            simulate_case(case_path, body,
                          args.E, args.nu, args.p_inner, args.f_axial, args.torque,
                          force=args.force)
    else:
        run(
            dataset_dir=args.dataset_dir,
            E=args.E, nu=args.nu,
            p_inner=args.p_inner, F_axial=args.f_axial, T_torque=args.torque,
            force=args.force,
        )
