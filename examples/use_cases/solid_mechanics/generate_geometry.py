"""
generate_geometry.py — NC50 Drill Pipe Geometry (PINNeAPPle)
============================================================

Gera geometrias 3D (STL) e meshes 2D axissimétricas (.msh) para o
problema do drill pipe NC50 em rotação, com suporte a datasets paramétricos.

Adaptação do pipeline original de CadQuery + gmsh ao framework PINNeAPPle:
  - Usa os parâmetros do preset `drill_pipe_nc50_rotating` como fonte de verdade
  - Os parâmetros de variação (clearance, thread_height, offset) geram o dataset
  - O mesh 2D (r, z) é o que o FEM e o PINN usam de fato

Instalação
----------
    pip install "PINNeAPPle[geom,cad]"
    # ou individualmente:
    pip install gmsh meshio cadquery

Uso rápido
----------
    # Gera todos os 27 casos (3×3×3):
    python generate_geometry.py

    # Só o mesh 2D (sem STL 3D — mais rápido, sem CadQuery):
    python generate_geometry.py --no-stl

    # Um caso específico:
    python generate_geometry.py --clearance 0.2 --thread-height 0.8 --offset 1.0

Estrutura de saída
------------------
    dataset/
      case_000/
        pin.stl          ← geometria 3D do PIN (CadQuery)
        box.stl          ← geometria 3D do BOX (CadQuery)
        mesh.msh         ← mesh 2D axissimétrico do BOX (gmsh)
        mesh_pin.msh     ← mesh 2D axissimétrico do PIN (gmsh)
      case_001/
        ...
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

# ── PINNeAPPle ─────────────────────────────────────────────────────────────
# Importa os parâmetros base do preset em vez de duplicar aqui
from pinneaple_environment import get_preset

# ══════════════════════════════════════════════════════════════════════════════
# Parâmetros NC50 base
# (devem coincidir com o preset drill_pipe_nc50_rotating)
# ══════════════════════════════════════════════════════════════════════════════

LENGTH_THREAD   = 120.0    # mm
LENGTH_SHOULDER = 20.0     # mm
PIN_MAJOR_D     = 114.3    # mm
PIN_BORE_R      = 48.0     # mm
BOX_OUTER_D     = 168.3    # mm
PITCH           = 6.35     # mm
TAPER           = 1.0 / 16.0

# Variações do dataset paramétrico
CLEARANCES     = [0.1, 0.2, 0.3]
THREAD_HEIGHTS = [0.6, 0.8, 1.0]
OFFSETS        = [0.5, 1.0, 1.5]

# Refinamento de mesh
SIZE_FINE   = 0.25    # mm — zona roscada
SIZE_COARSE = 3.0     # mm — corpo


# ══════════════════════════════════════════════════════════════════════════════
# Utilitário de perfil de rosca
# ══════════════════════════════════════════════════════════════════════════════

def _thread_r(base_r: float, phase: float, thread_height: float) -> float:
    """Raio no ponto de fase `phase` ∈ [0, 1] de uma rosca V."""
    if phase < 0.5:
        return base_r - thread_height * (phase * 2)
    else:
        return base_r - thread_height * (1.0 - (phase - 0.5) * 2)


# ══════════════════════════════════════════════════════════════════════════════
# Perfis 2D (r, z) para o gmsh
# ══════════════════════════════════════════════════════════════════════════════

def build_box_profile(
    clearance: float,
    thread_height: float,
) -> List[Tuple[float, float]]:
    """
    Polígono (r, z) do BOX — sentido CCW.

    O bore interno apresenta o perfil de rosca (dente de serra).
    """
    r_outer    = BOX_OUTER_D / 2           # 84.15 mm
    r_shoulder = PIN_MAJOR_D / 2 + 10 + 0.05   # ~67.2 mm
    z_top      = LENGTH_THREAD + LENGTH_SHOULDER
    dz         = PITCH / 6

    pts_inner: List[Tuple[float, float]] = []
    z = 0.0
    while z <= LENGTH_THREAD + 1e-9:
        base_r = (PIN_MAJOR_D / 2 + clearance + 0.05) - TAPER * z
        r = _thread_r(base_r, (z % PITCH) / PITCH, thread_height)
        pts_inner.append((r, z))
        z += dz

    pts_inner.append((r_shoulder, LENGTH_THREAD))
    pts_inner.append((r_shoulder, z_top))
    pts_inner.append((r_outer,    z_top))
    pts_inner.append((r_outer,    0.0))
    return pts_inner


def build_pin_profile(thread_height: float) -> List[Tuple[float, float]]:
    """
    Polígono (r, z) do PIN — sentido CCW.

    A superfície externa apresenta o perfil de rosca.
    """
    r_shoulder = PIN_MAJOR_D / 2 + 10.0 + 0.05
    z_top      = LENGTH_THREAD + LENGTH_SHOULDER
    dz         = PITCH / 6

    pts_up: List[Tuple[float, float]] = []
    z = 0.0
    while z <= LENGTH_THREAD + 1e-9:
        base_r = (PIN_MAJOR_D / 2) - TAPER * z
        r = _thread_r(base_r, (z % PITCH) / PITCH, thread_height)
        pts_up.append((r, z))
        z += dz

    pts_thread_down = list(reversed(pts_up))

    return (
        [(PIN_BORE_R, 0.0),
         (PIN_BORE_R, z_top),
         (r_shoulder, z_top),
         (r_shoulder, LENGTH_THREAD)]
        + pts_thread_down
    )


# ══════════════════════════════════════════════════════════════════════════════
# Geração de mesh gmsh
# ══════════════════════════════════════════════════════════════════════════════

def _add_gmsh_surface(gmsh, polygon: List[Tuple[float, float]]) -> int:
    """Adiciona um polígono como superfície plana no modelo gmsh ativo."""
    pt_tags = [gmsh.model.geo.addPoint(r, z, 0) for r, z in polygon]
    n = len(pt_tags)
    line_tags = [
        gmsh.model.geo.addLine(pt_tags[i], pt_tags[(i + 1) % n])
        for i in range(n)
    ]
    loop = gmsh.model.geo.addCurveLoop(line_tags)
    surf = gmsh.model.geo.addPlaneSurface([loop])
    return surf


def _apply_refinement(gmsh, r_max_thread: float, z_max_thread: float,
                       size_fine: float, size_coarse: float) -> None:
    """Campo Box de refinamento na zona roscada."""
    gmsh.model.mesh.field.add("Box", 1)
    gmsh.model.mesh.field.setNumber(1, "XMin",  0.0)
    gmsh.model.mesh.field.setNumber(1, "XMax",  r_max_thread)
    gmsh.model.mesh.field.setNumber(1, "YMin",  0.0)
    gmsh.model.mesh.field.setNumber(1, "YMax",  z_max_thread)
    gmsh.model.mesh.field.setNumber(1, "ZMin", -1.0)
    gmsh.model.mesh.field.setNumber(1, "ZMax",  1.0)
    gmsh.model.mesh.field.setNumber(1, "VIn",   size_fine)
    gmsh.model.mesh.field.setNumber(1, "VOut",  size_coarse)
    gmsh.model.mesh.field.setAsBackgroundMesh(1)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints",         0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",      0)
    gmsh.option.setNumber("Mesh.Algorithm",     6)   # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.ElementOrder",  1)


def generate_box_mesh(
    case_path: Path,
    clearance: float,
    thread_height: float,
    size_fine: float = SIZE_FINE,
    size_coarse: float = SIZE_COARSE,
    force: bool = False,
) -> bool:
    """
    Gera mesh 2D axissimétrico do BOX em `case_path/mesh.msh`.
    Retorna True se gerado, False se já existia.
    """
    msh_file = case_path / "mesh.msh"
    if not force and msh_file.exists():
        return False

    try:
        import gmsh
    except ImportError:
        raise ImportError("pip install 'PINNeAPPle[geom]'  ou  pip install gmsh")

    case_path.mkdir(parents=True, exist_ok=True)
    polygon = build_box_profile(clearance, thread_height)

    gmsh.model.add(f"box_{case_path.name}")
    _add_gmsh_surface(gmsh, polygon)
    gmsh.model.geo.synchronize()
    _apply_refinement(gmsh, PIN_MAJOR_D / 2 + 3.0,
                      LENGTH_THREAD + 2.0, size_fine, size_coarse)
    gmsh.model.mesh.generate(2)
    gmsh.write(str(msh_file))
    gmsh.model.remove()
    return True


def generate_pin_mesh(
    case_path: Path,
    thread_height: float,
    size_fine: float = SIZE_FINE,
    size_coarse: float = SIZE_COARSE,
    force: bool = False,
) -> bool:
    """
    Gera mesh 2D axissimétrico do PIN em `case_path/mesh_pin.msh`.
    Retorna True se gerado, False se já existia.
    """
    msh_file = case_path / "mesh_pin.msh"
    if not force and msh_file.exists():
        return False

    try:
        import gmsh
    except ImportError:
        raise ImportError("pip install 'PINNeAPPle[geom]'  ou  pip install gmsh")

    case_path.mkdir(parents=True, exist_ok=True)
    polygon = build_pin_profile(thread_height)

    gmsh.model.add(f"pin_{case_path.name}")
    _add_gmsh_surface(gmsh, polygon)
    gmsh.model.geo.synchronize()
    _apply_refinement(gmsh, PIN_MAJOR_D / 2 + 3.0,
                      LENGTH_THREAD + 2.0, size_fine, size_coarse)
    gmsh.model.mesh.generate(2)
    gmsh.write(str(msh_file))
    gmsh.model.remove()
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Geração de STL 3D (CadQuery — opcional)
# ══════════════════════════════════════════════════════════════════════════════

def generate_stl(
    case_path: Path,
    clearance: float,
    thread_height: float,
    offset: float,
    force: bool = False,
) -> bool:
    """
    Gera pin.stl e box.stl via revolução CadQuery.
    Retorna True se gerado, False se já existia.

    Requer: pip install 'PINNeAPPle[cad]'  ou  pip install cadquery
    """
    pin_file = case_path / "pin.stl"
    box_file = case_path / "box.stl"
    if not force and pin_file.exists() and box_file.exists():
        return False

    try:
        import cadquery as cq
    except ImportError:
        raise ImportError("pip install 'PINNeAPPle[cad]'  ou  pip install cadquery")

    case_path.mkdir(parents=True, exist_ok=True)

    dz = PITCH / 6

    # ── PIN ──────────────────────────────────────────────────────────────────
    pts_pin = []
    z = 0.0
    while z <= LENGTH_THREAD:
        base_r = PIN_MAJOR_D / 2 - TAPER * z
        phase  = (z % PITCH) / PITCH
        pts_pin.append((_thread_r(base_r, phase, thread_height), z))
        z += dz

    shoulder_r = PIN_MAJOR_D / 2 + 4.0
    pts_pin += [
        (shoulder_r, LENGTH_THREAD),
        (shoulder_r, LENGTH_THREAD + LENGTH_SHOULDER),
    ]

    pin = (cq.Workplane("XZ")
             .polyline(pts_pin)
             .close()
             .revolve(360))
    pin = pin.translate((0, 0, -offset))

    # ── BOX ──────────────────────────────────────────────────────────────────
    pts_box = []
    z = 0.0
    while z <= LENGTH_THREAD:
        base_r = (PIN_MAJOR_D / 2 + clearance + 0.05) - TAPER * z
        phase  = (z % PITCH) / PITCH
        pts_box.append((_thread_r(base_r, phase, thread_height), z))
        z += dz

    pts_box += [
        (PIN_MAJOR_D / 2 + 10 + 0.05, LENGTH_THREAD),
        (BOX_OUTER_D / 2,              LENGTH_THREAD + LENGTH_SHOULDER),
    ]

    box = (cq.Workplane("XZ")
             .polyline(pts_box)
             .close()
             .revolve(360))

    cq.exporters.export(pin, str(pin_file))
    cq.exporters.export(box, str(box_file))
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Runner do dataset completo
# ══════════════════════════════════════════════════════════════════════════════

def run(
    dataset_dir: str | Path = "dataset",
    clearances: List[float]     = CLEARANCES,
    thread_heights: List[float] = THREAD_HEIGHTS,
    offsets: List[float]        = OFFSETS,
    generate_stl_flag: bool     = True,
    force: bool                 = False,
) -> None:
    """
    Gera geometrias e meshes para todos os casos paramétricos.

    Para cada combinação (clearance, thread_height, offset):
      case_XXX/pin.stl       — 3D revolução CadQuery (se --stl)
      case_XXX/box.stl       — 3D revolução CadQuery (se --stl)
      case_XXX/mesh.msh      — mesh 2D BOX (gmsh)
      case_XXX/mesh_pin.msh  — mesh 2D PIN (gmsh)
    """
    try:
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
    except ImportError:
        raise ImportError("pip install 'PINNeAPPle[geom]'  ou  pip install gmsh")

    dataset_path = Path(dataset_dir)
    dataset_path.mkdir(parents=True, exist_ok=True)

    n_total = len(clearances) * len(thread_heights) * len(offsets)
    ok = skipped = failed = 0
    case_id = 0

    print(f"\n  Gerando {n_total} casos em '{dataset_path}/'")
    print(f"  STL 3D: {'sim' if generate_stl_flag else 'não'}  |  "
          f"Mesh 2D: sim  |  force={force}\n")

    for cl in clearances:
        for th in thread_heights:
            for off in offsets:
                case_path = dataset_path / f"case_{case_id:03d}"
                label     = f"case_{case_id:03d}  cl={cl} th={th} off={off}"
                try:
                    done = False

                    if generate_stl_flag:
                        done |= generate_stl(case_path, cl, th, off, force=force)

                    done |= generate_box_mesh(case_path, cl, th, force=force)
                    done |= generate_pin_mesh(case_path, th, force=force)

                    if done:
                        print(f"  [OK]    {label}")
                        ok += 1
                    else:
                        print(f"  [skip]  {label}")
                        skipped += 1

                except Exception as e:
                    print(f"  [ERROR] {label}  → {e}")
                    failed += 1

                case_id += 1

    gmsh.finalize()
    print(f"\n  Concluído: {ok} gerados | {skipped} pulados | {failed} erros")
    if failed:
        raise RuntimeError(f"{failed} caso(s) falharam.")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera geometrias NC50 para o pipeline PINNeAPPle"
    )
    parser.add_argument("--dataset-dir",    default="dataset", help="Pasta de saída")
    parser.add_argument("--clearance",      type=float, default=None,
                        help="Valor único de clearance (default: 3 valores)")
    parser.add_argument("--thread-height",  type=float, default=None)
    parser.add_argument("--offset",         type=float, default=None)
    parser.add_argument("--no-stl",         action="store_true",
                        help="Não gera STL 3D (só mesh 2D — mais rápido)")
    parser.add_argument("--force",          action="store_true",
                        help="Regera mesmo se já existir")
    args = parser.parse_args()

    run(
        dataset_dir    = args.dataset_dir,
        clearances     = [args.clearance]     if args.clearance     else CLEARANCES,
        thread_heights = [args.thread_height] if args.thread_height else THREAD_HEIGHTS,
        offsets        = [args.offset]        if args.offset        else OFFSETS,
        generate_stl_flag = not args.no_stl,
        force          = args.force,
    )
