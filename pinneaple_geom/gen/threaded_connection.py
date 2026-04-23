"""Threaded connection geometry for rotary coupling PINN simulations.

Generates 2D axisymmetric (r, z) meshes of rotary threaded connections
using gmsh. The mesh is compatible with FEniCS/dolfinx and the
SolidMechanicsPipeline.

Supported connections
---------------------
  TC50  — large-diameter coupling, 4 TPI, taper 1:16
  TC46  — medium-diameter coupling, 4 TPI, taper 1:16
  TC38  — standard coupling, 4 TPI, taper 1:16
  Custom — any V-thread via ThreadProfile

Thread profile (API V-0.038R)
------------------------------
  Pitch     p = 25.4 / TPI   mm
  Taper     T = 1:16  →  Δr/Δz = 1/32 per side
  Height    h_total = 0.8660 · p
  Clearance c = 0.038 · p     (root flat)
  Crest cut = 0.038 · p

In the 2D (r, z) cross-section the thread appears as a sawtooth on the
inner/outer cylindrical surfaces of the connection.

Usage
-----
    from pinneaple_geom.gen.threaded_connection import TC50Geometry

    geom = TC50Geometry(body="BOX")
    mesh_path = geom.generate_mesh(output="mesh_tc50_box.msh", lc=0.5)
    # → Mesh2D / .msh file ready for FEniCS or point sampling
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Thread profile data
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ThreadProfile:
    """Geometric parameters for a single API rotary shouldered thread form."""
    name: str
    tpi: float          # threads per inch
    taper: float        # radial change per unit axial length (= 1/32 for 1:16 taper)
    h_factor: float = 0.8660   # h_total = h_factor * pitch  (60° V thread)
    clearance_factor: float = 0.038  # c = factor * pitch (root/crest flat width)

    @property
    def pitch_mm(self) -> float:
        return 25.4 / self.tpi

    @property
    def thread_height_mm(self) -> float:
        return self.h_factor * self.pitch_mm

    @property
    def clearance_mm(self) -> float:
        return self.clearance_factor * self.pitch_mm


# Standard API V-0.038R profiles
API_V038R = ThreadProfile("API V-0.038R", tpi=4.0, taper=1.0 / 32.0)


@dataclass
class ConnectionSpec:
    """Full geometry specification for one body (BOX or PIN) of an NC connection."""
    name: str                    # e.g. "TC50_BOX"
    r_bore: float                # mm — bore radius (inner surface)
    r_outer: float               # mm — outer radius
    z_thread_start: float        # mm — z where thread begins
    z_thread_end: float          # mm — z where thread ends
    z_shoulder_top: float        # mm — total length (shoulder at top)
    thread: ThreadProfile
    body: str = "BOX"            # "BOX" or "PIN"
    n_threads: Optional[int] = None  # if None, computed from geometry

    @property
    def z_total(self) -> float:
        return self.z_shoulder_top

    @property
    def n_thread_turns(self) -> int:
        if self.n_threads is not None:
            return self.n_threads
        thread_length = self.z_thread_end - self.z_thread_start
        return max(1, int(thread_length / self.thread.pitch_mm))


# Standard NC connections
_TC_SPECS: Dict[str, Dict[str, ConnectionSpec]] = {
    "TC50": {
        "BOX": ConnectionSpec(
            name="TC50_BOX",
            r_bore=50.0, r_outer=84.15,
            z_thread_start=10.0, z_thread_end=115.0,
            z_shoulder_top=140.0,
            thread=API_V038R, body="BOX",
        ),
        "PIN": ConnectionSpec(
            name="TC50_PIN",
            r_bore=48.0, r_outer=62.0,
            z_thread_start=10.0, z_thread_end=115.0,
            z_shoulder_top=140.0,
            thread=API_V038R, body="PIN",
        ),
    },
    "TC46": {
        "BOX": ConnectionSpec(
            name="TC46_BOX",
            r_bore=44.0, r_outer=76.2,
            z_thread_start=8.0, z_thread_end=100.0,
            z_shoulder_top=125.0,
            thread=API_V038R, body="BOX",
        ),
        "PIN": ConnectionSpec(
            name="TC46_PIN",
            r_bore=42.0, r_outer=55.0,
            z_thread_start=8.0, z_thread_end=100.0,
            z_shoulder_top=125.0,
            thread=API_V038R, body="PIN",
        ),
    },
    "TC38": {
        "BOX": ConnectionSpec(
            name="TC38_BOX",
            r_bore=36.0, r_outer=63.5,
            z_thread_start=6.0, z_thread_end=85.0,
            z_shoulder_top=105.0,
            thread=API_V038R, body="BOX",
        ),
        "PIN": ConnectionSpec(
            name="TC38_PIN",
            r_bore=34.0, r_outer=46.0,
            z_thread_start=6.0, z_thread_end=85.0,
            z_shoulder_top=105.0,
            thread=API_V038R, body="PIN",
        ),
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Thread profile polygon builder
# ══════════════════════════════════════════════════════════════════════════════

def build_thread_polygon(
    spec: ConnectionSpec,
    n_threads: Optional[int] = None,
) -> List[Tuple[float, float]]:
    """
    Builds the (r, z) polygon that describes the threaded surface profile.

    The result is a list of (r, z) points forming the sawtooth thread profile
    on the relevant cylindrical surface:
      BOX: inner surface (r = r_bore side) → threads cut INTO the box bore
      PIN: outer surface (r = r_outer side) → threads on pin outer diameter

    The taper is applied: the thread reference diameter increases from
    z_thread_start toward z_thread_end at rate `taper` (radial/axial).

    Returns ordered (r, z) vertices for the threaded profile segment.
    The full boundary polygon is assembled by the mesh builder.
    """
    th  = spec.thread
    p   = th.pitch_mm
    h   = th.thread_height_mm
    c   = th.clearance_mm
    tap = th.taper     # Δr per Δz

    n = n_threads or spec.n_thread_turns
    z0 = spec.z_thread_start

    # For BOX: threads are on the bore (r-side), nominal surface at r_bore
    # For PIN: threads are on the outer surface, nominal at r_outer
    if spec.body == "BOX":
        r_nominal_at_start = spec.r_bore
    else:
        r_nominal_at_start = spec.r_outer

    pts = []
    for i in range(n):
        z_base = z0 + i * p
        r_base = r_nominal_at_start + tap * (z_base - z0)

        # Each thread tooth: flat root (width c) → flanks → flat crest (width c)
        # Simplified sawtooth in (z, r) going from root to crest:
        # points: (z_base, r_base) → (z_base+c, r_base) →
        #         (z_base+p/2, r_base±h) → (z_base+p-c, r_base+(p-c)*tap±h) →
        #         (z_base+p, r_base+p*tap)
        z1 = z_base
        z2 = z_base + c
        z3 = z_base + p / 2.0
        z4 = z_base + p - c
        z5 = z_base + p

        r1 = r_base
        r2 = r_base + tap * c
        r3 = r_base + tap * p / 2.0
        r4 = r_base + tap * (p - c)
        r5 = r_base + tap * p

        # Thread direction: BOX threads go inward (+r), PIN threads go outward (+r)
        sign = +1.0  # both BOX bore and PIN outer go toward larger r with taper

        if spec.body == "BOX":
            # Thread height goes INTO bore (increasing r)
            pts += [(r1, z1), (r2, z2), (r3 + h, z3), (r4, z4)]
        else:
            # Thread height protrudes outward (increasing r)
            pts += [(r1, z1), (r2, z2), (r3 + h, z3), (r4, z4)]

    # Close to end of thread zone
    z_end = z0 + n * p
    r_end = r_nominal_at_start + tap * (z_end - z0)
    pts.append((r_end, z_end))

    return pts


# ══════════════════════════════════════════════════════════════════════════════
# Gmsh-based mesh generator
# ══════════════════════════════════════════════════════════════════════════════

class ThreadedConnectionMesher:
    """
    Generates a 2D axisymmetric mesh of an API threaded connection using gmsh.

    The mesh represents the (r, z) cross-section. Boundary regions are tagged:
      - "bore"         : inner cylindrical surface (r = r_bore)
      - "outer"        : outer cylindrical surface (r = r_outer)
      - "nose"         : z = z_min face (thread start, fixed nose)
      - "shoulder_top" : z = z_max face (make-up shoulder / top)
      - "thread_flank" : threaded surface (sawtooth profile)

    Parameters
    ----------
    spec : ConnectionSpec
        Geometry parameters for the connection body.
    lc : float
        Characteristic mesh size (mm). Smaller = finer.
    lc_thread : float
        Finer mesh size at thread root/crest (default = lc / 3).
    """

    def __init__(
        self,
        spec: ConnectionSpec,
        lc: float = 1.0,
        lc_thread: Optional[float] = None,
    ):
        self.spec = spec
        self.lc = lc
        self.lc_thread = lc_thread or lc / 3.0

    def generate(self, output_path: str | Path = "mesh.msh") -> Path:
        """
        Build the mesh and write it to a .msh file.

        Requires: pip install "PINNeAPPle[geom]"  (installs gmsh)

        Returns the path to the generated .msh file.
        """
        try:
            import gmsh  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "gmsh is required for mesh generation.\n"
                "Install with: pip install 'PINNeAPPle[geom]'  or  pip install gmsh"
            )

        output_path = Path(output_path)
        spec = self.spec
        lc   = self.lc
        lc_t = self.lc_thread

        gmsh.initialize()
        gmsh.model.add(spec.name)
        gmsh.option.setNumber("General.Terminal", 0)   # silent
        gmsh.option.setNumber("Mesh.Algorithm", 6)     # Frontal-Delaunay 2D

        # ── Build boundary polygon ──────────────────────────────────────────
        # The 2D cross section in (r, z) is bounded by:
        #   bottom (z=0, nose)  :  r ∈ [r_bore, r_outer]
        #   outer right (r=r_outer): z ∈ [0, z_total]
        #   top (z=z_total, shoulder): r ∈ [r_bore, r_outer]
        #   left side (r=r_bore): z ∈ [0, z_total]  — with thread sawtooth
        #     (simplified: straight bore + thread notch approximation)

        s = spec
        r0, r1 = s.r_bore, s.r_outer
        z0, z1 = 0.0, s.z_total

        # Corner points (all at nominal bore/outer — no thread detail in simple mode)
        p_bore_bot   = gmsh.model.geo.addPoint(r0, z0, 0, lc)
        p_outer_bot  = gmsh.model.geo.addPoint(r1, z0, 0, lc)
        p_outer_top  = gmsh.model.geo.addPoint(r1, z1, 0, lc)
        p_bore_top   = gmsh.model.geo.addPoint(r0, z1, 0, lc)

        # Thread zone — add intermediate points with sawtooth profile
        zt_start = s.z_thread_start
        zt_end   = s.z_thread_end
        th_pts_inner = []  # points on bore side (r = r0) with thread notches

        p_bore_pre_thread  = gmsh.model.geo.addPoint(r0, zt_start, 0, lc)
        p_bore_post_thread = gmsh.model.geo.addPoint(r0, zt_end,   0, lc)

        # Add thread tooth points on bore surface
        thread_poly = build_thread_polygon(spec)
        for r_t, z_t in thread_poly:
            th_pts_inner.append(
                gmsh.model.geo.addPoint(r_t, z_t, 0, lc_t)
            )

        # ── Lines ────────────────────────────────────────────────────────────

        # Bottom face (nose)
        l_bot = gmsh.model.geo.addLine(p_bore_bot, p_outer_bot)

        # Outer face (r = r_outer, no threads on outer in this simplified model)
        l_outer = gmsh.model.geo.addLine(p_outer_bot, p_outer_top)

        # Top face (shoulder)
        l_top = gmsh.model.geo.addLine(p_outer_top, p_bore_top)

        # Bore face — built in segments: bore → pre_thread → thread → post_thread → bore_top
        l_bore_bot_seg   = gmsh.model.geo.addLine(p_bore_top, p_bore_post_thread)
        l_bore_top_seg   = gmsh.model.geo.addLine(p_bore_pre_thread, p_bore_bot)

        # Thread sawtooth: chain through all thread points
        l_thread_lines = []
        all_thread_pts = [p_bore_post_thread] + th_pts_inner[::-1] + [p_bore_pre_thread]
        for i in range(len(all_thread_pts) - 1):
            l_thread_lines.append(
                gmsh.model.geo.addLine(all_thread_pts[i], all_thread_pts[i + 1])
            )

        # ── Curve loop + surface ─────────────────────────────────────────────
        bore_chain = [l_bore_bot_seg] + l_thread_lines + [l_bore_top_seg]
        loop_tags  = [l_bot, l_outer, l_top] + bore_chain
        cl  = gmsh.model.geo.addCurveLoop(loop_tags)
        surf = gmsh.model.geo.addPlaneSurface([cl])

        gmsh.model.geo.synchronize()

        # ── Physical groups (boundary tags for FEniCS) ────────────────────────
        gmsh.model.addPhysicalGroup(1, [l_bot],              tag=1, name="nose")
        gmsh.model.addPhysicalGroup(1, [l_outer],            tag=2, name="outer")
        gmsh.model.addPhysicalGroup(1, [l_top],              tag=3, name="shoulder_top")
        gmsh.model.addPhysicalGroup(1, l_thread_lines,       tag=4, name="thread_flank")
        gmsh.model.addPhysicalGroup(1, [l_bore_bot_seg, l_bore_top_seg], tag=5, name="bore")
        gmsh.model.addPhysicalGroup(2, [surf],               tag=10, name="domain")

        # ── Mesh ──────────────────────────────────────────────────────────────
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Netgen")
        gmsh.write(str(output_path))
        gmsh.finalize()

        print(f"  [gmsh] Mesh gerado: {output_path}")
        print(f"    Conexão : {spec.name}  ({spec.body})")
        print(f"    r ∈ [{r0:.1f}, {r1:.1f}] mm   z ∈ [{z0:.1f}, {z1:.1f}] mm")
        print(f"    Roscas  : {spec.n_thread_turns}  |  pitch = {spec.thread.pitch_mm:.2f} mm")
        return output_path

    def sample_points(self, n: int = 5000, seed: int = 42) -> np.ndarray:
        """
        Amostra pontos dentro do domínio (sem gmsh) para uso como
        pontos de collocação no PINN.

        O domínio é tratado como retangular (bounding box). Para geometrias
        com perfil roscado real use o mesh gerado e amostre dos centroides
        dos triângulos.
        """
        rng = np.random.default_rng(seed)
        r = rng.uniform(self.spec.r_bore,   self.spec.r_outer,  n)
        z = rng.uniform(0.0,                self.spec.z_total,   n)
        return np.column_stack([r, z]).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Convenience classes per connection type
# ══════════════════════════════════════════════════════════════════════════════

class TC50Geometry(ThreadedConnectionMesher):
    """TC50 threaded coupling geometry (large-diameter)."""
    def __init__(self, body: str = "BOX", lc: float = 0.8, lc_thread: Optional[float] = None):
        body = body.upper()
        if body not in ("BOX", "PIN"):
            raise ValueError("body deve ser 'BOX' ou 'PIN'")
        super().__init__(_TC_SPECS["TC50"][body], lc=lc, lc_thread=lc_thread)


class TC46Geometry(ThreadedConnectionMesher):
    """TC46 threaded coupling geometry (medium-diameter)."""
    def __init__(self, body: str = "BOX", lc: float = 0.8, lc_thread: Optional[float] = None):
        body = body.upper()
        super().__init__(_TC_SPECS["TC46"][body], lc=lc, lc_thread=lc_thread)


class TC38Geometry(ThreadedConnectionMesher):
    """TC38 threaded coupling geometry (standard)."""
    def __init__(self, body: str = "BOX", lc: float = 0.8, lc_thread: Optional[float] = None):
        body = body.upper()
        super().__init__(_TC_SPECS["TC38"][body], lc=lc, lc_thread=lc_thread)


def get_connection_geometry(
    connection: str,
    body: str = "BOX",
    lc: float = 0.8,
) -> ThreadedConnectionMesher:
    """
    Factory: retorna o mesher para uma conexão pelo nome.

    Parameters
    ----------
    connection : "TC50", "TC46", "TC38"
    body : "BOX" ou "PIN"
    lc : characteristic mesh size in mm

    Example
    -------
    geom = get_connection_geometry("TC50", "BOX")
    mesh_path = geom.generate("mesh_tc50_box.msh")
    pts = geom.sample_points(n=10000)
    """
    connection = connection.upper()
    body       = body.upper()
    if connection not in _TC_SPECS:
        raise KeyError(f"Conexão '{connection}' não suportada. Disponíveis: {list(_TC_SPECS)}")
    return ThreadedConnectionMesher(_TC_SPECS[connection][body], lc=lc)
