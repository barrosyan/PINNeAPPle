"""CadQuery-based geometry synthesis: parametric templates and STL-driven variants."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Union
import math
import numpy as np
import torch

from .base import SynthConfig, SynthOutput
from .sample_adapter import to_physical_sample


def _try_import_cadquery():
    """
    Attempt to import CadQuery.

    Returns
    -------
    Any
        The imported `cadquery` module if available; otherwise None.
    """
    try:
        import cadquery as cq  # type: ignore
        return cq
    except Exception:
        return None


def _try_import_trimesh():
    """
    Attempt to import trimesh.

    Returns
    -------
    Any
        The imported `trimesh` module if available; otherwise None.
    """
    try:
        import trimesh  # type: ignore
        return trimesh
    except Exception:
        return None


def _mesh_from_cq_solid(cq, solid, *, linear_deflection=0.2, angular_deflection=0.3):
    """
    Tessellate a CadQuery solid into triangle mesh arrays.

    Uses CadQuery/OCP tessellation to obtain vertices and triangle indices.

    Parameters
    ----------
    cq : Any
        CadQuery module reference.
    solid : Any
        CadQuery Workplane/Shape-like object (or object exposing `.val()`).
    linear_deflection : float, optional
        Tessellation linear deflection parameter. Default is 0.2.
    angular_deflection : float, optional
        Tessellation angular deflection parameter. Default is 0.3.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Vertices V of shape (N, 3) as float32 and faces F of shape (M, 3) as int64.
    """
    """
    Convert a CadQuery solid to a triangle mesh (trimesh) if available, else numpy arrays.
    """
    # CadQuery has exporters; easiest is to tessellate via Shape.tessellate
    shape = solid.val() if hasattr(solid, "val") else solid
    # CadQuery/OCP tessellation
    pts, tris = shape.tessellate(linear_deflection, angular_deflection)  # pts: list[gp_Pnt] or tuples
    V = np.array([[p.x, p.y, p.z] if hasattr(p, "x") else [p[0], p[1], p[2]] for p in pts], dtype=np.float32)
    F = np.array(tris, dtype=np.int64)
    return V, F


def _extract_mesh_arrays(mesh: Any):
    """
    Extract vertices and faces arrays from a mesh-like object.

    Supports:
    - trimesh-like objects exposing `.vertices` and `.faces`
    - dict-based meshes with "vertices" and "faces" keys

    Parameters
    ----------
    mesh : Any
        Mesh object or dictionary.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Vertices array and faces array.

    Raises
    ------
    TypeError
        If the mesh does not match supported structures.
    """
    # trimesh
    if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        V = np.asarray(mesh.vertices)
        F = np.asarray(mesh.faces)
        return V, F
    # dict
    if isinstance(mesh, dict) and "vertices" in mesh and "faces" in mesh:
        return np.asarray(mesh["vertices"]), np.asarray(mesh["faces"])
    raise TypeError("mesh must be trimesh-like or dict(vertices,faces)")


def _bbox_params(V: np.ndarray):
    """
    Compute axis-aligned bounding-box statistics for a vertex cloud.

    Parameters
    ----------
    V : np.ndarray
        Vertex array of shape (N, 3).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - "min": per-axis minimum (3,)
        - "max": per-axis maximum (3,)
        - "size": bbox size (max-min) (3,)
        - "center": bbox center (3,)
        - "diag": Euclidean norm of bbox size (float)
    """
    mn = V.min(axis=0)
    mx = V.max(axis=0)
    size = mx - mn
    center = 0.5 * (mx + mn)
    diag = float(np.linalg.norm(size))
    return {"min": mn, "max": mx, "size": size, "center": center, "diag": diag}


class ParametricCadQuerySynthGenerator:
    """
    Generate geometry variants from a parametric CadQuery template.

    This generator samples parameter values within provided ranges, builds
    a CadQuery solid via `template_fn`, tessellates it to a triangle mesh, and
    wraps results as PhysicalSample-like objects using `to_physical_sample`.

    Inputs
    ------
    - template_fn(params) -> cadquery.Workplane or Shape
    - param_ranges: {name: (lo, hi)}
    - n_variants: number of variants to generate

    Outputs
    -------
    Samples carry:
    - fields: vertices (N,3), faces (F,3)
    - meta: generator name, template name, sampled params, and parameter ranges

    Parameters
    ----------
    cfg : Optional[SynthConfig]
        Configuration controlling reproducibility, device, and dtype.
    """
    """
    Generate geometry variants from a parametric CadQuery template.

    You provide:
      - template_fn(params) -> cadquery.Workplane or Shape
      - param_ranges: {name: (lo, hi)}
      - n_variants

    Output samples:
      fields: vertices (N,3), faces (F,3), params (dict)
      meta: template name, ranges
    """
    def __init__(self, cfg: Optional[SynthConfig] = None):
        """
        Initialize the parametric CadQuery geometry generator.

        Parameters
        ----------
        cfg : Optional[SynthConfig]
            Optional generator configuration. If not provided, defaults are used.
        """
        self.cfg = cfg or SynthConfig()

    def generate(
        self,
        *,
        template_fn: Callable[[Dict[str, float]], Any],
        param_ranges: Dict[str, Tuple[float, float]],
        n_variants: int = 16,
        tess_linear_deflection: float = 0.2,
        tess_angular_deflection: float = 0.3,
        seed_offset: int = 0,
        name: str = "cadquery_template",
    ) -> SynthOutput:
        """
        Sample parameters, generate CadQuery solids, tessellate, and return samples.

        Parameters
        ----------
        template_fn : Callable[[Dict[str, float]], Any]
            Function that builds a CadQuery solid/workplane from sampled parameters.
        param_ranges : Dict[str, Tuple[float, float]]
            Mapping of parameter name -> (lo, hi) sampling bounds.
        n_variants : int, optional
            Number of variants to generate. Default is 16.
        tess_linear_deflection : float, optional
            Tessellation linear deflection. Default is 0.2.
        tess_angular_deflection : float, optional
            Tessellation angular deflection. Default is 0.3.
        seed_offset : int, optional
            Offset added to cfg.seed for deterministic variation. Default is 0.
        name : str, optional
            Template name recorded in metadata. Default is "cadquery_template".

        Returns
        -------
        SynthOutput
            Output containing generated PhysicalSample-like objects and metadata.

        Raises
        ------
        ImportError
            If CadQuery is not available.
        """
        cq = _try_import_cadquery()
        if cq is None:
            raise ImportError("cadquery is not available. Install cadquery to use ParametricCadQuerySynthGenerator.")

        rng = np.random.default_rng(int(self.cfg.seed) + int(seed_offset))

        device = torch.device(self.cfg.device)
        dtype = getattr(torch, self.cfg.dtype)

        samples = []
        for i in range(int(n_variants)):
            params = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in param_ranges.items()}
            solid = template_fn(params)
            V, F = _mesh_from_cq_solid(cq, solid, linear_deflection=tess_linear_deflection, angular_deflection=tess_angular_deflection)

            ps_like = {
                "fields": {
                    "vertices": torch.tensor(V, device=device, dtype=torch.float32),
                    "faces": torch.tensor(F, device=device, dtype=torch.long),
                },
                "coords": {},
                "meta": {
                    "generator": "ParametricCadQuerySynthGenerator",
                    "template": name,
                    "params": params,
                    "param_ranges": {k: [float(a), float(b)] for k, (a, b) in param_ranges.items()},
                },
            }
            samples.append(to_physical_sample(ps_like, device=device, dtype=dtype))

        return SynthOutput(samples=samples, extras={"template": name, "n_variants": int(n_variants)})


class STLTemplateSynthGenerator:
    """
    Synthetic geometry generator driven by an existing STL/mesh template.

    Behavior
    --------
    1) Loads a mesh (via trimesh if a path is provided).
    2) Computes its axis-aligned bounding box.
    3) If CadQuery is available and `template` is one of ("box", "cylinder", "capsule"):
       builds a simple parametric primitive matched to bbox scale, jitters parameters,
       tessellates, and recenters to the original bbox center.
    4) Otherwise falls back to vertex-space augmentation of the original mesh:
       similarity-preserving transforms (uniform + axis scaling, small rotations)
       plus optional additive vertex noise.

    Supported templates (when CadQuery exists)
    -----------------------------------------
    - "box": length/width/height
    - "cylinder": radius/height (radius inferred from xy bbox)
    - "capsule": radius/height (approx)

    Parameters
    ----------
    cfg : Optional[SynthConfig]
        Configuration controlling reproducibility, device, and dtype.
    """
    """
    Fallback generator:
      - takes an STL/mesh, infers bbox,
      - optionally builds a simple parametric CadQuery primitive matching bbox,
      - or falls back to vertex-space augmentation if cadquery missing.

    Templates (when cadquery exists):
      - "box": length/width/height
      - "cylinder": radius/height (radius inferred from xy bbox)
      - "capsule": radius/height (approx)

    If cadquery not available:
      - does similarity-preserving transforms (scale/axis-scale/rot/noise) on original mesh vertices.
    """
    def __init__(self, cfg: Optional[SynthConfig] = None):
        """
        Initialize the STL template generator.

        Parameters
        ----------
        cfg : Optional[SynthConfig]
            Optional generator configuration. If not provided, defaults are used.
        """
        self.cfg = cfg or SynthConfig()

    def _load_mesh(self, mesh_or_path: Union[str, Any]) -> Any:
        """
        Load a mesh from a filesystem path or pass through an in-memory mesh object.

        Parameters
        ----------
        mesh_or_path : Union[str, Any]
            Path to a mesh file (e.g., STL/OBJ) or an already loaded mesh object.

        Returns
        -------
        Any
            Loaded mesh object.

        Raises
        ------
        ImportError
            If trimesh is required for path loading but not available.
        """
        tm = _try_import_trimesh()
        if isinstance(mesh_or_path, str):
            if tm is None:
                raise ImportError("trimesh is required to load STL/OBJ paths.")
            return tm.load(mesh_or_path)
        return mesh_or_path

    def generate(
        self,
        *,
        mesh_or_path: Union[str, Any],
        n_variants: int = 16,
        template: str = "box",  # "box" | "cylinder" | "capsule" | "augment"
        param_jitter: float = 0.08,
        rot_deg: float = 8.0,
        noise_amp: float = 0.001,
        tess_linear_deflection: float = 0.2,
        tess_angular_deflection: float = 0.3,
        seed_offset: int = 0,
        name: str = "stl_template",
    ) -> SynthOutput:
        """
        Generate geometry variants from an STL/mesh template.

        Parameters
        ----------
        mesh_or_path : Union[str, Any]
            Mesh object or path to mesh file.
        n_variants : int, optional
            Number of variants to generate. Default is 16.
        template : str, optional
            Generation mode/template:
            - "box", "cylinder", "capsule" (CadQuery primitive if available)
            - otherwise falls back to vertex augmentation. Default is "box".
        param_jitter : float, optional
            Relative jitter applied to primitive parameters and augment scaling.
            Default is 0.08.
        rot_deg : float, optional
            Maximum rotation magnitude (degrees) for vertex augmentation.
            Default is 8.0.
        noise_amp : float, optional
            Standard deviation of additive vertex noise (augmentation fallback).
            Default is 0.001.
        tess_linear_deflection : float, optional
            Tessellation linear deflection for CadQuery primitives. Default is 0.2.
        tess_angular_deflection : float, optional
            Tessellation angular deflection for CadQuery primitives. Default is 0.3.
        seed_offset : int, optional
            Offset added to cfg.seed for deterministic variation. Default is 0.
        name : str, optional
            Name recorded in metadata. Default is "stl_template".

        Returns
        -------
        SynthOutput
            Output containing generated PhysicalSample-like objects and metadata.

        Raises
        ------
        ImportError
            If trimesh is required to load paths but not available.
        """
        tm = _try_import_trimesh()
        mesh = self._load_mesh(mesh_or_path)
        V0, F0 = _extract_mesh_arrays(mesh)
        bb = _bbox_params(V0)

        cq = _try_import_cadquery()
        rng = np.random.default_rng(int(self.cfg.seed) + int(seed_offset))

        device = torch.device(self.cfg.device)
        dtype = getattr(torch, self.cfg.dtype)

        samples = []

        def augment_vertices(V: np.ndarray) -> np.ndarray:
            """
            Apply geometric augmentations to a vertex array.

            Augmentations include:
            - recentering around the source bbox center
            - uniform scaling and per-axis scaling jitter
            - small random Euler rotations
            - optional additive Gaussian noise
            - retranslation back to original center

            Parameters
            ----------
            V : np.ndarray
                Vertex array of shape (N, 3).

            Returns
            -------
            np.ndarray
                Augmented vertex array of shape (N, 3).
            """
            # centered
            center = bb["center"]
            Vc = V - center

            # scale
            s = float(rng.uniform(1.0 - param_jitter, 1.0 + param_jitter))
            axs = rng.uniform(1.0 - param_jitter, 1.0 + param_jitter, size=(3,))
            Vc = Vc * (s * axs)

            # rotation
            r = math.radians(rot_deg)
            rx, ry, rz = rng.uniform(-r, r), rng.uniform(-r, r), rng.uniform(-r, r)

            cx, sx = math.cos(rx), math.sin(rx)
            cy, sy = math.cos(ry), math.sin(ry)
            cz, sz = math.cos(rz), math.sin(rz)
            Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
            Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
            Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
            R = Rz @ Ry @ Rx
            Vc = (Vc @ R.T)

            # noise
            if noise_amp and noise_amp > 0:
                Vc = Vc + rng.normal(0.0, noise_amp, size=Vc.shape)

            return Vc + center

        for i in range(int(n_variants)):
            if cq is not None and template.lower() in ("box", "cylinder", "capsule"):
                # infer base params from bbox
                sx, sy, sz = bb["size"].tolist()
                # jitter params
                j = lambda v: float(v * rng.uniform(1.0 - param_jitter, 1.0 + param_jitter))

                if template.lower() == "box":
                    L, W, H = j(sx), j(sy), j(sz)
                    solid = cq.Workplane("XY").box(L, W, H)

                    params = {"L": L, "W": W, "H": H}

                elif template.lower() == "cylinder":
                    r0 = 0.25 * (sx + sy)
                    R = j(max(r0 * 0.5, 1e-6))
                    H = j(max(sz, 1e-6))
                    solid = cq.Workplane("XY").cylinder(H, R)
                    params = {"R": R, "H": H}

                else:  # capsule (approx): cylinder + two spheres
                    r0 = 0.25 * (sx + sy)
                    R = j(max(r0 * 0.5, 1e-6))
                    H = j(max(sz, 1e-6))
                    cyl_h = max(H - 2 * R, 1e-6)
                    solid = cq.Workplane("XY").cylinder(cyl_h, R).faces(">Z").workplane().sphere(R).translate((0,0,cyl_h/2))
                    solid2 = cq.Workplane("XY").sphere(R).translate((0,0,-cyl_h/2))
                    # union two halves
                    solid = solid.union(solid2)
                    params = {"R": R, "H": H}

                # tessellate
                V, F = _mesh_from_cq_solid(cq, solid, linear_deflection=tess_linear_deflection, angular_deflection=tess_angular_deflection)

                # translate to original bbox center
                V = V + bb["center"][None, :]

                ps_like = {
                    "fields": {
                        "vertices": torch.tensor(V, device=device, dtype=torch.float32),
                        "faces": torch.tensor(F, device=device, dtype=torch.long),
                    },
                    "coords": {},
                    "meta": {
                        "generator": "STLTemplateSynthGenerator",
                        "mode": "cadquery_template",
                        "template": template.lower(),
                        "params": params,
                        "source_bbox": {"size": bb["size"].tolist(), "center": bb["center"].tolist()},
                        "name": name,
                    },
                }
                samples.append(to_physical_sample(ps_like, device=device, dtype=dtype))

            else:
                # fallback: augment original mesh vertices
                V = augment_vertices(V0)
                ps_like = {
                    "fields": {
                        "vertices": torch.tensor(V, device=device, dtype=torch.float32),
                        "faces": torch.tensor(F0, device=device, dtype=torch.long),
                    },
                    "coords": {},
                    "meta": {
                        "generator": "STLTemplateSynthGenerator",
                        "mode": "vertex_augment",
                        "template": "augment",
                        "param_jitter": float(param_jitter),
                        "rot_deg": float(rot_deg),
                        "noise_amp": float(noise_amp),
                        "source_bbox": {"size": bb["size"].tolist(), "center": bb["center"].tolist()},
                        "name": name,
                    },
                }
                samples.append(to_physical_sample(ps_like, device=device, dtype=dtype))

        return SynthOutput(samples=samples, extras={"n_variants": int(n_variants), "template": template.lower(), "name": name})