"""Genesis AI scene construction for PINNeAPPle.

GenesisConfig captures the full scene specification in a plain dataclass
so that scene creation is reproducible, serialisable, and testable without
an active Genesis/GPU session.

build_scene() materialises the config into a gs.Scene ready for simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EntitySpec:
    """Specification for a single entity added to the Genesis scene.

    Parameters
    ----------
    morph : dict describing the shape, e.g.
        {"type": "URDF", "file": "robot.urdf", "fixed": True}
        {"type": "MJCF", "file": "panda.xml"}
        {"type": "Box",  "size": (0.1, 0.1, 0.1), "pos": (0, 0, 0.5)}
        {"type": "Plane"}
        {"type": "Sphere", "radius": 0.05}
    material : dict describing the physics material, e.g.
        {"type": "Rigid", "rho": 1000}
        {"type": "MPM.Elastic", "rho": 500}
        {"type": "SPH.Liquid"}
        {"type": "PBD.Cloth"}
    surface : optional surface properties dict, e.g. {"color": (0.8, 0.2, 0.2, 1.0)}
    vis_mode : "visual" or "particle"
    name : optional label used to retrieve the entity after build
    """
    morph: Dict[str, Any]
    material: Optional[Dict[str, Any]] = None
    surface: Optional[Dict[str, Any]] = None
    vis_mode: str = "visual"
    name: Optional[str] = None


@dataclass
class GenesisConfig:
    """Full configuration for a Genesis AI simulation scene.

    Parameters
    ----------
    backend : "gpu", "cpu", or "metal"
    precision : "32" or "64"
    dt : timestep in seconds (written into SimOptions)
    substeps : substeps per scene.step() call
    n_steps : total steps to run in simulate()
    n_envs : number of parallel environments (>1 enables vectorised mode)
    env_spacing : (x, y) spacing between envs in vectorised mode
    gravity : (gx, gy, gz) gravity vector
    requires_grad : enable differentiable simulation (MPM/Tool solvers only)
    entities : list of EntitySpec describing objects in the scene
    show_viewer : open the interactive viewer window
    rigid_options : extra kwargs forwarded to gs.options.RigidOptions
    mpm_options : extra kwargs forwarded to gs.options.MPMOptions (if used)
    sph_options : extra kwargs forwarded to gs.options.SPHOptions (if used)
    """
    backend: str = "gpu"
    precision: str = "32"
    dt: float = 0.01
    substeps: int = 10
    n_steps: int = 1000
    n_envs: int = 1
    env_spacing: Tuple[float, float] = (1.0, 1.0)
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    requires_grad: bool = False
    entities: List[EntitySpec] = field(default_factory=list)
    show_viewer: bool = False
    rigid_options: Dict[str, Any] = field(default_factory=dict)
    mpm_options: Optional[Dict[str, Any]] = None
    sph_options: Optional[Dict[str, Any]] = None


def _resolve_backend(name: str):
    import genesis as gs
    return {"gpu": gs.gpu, "cpu": gs.cpu, "metal": gs.metal}.get(name, gs.cpu)


def _build_morph(spec: dict):
    import genesis as gs
    morph_type = spec["type"]
    kw = {k: v for k, v in spec.items() if k != "type"}
    morph_cls = getattr(gs.morphs, morph_type)
    return morph_cls(**kw)


def _build_material(spec: Optional[dict]):
    if spec is None:
        return None
    import genesis as gs
    mat_type = spec["type"]       # e.g. "Rigid", "MPM.Elastic", "SPH.Liquid"
    kw = {k: v for k, v in spec.items() if k != "type"}
    parts = mat_type.split(".")
    obj = gs.materials
    for part in parts:
        obj = getattr(obj, part)
    return obj(**kw)


def _build_surface(spec: Optional[dict]):
    if spec is None:
        return None
    import genesis as gs
    kw = {k: v for k, v in spec.items()}
    return gs.surfaces.Default(**kw)


def build_scene(config: GenesisConfig):
    """Materialise a GenesisConfig into a gs.Scene.

    The scene is built (JIT compiled) with config.n_envs environments.
    Entity references are stored in scene._pinneaple_entities dict (keyed
    by EntitySpec.name or positional index) for later state extraction.

    Returns
    -------
    gs.Scene — ready to call scene.step() on
    """
    try:
        import genesis as gs
    except ImportError:
        raise ImportError(
            "genesis-world is required. Install with:\n"
            "  pip install genesis-world\n"
            "or from source: https://github.com/Genesis-Embodied-AI/Genesis"
        )

    backend = _resolve_backend(config.backend)
    gs.init(backend=backend, precision=config.precision, logging_level="warning")

    rigid_kw = {"gravity": config.gravity, **config.rigid_options}
    sim_opts = gs.options.SimOptions(
        dt=config.dt,
        substeps=config.substeps,
        requires_grad=config.requires_grad,
    )

    scene_kwargs: dict = dict(
        sim_options=sim_opts,
        rigid_options=gs.options.RigidOptions(**rigid_kw),
        show_viewer=config.show_viewer,
    )

    if config.mpm_options is not None:
        scene_kwargs["mpm_options"] = gs.options.MPMOptions(**config.mpm_options)
    if config.sph_options is not None:
        scene_kwargs["sph_options"] = gs.options.SPHOptions(**config.sph_options)

    scene = gs.Scene(**scene_kwargs)
    entity_map: dict = {}

    for idx, espec in enumerate(config.entities):
        morph = _build_morph(espec.morph)
        material = _build_material(espec.material)
        surface = _build_surface(espec.surface)

        add_kwargs: dict = dict(morph=morph, vis_mode=espec.vis_mode)
        if material is not None:
            add_kwargs["material"] = material
        if surface is not None:
            add_kwargs["surface"] = surface

        entity = scene.add_entity(**add_kwargs)
        key = espec.name if espec.name else str(idx)
        entity_map[key] = entity

    if config.n_envs > 1:
        scene.build(n_envs=config.n_envs, env_spacing=config.env_spacing)
    else:
        scene.build()

    scene._pinneaple_entities = entity_map  # type: ignore[attr-defined]
    return scene
