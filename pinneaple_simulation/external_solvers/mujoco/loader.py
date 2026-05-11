"""MuJoCo model loading and data initialization for PINNeAPPle.

Wraps mujoco.MjModel / mujoco.MjData construction with a structured config
so the rest of the pipeline can remain framework-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass
class MuJoCoConfig:
    """Configuration for a MuJoCo simulation.

    Parameters
    ----------
    model_path : path to a .xml (MJCF/URDF) or .mjb binary model file.
        Mutually exclusive with model_xml.
    model_xml : raw XML string (MJCF).
        Mutually exclusive with model_path.
    assets : dict mapping virtual filenames to bytes — forwarded to
        mujoco.MjModel.from_xml_string when using model_xml.
    dt : override for model.opt.timestep (seconds). None = use model default.
    n_steps : total number of mj_step calls per simulate() run.
    n_substeps : how many mj_step calls to batch per loop iteration
        (releases the GIL between batches, enabling parallelism).
    gravity : override gravity vector (3-tuple). None = use model default.
    extra_opts : additional key-value pairs written into model.opt before
        the simulation runs.
    """
    model_path: Optional[Union[str, Path]] = None
    model_xml: Optional[str] = None
    assets: Dict[str, bytes] = field(default_factory=dict)
    dt: Optional[float] = None
    n_steps: int = 1000
    n_substeps: int = 1
    gravity: Optional[tuple] = None
    extra_opts: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model_path is None and self.model_xml is None:
            raise ValueError("Provide either model_path or model_xml.")
        if self.model_path is not None and self.model_xml is not None:
            raise ValueError("Provide only one of model_path or model_xml.")


def load_model(config: MuJoCoConfig):
    """Load a MjModel from the config.

    Returns
    -------
    model : mujoco.MjModel
    """
    try:
        import mujoco
    except ImportError:
        raise ImportError(
            "mujoco is required. Install it with: pip install mujoco"
        )

    if config.model_path is not None:
        model = mujoco.MjModel.from_xml_path(str(config.model_path))
    else:
        model = mujoco.MjModel.from_xml_string(config.model_xml, config.assets)

    if config.dt is not None:
        model.opt.timestep = config.dt
    if config.gravity is not None:
        model.opt.gravity[:] = config.gravity
    for k, v in config.extra_opts.items():
        setattr(model.opt, k, v)

    return model


def make_data(model):
    """Allocate a fresh MjData for the given model.

    Returns
    -------
    data : mujoco.MjData
    """
    try:
        import mujoco
    except ImportError:
        raise ImportError(
            "mujoco is required. Install it with: pip install mujoco"
        )
    return mujoco.MjData(model)
