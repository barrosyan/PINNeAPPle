# -*- coding: utf-8 -*-
"""Physics Synthetic Data Factory for Physical AI.

End-to-end 9-stage pipeline that generates multimodal datasets for training
PINNs, Neural Operators, World Models, and Physical AI foundation models.

Pipeline
--------
::

    Stage 1  ScenarioGenerator   prompt / JSON  →  PhysicsScenario + metadata
    Stage 2  PhysicsSimulator    scenario        →  physical field trajectories
    Stage 3  DatasetPackager     fields          →  Zarr storage
    Stage 4  PhysicsRenderer     fields          →  RGB / thermal / depth frames
    Stage 5  CameraSystem        fields          →  multi-sensor observations
    Stage 6  RealityRandomizer   frames          →  sim-to-real augmented frames
    Stage 7  PhotorealisticEnhancer              →  photorealistic frames
    Stage 8  DatasetPackager     all outputs     →  canonical dataset structure
    Stage 9  TrainingHooks       dataset         →  PyTorch DataLoader interfaces

The factory reuses existing PINNeAPPle modules at every stage:
  - pinneapple_worldmodel.simulator      (PhysicsSimulator)
  - pinneapple_worldmodel.scenario       (PhysicsScenario, BUILTIN_SCENARIOS)
  - pinneapple_data.*                    (Zarr / UPD storage)
  - pinneapple_tools.visualization       (FlowVisualizer)

Dataset structure
-----------------
::

    <output_dir>/
      sample_000001/
        video_rgb.mp4
        video_thermal.mp4
        velocity.zarr/
        pressure.zarr/
        temperature.zarr/
        mesh.vtk
        metadata.json
        pde.json
        boundary_conditions.json
        camera.json
      dataset_manifest.json

Public API
----------
  SyntheticFactoryConfig           — single configuration dataclass for all stages
  SyntheticDataFactory    — main orchestrator class
  FactoryResult           — output of a generate_dataset() call

Quick start
-----------
::

    from pinneapple_worldmodel import SyntheticDataFactory, SyntheticFactoryConfig

    factory = SyntheticDataFactory(SyntheticFactoryConfig(
        scenario_input = {
            "domain":   "fluid_dynamics",
            "geometry": "pipe",
            "length":   10,
            "diameter": 0.5,
            "fluid":    "water",
            "Re":       5000,
        },
        n_samples    = 50,
        output_dir   = "./physics_dataset_pipe_flow",
        render_fps   = 24,
        sensors      = ["rgb", "thermal", "depth"],
        device       = "cpu",
    ))
    result = factory.generate_dataset()
    print(result.summary())

    # Use for training (Stage 9)
    hooks  = result.training_hooks()
    ds     = hooks.neural_operator(field_names=["u", "v", "p"], horizon=1)
    loader = ds.to_dataloader(batch_size=16)
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

# ── Existing PINNeAPPle modules ──────────────────────────────────────────────
from .scenario import PhysicsScenario, BUILTIN_SCENARIOS
from .simulator import PhysicsSimulator, TrajectoryData

# ── New factory modules ──────────────────────────────────────────────────────
from .scenario_generator import ScenarioGenerator, ScenarioSpec
from .physics_renderer   import PhysicsRenderer, RendererConfig, RenderResult
from .camera_system      import CameraSystem, MultiCameraArray
from .reality_randomizer import RealityRandomizer, RandomizerConfig
from .photorealistic_enhancer import PhotorealisticEnhancer, EnhancerConfig
from .dataset_packager       import DatasetPackager, PackagerConfig, PackagedSample
from .training_hooks         import TrainingHooks
from .physics_solver_bridge  import (
    PhysicsSolverBridge, SolverBridgeConfig, GroundTruthPackager,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SyntheticFactoryConfig:
    """Single configuration dataclass covering all 9 stages.

    Parameters
    ----------
    scenario_input : dict | str | PhysicsScenario
        Input for ScenarioGenerator.  Can be:
        - JSON dict  (``{"domain": "fluid_dynamics", "geometry": "pipe", …}``)
        - Text prompt (``"simulate turbulent pipe flow at Re=5000"``)
        - Built-in name (``"ns2d_cavity"``)
        - :class:`~pinneapple_worldmodel.scenario.PhysicsScenario` object
    n_samples : int
        Number of random-parameter samples to generate.
    output_dir : str | Path
        Root output directory.
    device : str
        Compute device for the physics simulator.
    render_fps : int
        Frames per second for rendered videos.
    sensors : list of str
        Sensor channels to render: any of ``"rgb"``, ``"thermal"``,
        ``"depth"``, ``"ir"``.
    render_resolution : (H, W)
        Output frame resolution.
    apply_randomization : bool
        Whether to apply reality-gap augmentations (Stage 6).
    apply_enhancement : bool
        Whether to apply photorealistic enhancement (Stage 7).
    enhancement_backend : str
        Enhancement backend: ``"stub"``, ``"local_diffusion"``, ``"cosmos"``.
    enhancement_strength : float
        Diffusion denoising strength (0 = no change, 1 = full re-gen).
    store_format : str
        Physical field storage format: ``"zarr"`` or ``"npy"``.
    save_vtk : bool
        Write VTK mesh files for ParaView.
    verbose : bool
        Print progress to stdout.
    seed : int or None
        Random seed for reproducibility.
    """
    scenario_input:       Any             = "heat_2d"
    n_samples:            int             = 10
    output_dir:           Union[str, Path] = "./physics_dataset"
    device:               str             = "cpu"
    render_fps:           int             = 24
    sensors:              List[str]       = field(default_factory=lambda: ["rgb", "thermal", "depth"])
    render_resolution:    tuple           = (256, 256)
    apply_randomization:  bool            = True
    apply_enhancement:    bool            = False
    enhancement_backend:  str             = "stub"
    enhancement_strength: float           = 0.20
    store_format:         str             = "zarr"
    save_vtk:             bool            = True
    verbose:              bool            = True
    seed:                 Optional[int]   = 42
    # Stage 2 — solver backend selection
    solver:               str             = "builtin"   # "builtin"|"openfoam"|"fenics"|"pinn"|"fno"
    openfoam_case_dir:    Optional[str]   = None
    pinn_checkpoint:      Optional[str]   = None
    fno_checkpoint:       Optional[str]   = None
    generate_temperature: bool            = True
    generate_concentration: bool          = False

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)


# ---------------------------------------------------------------------------
# FactoryResult
# ---------------------------------------------------------------------------

@dataclass
class FactoryResult:
    """Output of SyntheticDataFactory.generate_dataset().

    Attributes
    ----------
    config : SyntheticFactoryConfig
    spec : ScenarioSpec
    samples : list of PackagedSample
    output_dir : Path
    elapsed_s : float
    """
    config:     SyntheticFactoryConfig
    spec:       ScenarioSpec
    samples:    List[PackagedSample]
    output_dir: Path
    elapsed_s:  float = 0.0

    def __len__(self) -> int:
        return len(self.samples)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  Physics Synthetic Data Factory — Result",
            "=" * 60,
            f"  Scenario  : {self.spec.name}",
            f"  PDE kind  : {self.spec.pde_kind}",
            f"  Domain    : {self.spec.domain}",
            f"  Samples   : {len(self.samples)}",
            f"  Output    : {self.output_dir}",
            f"  Elapsed   : {self.elapsed_s:.1f}s",
            f"  Fields    : {', '.join(self.spec.field_names)}",
            f"  Sensors   : {', '.join(self.config.sensors)}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def training_hooks(self) -> TrainingHooks:
        """Return TrainingHooks for all 4 training objectives."""
        return TrainingHooks(self.output_dir, device=self.config.device)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name":  self.spec.name,
            "pde_kind":       self.spec.pde_kind,
            "n_samples":      len(self.samples),
            "output_dir":     str(self.output_dir),
            "elapsed_s":      self.elapsed_s,
            "field_names":    self.spec.field_names,
            "sensors":        self.config.sensors,
        }


# ---------------------------------------------------------------------------
# SyntheticDataFactory
# ---------------------------------------------------------------------------

class SyntheticDataFactory:
    """Physics Synthetic Data Factory for Physical AI.

    Orchestrates all 9 pipeline stages to produce a complete multimodal
    dataset aligned with physical ground truth.

    Parameters
    ----------
    config : SyntheticFactoryConfig

    Examples
    --------
    From JSON prompt::

        factory = SyntheticDataFactory(SyntheticFactoryConfig(
            scenario_input = {"domain": "fluid_dynamics", "geometry": "cylinder",
                              "fluid": "air", "Re": 200},
            n_samples = 100,
            output_dir = "./cylinder_dataset",
            sensors = ["rgb", "thermal"],
            device = "cpu",
        ))
        result = factory.generate_dataset()

    From built-in scenario::

        factory = SyntheticDataFactory(SyntheticFactoryConfig(
            scenario_input = "ns2d_cavity",
            n_samples = 50,
        ))
        result = factory.generate_dataset()

    One-shot generation::

        result = SyntheticDataFactory.from_prompt(
            "turbulent flow around a cylinder at Re=3900, fluid=air",
            n_samples=20,
            output_dir="./cyl_dataset",
        )
    """

    def __init__(self, config: Optional[SyntheticFactoryConfig] = None) -> None:
        self.cfg = config or SyntheticFactoryConfig()
        if self.cfg.seed is not None:
            np.random.seed(self.cfg.seed)
            torch.manual_seed(self.cfg.seed)
        self._init_stages()

    @classmethod
    def from_prompt(cls, prompt: str, **kwargs) -> "FactoryResult":
        """One-shot: from prompt string → FactoryResult."""
        cfg = SyntheticFactoryConfig(scenario_input=prompt, **kwargs)
        return cls(cfg).generate_dataset()

    @classmethod
    def from_dict(cls, spec_dict: Dict[str, Any], **kwargs) -> "FactoryResult":
        """One-shot: from JSON dict → FactoryResult."""
        cfg = SyntheticFactoryConfig(scenario_input=spec_dict, **kwargs)
        return cls(cfg).generate_dataset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_dataset(self) -> FactoryResult:
        """Run all 9 stages and return a :class:`FactoryResult`.

        This is the main entry point.  Call once to generate the full
        multi-sample dataset.
        """
        t0 = time.time()
        self._log(f"\n{'='*60}")
        self._log("  Physics Synthetic Data Factory")
        self._log(f"  Scenario : {self.spec.name}")
        self._log(f"  Samples  : {self.cfg.n_samples}")
        self._log(f"  Output   : {self.cfg.output_dir}")
        self._log(f"{'='*60}")

        samples: List[PackagedSample] = []
        for i in range(self.cfg.n_samples):
            sample = self._generate_one(i)
            samples.append(sample)
            pct = (i + 1) / self.cfg.n_samples * 100
            self._log(f"  [{i+1:>4}/{self.cfg.n_samples}]  sample_{str(i).zfill(6)}  ({pct:.0f}%)")

        elapsed = time.time() - t0
        result  = FactoryResult(
            config     = self.cfg,
            spec       = self.spec,
            samples    = samples,
            output_dir = self.cfg.output_dir,
            elapsed_s  = elapsed,
        )
        # Save factory config alongside the dataset
        (self.cfg.output_dir / "factory_config.json").write_text(
            json.dumps(result.to_dict(), indent=2)
        )
        self._log(result.summary())
        return result

    def generate_one(self, sample_idx: int = 0) -> PackagedSample:
        """Generate a single sample (useful for debugging)."""
        return self._generate_one(sample_idx)

    # ------------------------------------------------------------------
    # Internal stage runner
    # ------------------------------------------------------------------

    def _generate_one(self, idx: int) -> PackagedSample:
        # Stage 2: Physics solver → SolverOutput (velocity, pressure, temperature, …)
        solver_out = self._solver_bridge.solve(self.spec)

        # Convert SolverOutput → TrajectoryData (for downstream compatibility)
        traj = solver_out.to_trajectory(self.spec.name)

        # Stage 3: Ground truth packaging (zarr fields + VTK + metadata JSONs)
        self._gt_packager.write(solver_out, self.spec, sample_idx=idx)

        # Convert to numpy (T, C, Ny, Nx) for renderer
        states_np = traj.states.cpu().numpy()

        # Stage 4+5: Physics rendering + camera system
        render_result = self._render(states_np, idx)

        # Stage 6: Reality randomization
        if self.cfg.apply_randomization:
            render_result.frames = self._randomizer.augment(render_result.frames)

        # Stage 7: Photorealistic enhancement
        if self.cfg.apply_enhancement:
            render_result.frames = self._enhancer.enhance(render_result.frames)

        # Stage 8: Dataset packaging
        camera_meta = self._camera_system.to_dict()
        sample = self._packager.package(
            trajectory    = traj,
            spec          = self.spec,
            render_result = render_result,
            sample_idx    = idx,
            camera_config = camera_meta,
        )
        return sample

    # ------------------------------------------------------------------
    # Stage initialisation
    # ------------------------------------------------------------------

    def _init_stages(self) -> None:
        # Stage 1: Scenario generation
        gen        = ScenarioGenerator(device=self.cfg.device)
        self.spec  = gen.from_any(self.cfg.scenario_input)
        self._log(f"  Scenario parsed: {self.spec.name}  ({self.spec.pde_kind})")

        # Stage 2: Physics solver bridge (dispatches to builtin / OpenFOAM / FEniCS / PINN / FNO)
        self._solver_bridge = PhysicsSolverBridge(SolverBridgeConfig(
            solver                 = self.cfg.solver,
            generate_temperature   = self.cfg.generate_temperature,
            generate_concentration = self.cfg.generate_concentration,
            openfoam_case_dir      = Path(self.cfg.openfoam_case_dir) if self.cfg.openfoam_case_dir else None,
            pinn_checkpoint        = Path(self.cfg.pinn_checkpoint)   if self.cfg.pinn_checkpoint   else None,
            fno_checkpoint         = Path(self.cfg.fno_checkpoint)    if self.cfg.fno_checkpoint    else None,
        ))
        # Also keep PhysicsSimulator as fallback for trajectory→TrajectoryData conversions
        self._simulator = PhysicsSimulator(
            self.spec.scenario,
            device  = self.cfg.device,
            verbose = False,
        )

        # Stage 3: Ground truth packager (zarr / npy physical fields)
        self._gt_packager = GroundTruthPackager(
            output_dir  = self.cfg.output_dir,
            format      = self.cfg.store_format,
            overwrite   = True,
        )

        # Stage 4: Renderer
        self._renderer = PhysicsRenderer(RendererConfig(
            resolution  = tuple(self.cfg.render_resolution),
            fps         = self.cfg.render_fps,
            sensors     = self.cfg.sensors,
            dark_background = True,
        ))

        # Stage 5: Camera system
        camera_array         = MultiCameraArray.from_config(self.spec.sensor_config)
        self._camera_system  = CameraSystem(camera_array, self.spec.scenario.domain_bounds)

        # Stage 6: Reality randomizer
        self._randomizer = RealityRandomizer(RandomizerConfig(seed=self.cfg.seed))

        # Stage 7: Photorealistic enhancer
        self._enhancer = PhotorealisticEnhancer(EnhancerConfig(
            backend  = self.cfg.enhancement_backend,
            strength = self.cfg.enhancement_strength,
            device   = self.cfg.device,
        ))

        # Stage 8: Packager
        self._packager = DatasetPackager(PackagerConfig(
            output_dir  = self.cfg.output_dir,
            format      = self.cfg.store_format,
            save_vtk    = self.cfg.save_vtk,
            save_videos = True,
            overwrite   = True,   # renderer pre-creates the dir; packager must write into it
        ))

    # ------------------------------------------------------------------
    # Rendering helper (combines stages 4 + 5)
    # ------------------------------------------------------------------

    def _render(self, states_np: np.ndarray, idx: int) -> RenderResult:
        """Run renderer and camera system on the state sequence."""
        field_names = self.spec.field_names
        sample_dir  = self.cfg.output_dir / f"sample_{str(idx).zfill(6)}"

        # Stage 4: Physics renderer (matplotlib-based)
        render_result = self._renderer.render(
            states      = states_np,
            field_names = field_names,
            output_dir  = sample_dir,
            sample_id   = str(idx).zfill(6),
        )

        # Stage 5: Camera system observations
        # (merge camera observations into render result frames)
        cam_obs = self._camera_system.observe_sequence(states_np, field_names)
        for cam_name, cam_seq in cam_obs.items():
            # Camera observations supplement the renderer output
            if cam_name not in render_result.frames:
                render_result.frames[cam_name] = cam_seq

        return render_result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.cfg.verbose:
            print(msg)


# ---------------------------------------------------------------------------
# Convenience one-liners
# ---------------------------------------------------------------------------

def generate_pipe_flow(n_samples: int = 20, output_dir: str = "./pipe_dataset", **kw) -> FactoryResult:
    """Generate a pipe flow dataset (NS2D, Re=100-5000)."""
    return SyntheticDataFactory(SyntheticFactoryConfig(
        scenario_input = {"domain": "fluid_dynamics", "geometry": "pipe",
                          "fluid": "water", "Re": 2000},
        n_samples = n_samples, output_dir = output_dir, **kw,
    )).generate_dataset()


def generate_cylinder_flow(n_samples: int = 20, output_dir: str = "./cylinder_dataset", **kw) -> FactoryResult:
    """Generate flow-around-cylinder dataset (classic von Karman vortex shedding)."""
    return SyntheticDataFactory(SyntheticFactoryConfig(
        scenario_input = {"domain": "fluid_dynamics", "geometry": "cylinder",
                          "fluid": "air", "Re": 200},
        n_samples = n_samples, output_dir = output_dir, **kw,
    )).generate_dataset()


def generate_heat_transfer(n_samples: int = 20, output_dir: str = "./heat_dataset", **kw) -> FactoryResult:
    """Generate 2D heat conduction dataset."""
    return SyntheticDataFactory(SyntheticFactoryConfig(
        scenario_input = "heat_2d",
        n_samples = n_samples, output_dir = output_dir,
        sensors = ["thermal", "rgb"], **kw,
    )).generate_dataset()


def generate_chemical_mixing(n_samples: int = 20, output_dir: str = "./mixing_dataset", **kw) -> FactoryResult:
    """Generate chemical mixing / advection-diffusion dataset."""
    return SyntheticDataFactory(SyntheticFactoryConfig(
        scenario_input = {"domain": "mass_transfer", "geometry": "mixing",
                          "fluid": "water"},
        n_samples = n_samples, output_dir = output_dir,
        sensors = ["rgb", "depth"], **kw,
    )).generate_dataset()
