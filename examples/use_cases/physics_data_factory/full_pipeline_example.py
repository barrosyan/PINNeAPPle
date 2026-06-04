# -*- coding: utf-8 -*-
"""Physics Synthetic Data Factory -- Complete Pipeline Example
=============================================================

Demonstrates the full 9-stage Physical AI data generation pipeline:

  Stage 1  ScenarioGenerator    prompt / JSON / name  ->  PhysicsScenario
  Stage 2  PhysicsSolverBridge  scenario               ->  SolverOutput
           (velocity, pressure, temperature, concentration fields)
  Stage 3  GroundTruthPackager  SolverOutput           ->  zarr / npy on disk
  Stage 4  PhysicsRenderer      fields                 ->  RGB / thermal / depth frames
  Stage 5  CameraSystem         fields                 ->  multi-sensor observations
  Stage 6  RealityRandomizer    frames                 ->  sim-to-real augmentation
  Stage 7  PhotorealisticEnhancer                      ->  (stub here; hooks for diffusion/Cosmos)
  Stage 8  DatasetPackager      everything             ->  canonical dataset structure
  Stage 9  TrainingHooks        dataset                ->  PyTorch DataLoaders
           (NeuralOperator | InversePINN | CosmosEncoder | PhysicsDecoder)

Three scenarios are generated:
  A. Heat conduction      (built-in "heat_2d")
  B. Cavity flow NS2D     (JSON dict)
  C. Advection-diffusion  (natural-language prompt)

Run
---
  python full_pipeline_example.py

Outputs saved to ./outputs/
"""
from __future__ import annotations

import json
import sys
import time

# -- Path bootstrap: allow running directly from the examples directory --------
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

# -- PINNeAPPle world-model imports --------------------------------------------
from pinneapple_worldmodel import (
    # Stage 1
    ScenarioGenerator,
    # Stages 2 + 3
    PhysicsSolverBridge, SolverBridgeConfig,
    GroundTruthPackager,
    # Stage 4
    PhysicsRenderer, RendererConfig,
    # Stage 5
    CameraSystem, MultiCameraArray,
    # Stage 6
    RealityRandomizer, RandomizerConfig,
    # Stage 7
    PhotorealisticEnhancer, EnhancerConfig,
    # Stage 8
    DatasetPackager, PackagerConfig,
    # Stage 9
    TrainingHooks,
    # Main factory (all-in-one)
    SyntheticDataFactory, SyntheticFactoryConfig,
    FactoryResult,
)

import shutil

OUT_DIR = Path(__file__).parent / "outputs"
# Fresh run: remove old outputs so manifests don't accumulate across runs
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DARK   = "#0d1117"
PANEL  = "#161b22"
BLUE   = "#58a6ff"
ORANGE = "#f78166"
GREEN  = "#3fb950"
GOLD   = "#d29922"


# ===============================================================================
# Helpers
# ===============================================================================

def section(title: str, width: int = 66) -> None:
    bar = "=" * width
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def subsection(title: str) -> None:
    print(f"\n  -- {title} --")


def _load_npy_or_zarr(path: Path) -> np.ndarray:
    """Load a field array from .npy or .zarr."""
    npy = path.with_suffix(".npy")
    zarr_dir = path.with_suffix(".zarr")
    if npy.exists():
        return np.load(str(npy))
    try:
        import zarr
        return np.array(zarr.open(str(zarr_dir), mode="r"))
    except Exception:
        return None


# ===============================================================================
# STAGE 1 -- Scenario generation (three input styles)
# ===============================================================================

def demo_stage1():
    section("STAGE 1 -- Scenario Generator")
    gen = ScenarioGenerator()

    # 1-A: built-in name
    subsection("1-A: from built-in name")
    spec_a = gen.from_name("heat_2d")
    print(f"    name      : {spec_a.name}")
    print(f"    pde_kind  : {spec_a.pde_kind}")
    print(f"    grid      : {spec_a.scenario.grid_shape}")
    print(f"    fields    : {spec_a.field_names}")

    # 1-B: JSON dict
    subsection("1-B: from JSON dict")
    spec_b = gen.from_dict({
        "domain":   "fluid_dynamics",
        "geometry": "cavity",
        "fluid":    "air",
        "Re":       400,
        "Nx": 64, "Ny": 64,
        "n_steps": 50,
        "name": "cavity_Re400",
    })
    print(f"    name      : {spec_b.name}")
    print(f"    pde_kind  : {spec_b.pde_kind}")
    print(f"    Re range  : {spec_b.scenario.param_ranges}")
    print(f"    fluid     : {spec_b.fluid['name']}  rho={spec_b.fluid['rho']} kg/m3")

    # 1-C: natural-language prompt
    subsection("1-C: from natural-language prompt")
    spec_c = gen.from_prompt("simulate advection diffusion of a chemical tracer in water")
    print(f"    name      : {spec_c.name}")
    print(f"    pde_kind  : {spec_c.pde_kind}")
    print(f"    domain    : {spec_c.domain}")

    return spec_a, spec_b, spec_c


# ===============================================================================
# STAGES 2 + 3 -- Solver bridge + ground truth packaging (standalone)
# ===============================================================================

def demo_stages_2_3(spec_b):
    section("STAGES 2+3 -- Physics Solver Bridge + Ground Truth Packager")
    dataset_dir = OUT_DIR / "gt_dataset_cavity"

    # Stage 2: run the physics solver
    subsection("Stage 2: PhysicsSolverBridge (builtin FD solver)")
    bridge = PhysicsSolverBridge(SolverBridgeConfig(
        solver               = "builtin",
        generate_temperature = True,
        generate_concentration = True,
    ))
    t0  = time.perf_counter()
    out = bridge.solve(spec_b, params={"Re": 400.0})
    dt  = time.perf_counter() - t0

    print(f"    solver      : {out.solver_name}")
    print(f"    elapsed     : {dt:.2f}s")
    print(f"    fields      : {out.field_names}")
    for fname, arr in out.fields.items():
        print(f"      {fname:15s}  shape={arr.shape}  "
              f"range=[{arr.min():.3f}, {arr.max():.3f}]")

    # Stage 3: persist to disk
    subsection("Stage 3: GroundTruthPackager (zarr / npy + VTK + JSONs)")
    packager = GroundTruthPackager(dataset_dir, format="npy", overwrite=True)
    sdir = packager.write(out, spec_b, sample_idx=0)
    print(f"    sample dir  : {sdir}")
    print("    files written:")
    for f in sorted(sdir.iterdir()):
        size = f.stat().st_size if f.is_file() else "dir"
        print(f"      {f.name:<40s}  {size} bytes" if isinstance(size, int)
              else f"      {f.name}/")

    return out, sdir


# ===============================================================================
# STAGES 4+5 -- Physics renderer + camera system
# ===============================================================================

def demo_stages_4_5(solver_out, spec_b):
    section("STAGES 4+5 -- Physics Renderer + Camera System")
    sample_dir = OUT_DIR / "render_demo"
    states_np  = np.stack(
        [solver_out.fields.get("velocity", np.zeros((solver_out.n_timesteps, 1,
          *spec_b.scenario.grid_shape)))[:,0],
         solver_out.fields.get("pressure", np.zeros((solver_out.n_timesteps,
          *spec_b.scenario.grid_shape))),
         solver_out.fields.get("temperature", np.zeros((solver_out.n_timesteps,
          *spec_b.scenario.grid_shape))),
        ], axis=1)   # (T, 3, Ny, Nx)

    field_names = ["u", "p", "T"]

    # Stage 4: renderer
    subsection("Stage 4: PhysicsRenderer (matplotlib-based)")
    renderer = PhysicsRenderer(RendererConfig(
        resolution = (128, 128),
        fps        = 12,
        sensors    = ["rgb", "thermal", "depth"],
    ))
    render_result = renderer.render(states_np, field_names, sample_dir, "demo")
    print(f"    frames generated per sensor:")
    for sensor, frames in render_result.frames.items():
        print(f"      {sensor:<10s}  shape={frames.shape}  "
              f"dtype={frames.dtype}")

    # Stage 5: camera system
    subsection("Stage 5: CameraSystem (multi-sensor observations)")
    cam_array = MultiCameraArray.from_config(spec_b.sensor_config)
    cam_sys   = CameraSystem(cam_array, spec_b.scenario.domain_bounds)
    cam_obs   = cam_sys.observe_sequence(states_np, field_names)
    print(f"    cameras: {[c.name for c in cam_array.cameras]}")
    for name, seq in cam_obs.items():
        print(f"      {name:<20s}  shape={seq.shape}")

    return render_result, cam_obs


# ===============================================================================
# STAGES 6+7 -- Reality randomization + photorealistic enhancement
# ===============================================================================

def demo_stages_6_7(render_result):
    section("STAGES 6+7 -- Reality Randomizer + Photorealistic Enhancer")

    subsection("Stage 6: RealityRandomizer")
    randomizer = RealityRandomizer(RandomizerConfig(
        seed             = 42,
        p_sensor_noise   = 1.0,
        p_jpeg           = 0.8,
        p_lighting       = 1.0,
        p_shadow         = 0.6,
        p_occlusion      = 0.4,
        p_blur           = 0.5,
        p_vignette       = 0.8,
    ))
    aug_frames = randomizer.augment(render_result.frames)
    print("    augmentation applied to sensors:", list(aug_frames.keys()))
    for sensor in aug_frames:
        orig = render_result.frames[sensor]
        aug  = aug_frames[sensor]
        diff = np.abs(orig.astype(float) - aug.astype(float)).mean()
        print(f"      {sensor:<10s}  mean pixel diff = {diff:.2f}")

    subsection("Stage 7: PhotorealisticEnhancer (stub mode)")
    enhancer = PhotorealisticEnhancer(EnhancerConfig(backend="stub"))
    enhanced  = enhancer.enhance(aug_frames)
    print(f"    backend: {enhancer.cfg.backend}  (identity pass -- install diffusers for diffusion mode)")
    print(f"    frames unchanged: {all(np.array_equal(aug_frames[k], enhanced[k]) for k in enhanced)}")

    # Enhancement hooks available:
    print("    Available backends:")
    print("      'stub'            -- identity (default, always works)")
    print("      'local_diffusion' -- Stable Diffusion img2img (pip install diffusers)")
    print("      'sharpening'      -- CPU unsharp-mask (built-in custom backend)")
    print("      'cosmos'          -- NVIDIA Cosmos API (placeholder)")

    return aug_frames, enhanced


# ===============================================================================
# MAIN FACTORY -- all 9 stages via SyntheticDataFactory
# ===============================================================================

def demo_full_factory():
    section("FULL FACTORY -- SyntheticDataFactory (all 9 stages)")

    factory_dir = OUT_DIR / "factory_dataset"

    factory = SyntheticDataFactory(SyntheticFactoryConfig(
        scenario_input = {
            "domain":   "fluid_dynamics",
            "geometry": "cavity",
            "fluid":    "air",
            "Re":       300,
            "Nx": 64, "Ny": 64,
            "n_steps": 32,
            "name": "demo_cavity",
        },
        n_samples             = 5,
        output_dir            = factory_dir,
        device                = "cpu",
        render_fps            = 12,
        sensors               = ["rgb", "thermal", "depth"],
        render_resolution     = (128, 128),
        apply_randomization   = True,
        apply_enhancement     = False,   # set True + enhancement_backend="local_diffusion" for diffusion
        solver                = "builtin",
        generate_temperature  = True,
        generate_concentration = False,
        store_format          = "npy",   # "zarr" when zarr is installed
        save_vtk              = True,
        verbose               = True,
        seed                  = 0,
    ))

    result = factory.generate_dataset()
    return result, factory_dir


# ===============================================================================
# STAGE 9 -- Training hooks
# ===============================================================================

def demo_stage9(factory_dir: Path):
    section("STAGE 9 -- Training Hooks (PyTorch DataLoaders)")

    hooks = TrainingHooks(factory_dir, device="cpu")

    # 1. Neural Operator: (state_t, state_{t+1}) pairs
    subsection("A. NeuralOperatorDataset  ->  FNO / DeepONet training")
    ds_no = hooks.neural_operator(
        field_names = ["u", "p", "T"],
        horizon     = 1,
        max_samples = 5,
    )
    loader_no = ds_no.to_dataloader(batch_size=2, shuffle=False)
    batch = next(iter(loader_no))
    print(f"    dataset size  : {len(ds_no)} pairs")
    print(f"    state_t shape : {batch['state_t'].shape}   (batch, C, Ny, Nx)")
    print(f"    state_tp1     : {batch['state_tp1'].shape}")
    print(f"    params        : {batch['params'].shape}    (batch, P)")

    # 2. Inverse PINN: video -> PDE parameters
    subsection("B. InversePINNDataset  ->  estimate viscosity / Re from video")
    ds_inv = hooks.inverse_pinn(
        video_sensor = "rgb",
        max_samples  = 5,
    )
    loader_inv = ds_inv.to_dataloader(batch_size=2, shuffle=False)
    batch_inv  = next(iter(loader_inv))
    print(f"    dataset size  : {len(ds_inv)} samples")
    keys_present = [k for k in batch_inv if batch_inv[k] is not None]
    for k in keys_present:
        print(f"    {k:<20s}: {batch_inv[k].shape}")
    print(f"    param names   : {ds_inv.param_names}")

    # 3. Cosmos Encoder: video + fields -> latent embedding
    subsection("C. CosmosEncoderDataset  ->  video <-> field contrastive learning")
    ds_cos = hooks.cosmos_encoder(
        video_sensors = ["rgb"],
        field_names   = ["u", "p", "T"],
        max_samples   = 5,
    )
    loader_cos = ds_cos.to_dataloader(batch_size=2, shuffle=False)
    batch_cos  = next(iter(loader_cos))
    print(f"    dataset size  : {len(ds_cos)} samples")
    for k, v in batch_cos.items():
        if hasattr(v, "shape"):
            print(f"    {k:<20s}: {v.shape}")

    # 4. Physics Decoder: embedding -> physical fields
    subsection("D. PhysicsDecoderDataset  ->  embedding -> (T, p, u) reconstruction")
    ds_dec = hooks.physics_decoder(
        embedding_dim = 512,
        field_names   = ["u", "p", "T"],
        max_samples   = 5,
    )
    loader_dec = ds_dec.to_dataloader(batch_size=2, shuffle=False)
    batch_dec  = next(iter(loader_dec))
    print(f"    dataset size  : {len(ds_dec)} samples")
    print(f"    embedding     : {batch_dec['embedding'].shape}   (batch, D) -- zeros until encoder trained")
    print(f"    fields        : {batch_dec['fields'].shape}   (batch, C, Ny, Nx)")

    return ds_no, ds_inv


# ===============================================================================
# VISUALISATION -- side-by-side pipeline stages
# ===============================================================================

def visualise_pipeline(solver_out, render_result, aug_frames, ds_no, factory_dir):
    section("VISUALISATION -- Pipeline Output Collage")

    fig = plt.figure(figsize=(20, 12), facecolor=DARK)
    gs  = gridspec.GridSpec(3, 5, figure=fig, wspace=0.08, hspace=0.35)

    def ax_style(ax, title):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color="white", fontsize=9, pad=4)
        ax.axis("off")

    # -- Row 0: Physical fields (Stage 2) --------------------------------------
    t_idx = solver_out.n_timesteps // 2   # mid-trajectory snapshot

    field_order = [
        ("velocity",     "Velocity |u|",    "plasma"),
        ("pressure",     "Pressure p",      "RdBu_r"),
        ("temperature",  "Temperature T",   "inferno"),
        ("concentration","Concentration C", "viridis"),
    ]
    col = 0
    for fname, label, cmap in field_order:
        if fname not in solver_out.fields:
            continue
        arr = solver_out.fields[fname]
        snap = arr[t_idx] if arr.ndim == 3 else arr[t_idx, 0]   # (Ny, Nx)
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(snap, cmap=cmap, aspect="auto", origin="lower")
        ax_style(ax, f"Stage 2 -- {label}")
        col += 1
        if col >= 4:
            break

    # Row 0 col 4: velocity vector field (quiver)
    vel = solver_out.fields.get("velocity")
    if vel is not None and vel.shape[1] == 2:
        ax = fig.add_subplot(gs[0, 4])
        ax.set_facecolor(PANEL)
        u2d = vel[t_idx, 0][::4, ::4]
        v2d = vel[t_idx, 1][::4, ::4]
        ax.quiver(u2d, v2d, color=BLUE, scale=15, width=0.005)
        ax_style(ax, "Stage 2 -- Velocity field")

    # -- Row 1: Rendered frames (Stages 4-5) -----------------------------------
    sensor_cols = [("rgb", "Stage 4 -- RGB render", "viridis"),
                   ("thermal", "Stage 4 -- Thermal", "inferno"),
                   ("depth",   "Stage 4 -- Depth", "gray")]
    for c, (sensor, label, cmap) in enumerate(sensor_cols):
        if sensor not in render_result.frames:
            continue
        ax = fig.add_subplot(gs[1, c])
        frame = render_result.frames[sensor][min(t_idx, len(render_result.frames[sensor])-1)]
        ax.imshow(frame, aspect="auto")
        ax_style(ax, label)

    # Row 1 col 3: augmented (Stage 6)
    if "rgb" in aug_frames:
        ax = fig.add_subplot(gs[1, 3])
        frame_aug = aug_frames["rgb"][min(t_idx, len(aug_frames["rgb"])-1)]
        ax.imshow(frame_aug, aspect="auto")
        ax_style(ax, "Stage 6 -- Augmented RGB")

    # Row 1 col 4: frame comparison (original vs aug)
    if "rgb" in render_result.frames and "rgb" in aug_frames:
        ax = fig.add_subplot(gs[1, 4])
        orig = render_result.frames["rgb"][0].astype(float)
        aug  = aug_frames["rgb"][0].astype(float)
        diff = np.abs(orig - aug).mean(axis=-1)
        ax.imshow(diff, cmap="hot", aspect="auto")
        ax_style(ax, "Stage 6 -- Pixel diff (orig vs aug)")

    # -- Row 2: Training data (Stage 9) ----------------------------------------
    # Sample from NeuralOperatorDataset
    if len(ds_no) > 0:
        sample = ds_no[0]
        s_t   = sample["state_t"].numpy()   # (C, Ny, Nx)
        s_tp1 = sample["state_tp1"].numpy()

        for c, (ch, label, cmap) in enumerate([
            (0, "State t  (channel 0)", "viridis"),
            (min(1, s_t.shape[0]-1), "State t  (channel 1)", "plasma"),
        ]):
            ax = fig.add_subplot(gs[2, c])
            ax.imshow(s_t[ch], cmap=cmap, aspect="auto", origin="lower")
            ax_style(ax, f"Stage 9 NO -- {label}")

        # Target
        ax = fig.add_subplot(gs[2, 2])
        ax.imshow(s_tp1[0], cmap="viridis", aspect="auto", origin="lower")
        ax_style(ax, "Stage 9 NO -- State t+1")

        # Residual (prediction target)
        ax = fig.add_subplot(gs[2, 3])
        residual = s_tp1[0] - s_t[0]
        ax.imshow(residual, cmap="RdBu_r", aspect="auto", origin="lower")
        ax_style(ax, "Stage 9 NO -- Residual (t+1 - t)")

    # Col 4 row 2: dataset directory tree text
    ax_tree = fig.add_subplot(gs[2, 4])
    ax_tree.set_facecolor(PANEL)
    ax_tree.axis("off")
    sample_dir = factory_dir / "sample_000000"
    if sample_dir.exists():
        lines = [f.name for f in sorted(sample_dir.iterdir())]
        tree_text = "sample_000000/\n" + "\n".join(f"  {l}" for l in lines)
    else:
        tree_text = "sample_000000/\n  (no files)"
    ax_tree.text(0.05, 0.95, tree_text, transform=ax_tree.transAxes,
                 color="white", fontsize=7.5, family="monospace",
                 verticalalignment="top")
    ax_tree.set_title("Stage 8 -- Dataset structure", color="white", fontsize=9, pad=4)

    # Super-title
    fig.suptitle("Physics Synthetic Data Factory -- Full 9-Stage Pipeline",
                 color="white", fontsize=14, y=0.98)

    out_path = OUT_DIR / "pipeline_collage.png"
    fig.savefig(str(out_path), dpi=140, facecolor=DARK, bbox_inches="tight")
    plt.close(fig)
    print(f"  Collage saved -> {out_path}")


# ===============================================================================
# DATASET MANIFEST SUMMARY
# ===============================================================================

def print_dataset_summary(factory_dir: Path):
    section("DATASET SUMMARY")
    manifest_path = factory_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        print("  (no manifest found)")
        return
    manifest = json.loads(manifest_path.read_text())
    print(f"  Total samples       : {len(manifest)}")
    if manifest:
        entry = manifest[0]
        sdir  = Path(entry["sample_dir"])
        files = sorted(sdir.iterdir()) if sdir.exists() else []
        total_bytes = sum(
            f.stat().st_size for f in files if f.is_file()
        )
        print(f"  Fields per sample   : {list(entry.get('fields', {}).keys())}")
        print(f"  Videos per sample   : {list(entry.get('videos', {}).keys())}")
        print(f"  Disk size (sample 0): {total_bytes / 1024:.1f} KB")
        print(f"\n  Canonical file layout (sample_000000/):")
        for f in files:
            size_str = f"{f.stat().st_size:>10,} B" if f.is_file() else "         dir"
            print(f"    {f.name:<45s} {size_str}")


# ===============================================================================
# QUICK RECIPE SECTION
# ===============================================================================

def print_recipes():
    section("QUICK RECIPES -- copy-paste ready")
    print("""
  # -- Recipe 1: One-liner pipe flow ---------------------------------------
  from pinneapple_worldmodel import generate_pipe_flow
  result = generate_pipe_flow(n_samples=50, output_dir="./pipe_dataset")

  # -- Recipe 2: Custom NS2D cavity with OpenFOAM backend ------------------
  from pinneapple_worldmodel import SyntheticDataFactory, SyntheticFactoryConfig
  result = SyntheticDataFactory(SyntheticFactoryConfig(
      scenario_input    = {"domain": "fluid_dynamics", "geometry": "cavity",
                           "fluid": "air", "Re": 5000},
      solver            = "openfoam",
      openfoam_case_dir = "./my_case_template",
      n_samples         = 200,
      sensors           = ["rgb", "thermal"],
      apply_randomization = True,
      output_dir        = "./ns_dataset",
  )).generate_dataset()

  # -- Recipe 3: Train a Neural Operator on the dataset --------------------
  from pinneapple_worldmodel import TrainingHooks
  from pinneapple_neural.architectures import FNO

  hooks  = TrainingHooks("./ns_dataset")
  ds     = hooks.neural_operator(field_names=["u","v","p"], horizon=1)
  loader = ds.to_dataloader(batch_size=16, shuffle=True)
  model  = FNO(in_channels=3, out_channels=3, modes=16, width=64)
  # ... standard PyTorch training loop ...

  # -- Recipe 4: Inverse PINN -- estimate Re from video ---------------------
  hooks  = TrainingHooks("./ns_dataset")
  ds_inv = hooks.inverse_pinn(video_sensor="rgb")
  loader = ds_inv.to_dataloader(batch_size=8)
  # video_rgb (T,H,W,3) -> encoder -> params [Re]

  # -- Recipe 5: Photorealistic enhancement with diffusion ------------------
  from pinneapple_worldmodel import SyntheticFactoryConfig
  cfg = SyntheticFactoryConfig(
      scenario_input       = "heat_2d",
      apply_enhancement    = True,
      enhancement_backend  = "local_diffusion",   # pip install diffusers
      enhancement_strength = 0.20,                # keep physical structure
      device               = "cuda",
  )
""")


# ===============================================================================
# MAIN
# ===============================================================================

def main():
    t0 = time.time()
    print("\n" + "=" * 66)
    print("  Physics Synthetic Data Factory for Physical AI")
    print("  PINNeAPPle -- Full Pipeline Example")
    print("=" * 66)

    # Stage 1
    spec_a, spec_b, spec_c = demo_stage1()

    # Stages 2 + 3 (standalone)
    solver_out, sdir = demo_stages_2_3(spec_b)

    # Stages 4 + 5
    render_result, cam_obs = demo_stages_4_5(solver_out, spec_b)

    # Stages 6 + 7
    aug_frames, enhanced = demo_stages_6_7(render_result)

    # Stages 1-8 all-in-one via SyntheticDataFactory
    result, factory_dir = demo_full_factory()

    # Stage 9
    ds_no, ds_inv = demo_stage9(factory_dir)

    # Visualisation
    visualise_pipeline(solver_out, render_result, aug_frames, ds_no, factory_dir)

    # Summary
    print_dataset_summary(factory_dir)

    # Recipes
    print_recipes()

    elapsed = time.time() - t0
    section(f"DONE -- total elapsed: {elapsed:.1f}s")
    print(f"  All outputs in: {OUT_DIR}")
    print(f"  Factory dataset: {factory_dir}")
    print()


if __name__ == "__main__":
    main()
