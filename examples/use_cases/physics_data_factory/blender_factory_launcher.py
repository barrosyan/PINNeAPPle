# -*- coding: utf-8 -*-
"""Blender Factory Render — Launcher
=====================================

Orchestrates the full pipeline:

  1. Find Blender executable (auto-detects portable installation)
  2. Run blender --background --python blender_factory_render.py
  3. Assemble PNG frames into MP4 via imageio-ffmpeg

Usage
-----
  # Full render (192 frames @ 256 samples, ~8 min CPU)
  python blender_factory_launcher.py

  # Quick preview (48 frames @ 64 samples, ~2 min CPU)
  python blender_factory_launcher.py --frames 48 --samples 64

  # Assemble already-rendered frames into MP4
  python blender_factory_launcher.py --assemble-only

  # Custom resolution + samples
  python blender_factory_launcher.py --width 1920 --height 1080 --samples 512

Options
-------
  --frames N       Number of frames to render    (default: 192)
  --fps N          Frames per second              (default: 24)
  --samples N      Cycles samples per pixel       (default: 256)
  --width W        Render width                   (default: 1920)
  --height H       Render height                  (default: 1080)
  --output DIR     Output directory               (default: ./outputs/blender)
  --assemble-only  Skip Blender, just make MP4
  --blender PATH   Path to blender.exe            (auto-detected)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def find_blender() -> Optional[Path]:
    # 1. Check PATH
    b = shutil.which("blender") or shutil.which("blender.exe")
    if b:
        return Path(b)
    # 2. Check user home portable install
    candidates = [
        Path.home() / "blender-4.2" / "blender.exe",
        Path.home() / "blender-4.1" / "blender.exe",
        Path.home() / "blender-4.3" / "blender.exe",
        Path("C:/Program Files/Blender Foundation/Blender 4.2/blender.exe"),
        Path("C:/Program Files/Blender Foundation/Blender 4.1/blender.exe"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


from typing import Optional


def render(args: argparse.Namespace) -> Path:
    blender = args.blender or find_blender()
    if blender is None:
        print("ERROR: Blender not found. Install from https://blender.org or pass --blender PATH")
        sys.exit(1)
    blender = Path(blender)
    print(f"Using Blender: {blender}")

    script  = HERE / "blender_factory_render.py"
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(blender), "--background",
        "--python", str(script),
        "--",
        "--frames",  str(args.frames),
        "--fps",     str(args.fps),
        "--samples", str(args.samples),
        "--width",   str(args.width),
        "--height",  str(args.height),
        "--output",  str(out_dir),
    ]
    print("\nStarting Blender render...")
    print("Command:", " ".join(cmd))
    print(f"Output:  {out_dir}")
    print(f"Frames:  {args.frames}  |  Samples: {args.samples}  |  {args.width}x{args.height}")
    print("\nBlender console output:")
    print("-" * 60)

    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True, encoding="utf-8", errors="replace")
    frame_count = 0
    for line in proc.stdout:
        line_s = line.rstrip()
        # Show render progress lines
        if any(kw in line_s for kw in ["Fra:", "Rendered", "Saving", "Error",
                                        "WARNING", "Building", "Scene", "Cycles"]):
            print(line_s)
            if "Fra:" in line_s:
                frame_count += 1
                elapsed = time.time() - t0
                if frame_count > 0:
                    eta = elapsed / frame_count * (args.frames - frame_count)
                    print(f"  -> {frame_count}/{args.frames} frames  "
                          f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")
        elif "ERROR" in line_s.upper():
            print(line_s)

    proc.wait()
    elapsed = time.time() - t0
    print("-" * 60)
    if proc.returncode != 0:
        print(f"ERROR: Blender exited with code {proc.returncode}")
        sys.exit(proc.returncode)
    print(f"Render complete in {elapsed:.0f}s  ({elapsed/args.frames:.1f}s/frame)")
    return out_dir


def assemble_mp4(out_dir: Path, fps: int, width: int, height: int) -> Path:
    """Assemble PNG frames from Blender into an MP4 file."""
    frames_paths = sorted(out_dir.glob("frame_*.png"))
    if not frames_paths:
        print(f"ERROR: No frame_*.png files found in {out_dir}")
        sys.exit(1)

    print(f"\nAssembling {len(frames_paths)} frames into MP4...")
    mp4_path = out_dir / "factory_photorealistic.mp4"

    try:
        import imageio.v3 as iio
        from PIL import Image

        frames = []
        for i, fp in enumerate(frames_paths):
            img = np.array(Image.open(str(fp)).convert("RGB"))
            frames.append(img)
            if (i+1) % 24 == 0:
                print(f"  Loading {i+1}/{len(frames_paths)} frames...", end="\r")

        print(f"\n  Encoding MP4 @ {fps} fps...")
        iio.imwrite(str(mp4_path), np.stack(frames), fps=fps)
        print(f"  MP4 saved: {mp4_path}  ({mp4_path.stat().st_size/1024/1024:.1f} MB)")
        return mp4_path

    except Exception as e:
        print(f"  imageio failed: {e}")
        print("  Trying ffmpeg directly...")
        pattern = str(out_dir / "frame_%04d.png")
        ffmpeg  = shutil.which("ffmpeg")
        if ffmpeg:
            subprocess.run([
                ffmpeg, "-y", "-framerate", str(fps),
                "-i", pattern,
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p",
                str(mp4_path)
            ], check=True)
            print(f"  MP4 saved via ffmpeg: {mp4_path}")
            return mp4_path
        print("  ERROR: No video encoder available. PNG frames are in:", out_dir)
        return out_dir


def main():
    ap = argparse.ArgumentParser(description="Blender Factory Render Launcher")
    ap.add_argument("--frames",       type=int,   default=192)
    ap.add_argument("--fps",          type=int,   default=24)
    ap.add_argument("--samples",      type=int,   default=256)
    ap.add_argument("--width",        type=int,   default=1920)
    ap.add_argument("--height",       type=int,   default=1080)
    ap.add_argument("--output",       type=str,   default=str(HERE / "outputs" / "blender"))
    ap.add_argument("--assemble-only",action="store_true",
                    help="Skip Blender render, just assemble existing PNG frames into MP4")
    ap.add_argument("--blender",      type=str,   default=None,
                    help="Path to blender executable (auto-detected if not given)")
    args = ap.parse_args()

    out_dir = Path(args.output)

    if not args.assemble_only:
        out_dir = render(args)

    mp4 = assemble_mp4(out_dir, fps=args.fps, width=args.width, height=args.height)

    print("\n" + "="*60)
    print("  DONE")
    print(f"  Video: {mp4}")
    print(f"  Frames: {out_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
