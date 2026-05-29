"""Run all PINNeAPPle visualization examples.

Usage
-----
    python -m examples.visualizations.run_all            # all 6 scripts
    python -m examples.visualizations.run_all 1 3 5      # specific scripts

Output files (current directory):
    viz_01_flow_cylinder.png
    viz_02_heat_2d.png
    viz_03_vortex_dynamics.png
    viz_04_phase_field.png
    viz_05_wave_2d.png
    viz_06_structural.png

Notes
-----
  • Scripts 01, 02, 05, 06 train a PINN — expect ~5 minutes each on CPU,
    ~30 seconds on GPU.
  • Scripts 03 and 04 use analytical solutions / pseudospectral methods —
    no training, they run in seconds.
"""

import sys
import importlib
import time

# Ensure UTF-8 output on Windows (box-drawing characters in sub-scripts)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPTS = {
    1: ("viz_01_flow_cylinder",  "Potential Flow Past Cylinder"),
    2: ("viz_02_heat_2d",        "2D Heat Equation"),
    3: ("viz_03_vortex_dynamics","Vortex Dynamics (Lamb-Oseen)"),
    4: ("viz_04_phase_field",    "Allen-Cahn Phase Field"),
    5: ("viz_05_wave_2d",        "2D Wave Equation (3D Surface)"),
    6: ("viz_06_structural",     "Structural Mechanics — Plate + Von Mises"),
}


def run(nums: list[int]) -> None:
    for n in nums:
        mod_name, title = SCRIPTS[n]
        print(f"\n{'='*58}")
        print(f"  [{n}/6]  {title}")
        print(f"{'='*58}")
        t0  = time.perf_counter()
        mod = importlib.import_module(f"examples.visualizations.{mod_name}")
        mod.main()
        print(f"  Completed in {time.perf_counter() - t0:.1f}s")


def main() -> None:
    if len(sys.argv) > 1:
        nums = [int(a) for a in sys.argv[1:] if a.isdigit()]
    else:
        nums = list(SCRIPTS.keys())

    invalid = [n for n in nums if n not in SCRIPTS]
    if invalid:
        print(f"  Unknown script numbers: {invalid}. Valid: 1–6.")
        sys.exit(1)

    print("\n  PINNeAPPle Physics Visualization Suite")
    print(f"  Running scripts: {nums}\n")
    run(nums)
    print("\n  All done. PNG files saved in current directory.")


if __name__ == "__main__":
    main()
