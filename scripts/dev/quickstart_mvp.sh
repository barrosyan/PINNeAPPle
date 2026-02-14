#!/usr/bin/env bash
set -euo pipefail

echo "=== PINNeAPPle MVP Quickstart (Flow Obstacle 2D) ==="

python scripts/dev/sanity_check_install.py

echo
echo "1) Validate bundle..."
python scripts/mvp/validate_bundle.py

echo
echo "2) Run native benchmark..."
python scripts/arena/run_benchmark.py --run configs/arena/runs/vanilla_pinn_native.yaml

echo
echo "3) Show leaderboard..."
python scripts/arena/leaderboard.py

echo
echo "Done."
