from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    artifacts = Path("artifacts")
    lb_path = artifacts / "leaderboard.json"
    if not lb_path.exists():
        raise SystemExit("leaderboard.json não encontrado. Rode benchmarks primeiro.")

    rows = json.loads(lb_path.read_text(encoding="utf-8"))
    df = pd.DataFrame(rows)

    task_id = "flow_obstacle_2d"
    df = df[df["task_id"] == task_id].copy()

    sort_by = "test_pde_rms"
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=True, na_position="last")

    out_csv = artifacts / f"compare_{task_id}.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== Comparação (top) ===")
    cols = [c for c in ["run_name", "backend", "test_pde_rms", "test_div_rms", "bc_mse", "test_l2_uv"] if c in df.columns]
    print(df[cols].head(20).to_string(index=False))

    for metric in ["test_pde_rms", "test_div_rms", "bc_mse", "test_l2_uv"]:
        if metric not in df.columns:
            continue
        dd = df[["run_name", "backend", metric]].dropna()
        if len(dd) == 0:
            continue

        plt.figure()
        x = range(len(dd))
        plt.bar(x, dd[metric].to_numpy())
        plt.xticks(list(x), [f"{a}\n({b})" for a, b in zip(dd["run_name"], dd["backend"])], rotation=45, ha="right")
        plt.ylabel(metric)
        plt.title(f"{task_id} - {metric}")
        plt.tight_layout()
        plt.savefig(artifacts / f"{task_id}_{metric}.png", dpi=200)
        plt.close()

    print(f"\n✅ CSV saved in: {out_csv}")
    print(f"✅ Plots saved in: {artifacts}/ (arquivos PNG)")


if __name__ == "__main__":
    main()