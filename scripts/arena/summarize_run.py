from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--artifacts", default="data/artifacts")
    args = ap.parse_args()

    run_dir = Path(args.artifacts) / "runs" / args.run_id
    rep = run_dir / "report.json"
    summ = run_dir / "summary.json"
    metr = run_dir / "metrics.json"

    if rep.exists():
        print(rep.read_text(encoding="utf-8"))
        return 0

    out = {}
    if summ.exists():
        out["summary"] = json.loads(summ.read_text(encoding="utf-8"))
    if metr.exists():
        out["metrics"] = json.loads(metr.read_text(encoding="utf-8"))

    if not out:
        raise FileNotFoundError(f"No report/summary/metrics found in {run_dir}")

    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
